import copy
import os
import warnings

import dask
from dask.distributed import LocalCluster, Nanny, Worker
from distributed.utils import parse_bytes
from distributed.worker import parse_memory_limit

from .device_host_file import DeviceHostFile
from .initialize import initialize
from .proxify_host_file import ProxifyHostFile
from .utils import (
    CPUAffinity,
    RMMSetup,
    cuda_visible_devices,
    get_cpu_affinity,
    get_ucx_config,
    get_ucx_net_devices,
    parse_cuda_visible_device,
    parse_device_memory_limit,
)


class LoggedWorker(Worker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def start(self):
        await super().start()
        self.data.set_address(self.address)


class LoggedNanny(Nanny):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, worker_class=LoggedWorker, **kwargs)


class LocalCUDACluster(LocalCluster):
    """A variant of ``dask.distributed.LocalCluster`` that uses one GPU per process.

    This assigns a different ``CUDA_VISIBLE_DEVICES`` environment variable to each
    worker process.

    For machines with a complex architecture mapping CPUs, GPUs, and network
    hardware, such as NVIDIA DGX-1 and DGX-2, this class creates a local
    cluster that tries to respect this hardware as much as possible.

    It creates one Dask worker process per GPU, and assigns each worker process
    the correct CPU cores and Network interface cards to maximize performance.
    If UCX and UCX-Py are also available, it's possible to use InfiniBand and
    NVLink connections for optimal data transfer performance.

    Parameters
    ----------
    CUDA_VISIBLE_DEVICES: str or list
        String or list ``"0,1,2,3"`` or ``[0, 1, 2, 3]`` to restrict activity to
        different GPUs.
    device_memory_limit: int, float or str
        Specifies the size of the CUDA device LRU cache, which is used to
        determine when the worker starts spilling to host memory.  This can be
        a float (fraction of total device memory), an integer (bytes), a string
        (like ``"5GB"`` or ``"5000M"``), or ``"auto"``, ``0``, or ``None`` to disable
        spilling to host (i.e., allow full device memory usage). Default is ``0.8``,
        80% of the worker's total device memory.
    interface: str
        The external interface used to connect to the scheduler, usually
        an ethernet interface is used for connection, and not an InfiniBand
        interface (if one is available).
    threads_per_worker: int
        Number of threads to be used for each CUDA worker process.
    protocol: str
        Protocol to use for communication, e.g., ``"tcp"`` or ``"ucx"``.
    enable_tcp_over_ucx: bool
        Set environment variables to enable TCP over UCX, even if InfiniBand
        and NVLink are not supported or disabled.
    enable_infiniband: bool
        Set environment variables to enable UCX InfiniBand support, requires
        ``protocol="ucx"`` and implies ``enable_tcp_over_ucx=True``.
    enable_rdmacm: bool
        Set environment variables to enable UCX RDMA connection manager support,
        requires ``protocol="ucx"`` and ``enable_infiniband=True``.
    enable_nvlink: bool
        Set environment variables to enable UCX NVLink support, requires
        ``protocol="ucx"`` and implies ``enable_tcp_over_ucx=True``.
    ucx_net_devices: None, callable or str
        When ``None`` (default), ``"UCX_NET_DEVICES"`` will be left to its default.
        If callable, the function must take exactly one argument (the index of
        current GPU) that will be used to get the interface name, such as
        ``lambda dev: "mlx5_%d:1" % (dev // 2)``, returning ``"mlx5_1:1"`` for
        GPU 3, for example. If it's a string, it must be a non-empty string
        with the interface name, such as ``"eth0"`` or ``"auto"`` to allow for
        automatically choosing the closest interface based on the system's
        topology.

        .. warning::
            ``"auto"`` requires UCX-Py to be installed and compiled with hwloc
            support. Additionally that will always use the closest interface, and
            that may cause unexpected errors if that interface is not properly
            configured or is disconnected, for that reason it's limited to
            InfiniBand only and will still cause unpredictable errors if **all**
            interfaces are not connected and properly configured.
    rmm_pool_size: None, int or str
        When ``None`` (default), no RMM pool is initialized. If a different value
        is given, it can be an integer (bytes) or string (like ``"5GB"`` or
        ``"5000M"``).

        .. note::
            The size is a per worker (i.e., per GPU) configuration, and not
            cluster-wide!
    rmm_managed_memory: bool
        If ``True``, initialize each worker with RMM and set it to use managed
        memory. If ``False``, RMM may still be used if ``rmm_pool_size`` is specified,
        but in that case with default (non-managed) memory type.

        .. warning::
            Managed memory is currently incompatible with NVLink, trying to enable
            both will result in an exception.
    rmm_log_directory: str
        Directory to write per-worker RMM log files to; the client and scheduler
        are not logged here. Logging will only be enabled if ``rmm_pool_size`` or
        ``rmm_managed_memory`` are specified.
    jit_unspill: bool
        If ``True``, enable just-in-time unspilling. This is experimental and doesn't
        support memory spilling to disk. Please see ``proxy_object.ProxyObject`` and
        ``proxify_host_file.ProxifyHostFile``.
    log_spilling: bool
        If True, all spilling operations will be logged directly to
        distributed.worker with an INFO loglevel. This will eventually be
        replaced by a Dask configuration flag.


    Examples
    --------
    >>> from dask_cuda import LocalCUDACluster
    >>> from dask.distributed import Client
    >>> cluster = LocalCUDACluster()
    >>> client = Client(cluster)

    Raises
    ------
    TypeError
        If ``enable_infiniband`` or ``enable_nvlink`` is ``True`` and protocol is not
        ``"ucx"``.
    ValueError
        If ``ucx_net_devices`` is an empty string, or if it is ``"auto"`` and UCX-Py is
        not installed, or if it is ``"auto"`` and ``enable_infiniband=False``, or UCX-Py
        wasn't compiled with hwloc support, or both RMM managed memory and
        NVLink are enabled.

    See Also
    --------
    LocalCluster
    """

    def __init__(
        self,
        n_workers=None,
        threads_per_worker=1,
        processes=True,
        memory_limit="auto",
        device_memory_limit=0.8,
        CUDA_VISIBLE_DEVICES=None,
        data=None,
        local_directory=None,
        protocol=None,
        enable_tcp_over_ucx=False,
        enable_infiniband=False,
        enable_nvlink=False,
        enable_rdmacm=False,
        ucx_net_devices=None,
        rmm_pool_size=None,
        rmm_managed_memory=False,
        rmm_log_directory=None,
        jit_unspill=None,
        log_spilling=False,
        **kwargs,
    ):
        # Required by RAPIDS libraries (e.g., cuDF) to ensure no context
        # initialization happens before we can set CUDA_VISIBLE_DEVICES
        os.environ["RAPIDS_NO_INITIALIZE"] = "True"

        if threads_per_worker < 1:
            raise ValueError("threads_per_worker must be higher than 0.")

        if CUDA_VISIBLE_DEVICES is None:
            CUDA_VISIBLE_DEVICES = cuda_visible_devices(0)
        if isinstance(CUDA_VISIBLE_DEVICES, str):
            CUDA_VISIBLE_DEVICES = CUDA_VISIBLE_DEVICES.split(",")
        CUDA_VISIBLE_DEVICES = list(
            map(parse_cuda_visible_device, CUDA_VISIBLE_DEVICES)
        )
        if n_workers is None:
            n_workers = len(CUDA_VISIBLE_DEVICES)
        if n_workers < 1:
            raise ValueError("Number of workers cannot be less than 1.")
        self.host_memory_limit = parse_memory_limit(
            memory_limit, threads_per_worker, n_workers
        )
        self.device_memory_limit = parse_device_memory_limit(
            device_memory_limit, device_index=0
        )

        self.rmm_pool_size = rmm_pool_size
        self.rmm_managed_memory = rmm_managed_memory
        if rmm_pool_size is not None or rmm_managed_memory:
            try:
                import rmm  # noqa F401
            except ImportError:
                raise ValueError(
                    "RMM pool or managed memory requested but module 'rmm' "
                    "is not available. For installation instructions, please "
                    "see https://github.com/rapidsai/rmm"
                )  # pragma: no cover
            if self.rmm_pool_size is not None:
                self.rmm_pool_size = parse_bytes(self.rmm_pool_size)
        else:
            if enable_nvlink:
                warnings.warn(
                    "When using NVLink we recommend setting a "
                    "`rmm_pool_size`. Please see: "
                    "https://dask-cuda.readthedocs.io/en/latest/ucx.html"
                    "#important-notes for more details"
                )

        self.rmm_log_directory = rmm_log_directory

        if not processes:
            raise ValueError(
                "Processes are necessary in order to use multiple GPUs with Dask"
            )

        if jit_unspill is None:
            self.jit_unspill = dask.config.get("jit-unspill", default=False)
        else:
            self.jit_unspill = jit_unspill

        if data is None:
            if self.jit_unspill:
                data = (
                    ProxifyHostFile,
                    {"device_memory_limit": self.device_memory_limit,},
                )
            else:
                data = (
                    DeviceHostFile,
                    {
                        "device_memory_limit": self.device_memory_limit,
                        "memory_limit": self.host_memory_limit,
                        "local_directory": local_directory
                        or dask.config.get("temporary-directory")
                        or os.getcwd(),
                        "log_spilling": log_spilling,
                    },
                )

        if enable_tcp_over_ucx or enable_infiniband or enable_nvlink:
            if protocol is None:
                protocol = "ucx"
            elif protocol != "ucx":
                raise TypeError("Enabling InfiniBand or NVLink requires protocol='ucx'")

        if ucx_net_devices == "auto":
            try:
                from ucp._libs.topological_distance import TopologicalDistance  # NOQA
            except ImportError:
                raise ValueError(
                    "ucx_net_devices set to 'auto' but UCX-Py is not "
                    "installed or it's compiled without hwloc support"
                )
        elif ucx_net_devices == "":
            raise ValueError("ucx_net_devices can not be an empty string")
        self.ucx_net_devices = ucx_net_devices
        self.set_ucx_net_devices = enable_infiniband
        self.host = kwargs.get("host", None)

        initialize(
            enable_tcp_over_ucx=enable_tcp_over_ucx,
            enable_nvlink=enable_nvlink,
            enable_infiniband=enable_infiniband,
            enable_rdmacm=enable_rdmacm,
            net_devices=ucx_net_devices,
            cuda_device_index=0,
        )

        super().__init__(
            n_workers=0,
            threads_per_worker=threads_per_worker,
            memory_limit=self.host_memory_limit,
            processes=True,
            data=data,
            local_directory=local_directory,
            protocol=protocol,
            worker_class=LoggedNanny if log_spilling is True else Nanny,
            config={
                "ucx": get_ucx_config(
                    enable_tcp_over_ucx=enable_tcp_over_ucx,
                    enable_nvlink=enable_nvlink,
                    enable_infiniband=enable_infiniband,
                    enable_rdmacm=enable_rdmacm,
                )
            },
            **kwargs,
        )

        self.new_spec["options"]["preload"] = self.new_spec["options"].get(
            "preload", []
        ) + ["dask_cuda.initialize"]
        self.new_spec["options"]["preload_argv"] = self.new_spec["options"].get(
            "preload_argv", []
        ) + ["--create-cuda-context"]

        self.cuda_visible_devices = CUDA_VISIBLE_DEVICES
        self.scale(n_workers)
        self.sync(self._correct_state)

    def new_worker_spec(self):
        try:
            name = min(set(self.cuda_visible_devices) - set(self.worker_spec))
        except Exception:
            raise ValueError(
                "Can not scale beyond visible devices", self.cuda_visible_devices
            )

        spec = copy.deepcopy(self.new_spec)
        worker_count = self.cuda_visible_devices.index(name)
        visible_devices = cuda_visible_devices(worker_count, self.cuda_visible_devices)
        spec["options"].update(
            {
                "env": {"CUDA_VISIBLE_DEVICES": visible_devices,},
                "plugins": {
                    CPUAffinity(get_cpu_affinity(worker_count)),
                    RMMSetup(
                        self.rmm_pool_size,
                        self.rmm_managed_memory,
                        self.rmm_log_directory,
                    ),
                },
            }
        )

        if self.set_ucx_net_devices:
            cuda_device_index = visible_devices.split(",")[0]

            net_dev = get_ucx_net_devices(cuda_device_index, self.ucx_net_devices)
            if net_dev is not None:
                spec["options"]["env"]["UCX_NET_DEVICES"] = net_dev
                spec["options"]["config"]["ucx"]["net-devices"] = net_dev

            spec["options"]["interface"] = get_ucx_net_devices(
                cuda_device_index,
                self.ucx_net_devices,
                get_openfabrics=False,
                get_network=True,
            )

        return {name: spec}
