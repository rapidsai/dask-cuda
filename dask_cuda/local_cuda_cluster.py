import copy
import os
import warnings

import dask
from dask.utils import parse_bytes
from distributed import LocalCluster, Nanny, Worker
from distributed.worker import parse_memory_limit

from .device_host_file import DeviceHostFile
from .initialize import initialize
from .proxify_host_file import ProxifyHostFile
from .utils import (
    CPUAffinity,
    RMMSetup,
    _ucx_111,
    cuda_visible_devices,
    get_cpu_affinity,
    get_ucx_config,
    get_ucx_net_devices,
    nvml_device_index,
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

    This assigns a different ``CUDA_VISIBLE_DEVICES`` environment variable to each Dask
    worker process.

    For machines with a complex architecture mapping CPUs, GPUs, and network hardware,
    such as NVIDIA DGX-1 and DGX-2, this class creates a local cluster that tries to
    respect this hardware as much as possible.

    Each worker process is automatically assigned the correct CPU cores and network
    interface cards to maximize performance. If UCX and UCX-Py are available, InfiniBand
    and NVLink connections can be used to optimize data transfer performance.

    Parameters
    ----------
    CUDA_VISIBLE_DEVICES : str, list of int, or None, default None
        GPUs to restrict activity to. Can be a string (like ``"0,1,2,3"``), list (like
        ``[0, 1, 2, 3]``), or ``None`` to use all available GPUs.
    n_workers : int or None, default None
        Number of workers. Can be an integer or ``None`` to fall back on the GPUs
        specified by ``CUDA_VISIBLE_DEVICES``. Will override the value of
        ``CUDA_VISIBLE_DEVICES`` if specified.
    threads_per_worker : int, default 1
        Number of threads to be used for each Dask worker process.
    memory_limit : int, float, str, or None, default "auto"
        Bytes of memory per process that the worker can use. Can be an integer (bytes),
        float (fraction of total system memory), string (like ``"5GB"`` or ``"5000M"``),
        or ``"auto"``, 0, or ``None`` for no memory management.
    device_memory_limit : int, float, str, or None, default 0.8
        Size of the CUDA device LRU cache, which is used to determine when the worker
        starts spilling to host memory. Can be an integer (bytes), float (fraction of
        total device memory), string (like ``"5GB"`` or ``"5000M"``), or ``"auto"``, 0,
        or ``None`` to disable spilling to host (i.e. allow full device memory usage).
    local_directory : str or None, default None
        Path on local machine to store temporary files. Can be a string (like
        ``"path/to/files"``) or ``None`` to fall back on the value of
        ``dask.temporary-directory`` in the local Dask configuration, using the current
        working directory if this is not set.
    protocol : str or None, default None
        Protocol to use for communication. Can be a string (like ``"tcp"`` or
        ``"ucx"``), or ``None`` to automatically choose the correct protocol.
    enable_tcp_over_ucx : bool, default False
        Set environment variables to enable TCP over UCX, even if InfiniBand and NVLink
        are not supported or disabled.
    enable_infiniband : bool, default False
        Set environment variables to enable UCX over InfiniBand, requires
        ``protocol="ucx"`` and implies ``enable_tcp_over_ucx=True``.
    enable_nvlink : bool, default False
        Set environment variables to enable UCX over NVLink, requires ``protocol="ucx"``
        and implies ``enable_tcp_over_ucx=True``.
    enable_rdmacm : bool, default False
        Set environment variables to enable UCX RDMA connection manager support,
        requires ``protocol="ucx"`` and ``enable_infiniband=True``.
    ucx_net_devices : str, callable, or None, default None
        Interface(s) used by workers for UCX communication. Can be a string (like
        ``"eth0"`` for NVLink or ``"mlx5_0:1"``/``"ib0"`` for InfiniBand), a callable
        function that takes the index of the current GPU to return an interface name
        (like ``lambda dev: "mlx5_%d:1" % (dev // 2)``), ``"auto"`` (requires
        ``enable_infiniband=True``) to pick the optimal interface per-worker
        based on the system's topology, or ``None`` to stay with the default value of
        ``"all"`` (use all available interfaces).

        .. warning::
            ``"auto"`` requires UCX-Py to be installed and compiled with hwloc support.
            Unexpected errors can occur when using ``"auto"`` if any interfaces are
            disconnected or improperly configured.
    rmm_pool_size : int, str or None, default None
        RMM pool size to initialize each worker with. Can be an integer (bytes), string
        (like ``"5GB"`` or ``"5000M"``), or ``None`` to disable RMM pools.

        .. note::
            This size is a per-worker configuration, and not cluster-wide.
    rmm_managed_memory : bool, default False
        Initialize each worker with RMM and set it to use managed memory. If disabled,
        RMM may still be used by specifying ``rmm_pool_size``.

        .. warning::
            Managed memory is currently incompatible with NVLink. Trying to enable both
            will result in an exception.
    rmm_async: bool, default False
        Initialize each worker withh RMM and set it to use RMM's asynchronous allocator.
        See ``rmm.mr.CudaAsyncMemoryResource`` for more info.

        .. warning::
            The asynchronous allocator requires CUDA Toolkit 11.2 or newer. It is also
            incompatible with RMM pools and managed memory. Trying to enable both will
            result in an exception.
    rmm_log_directory : str or None, default None
        Directory to write per-worker RMM log files to. The client and scheduler are not
        logged here. Can be a string (like ``"/path/to/logs/"``) or ``None`` to
        disable logging.

        .. note::
            Logging will only be enabled if ``rmm_pool_size`` is specified or
            ``rmm_managed_memory=True``.
    jit_unspill : bool or None, default None
        Enable just-in-time unspilling. Can be a boolean or ``None`` to fall back on
        the value of ``dask.jit-unspill`` in the local Dask configuration, disabling
        unspilling if this is not set.

        .. note::
            This is experimental and doesn't support memory spilling to disk. See
            ``proxy_object.ProxyObject`` and ``proxify_host_file.ProxifyHostFile`` for
            more info.
    log_spilling : bool, default True
        Enable logging of spilling operations directly to ``distributed.Worker`` with an
        ``INFO`` log level.

    Examples
    --------
    >>> from dask_cuda import LocalCUDACluster
    >>> from dask.distributed import Client
    >>> cluster = LocalCUDACluster()
    >>> client = Client(cluster)

    Raises
    ------
    TypeError
        If InfiniBand or NVLink are enabled and ``protocol!="ucx"``.
    ValueError
        If ``ucx_net_devices=""``, if NVLink and RMM managed memory are
        both enabled, if RMM pools / managed memory and asynchronous allocator are both
        enabled, or if ``ucx_net_devices="auto"`` and:

            - UCX-Py is not installed or wasn't compiled with hwloc support; or
            - ``enable_infiniband=False``

    See Also
    --------
    LocalCluster
    """

    def __init__(
        self,
        CUDA_VISIBLE_DEVICES=None,
        n_workers=None,
        threads_per_worker=1,
        memory_limit="auto",
        device_memory_limit=0.8,
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
        rmm_async=False,
        rmm_log_directory=None,
        jit_unspill=None,
        log_spilling=False,
        worker_class=None,
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
            device_memory_limit, device_index=nvml_device_index(0, CUDA_VISIBLE_DEVICES)
        )

        self.rmm_pool_size = rmm_pool_size
        self.rmm_managed_memory = rmm_managed_memory
        self.rmm_async = rmm_async
        if rmm_pool_size is not None or rmm_managed_memory:
            try:
                import rmm  # noqa F401
            except ImportError:
                raise ValueError(
                    "RMM pool or managed memory requested but module 'rmm' "
                    "is not available. For installation instructions, please "
                    "see https://github.com/rapidsai/rmm"
                )  # pragma: no cover
            if rmm_async:
                raise ValueError(
                    "RMM pool and managed memory are incompatible with asynchronous "
                    "allocator"
                )
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

        if not kwargs.pop("processes", True):
            raise ValueError(
                "Processes are necessary in order to use multiple GPUs with Dask"
            )

        if jit_unspill is None:
            self.jit_unspill = dask.config.get("jit-unspill", default=False)
        else:
            self.jit_unspill = jit_unspill

        data = kwargs.pop("data", None)
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
            if _ucx_111:
                warnings.warn(
                    "Starting with UCX 1.11, `ucx_net_devices='auto' is deprecated, "
                    "it should now be left unspecified for the same behavior. "
                    "Please make sure to read the updated UCX Configuration section in "
                    "https://docs.rapids.ai/api/dask-cuda/nightly/ucx.html, "
                    "where significant performance considerations for InfiniBand with "
                    "UCX 1.11 and above is documented.",
                )
            else:
                try:
                    from ucp._libs.topological_distance import (  # NOQA
                        TopologicalDistance,
                    )
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

        if worker_class is not None:
            from functools import partial

            worker_class = partial(
                LoggedNanny if log_spilling is True else Nanny,
                worker_class=worker_class,
            )

        super().__init__(
            n_workers=0,
            threads_per_worker=threads_per_worker,
            memory_limit=self.host_memory_limit,
            processes=True,
            data=data,
            local_directory=local_directory,
            protocol=protocol,
            worker_class=worker_class,
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
                    CPUAffinity(
                        get_cpu_affinity(nvml_device_index(0, visible_devices))
                    ),
                    RMMSetup(
                        self.rmm_pool_size,
                        self.rmm_managed_memory,
                        self.rmm_async,
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
