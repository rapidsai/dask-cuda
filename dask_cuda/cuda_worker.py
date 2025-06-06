from __future__ import absolute_import, division, print_function

import asyncio
import atexit
import logging
import os
import warnings

from toolz import valmap

import dask
from distributed import Nanny
from distributed.core import Server
from distributed.deploy.cluster import Cluster
from distributed.proctitle import (
    enable_proctitle_on_children,
    enable_proctitle_on_current,
)
from distributed.worker_memory import parse_memory_limit

from .device_host_file import DeviceHostFile
from .initialize import initialize
from .plugins import CPUAffinity, CUDFSetup, PreImport, RMMSetup
from .proxify_host_file import ProxifyHostFile
from .utils import (
    cuda_visible_devices,
    get_cpu_affinity,
    get_n_gpus,
    get_ucx_config,
    nvml_device_index,
    parse_device_memory_limit,
)


class CUDAWorker(Server):
    def __init__(
        self,
        scheduler=None,
        host=None,
        nthreads=1,
        name=None,
        memory_limit="auto",
        device_memory_limit="auto",
        enable_cudf_spill=False,
        cudf_spill_stats=0,
        rmm_pool_size=None,
        rmm_maximum_pool_size=None,
        rmm_managed_memory=False,
        rmm_async=False,
        rmm_allocator_external_lib_list=None,
        rmm_release_threshold=None,
        rmm_log_directory=None,
        rmm_track_allocations=False,
        pid_file=None,
        resources=None,
        dashboard=True,
        dashboard_address=":0",
        local_directory=None,
        shared_filesystem=None,
        scheduler_file=None,
        interface=None,
        preload=[],
        dashboard_prefix=None,
        security=None,
        enable_tcp_over_ucx=None,
        enable_infiniband=None,
        enable_nvlink=None,
        enable_rdmacm=None,
        jit_unspill=None,
        worker_class=None,
        pre_import=None,
        **kwargs,
    ):
        # Required by RAPIDS libraries (e.g., cuDF) to ensure no context
        # initialization happens before we can set CUDA_VISIBLE_DEVICES
        os.environ["RAPIDS_NO_INITIALIZE"] = "True"

        enable_proctitle_on_current()
        enable_proctitle_on_children()

        try:
            nprocs = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
        except KeyError:
            nprocs = get_n_gpus()

        if nthreads < 1:
            raise ValueError("nthreads must be higher than 0.")

        # Set nthreads=1 when parsing mem_limit since it only depends on nprocs
        logger = logging.getLogger(__name__)
        memory_limit = parse_memory_limit(
            memory_limit=memory_limit, nthreads=1, total_cores=nprocs, logger=logger
        )

        if pid_file:
            with open(pid_file, "w") as f:
                f.write(str(os.getpid()))

            def del_pid_file():
                if os.path.exists(pid_file):
                    os.remove(pid_file)

            atexit.register(del_pid_file)

        if resources:
            resources = resources.replace(",", " ").split()
            resources = dict(pair.split("=") for pair in resources)
            resources = valmap(float, resources)
        else:
            resources = None

        preload_argv = kwargs.pop("preload_argv", [])
        kwargs = {"worker_port": None, "listen_address": None, **kwargs}

        if (
            scheduler is None
            and scheduler_file is None
            and dask.config.get("scheduler-address", None) is None
        ):
            raise ValueError(
                "No scheduler specified. A scheduler can be specified by "
                "passing an address through the SCHEDULER argument or "
                "'dask.scheduler-address' config option, or by passing the "
                "location of a scheduler file through the --scheduler-file "
                "option"
            )

        if isinstance(scheduler, Cluster):
            scheduler = scheduler.scheduler_address

        if interface and host:
            raise ValueError("Can not specify both interface and host")

        if rmm_pool_size is not None or rmm_managed_memory:
            try:
                import rmm  # noqa F401
            except ImportError:
                raise ValueError(
                    "RMM pool requested but module 'rmm' is not available. "
                    "For installation instructions, please see "
                    "https://github.com/rapidsai/rmm"
                )  # pragma: no cover
        else:
            if enable_nvlink:
                warnings.warn(
                    "When using NVLink we recommend setting a "
                    "`rmm_pool_size`.  Please see: "
                    "https://docs.rapids.ai/api/dask-cuda/nightly/ucx/ "
                    "for more details"
                )

        if enable_nvlink and rmm_managed_memory:
            raise ValueError(
                "RMM managed memory and NVLink are currently incompatible."
            )

        # Ensure this parent dask-cuda-worker process uses the same UCX
        # configuration as child worker processes created by it.
        initialize(
            create_cuda_context=False,
            enable_tcp_over_ucx=enable_tcp_over_ucx,
            enable_infiniband=enable_infiniband,
            enable_nvlink=enable_nvlink,
            enable_rdmacm=enable_rdmacm,
        )

        if jit_unspill is None:
            jit_unspill = dask.config.get("jit-unspill", default=False)
        if device_memory_limit is None and memory_limit is None:
            data = lambda _: {}
        elif jit_unspill:
            if enable_cudf_spill:
                warnings.warn(
                    "Enabling cuDF spilling and JIT-Unspill together is not "
                    "safe, consider disabling JIT-Unspill."
                )

            data = lambda i: (
                ProxifyHostFile,
                {
                    "device_memory_limit": parse_device_memory_limit(
                        device_memory_limit, device_index=i
                    ),
                    "memory_limit": memory_limit,
                    "shared_filesystem": shared_filesystem,
                },
            )
        else:
            data = lambda i: (
                DeviceHostFile,
                {
                    "device_memory_limit": parse_device_memory_limit(
                        device_memory_limit, device_index=i
                    ),
                    "memory_limit": memory_limit,
                },
            )

        cudf_spill_warning = dask.config.get("cudf-spill-warning", default=True)
        if enable_cudf_spill and cudf_spill_warning:
            warnings.warn(
                "cuDF spilling is enabled, please ensure the client and scheduler "
                "processes set `CUDF_SPILL=on` as well. To disable this warning "
                "set `DASK_CUDF_SPILL_WARNING=False`."
            )

        self.nannies = [
            Nanny(
                scheduler,
                scheduler_file=scheduler_file,
                nthreads=nthreads,
                dashboard=dashboard,
                dashboard_address=dashboard_address,
                http_prefix=dashboard_prefix,
                resources=resources,
                memory_limit=memory_limit,
                interface=interface,
                host=host,
                preload=(list(preload) or []) + ["dask_cuda.initialize"],
                preload_argv=(list(preload_argv) or []) + ["--create-cuda-context"],
                security=security,
                env={"CUDA_VISIBLE_DEVICES": cuda_visible_devices(i)},
                plugins={
                    CPUAffinity(
                        get_cpu_affinity(nvml_device_index(i, cuda_visible_devices(i)))
                    ),
                    RMMSetup(
                        initial_pool_size=rmm_pool_size,
                        maximum_pool_size=rmm_maximum_pool_size,
                        managed_memory=rmm_managed_memory,
                        async_alloc=rmm_async,
                        release_threshold=rmm_release_threshold,
                        log_directory=rmm_log_directory,
                        track_allocations=rmm_track_allocations,
                        external_lib_list=rmm_allocator_external_lib_list,
                    ),
                    PreImport(pre_import),
                    CUDFSetup(spill=enable_cudf_spill, spill_stats=cudf_spill_stats),
                },
                name=name if nprocs == 1 or name is None else str(name) + "-" + str(i),
                local_directory=local_directory,
                config={
                    "distributed.comm.ucx": get_ucx_config(
                        enable_tcp_over_ucx=enable_tcp_over_ucx,
                        enable_infiniband=enable_infiniband,
                        enable_nvlink=enable_nvlink,
                        enable_rdmacm=enable_rdmacm,
                    )
                },
                data=data(nvml_device_index(i, cuda_visible_devices(i))),
                worker_class=worker_class,
                **kwargs,
            )
            for i in range(nprocs)
        ]

    def __await__(self):
        return self._wait().__await__()

    async def _wait(self):
        await asyncio.gather(*self.nannies)

    async def finished(self):
        await asyncio.gather(*[n.finished() for n in self.nannies])

    async def close(self, timeout=5):
        await asyncio.gather(*[n.close(timeout=timeout) for n in self.nannies])
