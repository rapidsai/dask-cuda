from __future__ import absolute_import, division, print_function

import asyncio
import atexit
import multiprocessing
import os

from toolz import valmap
from tornado.ioloop import IOLoop

from distributed import Nanny
from distributed.config import config
from distributed.proctitle import (
    enable_proctitle_on_children,
    enable_proctitle_on_current,
)
from distributed.utils import parse_bytes
from distributed.worker import parse_memory_limit

from .device_host_file import DeviceHostFile
from .initialize import initialize
from .local_cuda_cluster import cuda_visible_devices
from .utils import (
    CPUAffinity,
    RMMSetup,
    get_cpu_affinity,
    get_device_total_memory,
    get_n_gpus,
    get_ucx_config,
    get_ucx_net_devices,
)


def _get_interface(interface, host, cuda_device_index, ucx_net_devices):
    if host:
        return None
    else:
        return interface or get_ucx_net_devices(
            cuda_device_index=cuda_device_index,
            ucx_net_devices=ucx_net_devices,
            get_openfabrics=False,
            get_network=True,
        )


class CUDAWorker:
    def __init__(
        self,
        scheduler,
        host=None,
        nthreads=0,
        name=None,
        memory_limit="auto",
        device_memory_limit="auto",
        rmm_pool_size=None,
        rmm_managed_memory=False,
        pid_file=None,
        resources=None,
        dashboard=True,
        dashboard_address=":0",
        local_directory=None,
        scheduler_file=None,
        interface=None,
        death_timeout=None,
        preload=[],
        dashboard_prefix=None,
        security=None,
        enable_tcp_over_ucx=False,
        enable_infiniband=False,
        enable_nvlink=False,
        enable_rdmacm=False,
        net_devices=None,
        **kwargs,
    ):
        enable_proctitle_on_current()
        enable_proctitle_on_children()

        try:
            nprocs = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
        except KeyError:
            nprocs = get_n_gpus()

        if not nthreads:
            nthreads = min(1, multiprocessing.cpu_count() // nprocs)

        memory_limit = parse_memory_limit(memory_limit, nthreads, total_cores=nprocs)

        if pid_file:
            with open(pid_file, "w") as f:
                f.write(str(os.getpid()))

            def del_pid_file():
                if os.path.exists(pid_file):
                    os.remove(pid_file)

            atexit.register(del_pid_file)

        services = {}

        if dashboard:
            try:
                from distributed.dashboard import BokehWorker
            except ImportError:
                pass
            else:
                if dashboard_prefix:
                    result = (BokehWorker, {"prefix": dashboard_prefix})
                else:
                    result = BokehWorker
                services[("dashboard", dashboard_address)] = result

        if resources:
            resources = resources.replace(",", " ").split()
            resources = dict(pair.split("=") for pair in resources)
            resources = valmap(float, resources)
        else:
            resources = None

        loop = IOLoop.current()

        preload_argv = kwargs.get("preload_argv", [])
        kwargs = {"worker_port": None, "listen_address": None}
        t = Nanny

        if not scheduler and not scheduler_file and "scheduler-address" not in config:
            raise ValueError(
                "Need to provide scheduler address like\n"
                "dask-worker SCHEDULER_ADDRESS:8786"
            )

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
            if rmm_pool_size is not None:
                rmm_pool_size = parse_bytes(rmm_pool_size)

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
            net_devices=net_devices,
            cuda_device_index=0,
        )

        self.nannies = [
            t(
                scheduler,
                scheduler_file=scheduler_file,
                nthreads=nthreads,
                services=services,
                loop=loop,
                resources=resources,
                memory_limit=memory_limit,
                interface=_get_interface(interface, host, i, net_devices),
                host=host,
                preload=(list(preload) or []) + ["dask_cuda.initialize"],
                preload_argv=(list(preload_argv) or []) + ["--create-cuda-context"],
                security=security,
                env={"CUDA_VISIBLE_DEVICES": cuda_visible_devices(i)},
                plugins={
                    CPUAffinity(get_cpu_affinity(i)),
                    RMMSetup(rmm_pool_size, rmm_managed_memory),
                },
                name=name if nprocs == 1 or not name else name + "-" + str(i),
                local_directory=local_directory,
                config={
                    "ucx": get_ucx_config(
                        enable_tcp_over_ucx=enable_tcp_over_ucx,
                        enable_infiniband=enable_infiniband,
                        enable_nvlink=enable_nvlink,
                        enable_rdmacm=enable_rdmacm,
                        net_devices=net_devices,
                        cuda_device_index=i,
                    )
                },
                data=(
                    DeviceHostFile,
                    {
                        "device_memory_limit": get_device_total_memory(index=i)
                        if (
                            device_memory_limit == "auto"
                            or device_memory_limit == int(0)
                        )
                        else parse_bytes(device_memory_limit),
                        "memory_limit": memory_limit,
                        "local_directory": local_directory,
                    },
                ),
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

    async def close(self, timeout=2):
        await asyncio.gather(*[n.close(timeout=timeout) for n in self.nannies])
