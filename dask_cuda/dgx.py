import os

from dask.distributed import Nanny, SpecCluster, Scheduler
from distributed.system import MEMORY_LIMIT

from .local_cuda_cluster import cuda_visible_devices
from .utils import get_cpu_affinity_list


class CPUAffinity:
    def __init__(self, cores):
        self.cores = cores

    def setup(self, worker=None):
        os.sched_setaffinity(0, self.cores)


def DGX(
    interface="ib",
    dashboard_address=":8787",
    threads_per_worker=1,
    silence_logs=True,
    CUDA_VISIBLE_DEVICES=None,
    **kwargs
):
    """ A Local Cluster for a DGX 1 machine

    NVIDIA's DGX-1 machine has a complex architecture mapping CPUs, GPUs, and
    network hardware.  This function creates a local cluster that tries to
    respect this hardware as much as possible.

    It creates one Dask worker process per GPU, and assigns each worker process
    the correct CPU cores and Network interface cards to maximize performance.

    That being said, things aren't perfect.  Today a DGX has very high
    performance between certain sets of GPUs and not others.  A Dask DGX
    cluster that uses only certain tightly coupled parts of the computer will
    have significantly higher bandwidth than a deployment on the entire thing.

    Parameters
    ----------
    interface: str
        The interface prefix for the infiniband networking cards.  This is
        often "ib"` or "bond".  We will add the numeric suffix 0,1,2,3 as
        appropriate.  Defaults to "ib".
    dashboard_address: str
        The address for the scheduler dashboard.  Defaults to ":8787".
    CUDA_VISIBLE_DEVICES: str
        String like ``"0,1,2,3"`` or ``[0, 1, 2, 3]`` to restrict activity to
        different GPUs

    Examples
    --------
    >>> from dask_cuda import DGX
    >>> from dask.distributed import Client
    >>> cluster = DGX(interface='ib')
    >>> client = Client(cluster)
    """
    if CUDA_VISIBLE_DEVICES is None:
        CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    if isinstance(CUDA_VISIBLE_DEVICES, str):
        CUDA_VISIBLE_DEVICES = CUDA_VISIBLE_DEVICES.split(",")
    CUDA_VISIBLE_DEVICES = list(map(int, CUDA_VISIBLE_DEVICES))
    memory_limit = MEMORY_LIMIT / 8

    spec = {
        i: {
            "cls": Nanny,
            "options": {
                "env": {
                    "CUDA_VISIBLE_DEVICES": cuda_visible_devices(
                        ii, CUDA_VISIBLE_DEVICES
                    ),
                    # 'UCX_NET_DEVICES': 'mlx5_%d:1' % (i // 2)
                    "UCX_TLS": "rc,cuda_copy,cuda_ipc",
                },
                "interface": interface + str(i // 2),
                "protocol": "ucx",
                "nthreads": threads_per_worker,
                "data": dict,
                "preload": ["dask_cuda.initialize_context"],
                "dashboard_address": ":0",
                "plugins": [CPUAffinity(get_cpu_affinity_list(i))],
                "silence_logs": silence_logs,
                "memory_limit": memory_limit,
            },
        }
        for ii, i in enumerate(CUDA_VISIBLE_DEVICES)
    }

    scheduler = {
        "cls": Scheduler,
        "options": {
            "interface": interface + str(CUDA_VISIBLE_DEVICES[0] // 2),
            "protocol": "ucx",
            "dashboard_address": dashboard_address,
        },
    }

    return SpecCluster(
        workers=spec, scheduler=scheduler, silence_logs=silence_logs, **kwargs
    )
