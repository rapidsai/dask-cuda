import os

from dask.distributed import SpecCluster, Nanny, Scheduler
from distributed.worker import TOTAL_MEMORY

from .utils import get_n_gpus


def cuda_visible_devices(i, visible=None):
    """ Cycling values for CUDA_VISIBLE_DEVICES environment variable

    Examples
    --------
    >>> cuda_visible_devices(0, range(4))
    '0,1,2,3'
    >>> cuda_visible_devices(3, range(8))
    '3,4,5,6,7,0,1,2'
    """
    if visible is None:
        try:
            visible = map(int, os.environ["CUDA_VISIBLE_DEVICES"].split(","))
        except KeyError:
            visible = range(get_n_gpus())
    visible = list(visible)

    L = visible[i:] + visible[:i]
    return ",".join(map(str, L))


def LocalCUDACluster(
    n_workers=None,
    threads_per_worker=1,
    processes=True,
    memory_limit=None,
    interface=None,
    protocol=None,
    data=None,
    CUDA_VISIBLE_DEVICES=None,
    silence_logs=True,
    dashboard_address=":8787",
    **kwargs,
):
    if n_workers is None:
        n_workers = get_n_gpus()
    if CUDA_VISIBLE_DEVICES is None:
        CUDA_VISIBLE_DEVICES = cuda_visible_devices(0)
    if isinstance(CUDA_VISIBLE_DEVICES, str):
        CUDA_VISIBLE_DEVICES = CUDA_VISIBLE_DEVICES.split(",")
    CUDA_VISIBLE_DEVICES = list(map(int, CUDA_VISIBLE_DEVICES))
    if memory_limit is None:
        memory_limit = TOTAL_MEMORY / n_workers

    workers = {
        i: {
            "cls": Nanny,
            "options": {
                "env": {
                    "CUDA_VISIBLE_DEVICES": cuda_visible_devices(
                        ii, CUDA_VISIBLE_DEVICES
                    )
                },
                "ncores": threads_per_worker,
                "data": data,
                "preload": ["dask_cuda.initialize_context"],
                "dashboard_address": ":0",
                "silence_logs": silence_logs,
                "interface": interface,
                "protocol": protocol,
                "memory_limit": memory_limit,
            },
        }
        for ii, i in enumerate(CUDA_VISIBLE_DEVICES)
    }

    scheduler = {
        "cls": Scheduler,
        "options": {
            "dashboard_address": dashboard_address,
            "interface": interface,
            "protocol": protocol,
        },
    }

    return SpecCluster(
        workers=workers, scheduler=scheduler, silence_logs=silence_logs, **kwargs
    )
