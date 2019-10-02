import os

from dask.distributed import SpecCluster, Scheduler

from .worker_spec import worker_spec


def DGX(
    interface=None,
    dashboard_address=":8787",
    threads_per_worker=1,
    silence_logs=True,
    CUDA_VISIBLE_DEVICES=None,
    enable_infiniband=True,
    enable_nvlink=True,
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
        The external interface used to connect to the scheduler, usually
        the ethernet interface is used for connection (not the InfiniBand!).
    dashboard_address: str
        The address for the scheduler dashboard.  Defaults to ":8787".
    threads_per_worker: int
        Number of threads to be used for each CUDA worker process.
    silence_logs: bool
        Disable logging for all worker processes
    CUDA_VISIBLE_DEVICES: str
        String like ``"0,1,2,3"`` or ``[0, 1, 2, 3]`` to restrict activity to
        different GPUs
    enable_infiniband: bool
        Set environment variables to enable UCX InfiniBand support
    enable_nvlink: bool
        Set environment variables to enable UCX NVLink support

    Examples
    --------
    >>> from dask_cuda import DGX
    >>> from dask.distributed import Client
    >>> cluster = DGX(interface='ib')
    >>> client = Client(cluster)
    """
    spec = worker_spec(
        interface=interface,
        dashboard_address=dashboard_address,
        threads_per_worker=threads_per_worker,
        silence_logs=silence_logs,
        CUDA_VISIBLE_DEVICES=CUDA_VISIBLE_DEVICES,
        enable_infiniband=enable_infiniband,
        ucx_net_device=lambda i: "mlx5_%d:1" % (i // 2),
        enable_nvlink=enable_nvlink,
        **kwargs,
    )

    scheduler = {
        "cls": Scheduler,
        "options": {
            "interface": interface,
            "protocol": "ucx",
            "dashboard_address": dashboard_address,
        },
    }

    return SpecCluster(
        workers=spec, scheduler=scheduler, silence_logs=silence_logs, **kwargs
    )
