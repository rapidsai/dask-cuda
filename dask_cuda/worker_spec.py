import os

from dask.distributed import Nanny
from distributed.system import MEMORY_LIMIT

from .local_cuda_cluster import cuda_visible_devices
from .utils import CPUAffinity, get_cpu_affinity


def worker_spec(
    interface="enp1s0f0",
    dashboard_address=":8787",
    threads_per_worker=1,
    silence_logs=True,
    CUDA_VISIBLE_DEVICES=None,
    enable_infiniband=False,
    enable_nvlink=False,
    **kwargs
):
    """ Create a Spec for a CUDA worker.

    The Spec created by this function can be used as a recipe for CUDA workers
    that can be passed to a SpecCluster.

    Parameters
    ----------
    interface: str
        The external interface used to connect to the scheduler.
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
    >>> from dask_cuda.worker_spec import worker_spec
    >>> worker_spec(interface="enp1s0f0", CUDA_VISIBLE_DEVICES=[0, 1])
    {0: {'cls': distributed.nanny.Nanny,
      'options': {'env': {'CUDA_VISIBLE_DEVICES': '0,1',
        'UCX_NET_DEVICES': 'mlx5_0:1'},
       'interface': 'enp1s0f0',
       'protocol': 'ucx',
       'nthreads': 1,
       'data': dict,
       'preload': ['dask_cuda.initialize_context'],
       'dashboard_address': ':0',
       'plugins': [<dask_cuda.utils.CPUAffinity at 0x7f2d6c16ed90>],
       'silence_logs': True,
       'memory_limit': 135263611392.0}},
     1: {'cls': distributed.nanny.Nanny,
      'options': {'env': {'CUDA_VISIBLE_DEVICES': '1,0',
        'UCX_NET_DEVICES': 'mlx5_0:1'},
       'interface': 'enp1s0f0',
       'protocol': 'ucx',
       'nthreads': 1,
       'data': dict,
       'preload': ['dask_cuda.initialize_context'],
       'dashboard_address': ':0',
       'plugins': [<dask_cuda.utils.CPUAffinity at 0x7f2d47f9fc50>],
       'silence_logs': True,
       'memory_limit': 135263611392.0}}}
    """
    if CUDA_VISIBLE_DEVICES is None:
        CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    if isinstance(CUDA_VISIBLE_DEVICES, str):
        CUDA_VISIBLE_DEVICES = CUDA_VISIBLE_DEVICES.split(",")
    CUDA_VISIBLE_DEVICES = list(map(int, CUDA_VISIBLE_DEVICES))
    memory_limit = MEMORY_LIMIT / 8

    ucx_env = {}
    if enable_infiniband:
        ucx_env["UCX_SOCKADDR_TLS_PRIORITY"] = "sockcm"
        ucx_env["UCX_TLS"] = "rc,tcp,sockcm"
        ucx_env["UCXPY_IFNAME"] = interface
    if enable_nvlink:
        ucx_tls = "cuda_copy,cuda_ipc"
        if "UCX_TLS" in ucx_env:
            ucx_env["UCX_TLS"] = ucx_env["UCX_TLS"] + "," + ucx_tls
        else:
            ucx_env["UCX_TLS"] = ucx_tls

    spec = {
        i: {
            "cls": Nanny,
            "options": {
                "env": {
                    "CUDA_VISIBLE_DEVICES": cuda_visible_devices(
                        ii, CUDA_VISIBLE_DEVICES
                    ),
                    "UCX_NET_DEVICES": "mlx5_%d:1" % (i // 2),
                    **ucx_env,
                },
                "interface": interface,
                "protocol": "ucx",
                "nthreads": threads_per_worker,
                "data": dict,
                "preload": ["dask_cuda.initialize_context"],
                "dashboard_address": ":0",
                "plugins": [CPUAffinity(get_cpu_affinity(i))],
                "silence_logs": silence_logs,
                "memory_limit": memory_limit,
            },
        }
        for ii, i in enumerate(CUDA_VISIBLE_DEVICES)
    }

    return spec
