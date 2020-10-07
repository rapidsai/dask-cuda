import math
import os
import time
import warnings
from multiprocessing import cpu_count

import numpy as np
import pynvml
import toolz

try:
    from nvtx import annotate as nvtx_annotate
except ImportError:
    # If nvtx module is not installed, `annotate` yields only.
    from contextlib import contextmanager

    @contextmanager
    def nvtx_annotate(message=None, color="blue", domain=None):
        yield


class CPUAffinity:
    def __init__(self, cores):
        self.cores = cores

    def setup(self, worker=None):
        os.sched_setaffinity(0, self.cores)


class RMMSetup:
    def __init__(self, nbytes, managed_memory):
        self.nbytes = nbytes
        self.managed_memory = managed_memory

    def setup(self, worker=None):
        if self.nbytes is not None or self.managed_memory is True:
            import rmm

            pool_allocator = False if self.nbytes is None else True

            rmm.reinitialize(
                pool_allocator=pool_allocator,
                managed_memory=self.managed_memory,
                initial_pool_size=self.nbytes,
            )


def unpack_bitmask(x, mask_bits=64):
    """Unpack a list of integers containing bitmasks.

    Parameters
    ----------
    x: list of int
        A list of integers
    mask_bits: int
        An integer determining the bitwidth of `x`

    Examples
    --------
    >>> from dask_cuda.utils import unpack_bitmaps
    >>> unpack_bitmask([1 + 2 + 8])
    [0, 1, 3]
    >>> unpack_bitmask([1 + 2 + 16])
    [0, 1, 4]
    >>> unpack_bitmask([1 + 2 + 16, 2 + 4])
    [0, 1, 4, 65, 66]
    >>> unpack_bitmask([1 + 2 + 16, 2 + 4], mask_bits=32)
    [0, 1, 4, 33, 34]
    """
    res = []

    for i, mask in enumerate(x):
        if not isinstance(mask, int):
            raise TypeError("All elements of the list `x` must be integers")

        cpu_offset = i * mask_bits

        bytestr = np.frombuffer(
            bytes(np.binary_repr(mask, width=mask_bits), "utf-8"), "u1"
        )
        mask = np.flip(bytestr - ord("0")).astype(np.bool)
        unpacked_mask = np.where(
            mask, np.arange(mask_bits) + cpu_offset, np.full(mask_bits, -1)
        )

        res += unpacked_mask[(unpacked_mask >= 0)].tolist()

    return res


@toolz.memoize
def get_cpu_count():
    return cpu_count()


@toolz.memoize
def get_gpu_count():
    pynvml.nvmlInit()
    return pynvml.nvmlDeviceGetCount()


def get_cpu_affinity(device_index):
    """Get a list containing the CPU indices to which a GPU is directly connected.

    Parameters
    ----------
    device_index: int
        Index of the GPU device

    Examples
    --------
    >>> from dask_cuda.utils import get_cpu_affinity
    >>> get_cpu_affinity(0)  # DGX-1 has GPUs 0-3 connected to CPUs [0-19, 20-39]
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
     40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
    >>> get_cpu_affinity(5)  # DGX-1 has GPUs 5-7 connected to CPUs [20-39, 60-79]
    [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
     60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]
    >>> get_cpu_affinity(1000)  # DGX-1 has no device on index 1000
    dask_cuda/utils.py:96: UserWarning: Cannot get CPU affinity for device with index
    1000, setting default affinity
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
     20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
     40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
     60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]
    """
    pynvml.nvmlInit()

    try:
        # Result is a list of 64-bit integers, thus ceil(get_cpu_count() / 64)
        affinity = pynvml.nvmlDeviceGetCpuAffinity(
            pynvml.nvmlDeviceGetHandleByIndex(device_index),
            math.ceil(get_cpu_count() / 64),
        )
        return unpack_bitmask(affinity)
    except pynvml.NVMLError:
        warnings.warn(
            "Cannot get CPU affinity for device with index %d, setting default affinity"
            % device_index
        )
        return list(range(get_cpu_count()))


def get_n_gpus():
    try:
        return len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    except KeyError:
        return get_gpu_count()


def get_device_total_memory(index=0):
    """
    Return total memory of CUDA device with index
    """
    pynvml.nvmlInit()
    return pynvml.nvmlDeviceGetMemoryInfo(
        pynvml.nvmlDeviceGetHandleByIndex(index)
    ).total


def get_ucx_net_devices(
    cuda_device_index, ucx_net_devices, get_openfabrics=True, get_network=False
):
    if cuda_device_index is None and (
        callable(ucx_net_devices) or ucx_net_devices == "auto"
    ):
        raise ValueError(
            "A CUDA device index must be specified if the "
            "ucx_net_devices variable is either callable or 'auto'"
        )
    elif cuda_device_index is not None:
        dev = int(cuda_device_index)

    net_dev = None
    if callable(ucx_net_devices):
        net_dev = ucx_net_devices(int(cuda_device_index))
    elif isinstance(ucx_net_devices, str):
        if ucx_net_devices == "auto":
            # If TopologicalDistance from ucp is available, we set the UCX
            # net device to the closest network device explicitly.
            from ucp._libs.topological_distance import TopologicalDistance

            net_dev = ""
            td = TopologicalDistance()
            if get_openfabrics:
                ibs = td.get_cuda_distances_from_device_index(dev, "openfabrics")
                if len(ibs) > 0:
                    net_dev += ibs[0]["name"] + ":1"
            if get_network:
                ifnames = td.get_cuda_distances_from_device_index(dev, "network")
                if len(ifnames) > 0:
                    if len(net_dev) > 0:
                        net_dev += ","
                    net_dev += ifnames[0]["name"]
        else:
            net_dev = ucx_net_devices
    return net_dev


def get_ucx_config(
    enable_tcp_over_ucx=False,
    enable_infiniband=False,
    enable_nvlink=False,
    enable_rdmacm=False,
    net_devices="",
    cuda_device_index=None,
):
    if net_devices == "auto" and enable_infiniband is False:
        raise ValueError(
            "Using ucx_net_devices='auto' is currently only "
            "supported when enable_infiniband=True."
        )

    ucx_config = {
        "tcp": None,
        "infiniband": None,
        "nvlink": None,
        "rdmacm": None,
        "net-devices": None,
        "cuda_copy": None,
        "reuse-endpoints": True,
    }
    if enable_tcp_over_ucx or enable_infiniband or enable_nvlink:
        ucx_config["cuda_copy"] = True
    if enable_tcp_over_ucx:
        ucx_config["tcp"] = True
    if enable_infiniband:
        ucx_config["infiniband"] = True
    if enable_nvlink:
        ucx_config["nvlink"] = True
    if enable_rdmacm:
        ucx_config["rdmacm"] = True

    if net_devices is not None and net_devices != "":
        ucx_config["net-devices"] = get_ucx_net_devices(cuda_device_index, net_devices)
    return ucx_config


def get_preload_options(
    protocol=None,
    create_cuda_context=False,
    enable_tcp_over_ucx=False,
    enable_infiniband=False,
    enable_nvlink=False,
    enable_rdmacm=False,
    ucx_net_devices="",
    cuda_device_index=0,
):
    """
    Return a dictionary with the preload and preload_argv options required to
    create CUDA context and enabling UCX communication.

    Parameters
    ----------
    protocol: None or str
        If "ucx", options related to UCX (enable_tcp_over_ucx, enable_infiniband,
        enable_nvlink and ucx_net_devices) are added to preload_argv.
    create_cuda_context: bool
        Ensure the CUDA context gets created at initialization, generally
        needed by Dask workers.
    enable_tcp: bool
        Set environment variables to enable TCP over UCX, even when InfiniBand or
        NVLink support are disabled.
    enable_infiniband: bool
        Set environment variables to enable UCX InfiniBand support. Implies
        enable_tcp=True.
    enable_rdmacm: bool
        Set environment variables to enable UCX RDMA connection manager support.
        Currently requires enable_infiniband=True.
    enable_nvlink: bool
        Set environment variables to enable UCX NVLink support. Implies
        enable_tcp=True.
    ucx_net_devices: str or callable
        A string with the interface name to be used for all devices (empty
        string means use default), or a callable function taking an integer
        identifying the GPU index.
    cuda_device_index: int
        The index identifying the CUDA device used by this worker, only used
        when ucx_net_devices is callable.

    Example
    -------
    >>> from dask_cuda.utils import get_preload_options
    >>> get_preload_options()
    {'preload': ['dask_cuda.initialize'], 'preload_argv': []}
    >>> get_preload_options(protocol="ucx", create_cuda_context=True,
    ...                     enable_infiniband=True, cuda_device_index=5,
    ...                     ucx_net_devices=lambda i: "mlx5_%d:1" % (i // 2))
    {'preload': ['dask_cuda.initialize'],
     'preload_argv': ['--create-cuda-context',
      '--enable-infiniband',
      '--net-devices=mlx5_2:1']}
    """
    preload_options = {"preload": ["dask_cuda.initialize"], "preload_argv": []}

    if create_cuda_context:
        preload_options["preload_argv"].append("--create-cuda-context")

    if protocol == "ucx":
        initialize_ucx_argv = []
        if enable_tcp_over_ucx:
            initialize_ucx_argv.append("--enable-tcp-over-ucx")
        if enable_infiniband:
            initialize_ucx_argv.append("--enable-infiniband")
        if enable_rdmacm:
            initialize_ucx_argv.append("--enable-rdmacm")
        if enable_nvlink:
            initialize_ucx_argv.append("--enable-nvlink")
        if ucx_net_devices is not None and ucx_net_devices != "":
            net_dev = get_ucx_net_devices(cuda_device_index, ucx_net_devices)
            initialize_ucx_argv.append("--net-devices=%s" % net_dev)

        preload_options["preload_argv"].extend(initialize_ucx_argv)

    return preload_options


def wait_workers(
    client, min_timeout=10, seconds_per_gpu=2, n_gpus=None, timeout_callback=None
):
    """
    Wait for workers to be available. When a timeout occurs, a callback
    is executed if specified. Generally used for tests.

    Parameters
    ----------
    client: distributed.Client
        Instance of client, used to query for number of workers connected.
    min_timeout: float
        Minimum number of seconds to wait before timeout.
    seconds_per_gpu: float
        Seconds to wait for each GPU on the system. For example, if its
        value is 2 and there is a total of 8 GPUs (workers) being started,
        a timeout will occur after 16 seconds. Note that this value is only
        used as timeout when larger than min_timeout.
    n_gpus: None or int
        If specified, will wait for a that amount of GPUs (i.e., Dask workers)
        to come online, else waits for a total of `get_n_gpus` workers.
    timeout_callback: None or callable
        A callback function to be executed if a timeout occurs, ignored if
        None.

    Returns
    -------
    True if all workers were started, False if a timeout occurs.
    """
    n_gpus = n_gpus or get_n_gpus()
    timeout = max(min_timeout, seconds_per_gpu * n_gpus)

    start = time.time()
    while True:
        if len(client.scheduler_info()["workers"]) == n_gpus:
            return True
        elif time.time() - start > timeout:
            if callable(timeout_callback):
                timeout_callback()
            return False
        else:
            time.sleep(0.1)
