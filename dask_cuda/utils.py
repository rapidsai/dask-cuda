import importlib
import math
import operator
import os
import pickle
import time
import warnings
from contextlib import suppress
from functools import singledispatch
from multiprocessing import cpu_count
from typing import Optional

import numpy as np
import pynvml
import toolz

import dask
import distributed  # noqa: required for dask.config.get("distributed.comm.ucx")
from dask.config import canonical_name
from dask.utils import format_bytes, parse_bytes
from distributed import Worker, wait
from distributed.comm import parse_address

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
    def __init__(
        self,
        initial_pool_size,
        maximum_pool_size,
        managed_memory,
        async_alloc,
        release_threshold,
        log_directory,
        track_allocations,
    ):
        if initial_pool_size is None and maximum_pool_size is not None:
            raise ValueError(
                "`rmm_maximum_pool_size` was specified without specifying "
                "`rmm_pool_size`.`rmm_pool_size` must be specified to use RMM pool."
            )
        if async_alloc is True:
            if managed_memory is True:
                raise ValueError(
                    "`rmm_managed_memory` is incompatible with the `rmm_async`."
                )
        if async_alloc is False and release_threshold is not None:
            raise ValueError("`rmm_release_threshold` requires `rmm_async`.")

        self.initial_pool_size = initial_pool_size
        self.maximum_pool_size = maximum_pool_size
        self.managed_memory = managed_memory
        self.async_alloc = async_alloc
        self.release_threshold = release_threshold
        self.logging = log_directory is not None
        self.log_directory = log_directory
        self.rmm_track_allocations = track_allocations

    def setup(self, worker=None):
        if self.initial_pool_size is not None:
            self.initial_pool_size = parse_device_memory_limit(
                self.initial_pool_size, alignment_size=256
            )

        if self.async_alloc:
            import rmm

            if self.release_threshold is not None:
                self.release_threshold = parse_device_memory_limit(
                    self.release_threshold, alignment_size=256
                )

            mr = rmm.mr.CudaAsyncMemoryResource(
                initial_pool_size=self.initial_pool_size,
                release_threshold=self.release_threshold,
            )

            if self.maximum_pool_size is not None:
                self.maximum_pool_size = parse_device_memory_limit(
                    self.maximum_pool_size, alignment_size=256
                )
                mr = rmm.mr.LimitingResourceAdaptor(
                    mr, allocation_limit=self.maximum_pool_size
                )

            rmm.mr.set_current_device_resource(mr)
            if self.logging:
                rmm.enable_logging(
                    log_file_name=get_rmm_log_file_name(
                        worker, self.logging, self.log_directory
                    )
                )
        elif self.initial_pool_size is not None or self.managed_memory:
            import rmm

            pool_allocator = False if self.initial_pool_size is None else True

            if self.initial_pool_size is not None:
                if self.maximum_pool_size is not None:
                    self.maximum_pool_size = parse_device_memory_limit(
                        self.maximum_pool_size, alignment_size=256
                    )

            rmm.reinitialize(
                pool_allocator=pool_allocator,
                managed_memory=self.managed_memory,
                initial_pool_size=self.initial_pool_size,
                maximum_pool_size=self.maximum_pool_size,
                logging=self.logging,
                log_file_name=get_rmm_log_file_name(
                    worker, self.logging, self.log_directory
                ),
            )
        if self.rmm_track_allocations:
            import rmm

            mr = rmm.mr.get_current_device_resource()
            rmm.mr.set_current_device_resource(rmm.mr.TrackingResourceAdaptor(mr))


class PreImport:
    def __init__(self, libraries):
        if libraries is None:
            libraries = []
        elif isinstance(libraries, str):
            libraries = libraries.split(",")
        self.libraries = libraries

    def setup(self, worker=None):
        for l in self.libraries:
            importlib.import_module(l)


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
        mask = np.flip(bytestr - ord("0")).astype(bool)
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


@toolz.memoize
def get_gpu_count_mig(return_uuids=False):
    """Return the number of MIG instances available

    Parameters
    ----------
    return_uuids: bool
        Returns the uuids of the MIG instances available optionally

    """
    pynvml.nvmlInit()
    uuids = []
    for index in range(get_gpu_count()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        try:
            is_mig_mode = pynvml.nvmlDeviceGetMigMode(handle)[0]
        except pynvml.NVMLError:
            # if not a MIG device, i.e. a normal GPU, skip
            continue
        if is_mig_mode:
            count = pynvml.nvmlDeviceGetMaxMigDeviceCount(handle)
            miguuids = []
            for i in range(count):
                try:
                    mighandle = pynvml.nvmlDeviceGetMigDeviceHandleByIndex(
                        device=handle, index=i
                    )
                    miguuids.append(mighandle)
                    uuids.append(pynvml.nvmlDeviceGetUUID(mighandle))
                except pynvml.NVMLError:
                    pass
    if return_uuids:
        return len(uuids), uuids
    return len(uuids)


def get_cpu_affinity(device_index=None):
    """Get a list containing the CPU indices to which a GPU is directly connected.
    Use either the device index or the specified device identifier UUID.

    Parameters
    ----------
    device_index: int or str
        Index or UUID of the GPU device

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
        if device_index and not str(device_index).isnumeric():
            # This means device_index is UUID.
            # This works for both MIG and non-MIG device UUIDs.
            handle = pynvml.nvmlDeviceGetHandleByUUID(str.encode(device_index))
            if pynvml.nvmlDeviceIsMigDeviceHandle(handle):
                # Additionally get parent device handle
                # if the device itself is a MIG instance
                handle = pynvml.nvmlDeviceGetDeviceHandleFromMigDeviceHandle(handle)
        else:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        # Result is a list of 64-bit integers, thus ceil(get_cpu_count() / 64)
        affinity = pynvml.nvmlDeviceGetCpuAffinity(
            handle,
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
    Return total memory of CUDA device with index or with device identifier UUID
    """
    pynvml.nvmlInit()

    if index and not str(index).isnumeric():
        # This means index is UUID. This works for both MIG and non-MIG device UUIDs.
        handle = pynvml.nvmlDeviceGetHandleByUUID(str.encode(str(index)))
    else:
        # This is a device index
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
    return pynvml.nvmlDeviceGetMemoryInfo(handle).total


def get_ucx_config(
    enable_tcp_over_ucx=None,
    enable_infiniband=None,
    enable_nvlink=None,
    enable_rdmacm=None,
):
    ucx_config = dask.config.get("distributed.comm.ucx")

    ucx_config[canonical_name("create-cuda-context", ucx_config)] = True
    ucx_config[canonical_name("reuse-endpoints", ucx_config)] = False

    # If any transport is explicitly disabled (`False`) by the user, others that
    # are not specified should be enabled (`True`). If transports are explicitly
    # enabled (`True`), then default (`None`) or an explicit `False` will suffice
    # in disabling others. However, if there's a mix of enable (`True`) and
    # disable (`False`), then those choices can be assumed as intended by the
    # user.
    #
    # This may be handled more gracefully in Distributed in the future.
    opts = [enable_tcp_over_ucx, enable_infiniband, enable_nvlink]
    if any(opt is False for opt in opts) and not any(opt is True for opt in opts):
        if enable_tcp_over_ucx is None:
            enable_tcp_over_ucx = True
        if enable_nvlink is None:
            enable_nvlink = True
        if enable_infiniband is None:
            enable_infiniband = True

    ucx_config[canonical_name("tcp", ucx_config)] = enable_tcp_over_ucx
    ucx_config[canonical_name("infiniband", ucx_config)] = enable_infiniband
    ucx_config[canonical_name("nvlink", ucx_config)] = enable_nvlink
    ucx_config[canonical_name("rdmacm", ucx_config)] = enable_rdmacm

    if enable_tcp_over_ucx or enable_infiniband or enable_nvlink:
        ucx_config[canonical_name("cuda-copy", ucx_config)] = True
    else:
        ucx_config[canonical_name("cuda-copy", ucx_config)] = None

    return ucx_config


def get_preload_options(
    protocol=None,
    create_cuda_context=None,
    enable_tcp_over_ucx=None,
    enable_infiniband=None,
    enable_nvlink=None,
    enable_rdmacm=None,
):
    """
    Return a dictionary with the preload and preload_argv options required to
    create CUDA context and enabling UCX communication.

    Parameters
    ----------
    protocol: None or str, default None
        If "ucx", options related to UCX (enable_tcp_over_ucx, enable_infiniband,
        enable_nvlink) are added to preload_argv.
    create_cuda_context: bool, default None
        Ensure the CUDA context gets created at initialization, generally
        needed by Dask workers.
    enable_tcp: bool, default None
        Set environment variables to enable TCP over UCX, even when InfiniBand or
        NVLink support are disabled.
    enable_infiniband: bool, default None
        Set environment variables to enable UCX InfiniBand support. Implies
        enable_tcp=True.
    enable_rdmacm: bool, default None
        Set environment variables to enable UCX RDMA connection manager support.
        Currently requires enable_infiniband=True.
    enable_nvlink: bool, default None
        Set environment variables to enable UCX NVLink support. Implies
        enable_tcp=True.

    Example
    -------
    >>> from dask_cuda.utils import get_preload_options
    >>> get_preload_options()
    {'preload': ['dask_cuda.initialize'], 'preload_argv': []}
    >>> get_preload_options(protocol="ucx",
    ...                     create_cuda_context=True,
    ...                     enable_infiniband=True)
    {'preload': ['dask_cuda.initialize'],
     'preload_argv': ['--create-cuda-context',
      '--enable-infiniband']}
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

        preload_options["preload_argv"].extend(initialize_ucx_argv)

    return preload_options


def get_rmm_log_file_name(dask_worker, logging=False, log_directory=None):
    return (
        os.path.join(
            log_directory,
            "rmm_log_%s.txt"
            % (
                (
                    dask_worker.name.split("/")[-1]
                    if isinstance(dask_worker.name, str)
                    else dask_worker.name
                )
                if hasattr(dask_worker, "name")
                else "scheduler"
            ),
        )
        if logging
        else None
    )


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
        Minimum number of seconds to wait before timeout. This value may be
        overridden by setting the `DASK_CUDA_WAIT_WORKERS_MIN_TIMEOUT` with
        a positive integer.
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
    min_timeout_env = os.environ.get("DASK_CUDA_WAIT_WORKERS_MIN_TIMEOUT", None)
    min_timeout = min_timeout if min_timeout_env is None else int(min_timeout_env)
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


async def _all_to_all(client):
    """
    Trigger all to all communication between workers and scheduler
    """
    workers = list(client.scheduler_info()["workers"])
    futs = []
    for w in workers:
        bit_of_data = b"0" * 1
        data = client.map(lambda x: bit_of_data, range(1), pure=False, workers=[w])
        futs.append(data[0])

    await wait(futs)

    def f(x):
        pass

    new_futs = []
    for w in workers:
        for future in futs:
            data = client.submit(f, future, workers=[w], pure=False)
            new_futs.append(data)

    await wait(new_futs)


def all_to_all(client):
    return client.sync(_all_to_all, client=client, asynchronous=client.asynchronous)


def parse_cuda_visible_device(dev):
    """Parses a single CUDA device identifier

    A device identifier must either be an integer, a string containing an
    integer or a string containing the device's UUID, beginning with prefix
    'GPU-' or 'MIG-'.

    >>> parse_cuda_visible_device(2)
    2
    >>> parse_cuda_visible_device('2')
    2
    >>> parse_cuda_visible_device('GPU-9baca7f5-0f2f-01ac-6b05-8da14d6e9005')
    'GPU-9baca7f5-0f2f-01ac-6b05-8da14d6e9005'
    >>> parse_cuda_visible_device('Foo')
    Traceback (most recent call last):
    ...
    ValueError: Devices in CUDA_VISIBLE_DEVICES must be comma-separated integers or
    strings beginning with 'GPU-' or 'MIG-' prefixes.
    """
    try:
        return int(dev)
    except ValueError:
        if any(
            dev.startswith(prefix)
            for prefix in [
                "GPU-",
                "MIG-",
            ]
        ):
            return dev
        else:
            raise ValueError(
                "Devices in CUDA_VISIBLE_DEVICES must be comma-separated integers "
                "or strings beginning with 'GPU-' or 'MIG-' prefixes."
            )


def cuda_visible_devices(i, visible=None):
    """Cycling values for CUDA_VISIBLE_DEVICES environment variable

    Examples
    --------
    >>> cuda_visible_devices(0, range(4))
    '0,1,2,3'
    >>> cuda_visible_devices(3, range(8))
    '3,4,5,6,7,0,1,2'
    """
    if visible is None:
        try:
            visible = map(
                parse_cuda_visible_device, os.environ["CUDA_VISIBLE_DEVICES"].split(",")
            )
        except KeyError:
            visible = range(get_n_gpus())
    visible = list(visible)

    L = visible[i:] + visible[:i]
    return ",".join(map(str, L))


def nvml_device_index(i, CUDA_VISIBLE_DEVICES):
    """Get the device index for NVML addressing

    NVML expects the index of the physical device, unlike CUDA runtime which
    expects the address relative to `CUDA_VISIBLE_DEVICES`. This function
    returns the i-th device index from the `CUDA_VISIBLE_DEVICES`
    comma-separated string of devices or list.

    Examples
    --------
    >>> nvml_device_index(1, "0,1,2,3")
    1
    >>> nvml_device_index(1, "1,2,3,0")
    2
    >>> nvml_device_index(1, [0,1,2,3])
    1
    >>> nvml_device_index(1, [1,2,3,0])
    2
    >>> nvml_device_index(1, ["GPU-84fd49f2-48ad-50e8-9f2e-3bf0dfd47ccb",
                              "GPU-d6ac2d46-159b-5895-a854-cb745962ef0f",
                              "GPU-158153b7-51d0-5908-a67c-f406bc86be17"])
    "MIG-d6ac2d46-159b-5895-a854-cb745962ef0f"
    >>> nvml_device_index(2, ["MIG-41b3359c-e721-56e5-8009-12e5797ed514",
                              "MIG-65b79fff-6d3c-5490-a288-b31ec705f310",
                              "MIG-c6e2bae8-46d4-5a7e-9a68-c6cf1f680ba0"])
    "MIG-c6e2bae8-46d4-5a7e-9a68-c6cf1f680ba0"
    >>> nvml_device_index(1, 2)
    Traceback (most recent call last):
    ...
    ValueError: CUDA_VISIBLE_DEVICES must be `str` or `list`
    """
    if isinstance(CUDA_VISIBLE_DEVICES, str):
        ith_elem = CUDA_VISIBLE_DEVICES.split(",")[i]
        if ith_elem.isnumeric():
            return int(ith_elem)
        else:
            return ith_elem
    elif isinstance(CUDA_VISIBLE_DEVICES, list):
        return CUDA_VISIBLE_DEVICES[i]
    else:
        raise ValueError("`CUDA_VISIBLE_DEVICES` must be `str` or `list`")


def parse_device_memory_limit(device_memory_limit, device_index=0, alignment_size=1):
    """Parse memory limit to be used by a CUDA device.

    Parameters
    ----------
    device_memory_limit: float, int, str or None
        This can be a float (fraction of total device memory), an integer (bytes),
        a string (like 5GB or 5000M), and "auto", 0 or None for the total device
        size.
    device_index: int or str
        The index or UUID of the device from which to obtain the total memory amount.
        Default: 0.
    alignment_size: int
        Number of bytes of alignment to use, i.e., allocation must be a multiple of
        that size. RMM pool requires 256 bytes alignment.

    Examples
    --------
    >>> # On a 32GB CUDA device
    >>> parse_device_memory_limit(None)
    34089730048
    >>> parse_device_memory_limit(0.8)
    27271784038
    >>> parse_device_memory_limit(1000000000)
    1000000000
    >>> parse_device_memory_limit("1GB")
    1000000000
    """

    def _align(size, alignment_size):
        return size // alignment_size * alignment_size

    if device_memory_limit in {0, "0", None, "auto"}:
        return _align(get_device_total_memory(device_index), alignment_size)

    with suppress(ValueError, TypeError):
        device_memory_limit = float(device_memory_limit)
        if isinstance(device_memory_limit, float) and device_memory_limit <= 1:
            return _align(
                int(get_device_total_memory(device_index) * device_memory_limit),
                alignment_size,
            )

    if isinstance(device_memory_limit, str):
        return _align(parse_bytes(device_memory_limit), alignment_size)
    else:
        return _align(int(device_memory_limit), alignment_size)


class MockWorker(Worker):
    """Mock Worker class preventing NVML from getting used by SystemMonitor.

    By preventing the Worker from initializing NVML in the SystemMonitor, we can
    mock test multiple devices in `CUDA_VISIBLE_DEVICES` behavior with single-GPU
    machines.
    """

    def __init__(self, *args, **kwargs):
        distributed.diagnostics.nvml.device_get_count = MockWorker.device_get_count
        self._device_get_count = distributed.diagnostics.nvml.device_get_count
        super().__init__(*args, **kwargs)

    def __del__(self):
        distributed.diagnostics.nvml.device_get_count = self._device_get_count

    @staticmethod
    def device_get_count():
        return 0


def get_gpu_uuid_from_index(device_index=0):
    """Get GPU UUID from CUDA device index.

    Parameters
    ----------
    device_index: int or str
        The index of the device from which to obtain the UUID. Default: 0.

    Examples
    --------
    >>> get_gpu_uuid_from_index()
    'GPU-9baca7f5-0f2f-01ac-6b05-8da14d6e9005'

    >>> get_gpu_uuid_from_index(3)
    'GPU-9fb42d6f-7d6b-368f-f79c-3c3e784c93f6'
    """
    import pynvml

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    try:
        return pynvml.nvmlDeviceGetUUID(handle).decode("utf-8")
    except AttributeError:
        return pynvml.nvmlDeviceGetUUID(handle)


def get_worker_config(dask_worker):
    from .proxify_host_file import ProxifyHostFile

    # assume homogeneous cluster
    plugin_vals = dask_worker.plugins.values()
    ret = {}
    # device and host memory configuration
    for p in plugin_vals:
        config = {
            v: getattr(p, v)
            for v in dir(p)
            if not (v.startswith("_") or v in {"setup", "cores"})
        }
        # To send this back to the client the data will be serialised
        # which might fail, so pre-emptively check
        try:
            pickle.dumps(config)
        except TypeError:
            config = "UNKNOWN CONFIG"
        ret[f"[plugin] {type(p).__name__}"] = config

    for mem in [
        "memory_limit",
        "memory_pause_fraction",
        "memory_spill_fraction",
        "memory_target_fraction",
    ]:
        ret[mem] = getattr(dask_worker.memory_manager, mem)

    # jit unspilling set
    ret["jit-unspill"] = isinstance(dask_worker.data, ProxifyHostFile)

    # get optional device-memory-limit
    if ret["jit-unspill"]:
        ret["device-memory-limit"] = dask_worker.data.manager._device_memory_limit
    else:
        has_device = hasattr(dask_worker.data, "device_buffer")
        if has_device:
            ret["device-memory-limit"] = dask_worker.data.device_buffer.n

    # using ucx ?
    scheme, loc = parse_address(dask_worker.scheduler.address)
    ret["protocol"] = scheme
    if scheme == "ucx":
        import ucp

        ret["ucx-transports"] = ucp.get_active_transports()

    # comm timeouts
    ret["distributed.comm.timeouts"] = dask.config.get("distributed.comm.timeouts")

    return ret


async def get_scheduler_configuration(client):
    worker_ttl = await client.run_on_scheduler(
        lambda dask_scheduler: dask_scheduler.worker_ttl
    )
    extensions = list(
        await client.run_on_scheduler(
            lambda dask_scheduler: dask_scheduler.extensions.keys()
        )
    )
    ret = {}
    ret["distributed.scheduler.worker-ttl"] = worker_ttl
    ret["active-extensions"] = extensions

    return ret


async def _get_cluster_configuration(client):
    worker_config = await client.run(get_worker_config)
    ret = await get_scheduler_configuration(client)

    # does the cluster have any workers ?
    if worker_config:
        w = list(worker_config.values())[0]
        ret.update(w)
        info = client.scheduler_info()
        workers = info.get("workers", {})
        ret["nworkers"] = len(workers)
        ret["nthreads"] = sum(w["nthreads"] for w in workers.values())

    return ret


@singledispatch
def pretty_print(obj, toplevel):
    from rich.pretty import Pretty

    return Pretty(obj)


@pretty_print.register(str)
def pretty_print_str(obj, toplevel):
    from rich.markup import escape

    return escape(obj)


@pretty_print.register(dict)
def pretty_print_dict(obj, toplevel):
    from rich.table import Table

    if not obj:
        return "No known settings"
    formatted_byte_keys = {
        "memory_limit",
        "device-memory-limit",
        "initial_pool_size",
        "maximum_pool_size",
    }
    t = Table(
        show_header=toplevel, title="Dask Cluster Configuration" if toplevel else None
    )
    t.add_column("Parameter", justify="left", style="bold bright_green")
    t.add_column("Value", justify="left", style="bold bright_green")
    for k, v in sorted(obj.items(), key=operator.itemgetter(0)):
        if k in formatted_byte_keys and v is not None:
            v = format_bytes(v)
        # need to escape tags: []
        # https://rich.readthedocs.io/en/stable/markup.html?highlight=escape#escaping
        t.add_row(pretty_print(k, False), pretty_print(v, False))
    return t


def print_cluster_config(client):
    """print current Dask cluster configuration"""
    if client.asynchronous:
        print("Printing cluster configuration works only with synchronous Dask clients")

    data = get_cluster_configuration(client)
    try:
        from rich.console import Console
    except ModuleNotFoundError as e:
        error_msg = (
            "Please install rich `python -m pip install rich` "
            "to print a table of the current Dask Cluster Configuration"
        )
        raise ModuleNotFoundError(error_msg) from e

    formatted = pretty_print(data, True)
    Console().print(formatted)


def get_cluster_configuration(client):
    data = client.sync(
        _get_cluster_configuration, client=client, asynchronous=client.asynchronous
    )
    return data


def get_rmm_device_memory_usage() -> Optional[int]:
    """Get current bytes allocated on current device through RMM

    Check the current RMM resource stack for resources such as
    `StatisticsResourceAdaptor` and `TrackingResourceAdaptor`
    that can report the current allocated bytes. Returns None,
    if no such resources exist.

    Return
    ------
    nbytes: int or None
        Number of bytes allocated on device through RMM or None
    """

    def get_rmm_memory_resource_stack(mr) -> list:
        if hasattr(mr, "upstream_mr"):
            return [mr] + get_rmm_memory_resource_stack(mr.upstream_mr)
        return [mr]

    try:
        import rmm
    except ImportError:
        return None

    for mr in get_rmm_memory_resource_stack(rmm.mr.get_current_device_resource()):
        if isinstance(mr, rmm.mr.TrackingResourceAdaptor):
            return mr.get_allocated_bytes()
        if isinstance(mr, rmm.mr.StatisticsResourceAdaptor):
            return mr.allocation_counts["current_bytes"]
    return None
