import importlib
import logging
import os
from typing import Callable, Dict

from distributed import WorkerPlugin

from .utils import get_rmm_log_file_name, parse_device_memory_limit


class CPUAffinity(WorkerPlugin):
    def __init__(self, cores):
        self.cores = cores

    def setup(self, worker=None):
        try:
            os.sched_setaffinity(0, self.cores)
        except Exception:
            logger = logging.getLogger("distributed.worker")
            logger.warning(
                "Setting CPU affinity for GPU failed. Please refer to the following "
                "link for troubleshooting information: "
                "https://docs.rapids.ai/api/dask-cuda/nightly/troubleshooting/#setting-cpu-affinity-failure"  # noqa: E501
            )


class CUDFSetup(WorkerPlugin):
    def __init__(self, spill, spill_stats):
        self.spill = spill
        self.spill_stats = spill_stats

    def setup(self, worker=None):
        try:
            import cudf

            cudf.set_option("spill", self.spill)
            cudf.set_option("spill_stats", self.spill_stats)
        except ImportError:
            pass


class RMMSetup(WorkerPlugin):
    def __init__(
        self,
        initial_pool_size,
        maximum_pool_size,
        managed_memory,
        async_alloc,
        release_threshold,
        log_directory,
        track_allocations,
        external_lib_list,
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
        self.external_lib_list = external_lib_list

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

        if self.external_lib_list is not None:
            for lib in self.external_lib_list:
                enable_rmm_memory_for_library(lib)


def enable_rmm_memory_for_library(lib_name: str) -> None:
    """Enable RMM memory pool support for a specified third-party library.

    This function allows the given library to utilize RMM's memory pool if it supports
    integration with RMM. The library name is passed as a string argument, and if the
    library is compatible, its memory allocator will be configured to use RMM.

    Parameters
    ----------
    lib_name : str
        The name of the third-party library to enable RMM memory pool support for.
        Supported libraries are "cupy" and "torch".

    Raises
    ------
    ValueError
        If the library name is not supported or does not have RMM integration.
    ImportError
        If the required library is not installed.
    """

    # Mapping of supported libraries to their respective setup functions
    setup_functions: Dict[str, Callable[[], None]] = {
        "torch": _setup_rmm_for_torch,
        "cupy": _setup_rmm_for_cupy,
    }

    if lib_name not in setup_functions:
        supported_libs = ", ".join(setup_functions.keys())
        raise ValueError(
            f"The library '{lib_name}' is not supported for RMM integration. "
            f"Supported libraries are: {supported_libs}."
        )

    # Call the setup function for the specified library
    setup_functions[lib_name]()


def _setup_rmm_for_torch() -> None:
    try:
        import torch
    except ImportError as e:
        raise ImportError("PyTorch is not installed.") from e

    from rmm.allocators.torch import rmm_torch_allocator

    torch.cuda.memory.change_current_allocator(rmm_torch_allocator)


def _setup_rmm_for_cupy() -> None:
    try:
        import cupy
    except ImportError as e:
        raise ImportError("CuPy is not installed.") from e

    from rmm.allocators.cupy import rmm_cupy_allocator

    cupy.cuda.set_allocator(rmm_cupy_allocator)


class PreImport(WorkerPlugin):
    def __init__(self, libraries):
        if libraries is None:
            libraries = []
        elif isinstance(libraries, str):
            libraries = libraries.split(",")
        self.libraries = libraries

    def setup(self, worker=None):
        for l in self.libraries:
            importlib.import_module(l)
