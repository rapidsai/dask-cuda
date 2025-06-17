# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import os
import warnings

from .device_host_file import DeviceHostFile
from .plugins import CPUAffinity, CUDFSetup, PreImport, RMMSetup
from .proxify_host_file import ProxifyHostFile
from .utils import (
    get_cpu_affinity,
    has_device_memory_resource,
    parse_device_memory_limit,
)


def worker_data_function(
    device_memory_limit=None,
    memory_limit=None,
    jit_unspill=False,
    enable_cudf_spill=False,
    shared_filesystem=None,
):
    """
    Create a data function for CUDA workers based on memory configuration.

    This function creates and returns a callable that generates data configuration
    for CUDA workers. The returned callable takes a device index parameter and
    returns the appropriate data configuration for that device.

    Parameters
    ----------
    device_memory_limit : str or int, optional
        Limit of device memory, defaults to None
    memory_limit : str or int, optional
        Limit of host memory, defaults to None
    jit_unspill : bool, optional
        Whether to enable JIT unspill functionality, defaults to False
    enable_cudf_spill : bool, optional
        Whether to enable cuDF spilling, defaults to False
    shared_filesystem : str or bool, optional
        Whether to use shared filesystem for spilling, defaults to None

    Returns
    -------
    callable
        A function that takes device index `device_index` and returns appropriate
        data configuration based on the availability of an dedicated device memory
        resource and arguments passed to the worker.
    """

    def data(device_index):
        if int(os.environ.get("DASK_CUDA_TEST_DISABLE_DEVICE_SPECIFIC", "0")) != 0:
            return {}

        # First raise errors for invalid configurations
        if not has_device_memory_resource(device_index):
            if jit_unspill:
                raise ValueError(
                    "JIT-Unspill is not supported on devices without dedicated memory, "
                    "such as system on a chip (SoC) devices."
                )
            elif enable_cudf_spill:
                raise ValueError(
                    "cuDF spilling is not supported on devices without dedicated "
                    "memory, such as system on a chip (SoC) devices."
                )
            elif device_memory_limit not in [None, "default"]:
                raise ValueError(
                    "device_memory_limit is set but device has no dedicated memory."
                )

        if device_memory_limit is None and memory_limit is None:
            # All spilling is disabled
            return {}
        elif not has_device_memory_resource(device_index):
            if device_memory_limit == "default" and memory_limit is None:
                # Devices without a dedicated memory resource only support default
                # host<->disk spilling via Dask's default mechanism.
                return {}
            # Devices without a dedicated memory resource only support default
            # host<->disk spilling via Dask's default mechanism.
            return None
        else:
            if jit_unspill:
                # JIT-Unspill is enabled
                if enable_cudf_spill:
                    warnings.warn(
                        "Enabling cuDF spilling and JIT-Unspill together is not "
                        "safe, consider disabling JIT-Unspill."
                    )

                return (
                    ProxifyHostFile,
                    {
                        "device_memory_limit": parse_device_memory_limit(
                            device_memory_limit, device_index=device_index
                        ),
                        "memory_limit": memory_limit,
                        "shared_filesystem": shared_filesystem,
                    },
                )
            else:
                # Device has dedicated memory and host memory is limited
                return (
                    DeviceHostFile,
                    {
                        "device_memory_limit": parse_device_memory_limit(
                            device_memory_limit, device_index=device_index
                        ),
                        "memory_limit": memory_limit,
                    },
                )

    return data


def worker_plugins(
    *,
    device_index,
    rmm_initial_pool_size,
    rmm_maximum_pool_size,
    rmm_managed_memory,
    rmm_async_alloc,
    rmm_release_threshold,
    rmm_log_directory,
    rmm_track_allocations,
    rmm_allocator_external_lib_list,
    pre_import,
    enable_cudf_spill,
    cudf_spill_stats,
):
    if int(os.environ.get("DASK_CUDA_TEST_DISABLE_DEVICE_SPECIFIC", "0")) != 0:
        return {
            PreImport(pre_import),
            CUDFSetup(spill=enable_cudf_spill, spill_stats=cudf_spill_stats),
        }
    return {
        CPUAffinity(
            get_cpu_affinity(device_index),
        ),
        RMMSetup(
            initial_pool_size=rmm_initial_pool_size,
            maximum_pool_size=rmm_maximum_pool_size,
            managed_memory=rmm_managed_memory,
            async_alloc=rmm_async_alloc,
            release_threshold=rmm_release_threshold,
            log_directory=rmm_log_directory,
            track_allocations=rmm_track_allocations,
            external_lib_list=rmm_allocator_external_lib_list,
        ),
        PreImport(pre_import),
        CUDFSetup(spill=enable_cudf_spill, spill_stats=cudf_spill_stats),
    }
