# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import warnings

from .device_host_file import DeviceHostFile
from .proxify_host_file import ProxifyHostFile
from .utils import get_device_total_memory, parse_device_memory_limit


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
        has_device_memory = (
            get_device_total_memory(device_index=device_index) is not None
        )

        # First raise errors for invalid configurations
        if not has_device_memory:
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
            elif device_memory_limit is not None:
                raise ValueError(
                    "device_memory_limit is set but device has no dedicated memory."
                )

        if device_memory_limit is None and memory_limit is None:
            # All spilling is disabled
            return {}
        elif not has_device_memory:
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
