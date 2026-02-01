# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import logging
import os

import click
from cuda.core import Device

import dask
from distributed.diagnostics.nvml import (
    CudaDeviceInfo,
    get_device_index_and_uuid,
    has_cuda_context,
)

from .utils import get_ucx_config

logger = logging.getLogger(__name__)


pre_existing_cuda_context = None
cuda_context_created = None


_warning_suffix = (
    "This is often the result of a CUDA-enabled library calling a CUDA runtime "
    "function before Dask-CUDA can spawn worker processes. Please make sure any such "
    "function calls don't happen at import time or in the global scope of a program."
)


def _get_device_and_uuid_str(device_info: CudaDeviceInfo) -> str:
    return f"{device_info.device_index} ({str(device_info.uuid)})"


def _warn_existing_cuda_context(device_info: CudaDeviceInfo, pid: int) -> None:
    device_uuid_str = _get_device_and_uuid_str(device_info)
    logger.warning(
        f"A CUDA context for device {device_uuid_str} already exists "
        f"on process ID {pid}. {_warning_suffix}"
    )


def _warn_cuda_context_wrong_device(
    device_info_expected: CudaDeviceInfo, device_info_actual: CudaDeviceInfo, pid: int
) -> None:
    expected_device_uuid_str = _get_device_and_uuid_str(device_info_expected)
    actual_device_uuid_str = _get_device_and_uuid_str(device_info_actual)
    logger.warning(
        f"Worker with process ID {pid} should have a CUDA context assigned to device "
        f"{expected_device_uuid_str}, but instead the CUDA context is on device "
        f"{actual_device_uuid_str}. {_warning_suffix}"
    )


def _mock_test_device() -> bool:
    """Check whether running tests in a single-GPU environment.


    Returns
    -------
    Whether running tests in a single-GPU environment, determined by checking whether
    `DASK_CUDA_TEST_SINGLE_GPU` environment variable is set to a value different than
    `"0"`.
    """
    return int(os.environ.get("DASK_CUDA_TEST_SINGLE_GPU", "0")) != 0


def _get_device_str() -> str:
    """Get the device string.

    Get a string with the first device (first element before the comma), which may be
    an index or a UUID.

    Always returns "0" when running tests in a single-GPU environment, determined by
    the result returned by `_mock_test_device()`.

    Returns
    -------
    The device string.
    """
    if _mock_test_device():
        return "0"
    else:
        return os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]


def _create_cuda_context_handler():
    """Create a CUDA context on the current device.

    A CUDA context is created on the current device if one does not exist yet, and not
    running tests on a single-GPU environment, determined by the result returned by
    `_mock_test_device()`.

    Returns
    -------
    The device string.
    """
    if _mock_test_device():
        try:
            Device().set_current()
        except Exception:
            pass
    else:
        Device().set_current()


def _create_cuda_context_and_warn():
    """Create CUDA context and warn depending on certain conditions.

    Warns if a pre-existing CUDA context already existed or if the resulting CUDA
    context was created in the wrong device.

    This function is almost an identical duplicate from
    `distributed_ucxx.ucxx.init_once`, the duplication is necessary because Dask-CUDA
    needs to support `protocol="tcp"` as well, even when distributed-ucxx is not
    installed, but this here runs _after_ comms have started, which is fine for TCP
    because the time when CUDA context is created is not important. The code needs to
    live also in distributed-ucxx because there the time when a CUDA context is created
    matters, and it needs to happen _before_ UCX is initialized, but comms in
    Distributed is initialized before preload, and thus only after this function
    executes.

    Raises
    ------
    Exception
        If anything wrong happened during context initialization.

    Returns
    -------
    None
    """
    global pre_existing_cuda_context, cuda_context_created

    cuda_visible_device = get_device_index_and_uuid(_get_device_str())
    pre_existing_cuda_context = has_cuda_context()
    if pre_existing_cuda_context.has_context:
        _warn_existing_cuda_context(pre_existing_cuda_context.device_info, os.getpid())

    _create_cuda_context_handler()

    cuda_context_created = has_cuda_context()
    if (
        cuda_context_created.has_context
        and cuda_context_created.device_info.uuid != cuda_visible_device.uuid
    ):
        _warn_cuda_context_wrong_device(
            cuda_visible_device, cuda_context_created.device_info, os.getpid()
        )


def _create_cuda_context():
    try:
        # Added here to ensure the parent `LocalCUDACluster` process creates the CUDA
        # context directly from the UCX module, thus avoiding a similar warning there.
        import distributed_ucxx.ucxx
    except ImportError:
        pass
    else:
        if distributed_ucxx.ucxx.ucxx is not None:
            # UCXX has already initialized (and warned if necessary)
            return

    try:
        _create_cuda_context_and_warn()
    except Exception:
        logger.error("Unable to start CUDA Context", exc_info=True)


def initialize(
    create_cuda_context=True,
    enable_tcp_over_ucx=None,
    enable_infiniband=None,
    enable_nvlink=None,
    enable_rdmacm=None,
):
    """Create CUDA context and initialize UCXX configuration.

    Sometimes it is convenient to initialize the CUDA context, particularly before
    starting up Dask worker processes which create a variety of threads.

    To ensure UCX works correctly, it is important to ensure it is initialized with the
    correct options. This is especially important for the client, which cannot be
    configured to use UCX with arguments like ``LocalCUDACluster`` and
    ``dask cuda worker``. This function will ensure that they are provided a UCX
    configuration based on the flags and options passed by the user.

    This function can also be used within a worker preload script for UCX configuration
    of mainline Dask.distributed.
    https://docs.dask.org/en/latest/setup/custom-startup.html

    You can add it to your global config with the following YAML:

    .. code-block:: yaml

        distributed:
          worker:
            preload:
              - dask_cuda.initialize

    See https://docs.dask.org/en/latest/configuration.html for more information about
    Dask configuration.

    Parameters
    ----------
    create_cuda_context : bool, default True
        Create CUDA context on initialization.
    enable_tcp_over_ucx : bool, default None
        Set environment variables to enable TCP over UCX, even if InfiniBand and NVLink
        are not supported or disabled.
    enable_infiniband : bool, default None
        Set environment variables to enable UCX over InfiniBand, implies
        ``enable_tcp_over_ucx=True`` when ``True``.
    enable_nvlink : bool, default None
        Set environment variables to enable UCX over NVLink, implies
        ``enable_tcp_over_ucx=True`` when ``True``.
    enable_rdmacm : bool, default None
        Set environment variables to enable UCX RDMA connection manager support,
        requires ``enable_infiniband=True``.
    """
    ucx_config = get_ucx_config(
        enable_tcp_over_ucx=enable_tcp_over_ucx,
        enable_infiniband=enable_infiniband,
        enable_nvlink=enable_nvlink,
        enable_rdmacm=enable_rdmacm,
    )
    dask.config.set({"distributed-ucxx": ucx_config})

    if create_cuda_context:
        _create_cuda_context()


@click.command()
@click.option(
    "--create-cuda-context/--no-create-cuda-context",
    default=False,
    help="Create CUDA context",
)
def dask_setup(
    worker,
    create_cuda_context,
):
    if create_cuda_context:
        _create_cuda_context()
