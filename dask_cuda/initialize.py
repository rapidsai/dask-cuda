import logging
import os
import warnings

import click
import numba.cuda

import dask
import distributed.comm.ucx
from distributed.diagnostics.nvml import has_cuda_context

from .utils import get_ucx_config

logger = logging.getLogger(__name__)


def _create_cuda_context_handler():
    if int(os.environ.get("DASK_CUDA_TEST_SINGLE_GPU", "0")) != 0:
        try:
            numba.cuda.current_context()
        except numba.cuda.cudadrv.error.CudaSupportError:
            pass
    else:
        numba.cuda.current_context()


def _create_cuda_context():
    try:
        # Added here to ensure the parent `LocalCUDACluster` process creates the CUDA
        # context directly from the UCX module, thus avoiding a similar warning there.
        try:
            distributed.comm.ucx.init_once()
        except ModuleNotFoundError:
            # UCX intialization has to be delegated to Distributed, it will take care
            # of setting correct environment variables and importing `ucp` after that.
            # Therefore if ``import ucp`` fails we can just continue here.
            pass

        cuda_visible_device = int(
            os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        )
        ctx = has_cuda_context()
        if ctx is not False and distributed.comm.ucx.cuda_context_created is False:
            warnings.warn(
                f"A CUDA context for device {ctx} already exists on process ID "
                f"{os.getpid()}. This is often the result of a CUDA-enabled library "
                "calling a CUDA runtime function before Dask-CUDA can spawn worker "
                "processes. Please make sure any such function calls don't happen at "
                "import time or in the global scope of a program."
            )

        _create_cuda_context_handler()

        if distributed.comm.ucx.cuda_context_created is False:
            ctx = has_cuda_context()
            if ctx is not False and ctx != cuda_visible_device:
                warnings.warn(
                    f"Worker with process ID {os.getpid()} should have a CUDA context "
                    f"assigned to device {cuda_visible_device}, but instead the CUDA "
                    f"context is on device {ctx}. This is often the result of a "
                    "CUDA-enabled library calling a CUDA runtime function before "
                    "Dask-CUDA can spawn worker processes. Please make sure any such "
                    "function calls don't happen at import time or in the global scope "
                    "of a program."
                )
    except Exception:
        logger.error("Unable to start CUDA Context", exc_info=True)


def initialize(
    create_cuda_context=True,
    enable_tcp_over_ucx=False,
    enable_infiniband=False,
    enable_nvlink=False,
    enable_rdmacm=False,
    net_devices="",
    cuda_device_index=None,
):
    """Create CUDA context and initialize UCX-Py, depending on user parameters.

    Sometimes it is convenient to initialize the CUDA context, particularly before
    starting up Dask worker processes which create a variety of threads.

    To ensure UCX works correctly, it is important to ensure it is initialized with the
    correct options. This is especially important for the client, which cannot be
    configured to use UCX with arguments like ``LocalCUDACluster`` and
    ``dask-cuda-worker``. This function will ensure that they are provided a UCX
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
    enable_tcp_over_ucx : bool, default False
        Set environment variables to enable TCP over UCX, even if InfiniBand and NVLink
        are not supported or disabled.
    enable_infiniband : bool, default False
        Set environment variables to enable UCX over InfiniBand, implies
        ``enable_tcp_over_ucx=True``.
    enable_nvlink : bool, default False
        Set environment variables to enable UCX over NVLink, implies
        ``enable_tcp_over_ucx=True``.
    enable_rdmacm : bool, default False
        Set environment variables to enable UCX RDMA connection manager support,
        requires ``enable_infiniband=True``.
    net_devices : str or callable, default ""
        Interface(s) used by workers for UCX communication. Can be a string (like
        ``"eth0"`` for NVLink, ``"mlx5_0:1"``/``"ib0"`` for InfiniBand, or ``""`` to use
        all available devices), or a callable function that takes the index of the
        current GPU to return an interface name (like
        ``lambda dev: "mlx5_%d:1" % (dev // 2)``).

        .. note::
            If ``net_devices`` is callable, a GPU index must be supplied through
            ``cuda_device_index``.
    cuda_device_index : int or None, default None
        Index of the current GPU, which must be specified for ``net_devices`` if
        it is callable. Can be an integer or ``None`` if ``net_devices`` is not
        callable.
    """
    ucx_config = get_ucx_config(
        enable_tcp_over_ucx=enable_tcp_over_ucx,
        enable_infiniband=enable_infiniband,
        enable_nvlink=enable_nvlink,
        enable_rdmacm=enable_rdmacm,
        net_devices=net_devices,
        cuda_device_index=cuda_device_index,
    )
    dask.config.set({"distributed.comm.ucx": ucx_config})

    if create_cuda_context:
        _create_cuda_context()


@click.command()
@click.option(
    "--create-cuda-context/--no-create-cuda-context",
    default=False,
    help="Create CUDA context",
)
@click.option(
    "--enable-tcp-over-ucx/--disable-tcp-over-ucx",
    default=False,
    help="Enable TCP communication over UCX",
)
@click.option(
    "--enable-infiniband/--disable-infiniband",
    default=False,
    help="Enable InfiniBand communication",
)
@click.option(
    "--enable-nvlink/--disable-nvlink",
    default=False,
    help="Enable NVLink communication",
)
@click.option(
    "--enable-rdmacm/--disable-rdmacm",
    default=False,
    help="Enable RDMA connection manager, currently requires InfiniBand enabled.",
)
@click.option(
    "--net-devices",
    type=str,
    default=None,
    help="Network interface to establish UCX connection, "
    "usually the Ethernet interface, like 'eth0' or 'enp1s0f0'",
)
def dask_setup(
    service,
    create_cuda_context,
    enable_tcp_over_ucx,
    enable_infiniband,
    enable_nvlink,
    enable_rdmacm,
    net_devices,
):
    if create_cuda_context:
        _create_cuda_context()
