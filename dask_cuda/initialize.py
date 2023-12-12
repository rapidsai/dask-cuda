import logging
import os

import click
import numba.cuda

import dask
from distributed.diagnostics.nvml import get_device_index_and_uuid, has_cuda_context

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


def _create_cuda_context(protocol="ucx"):
    if protocol not in ["ucx", "ucxx"]:
        return
    try:
        # Added here to ensure the parent `LocalCUDACluster` process creates the CUDA
        # context directly from the UCX module, thus avoiding a similar warning there.
        try:
            if protocol == "ucx":
                import distributed.comm.ucx

                distributed.comm.ucx.init_once()
            elif protocol == "ucxx":
                import distributed_ucxx.ucxx

                distributed_ucxx.ucxx.init_once()
        except ModuleNotFoundError:
            # UCX initialization has to be delegated to Distributed, it will take care
            # of setting correct environment variables and importing `ucp` after that.
            # Therefore if ``import ucp`` fails we can just continue here.
            pass

        cuda_visible_device = get_device_index_and_uuid(
            os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        )
        ctx = has_cuda_context()
        if protocol == "ucx":
            if (
                ctx.has_context
                and not distributed.comm.ucx.cuda_context_created.has_context
            ):
                distributed.comm.ucx._warn_existing_cuda_context(ctx, os.getpid())
        elif protocol == "ucxx":
            if (
                ctx.has_context
                and not distributed_ucxx.ucxx.cuda_context_created.has_context
            ):
                distributed_ucxx.ucxx._warn_existing_cuda_context(ctx, os.getpid())

        _create_cuda_context_handler()

        if protocol == "ucx":
            if not distributed.comm.ucx.cuda_context_created.has_context:
                ctx = has_cuda_context()
                if ctx.has_context and ctx.device_info != cuda_visible_device:
                    distributed.comm.ucx._warn_cuda_context_wrong_device(
                        cuda_visible_device, ctx.device_info, os.getpid()
                    )
        elif protocol == "ucxx":
            if not distributed_ucxx.ucxx.cuda_context_created.has_context:
                ctx = has_cuda_context()
                if ctx.has_context and ctx.device_info != cuda_visible_device:
                    distributed_ucxx.ucxx._warn_cuda_context_wrong_device(
                        cuda_visible_device, ctx.device_info, os.getpid()
                    )

    except Exception:
        logger.error("Unable to start CUDA Context", exc_info=True)


def initialize(
    create_cuda_context=True,
    enable_tcp_over_ucx=None,
    enable_infiniband=None,
    enable_nvlink=None,
    enable_rdmacm=None,
    protocol="ucx",
):
    """Create CUDA context and initialize UCX-Py, depending on user parameters.

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
    dask.config.set({"distributed.comm.ucx": ucx_config})

    if create_cuda_context:
        _create_cuda_context(protocol=protocol)


@click.command()
@click.option(
    "--create-cuda-context/--no-create-cuda-context",
    default=False,
    help="Create CUDA context",
)
@click.option(
    "--protocol",
    default=None,
    type=str,
    help="Communication protocol, such as: 'tcp', 'tls', 'ucx' or 'ucxx'.",
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
def dask_setup(
    service,
    create_cuda_context,
    protocol,
    enable_tcp_over_ucx,
    enable_infiniband,
    enable_nvlink,
    enable_rdmacm,
):
    if create_cuda_context:
        _create_cuda_context(protocol=protocol)
