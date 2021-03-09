import logging

import click
import numba.cuda

import dask

from .utils import get_ucx_config

logger = logging.getLogger(__name__)


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
    starting up Dask workers which create a variety of threads.

    To ensure UCX works correctly, it is important to ensure it is initialized with
    the correct options. This is especially important for the client, which cannot be
    configured to use UCX with arguments like ``LocalCUDACluster`` and
    ``dask-cuda-worker``. This function will ensure that they are provided a UCX
    configuration based on the flags and options passed by the user.

    This function can also be used within a worker preload script for UCX configuration
    of mainline Dask/Distributed.
    https://docs.dask.org/en/latest/setup/custom-startup.html

    You can add it to your global config with the following YAML:

    .. code-block:: yaml

        distributed:
          worker:
            preload:
              - dask_cuda.initialize

    See https://docs.dask.org/en/latest/configuration.html for more information
    about Dask configuration.

    Parameters
    ----------
    create_cuda_context: bool
        Create CUDA context on initialization.
        Default is ``True``.
    enable_tcp_over_ucx: bool
        Set environment variables to enable TCP over UCX, even if InfiniBand
        and NVLink are not supported or disabled.
        Default is ``False``.
    enable_infiniband: bool
        Set environment variables to enable UCX InfiniBand support, implies
        ``enable_tcp_over_ucx=True``.
        Default is ``False``.
    enable_nvlink: bool
        Set environment variables to enable UCX NVLink support, implies
        ``enable_tcp_over_ucx=True``.
        Default is ``False``.
    enable_rdmacm: bool
        Set environment variables to enable UCX RDMA connection manager support,
        implies ``enable_infiniband=True``.
        Default is ``False``.
    net_devices: callable or str
        If callable, the function must take exactly one argument (the index of
        current GPU) that will be used to get the interface name, such as
        ``lambda dev: "mlx5_%d:1" % (dev // 2)``, which would return
        ``"mlx5_1:1"`` for GPU 3.
        If a string, must be an explicit interface name, such as ``"ib0"``
        for InfiniBand or ``"eth0"`` if InfiniBand is disabled.
        Default is ``""``, which will result in all available devices being used.
    cuda_device_index: None or int
        Index of the current GPU, which will be supplied to ``net_devices`` if
        it is callable.
        Default is ``None``.
    """

    if create_cuda_context:
        try:
            numba.cuda.current_context()
        except Exception:
            logger.error("Unable to start CUDA Context", exc_info=True)

    ucx_config = get_ucx_config(
        enable_tcp_over_ucx=enable_tcp_over_ucx,
        enable_infiniband=enable_infiniband,
        enable_nvlink=enable_nvlink,
        enable_rdmacm=enable_rdmacm,
        net_devices=net_devices,
        cuda_device_index=cuda_device_index,
    )
    dask.config.update(dask.config.global_config, {"ucx": ucx_config}, priority="new")


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
        try:
            numba.cuda.current_context()
        except Exception:
            logger.error("Unable to start CUDA Context", exc_info=True)
