"""
This initialization scripts will create CUDA context and initialize UCX-Py,
depending on user parameters.

It is sometimes convenient to initialize the CUDA context, particularly before
starting up Dask workers which create a variety of threads.

To ensure UCX works correctly, it is important to ensure it is initialized with
the correct options. This is important for scheduler, workers and client. This
initialization script will ensure that based on the flags and options passed by
the user.

This module is intended to be used within a Worker preload script.
https://docs.dask.org/en/latest/setup/custom-startup.html

You can add it to your global config with the following yaml

    distributed:
      worker:
        preload:
          - dask_cuda.initialize_ucx

See https://docs.dask.org/en/latest/configuration.html for more information
about Dask configuration.
"""
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
