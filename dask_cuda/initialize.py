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
import click
import logging
import warnings

import numba.cuda


logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--create-cuda-context/--no-create-cuda-context",
    default=False,
    help="Create CUDA context",
)
@click.option(
    "--enable-tcp/--disable-tcp", default=True, help="Enable TCP communication over UCX"
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
    "--net-devices",
    type=str,
    default=None,
    help="Network interface to establish UCX connection, "
    "usually the Ethernet interface, like 'eth0' or 'enp1s0f0'",
)
def dask_setup(
    service,
    create_cuda_context,
    enable_tcp,
    enable_infiniband,
    enable_nvlink,
    net_devices,
):
    if create_cuda_context:
        try:
            numba.cuda.current_context()
        except Exception:
            logger.error("Unable to start CUDA Context", exc_info=True)

    if enable_tcp or enable_infiniband or enable_nvlink:
        try:
            import ucp
        except ImportError:
            logger.error(
                "UCX protocol requested but ucp module is not available",
                exc_info=True,
            )
        else:
            print("UCP ELSE")
            options = {}
            if enable_tcp or enable_infiniband or enable_nvlink:
                tls = "tcp,sockcm,cuda_copy"
                tls_priority = "sockcm"

                if enable_infiniband:
                    tls = "rc," + tls
                if enable_nvlink:
                    tls = tls + ",cuda_ipc"

                options = {"TLS": tls, "SOCKADDR_TLS_PRIORITY": tls_priority}

                if net_devices is not None and net_devices != "":
                    options["NET_DEVICES"] = net_devices

            print("NET_DEVICES:", net_devices)
            print(options)

            ucp.reset()
            ucp.init(options=options)
