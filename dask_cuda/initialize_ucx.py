"""
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
import ucp
import warnings

from distributed.cli.utils import check_python_3


@click.command()
@click.option(
    "--enable-tcp/--disable-tcp",
    default=True,
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
    "--interface", type=str, default=None,
    help="Network interface to establish UCX connection, "
    "usually the Ethernet interface, like 'eth0' or 'enp1s0f0'"
)
@click.option(
    "--net-devices", type=str, default=None,
    help="Network interface to establish UCX connection, "
    "usually the Ethernet interface, like 'eth0' or 'enp1s0f0'"
)
def dask_setup(service, enable_tcp, enable_infiniband, enable_nvlink, interface, net_devices):
    options = {}
    if enable_tcp or enable_infiniband or enable_nvlink:
        tls = "tcp,sockcm,cuda_copy"
        tls_priority = "sockcm"
        ifname = ""

        if enable_infiniband:
            if interface is None or interface == "":
                warnings.warn(
                    "InfiniBand requested but no interface specified, this may cause issues "
                    "for UCX."
                )
            tls = "rc," + tls
            ifname = interface or ""
        if enable_nvlink:
            tls = tls + ",cuda_ipc"

        options = {
            "TLS": tls,
            "SOCKADDR_TLS_PRIORITY": tls_priority,
        }

        if net_devices is not None and net_devices != "":
            options["NET_DEVICES"] = net_devices

    ucp.reset()
    ucp.init(options=options)
