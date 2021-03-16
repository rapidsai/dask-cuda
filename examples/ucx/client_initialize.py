import click

from dask.distributed import Client

from dask_cuda.initialize import initialize


@click.command(context_settings=dict(ignore_unknown_options=True))
@click.argument(
    "address", required=True, type=str,
)
@click.option(
    "--enable-nvlink/--disable-nvlink",
    default=False,
    help="Enable NVLink communication",
)
@click.option(
    "--enable-infiniband/--disable-infiniband",
    default=False,
    help="Enable InfiniBand communication with RDMA",
)
def main(
    address, enable_nvlink, enable_infiniband,
):

    enable_rdmacm = False
    ucx_net_devices = None

    if enable_infiniband:
        enable_rdmacm = True
        ucx_net_devices = "mlx5_0:1"

    initialize(
        enable_tcp_over_ucx=True,
        enable_nvlink=enable_nvlink,
        enable_infiniband=enable_infiniband,
        enable_rdmacm=enable_rdmacm,
        net_devices=ucx_net_devices,
    )

    client = Client(address)  # noqa F841


if __name__ == "__main__":
    main()
