import click
import cupy

from dask import array as da
from dask.distributed import Client

from dask_cuda.initialize import initialize


@click.command(context_settings=dict(ignore_unknown_options=True))
@click.argument(
    "address",
    required=True,
    type=str,
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
@click.option(
    "--enable-rdmacm/--disable-rdmacm",
    default=False,
    help="Enable RDMA connection manager, requires --enable-infiniband",
)
def main(
    address,
    enable_nvlink,
    enable_infiniband,
    enable_rdmacm,
):

    # set up environment
    initialize(
        enable_tcp_over_ucx=True,
        enable_nvlink=enable_nvlink,
        enable_infiniband=enable_infiniband,
        enable_rdmacm=enable_rdmacm,
    )

    # initialize client
    client = Client(address)

    # user code here
    rs = da.random.RandomState(RandomState=cupy.random.RandomState)
    x = rs.random((10000, 10000), chunks=1000)
    x.sum().compute()

    # shutdown cluster
    client.shutdown()


if __name__ == "__main__":
    main()
