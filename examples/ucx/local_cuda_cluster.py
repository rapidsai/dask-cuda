import click
import cupy

from dask import array as da
from dask.distributed import Client
from dask.utils import parse_bytes

from dask_cuda import LocalCUDACluster


@click.command(context_settings=dict(ignore_unknown_options=True))
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
@click.option(
    "--interface",
    default=None,
    type=str,
    help="Interface used by scheduler for communication. Must be "
    "specified if NVLink or InfiniBand are enabled.",
)
@click.option(
    "--rmm-pool-size",
    default="1GB",
    type=parse_bytes,
    help="If specified, initialize each worker with an RMM pool of "
    "the given size, otherwise no RMM pool is created. This can be "
    "an integer (bytes) or string (like 5GB or 5000M).",
)
def main(
    enable_nvlink,
    enable_infiniband,
    enable_rdmacm,
    interface,
    rmm_pool_size,
):

    if (enable_infiniband or enable_nvlink) and not interface:
        raise ValueError(
            "Interface must be specified if NVLink or Infiniband are enabled"
        )

    # initialize scheduler & workers
    cluster = LocalCUDACluster(
        enable_tcp_over_ucx=True,
        enable_nvlink=enable_nvlink,
        enable_infiniband=enable_infiniband,
        enable_rdmacm=enable_rdmacm,
        interface=interface,
        rmm_pool_size=rmm_pool_size,
    )

    # initialize client
    client = Client(cluster)

    # user code here
    rs = da.random.RandomState(RandomState=cupy.random.RandomState)
    x = rs.random((10000, 10000), chunks=1000)
    x.sum().compute()

    # shutdown cluster
    client.shutdown()


if __name__ == "__main__":
    main()
