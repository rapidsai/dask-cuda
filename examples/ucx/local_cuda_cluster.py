import click

from dask.distributed import Client
from dask.utils import parse_bytes

from dask_cuda import LocalCUDACluster
from dask_cuda.initialize import initialize


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
    enable_nvlink, enable_infiniband, interface, rmm_pool_size,
):

    enable_rdmacm = False
    client_devices = worker_devices = None

    if enable_infiniband:
        enable_rdmacm = True
        client_devices = "mlx5_0:1"
        worker_devices = "auto"

    if (enable_infiniband or enable_nvlink) and not interface:
        raise ValueError(
            "Interface must be specified if NVLink or Infiniband are enabled"
        )

    # set up environment
    initialize(
        enable_tcp_over_ucx=True,
        enable_nvlink=enable_nvlink,
        enable_infiniband=enable_infiniband,
        enable_rdmacm=enable_rdmacm,
        net_devices=client_devices,
    )

    # initialize scheduler & workers
    cluster = LocalCUDACluster(
        enable_tcp_over_ucx=True,
        enable_nvlink=enable_nvlink,
        enable_infiniband=enable_infiniband,
        enable_rdmacm=enable_rdmacm,
        ucx_net_devices=worker_devices,
        interface=interface,
        rmm_pool_size=rmm_pool_size,
    )

    # initialize client
    client = Client(cluster)  # noqa F842


if __name__ == "__main__":
    main()
