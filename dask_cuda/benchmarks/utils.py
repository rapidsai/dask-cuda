import argparse

from dask.distributed import SSHCluster

from dask_cuda.local_cuda_cluster import LocalCUDACluster


def parse_benchmark_args(description="Generic dask-cuda Benchmark", args_list=[]):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-d", "--devs", default="0", type=str, help='GPU devices to use (default "0").'
    )
    parser.add_argument(
        "-p",
        "--protocol",
        choices=["tcp", "ucx"],
        default="tcp",
        type=str,
        help="The communication protocol to use.",
    )
    parser.add_argument(
        "--profile",
        metavar="PATH",
        default=None,
        type=str,
        help="Write dask profile report (E.g. dask-report.html)",
    )
    parser.add_argument(
        "--rmm-pool-size",
        default=None,
        type=float,
        help="The size of the RMM memory pool. By default, 1/2 of "
        "the total GPU memory is used.",
    )
    parser.add_argument(
        "--disable-rmm-pool", action="store_true", help="Disable the RMM memory pool"
    )
    parser.add_argument(
        "--all-to-all", action="store_true", help="Run all-to-all before computation",
    )
    parser.add_argument(
        "--enable-tcp-over-ucx",
        action="store_true",
        dest="enable_tcp_over_ucx",
        help="Enable tcp over ucx.",
    )
    parser.add_argument(
        "--enable-infiniband",
        action="store_true",
        dest="enable_infiniband",
        help="Enable infiniband over ucx.",
    )
    parser.add_argument(
        "--enable-nvlink",
        action="store_true",
        dest="enable_nvlink",
        help="Enable NVLink over ucx.",
    )
    parser.add_argument(
        "--disable-tcp-over-ucx",
        action="store_false",
        dest="enable_tcp_over_ucx",
        help="Disable tcp over ucx.",
    )
    parser.add_argument(
        "--disable-infiniband",
        action="store_false",
        dest="enable_infiniband",
        help="Disable infiniband over ucx.",
    )
    parser.add_argument(
        "--disable-nvlink",
        action="store_false",
        dest="enable_nvlink",
        help="Disable NVLink over ucx.",
    )
    parser.add_argument(
        "--ucx-net-devices",
        default=None,
        type=str,
        help="The device to be used for UCX communication, or 'auto'. "
        "Ignored if protocol is 'tcp'",
    )
    parser.add_argument(
        "--multi-node",
        action="store_true",
        dest="multi_node",
        help="Runs a multi-node cluster on the hosts specified by --hosts.",
    )
    parser.add_argument(
        "--scheduler-address",
        default=None,
        type=str,
        dest="sched_addr",
        help="Scheduler Address -- assumes cluster is created outside of benchmark.",
    )
    parser.add_argument(
        "--hosts",
        default=None,
        type=str,
        help="Specifies a comma-separated list of IP addresses or hostnames. "
        "The list begins with the host where the scheduler will be launched "
        "followed by any number of workers, with a minimum of 1 worker. "
        "Requires --multi-node, ignored otherwise. "
        "Usage example: --multi-node --hosts 'dgx12,dgx12,10.10.10.10,dgx13' . "
        "In the example, the benchmark is launched with scheduler on host "
        "'dgx12' (first in the list), and workers on three hosts being 'dgx12', "
        "'10.10.10.10', and 'dgx13'. "
        "Note: --devs is currently ignored in multi-node mode and for each host "
        "one worker per GPU will be launched.",
    )

    for args in args_list:
        name = args.pop("name")
        if not isinstance(name, list):
            name = [name]
        parser.add_argument(*name, **args)

    parser.set_defaults(
        enable_tcp_over_ucx=True, enable_infiniband=True, enable_nvlink=True
    )
    args = parser.parse_args()

    if args.protocol == "tcp":
        args.enable_tcp_over_ucx = False
        args.enable_infiniband = False
        args.enable_nvlink = False

    if args.multi_node and len(args.hosts.split(",")) < 2:
        raise ValueError("--multi-node requires at least 2 hosts")

    return args


def get_cluster_options(args):
    if args.multi_node is True:
        Cluster = SSHCluster
        cluster_args = [args.hosts.split(",")]
        scheduler_addr = args.protocol + "://" + cluster_args[0][0] + ":8786"

        worker_options = {}

        # This looks counterintuitive but adding the variable name with
        # an empty string is how we can enable CLI booleans currently,
        # note that SSHCluster uses the dask-cuda-worker CLI.
        if args.enable_tcp_over_ucx:
            worker_options["enable_tcp_over_ucx"] = ""
        if args.enable_nvlink:
            worker_options["enable_nvlink"] = ""
        if args.enable_infiniband:
            worker_options["enable_infiniband"] = ""

        if args.ucx_net_devices:
            worker_options["ucx_net_devices"] = args.ucx_net_devices

        cluster_kwargs = {
            "connect_options": {"known_hosts": None},
            "scheduler_options": {"protocol": args.protocol},
            "worker_module": "dask_cuda.dask_cuda_worker",
            "worker_options": worker_options,
            # "n_workers": len(args.devs.split(",")),
            # "CUDA_VISIBLE_DEVICES": args.devs,
        }
    else:
        Cluster = LocalCUDACluster
        scheduler_addr = None
        cluster_args = []
        cluster_kwargs = {
            "protocol": args.protocol,
            "n_workers": len(args.devs.split(",")),
            "CUDA_VISIBLE_DEVICES": args.devs,
            "ucx_net_devices": args.ucx_net_devices,
            "enable_tcp_over_ucx": args.enable_tcp_over_ucx,
            "enable_infiniband": args.enable_infiniband,
            "enable_nvlink": args.enable_nvlink,
        }

    return {
        "class": Cluster,
        "args": cluster_args,
        "kwargs": cluster_kwargs,
        "scheduler_addr": scheduler_addr,
    }


def get_scheduler_workers(dask_scheduler=None):
    return dask_scheduler.workers


def setup_memory_pool(pool_size=None, disable_pool=False):
    import cupy

    import rmm

    if not disable_pool:
        rmm.reinitialize(
            pool_allocator=True, devices=0, initial_pool_size=pool_size,
        )
        cupy.cuda.set_allocator(rmm.rmm_cupy_allocator)
