import argparse
import os
from datetime import datetime

from dask.distributed import SSHCluster
from dask.utils import parse_bytes

from dask_cuda.local_cuda_cluster import LocalCUDACluster


def parse_benchmark_args(description="Generic dask-cuda Benchmark", args_list=[]):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-d", "--devs", default="0", type=str, help='GPU devices to use (default "0").'
    )
    parser.add_argument(
        "--threads-per-worker",
        default=1,
        type=int,
        help="Number of Dask threads per worker (i.e., GPU).",
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
        "--device-memory-limit",
        default=None,
        type=parse_bytes,
        help="Size of the CUDA device LRU cache, which is used to determine when the "
        "worker starts spilling to host memory. Can be an integer (bytes), float "
        "(fraction of total device memory), string (like ``'5GB'`` or ``'5000M'``), or "
        "``'auto'``, 0, or ``None`` to disable spilling to host (i.e. allow full "
        "device memory usage).",
    )
    parser.add_argument(
        "--rmm-pool-size",
        default=None,
        type=parse_bytes,
        help="The size of the RMM memory pool. Can be an integer (bytes) or a string "
        "(like '4GB' or '5000M'). By default, 1/2 of the total GPU memory is used.",
    )
    parser.add_argument(
        "--disable-rmm-pool", action="store_true", help="Disable the RMM memory pool"
    )
    parser.add_argument(
        "--rmm-log-directory",
        default=None,
        type=str,
        help="Directory to write worker and scheduler RMM log files to. "
        "Logging is only enabled if RMM memory pool is enabled.",
    )
    parser.add_argument(
        "--all-to-all", action="store_true", help="Run all-to-all before computation",
    )
    parser.add_argument(
        "--enable-tcp-over-ucx",
        action="store_true",
        dest="enable_tcp_over_ucx",
        help="Enable TCP over UCX.",
    )
    parser.add_argument(
        "--enable-infiniband",
        action="store_true",
        dest="enable_infiniband",
        help="Enable InfiniBand over UCX.",
    )
    parser.add_argument(
        "--enable-nvlink",
        action="store_true",
        dest="enable_nvlink",
        help="Enable NVLink over UCX.",
    )
    parser.add_argument(
        "--enable-rdmacm",
        action="store_true",
        dest="enable_rdmacm",
        help="Enable RDMACM with UCX.",
    )
    parser.add_argument(
        "--disable-tcp-over-ucx",
        action="store_false",
        dest="enable_tcp_over_ucx",
        help="Disable TCP over UCX.",
    )
    parser.add_argument(
        "--disable-infiniband",
        action="store_false",
        dest="enable_infiniband",
        help="Disable InfiniBand over UCX.",
    )
    parser.add_argument(
        "--disable-nvlink",
        action="store_false",
        dest="enable_nvlink",
        help="Disable NVLink over UCX.",
    )
    parser.add_argument(
        "--disable-rdmacm",
        action="store_false",
        dest="enable_rdmacm",
        help="Disable RDMACM with UCX.",
    )
    parser.add_argument(
        "--ucx-net-devices",
        default=None,
        type=str,
        help="The device to be used for UCX communication, or 'auto'. "
        "Ignored if protocol is 'tcp'",
    )
    parser.add_argument(
        "--interface",
        default=None,
        type=str,
        dest="interface",
        help="Network interface Dask processes will use to listen for connections.",
    )
    parser.add_argument(
        "--no-silence-logs",
        action="store_true",
        help="By default Dask logs are silenced, this argument unsilence them.",
    )
    parser.add_argument(
        "--multi-node",
        action="store_true",
        dest="multi_node",
        help="Runs a multi-node cluster on the hosts specified by --hosts."
        "Requires the ``asyncssh`` module to be installed.",
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
    parser.add_argument(
        "--plot",
        metavar="PATH",
        default=None,
        type=str,
        help="Generate plot output written to defined directory",
    )
    parser.add_argument(
        "--benchmark-json",
        default=None,
        type=str,
        help="Dump a line-delimited JSON report of benchmarks to this file (optional). "
        "Creates file if it does not exist, appends otherwise.",
    )

    for args in args_list:
        name = args.pop("name")
        if not isinstance(name, list):
            name = [name]
        parser.add_argument(*name, **args)

    parser.set_defaults(
        enable_tcp_over_ucx=True,
        enable_infiniband=True,
        enable_nvlink=True,
        enable_rdmacm=False,
    )
    args = parser.parse_args()

    if args.protocol == "tcp":
        args.enable_tcp_over_ucx = False
        args.enable_infiniband = False
        args.enable_nvlink = False
        args.enable_rdmacm = False

    if args.multi_node and len(args.hosts.split(",")) < 2:
        raise ValueError("--multi-node requires at least 2 hosts")

    return args


def get_cluster_options(args):
    if args.multi_node is True:
        Cluster = SSHCluster
        cluster_args = [args.hosts.split(",")]
        scheduler_addr = args.protocol + "://" + cluster_args[0][0] + ":8786"

        cluster_kwargs = {
            "connect_options": {"known_hosts": None},
            "scheduler_options": {"protocol": args.protocol, "port": 8786},
            "worker_class": "dask_cuda.CUDAWorker",
            "worker_options": {
                "protocol": args.protocol,
                "nthreads": args.threads_per_worker,
                "net_devices": args.ucx_net_devices,
                "enable_tcp_over_ucx": args.enable_tcp_over_ucx,
                "enable_infiniband": args.enable_infiniband,
                "enable_nvlink": args.enable_nvlink,
                "enable_rdmacm": args.enable_rdmacm,
                "interface": args.interface,
                "device_memory_limit": args.device_memory_limit,
            },
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
            "threads_per_worker": args.threads_per_worker,
            "CUDA_VISIBLE_DEVICES": args.devs,
            "ucx_net_devices": args.ucx_net_devices,
            "enable_tcp_over_ucx": args.enable_tcp_over_ucx,
            "enable_infiniband": args.enable_infiniband,
            "enable_nvlink": args.enable_nvlink,
            "enable_rdmacm": args.enable_rdmacm,
            "interface": args.interface,
            "device_memory_limit": args.device_memory_limit,
        }
        if args.no_silence_logs:
            cluster_kwargs["silence_logs"] = False

    return {
        "class": Cluster,
        "args": cluster_args,
        "kwargs": cluster_kwargs,
        "scheduler_addr": scheduler_addr,
    }


def get_scheduler_workers(dask_scheduler=None):
    return dask_scheduler.workers


def setup_memory_pool(
    dask_worker=None, pool_size=None, disable_pool=False, log_directory=None,
):
    import cupy

    import rmm

    from dask_cuda.utils import get_rmm_log_file_name

    logging = log_directory is not None

    if not disable_pool:
        rmm.reinitialize(
            pool_allocator=True,
            devices=0,
            initial_pool_size=pool_size,
            logging=logging,
            log_file_name=get_rmm_log_file_name(dask_worker, logging, log_directory),
        )
        cupy.cuda.set_allocator(rmm.rmm_cupy_allocator)


def plot_benchmark(t_runs, path, historical=False):
    """
    Plot the throughput the benchmark for each run.  If historical=True,
    Load historical data from ~/benchmark-historic-runs.csv
    """
    try:
        import pandas as pd
        import seaborn as sns
    except ImportError:
        print(
            "Plotting libraries are not installed.  Please install pandas, "
            "seaborn, and matplotlib"
        )
        return

    x = [str(x) for x in range(len(t_runs))]
    df = pd.DataFrame(dict(t_runs=t_runs, x=x))
    avg = round(df.t_runs.mean(), 2)

    ax = sns.barplot(x="x", y="t_runs", data=df, color="purple")

    ax.set(
        xlabel="Run Iteration",
        ylabel="Merge Throughput in GB/s",
        title=f"cudf Merge Throughput -- Average {avg} GB/s",
    )
    fig = ax.get_figure()
    today = datetime.now().strftime("%Y%m%d")
    fname_bench = today + "-benchmark.png"
    d = os.path.expanduser(path)
    bench_path = os.path.join(d, fname_bench)
    fig.savefig(bench_path)

    if historical:
        # record average tohroughput and plot historical averages
        history_file = os.path.join(
            os.path.expanduser("~"), "benchmark-historic-runs.csv"
        )
        with open(history_file, "a+") as f:
            f.write(f"{today},{avg}\n")

        df = pd.read_csv(
            history_file, names=["date", "throughput"], parse_dates=["date"]
        )
        ax = df.plot(
            x="date", y="throughput", marker="o", title="Historical Throughput"
        )

        ax.set_ylim(0, 30)

        fig = ax.get_figure()
        fname_hist = today + "-benchmark-history.png"
        hist_path = os.path.join(d, fname_hist)
        fig.savefig(hist_path)
