import argparse
import itertools
import json
import os
import time
from collections import defaultdict
from datetime import datetime
from operator import itemgetter
from typing import Any, Callable, Mapping, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd

from dask.distributed import Client, SSHCluster
from dask.utils import format_bytes, format_time, parse_bytes
from distributed.comm.addressing import get_address_host

from dask_cuda.local_cuda_cluster import LocalCUDACluster


def as_noop(dsk):
    """
    Turn the given dask computation into a noop.

    Uses dask-noop (https://github.com/gjoseph92/dask-noop/)

    Parameters
    ----------
    dsk
        Dask object (on which one could call compute)

    Returns
    -------
    New dask object representing the same task graph with no
    computation/data attached.

    Raises
    ------
    RuntimeError
        If dask_noop is not importable
    """
    try:
        from dask_noop import as_noop

        return as_noop(dsk)
    except ImportError:
        raise RuntimeError("Requested noop computation but dask-noop not installed.")


def parse_benchmark_args(description="Generic dask-cuda Benchmark", args_list=[]):
    parser = argparse.ArgumentParser(description=description)
    worker_args = parser.add_argument_group(description="Worker configuration")
    worker_args.add_argument(
        "-d", "--devs", default="0", type=str, help='GPU devices to use (default "0").'
    )
    worker_args.add_argument(
        "--threads-per-worker",
        default=1,
        type=int,
        help="Number of Dask threads per worker (i.e., GPU).",
    )
    worker_args.add_argument(
        "--device-memory-limit",
        default=None,
        type=parse_bytes,
        help="Size of the CUDA device LRU cache, which is used to determine when the "
        "worker starts spilling to host memory. Can be an integer (bytes), float "
        "(fraction of total device memory), string (like ``'5GB'`` or ``'5000M'``), or "
        "``'auto'``, 0, or ``None`` to disable spilling to host (i.e. allow full "
        "device memory usage).",
    )
    cluster_args = parser.add_argument_group(description="Cluster configuration")
    cluster_args.add_argument(
        "-p",
        "--protocol",
        choices=["tcp", "ucx"],
        default="tcp",
        type=str,
        help="The communication protocol to use.",
    )
    cluster_args.add_argument(
        "--multiprocessing-method",
        default="spawn",
        choices=["spawn", "fork", "forkserver"],
        type=str,
        help="Which method should multiprocessing use to start child processes? "
        "On supercomputing systems with a high-performance interconnect, "
        "'forkserver' can be used to avoid issues with fork not being allowed "
        "after the networking stack has been initialised.",
    )
    cluster_args.add_argument(
        "--rmm-pool-size",
        default=None,
        type=parse_bytes,
        help="The size of the RMM memory pool. Can be an integer (bytes) or a string "
        "(like '4GB' or '5000M'). By default, 1/2 of the total GPU memory is used.",
    )
    cluster_args.add_argument(
        "--disable-rmm-pool", action="store_true", help="Disable the RMM memory pool"
    )
    cluster_args.add_argument(
        "--enable-rmm-managed",
        action="store_true",
        help="Enable RMM managed memory allocator",
    )
    cluster_args.add_argument(
        "--enable-rmm-async",
        action="store_true",
        help="Enable RMM async memory allocator (implies --disable-rmm-pool)",
    )
    cluster_args.add_argument(
        "--rmm-release-threshold",
        default=None,
        type=parse_bytes,
        help="When --enable-rmm-async is set and the pool size grows beyond this "
        "value, unused memory held by the pool will be released at the next "
        "synchronization point. Can be an integer (bytes), or a string string (like "
        "'4GB' or '5000M'). By default, this feature is disabled.",
    )
    cluster_args.add_argument(
        "--rmm-log-directory",
        default=None,
        type=str,
        help="Directory to write worker and scheduler RMM log files to. "
        "Logging is only enabled if RMM memory pool is enabled.",
    )
    cluster_args.add_argument(
        "--enable-rmm-statistics",
        action="store_true",
        help="Use RMM's StatisticsResourceAdaptor to gather allocation statistics. "
        "This enables spilling implementations such as JIT-Unspill to provides more "
        "information on out-of-memory errors",
    )
    cluster_args.add_argument(
        "--enable-rmm-track-allocations",
        action="store_true",
        help="When enabled, wraps the memory resource used by each worker with a "
        "``rmm.mr.TrackingResourceAdaptor``, which tracks the amount of memory "
        "allocated."
        "NOTE: This option enables additional diagnostics to be collected and "
        "reported by the Dask dashboard. However, there is significant overhead "
        "associated with this and it should only be used for debugging and memory "
        "profiling.",
    )
    cluster_args.add_argument(
        "--enable-tcp-over-ucx",
        default=None,
        action="store_true",
        dest="enable_tcp_over_ucx",
        help="Enable TCP over UCX.",
    )
    cluster_args.add_argument(
        "--enable-infiniband",
        default=None,
        action="store_true",
        dest="enable_infiniband",
        help="Enable InfiniBand over UCX.",
    )
    cluster_args.add_argument(
        "--enable-nvlink",
        default=None,
        action="store_true",
        dest="enable_nvlink",
        help="Enable NVLink over UCX.",
    )
    cluster_args.add_argument(
        "--enable-rdmacm",
        default=None,
        action="store_true",
        dest="enable_rdmacm",
        help="Enable RDMACM with UCX.",
    )
    cluster_args.add_argument(
        "--disable-tcp-over-ucx",
        action="store_false",
        dest="enable_tcp_over_ucx",
        help="Disable TCP over UCX.",
    )
    cluster_args.add_argument(
        "--disable-infiniband",
        action="store_false",
        dest="enable_infiniband",
        help="Disable InfiniBand over UCX.",
    )
    cluster_args.add_argument(
        "--disable-nvlink",
        action="store_false",
        dest="enable_nvlink",
        help="Disable NVLink over UCX.",
    )
    cluster_args.add_argument(
        "--disable-rdmacm",
        action="store_false",
        dest="enable_rdmacm",
        help="Disable RDMACM with UCX.",
    )
    cluster_args.add_argument(
        "--interface",
        default=None,
        type=str,
        dest="interface",
        help="Network interface Dask processes will use to listen for connections.",
    )
    group = cluster_args.add_mutually_exclusive_group()
    group.add_argument(
        "--scheduler-address",
        default=None,
        type=str,
        help="Scheduler Address -- assumes cluster is created outside of benchmark. "
        "If provided, worker configuration options provided to this script are ignored "
        "since the workers are assumed to be started separately. Similarly the other "
        "cluster configuration options have no effect.",
    )
    group.add_argument(
        "--scheduler-file",
        default=None,
        type=str,
        dest="scheduler_file",
        help="Read cluster configuration from specified file. "
        "If provided, worker configuration options provided to this script are ignored "
        "since the workers are assumed to be started separately. Similarly the other "
        "cluster configuration options have no effect.",
    )
    group.add_argument(
        "--dashboard-address",
        default=None,
        type=str,
        help="Address on which to listen for diagnostics dashboard, ignored if "
        "either ``--scheduler-address`` or ``--scheduler-file`` is specified.",
    )
    cluster_args.add_argument(
        "--shutdown-external-cluster-on-exit",
        default=False,
        action="store_true",
        dest="shutdown_cluster",
        help="If connecting to an external cluster, should we shut down the cluster "
        "when the benchmark exits?",
    )
    cluster_args.add_argument(
        "--multi-node",
        action="store_true",
        dest="multi_node",
        help="Runs a multi-node cluster on the hosts specified by --hosts."
        "Requires the ``asyncssh`` module to be installed.",
    )
    cluster_args.add_argument(
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
        "--no-show-p2p-bandwidth",
        action="store_true",
        help="Do not produce detailed point to point bandwidth stats in output",
    )
    parser.add_argument(
        "--all-to-all",
        action="store_true",
        help="Run all-to-all before computation",
    )
    parser.add_argument(
        "--no-silence-logs",
        action="store_true",
        help="By default Dask logs are silenced, this argument unsilence them.",
    )
    parser.add_argument(
        "--plot",
        metavar="PATH",
        default=None,
        type=str,
        help="Generate plot output written to defined directory",
    )
    parser.add_argument(
        "--markdown",
        default=False,
        action="store_true",
        help="Write output as markdown",
    )
    parser.add_argument(
        "--profile",
        metavar="PATH",
        default=None,
        type=str,
        help="Write dask profile report (E.g. dask-report.html)",
    )
    # See save_benchmark_data for more information
    parser.add_argument(
        "--output-basename",
        default=None,
        type=str,
        help="Dump a benchmark data to files using this basename. "
        "Produces three files, BASENAME.json (containing timing data); "
        "BASENAME.npy (point to point bandwidth statistics); "
        "BASENAME.address_map.json (mapping from worker addresses to indices). "
        "If the files already exist, new files are created with a uniquified "
        "BASENAME.",
    )

    for args in args_list:
        name = args.pop("name")
        if not isinstance(name, list):
            name = [name]
        parser.add_argument(*name, **args)

    args = parser.parse_args()

    if args.multi_node and len(args.hosts.split(",")) < 2:
        raise ValueError("--multi-node requires at least 2 hosts")

    return args


def get_cluster_options(args):
    ucx_options = {
        "enable_tcp_over_ucx": args.enable_tcp_over_ucx,
        "enable_infiniband": args.enable_infiniband,
        "enable_nvlink": args.enable_nvlink,
        "enable_rdmacm": args.enable_rdmacm,
    }

    if args.multi_node is True:
        Cluster = SSHCluster
        cluster_args = [args.hosts.split(",")]
        scheduler_addr = args.protocol + "://" + cluster_args[0][0] + ":8786"

        cluster_kwargs = {
            "connect_options": {"known_hosts": None},
            "scheduler_options": {
                "protocol": args.protocol,
                "port": 8786,
                "dashboard_address": args.dashboard_address,
            },
            "worker_class": "dask_cuda.CUDAWorker",
            "worker_options": {
                "protocol": args.protocol,
                "nthreads": args.threads_per_worker,
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
            "dashboard_address": args.dashboard_address,
            "n_workers": len(args.devs.split(",")),
            "threads_per_worker": args.threads_per_worker,
            "CUDA_VISIBLE_DEVICES": args.devs,
            "interface": args.interface,
            "device_memory_limit": args.device_memory_limit,
            **ucx_options,
        }
        if args.no_silence_logs:
            cluster_kwargs["silence_logs"] = False

    return {
        "class": Cluster,
        "args": cluster_args,
        "kwargs": cluster_kwargs,
        "scheduler_addr": scheduler_addr,
    }


def get_worker_device():
    try:
        device, *_ = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        return int(device)
    except (KeyError, ValueError):
        # No CUDA_VISIBILE_DEVICES in environment, or else no appropriate value
        return -1


def setup_memory_pool(
    dask_worker=None,
    pool_size=None,
    disable_pool=False,
    rmm_async=False,
    rmm_managed=False,
    release_threshold=None,
    log_directory=None,
    statistics=False,
    rmm_track_allocations=False,
):
    import cupy

    import rmm
    from rmm.allocators.cupy import rmm_cupy_allocator

    from dask_cuda.utils import get_rmm_log_file_name

    logging = log_directory is not None

    if rmm_async:
        rmm.mr.set_current_device_resource(
            rmm.mr.CudaAsyncMemoryResource(
                initial_pool_size=pool_size, release_threshold=release_threshold
            )
        )
    else:
        rmm.reinitialize(
            pool_allocator=not disable_pool,
            managed_memory=rmm_managed,
            initial_pool_size=pool_size,
            logging=logging,
            log_file_name=get_rmm_log_file_name(dask_worker, logging, log_directory),
        )
    cupy.cuda.set_allocator(rmm_cupy_allocator)
    if statistics:
        rmm.mr.set_current_device_resource(
            rmm.mr.StatisticsResourceAdaptor(rmm.mr.get_current_device_resource())
        )
    if rmm_track_allocations:
        rmm.mr.set_current_device_resource(
            rmm.mr.TrackingResourceAdaptor(rmm.mr.get_current_device_resource())
        )


def setup_memory_pools(
    client,
    is_gpu,
    pool_size,
    disable_pool,
    rmm_async,
    rmm_managed,
    release_threshold,
    log_directory,
    statistics,
    rmm_track_allocations,
):
    if not is_gpu:
        return
    client.run(
        setup_memory_pool,
        pool_size=pool_size,
        disable_pool=disable_pool,
        rmm_async=rmm_async,
        rmm_managed=rmm_managed,
        release_threshold=release_threshold,
        log_directory=log_directory,
        statistics=statistics,
        rmm_track_allocations=rmm_track_allocations,
    )
    # Create an RMM pool on the scheduler due to occasional deserialization
    # of CUDA objects. May cause issues with InfiniBand otherwise.
    client.run_on_scheduler(
        setup_memory_pool,
        pool_size=1e9,
        disable_pool=disable_pool,
        rmm_async=rmm_async,
        rmm_managed=rmm_managed,
        release_threshold=release_threshold,
        log_directory=log_directory,
        statistics=statistics,
        rmm_track_allocations=rmm_track_allocations,
    )


def save_benchmark_data(
    basename,
    address_to_index: Mapping[str, int],
    timing_data: pd.DataFrame,
    p2p_data: np.ndarray,
):
    """Save benchmark data to files

    Parameters
    ----------
    basename: str
        Output file basename
    address_to_index
        Mapping from worker addresses to indices (in the p2p_data array)
    timing_data
        DataFrame containing timing and configuration data
    p2p_data
        numpy array of point to point bandwidth statistics

    Notes
    -----
    Produces ``BASENAME.json``, ``BASENAME.npy``, ``BASENAME.address_map.json``.
    If any of these files exist then ``basename`` is uniquified by
    appending the ISO date and a sequence number.
    """

    def exists(basename):
        return any(
            os.path.exists(f"{basename}{ext}")
            for ext in [".json", ".npy", ".address_map.json"]
        )

    new_basename = basename
    sequence = itertools.count()
    while exists(new_basename):
        now = datetime.now().strftime("%Y%m%d")
        new_basename = f"{basename}-{now}.{next(sequence)}"
    timing_data.to_json(f"{new_basename}.json")
    np.save(f"{new_basename}.npy", p2p_data)
    with open(f"{new_basename}.address_map.json", "w") as f:
        f.write(json.dumps(address_to_index))


def wait_for_cluster(client, timeout=120, shutdown_on_failure=True):
    """Wait for the cluster to come up.

    Parameters
    ----------
    client
        The distributed Client object
    timeout: int (optional)
        Timeout in seconds before we give up
    shutdown_on_failure: bool (optional)
        Should we call ``client.shutdown()`` if not all workers are
        found after the timeout is reached?

    Raises
    ------
    RuntimeError:
        If the timeout finishes and not all expected workers have
        appeared.
    """
    expected = os.environ.get("EXPECTED_NUM_WORKERS")
    if expected is None:
        return
    expected = int(expected)
    nworkers = 0
    for _ in range(timeout // 5):
        print(
            "Waiting for workers to come up, "
            f"have {len(client.scheduler_info().get('workers', []))}, "
            f"want {expected}"
        )
        time.sleep(5)
        nworkers = len(client.scheduler_info().get("workers", []))
        if nworkers == expected:
            return
    else:
        if shutdown_on_failure:
            client.shutdown()
        raise RuntimeError(
            f"Not all workers up after {timeout}s; "
            f"got {nworkers}, wanted {expected}"
        )


def address_to_index(client: Client) -> Mapping[str, int]:
    """Produce a mapping from worker addresses to unique indices

    Parameters
    ----------
    client: Client
        distributed client

    Returns
    -------
    Mapping from worker addresses to int, with workers on the same
    host numbered contiguously, and sorted by device index on each host.
    """
    # Group workers by hostname and then device index
    addresses = client.run(get_worker_device)
    return dict(
        zip(
            sorted(addresses, key=lambda k: (get_address_host(k), addresses[k])),
            itertools.count(),
        )
    )


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


def print_separator(separator="-", length=80):
    print(separator * length)


def print_key_value(key, value, key_length=25):
    print(f"{key: <{key_length}} | {value}")


def print_throughput_bandwidth(
    args, durations, data_processed, p2p_bw, address_to_index
):
    print_key_value(key="Number of workers", value=f"{len(address_to_index)}")
    print_separator(separator="=")
    print_key_value(key="Wall clock", value="Throughput")
    print_separator(separator="-")
    durations = np.asarray(durations)
    data_processed = np.asarray(data_processed)
    throughputs = data_processed / durations
    for duration, throughput in zip(durations, throughputs):
        print_key_value(
            key=f"{format_time(duration)}", value=f"{format_bytes(throughput)}/s"
        )
    print_separator(separator="=")
    print_key_value(
        key="Throughput",
        value=f"{format_bytes(hmean(throughputs))}/s "
        f"+/- {format_bytes(hstd(throughputs))}/s",
    )
    bandwidth_hmean = p2p_bw[..., BandwidthStats._fields.index("hmean")].reshape(-1)
    bandwidths_all = bandwidth_hmean[bandwidth_hmean > 0]
    print_key_value(
        key="Bandwidth",
        value=f"{format_bytes(hmean(bandwidths_all))}/s +/- "
        f"{format_bytes(hstd(bandwidths_all))}/s",
    )
    print_key_value(
        key="Wall clock",
        value=f"{format_time(durations.mean())} +/- {format_time(durations.std()) }",
    )
    if not args.no_show_p2p_bandwidth:
        print_separator(separator="=")
        if args.markdown:
            print("<details>\n<summary>Worker-Worker Transfer Rates</summary>\n\n```")

        print_key_value(key="(w1,w2)", value="25% 50% 75% (total nbytes)")
        print_separator(separator="-")
        for (source, dest) in np.ndindex(p2p_bw.shape[:2]):
            bw = BandwidthStats(*p2p_bw[source, dest, ...])
            if bw.total_bytes > 0:
                print_key_value(
                    key=f"({source},{dest})",
                    value=f"{format_bytes(bw.q25)}/s {format_bytes(bw.q50)}/s "
                    f"{format_bytes(bw.q75)}/s ({format_bytes(bw.total_bytes)})",
                )
        print_separator(separator="=")
        print_key_value(key="Worker index", value="Worker address")
        print_separator(separator="-")
        for address, index in sorted(address_to_index.items(), key=itemgetter(1)):
            print_key_value(key=index, value=address)
        print_separator(separator="=")
        if args.markdown:
            print("```\n</details>\n")
    if args.plot:
        plot_benchmark(throughputs, args.plot, historical=True)


class BandwidthStats(NamedTuple):
    hmean: float
    hstd: float
    q25: float
    q50: float
    q75: float
    min: float
    max: float
    median: float
    total_bytes: int


def bandwidth_statistics(
    logs, ignore_size: Optional[int] = None
) -> Mapping[str, BandwidthStats]:
    """Return bandwidth statistics from logs on a single worker.

    Parameters
    ----------
    logs:
        the ``dask_worker.incoming_transfer_log`` object
    ignore_size: int (optional)
        ignore messages whose total byte count is smaller than this
        value (if provided)

    Returns
    -------
    dict

        mapping worker names to a :class:`BandwidthStats`
        object summarising incoming messages (bandwidth and total bytes)

    """
    bandwidth = defaultdict(list)
    total_nbytes = defaultdict(int)
    for data in logs:
        if ignore_size is None or data["total"] >= ignore_size:
            bandwidth[data["who"]].append(data["bandwidth"])
            total_nbytes[data["who"]] += data["total"]
    aggregate = {}
    for address, data in bandwidth.items():
        data = np.asarray(data)
        q25, q50, q75 = np.quantile(data, [0.25, 0.50, 0.75])
        aggregate[address] = BandwidthStats(
            hmean=hmean(data),
            hstd=hstd(data),
            q25=q25,
            q50=q50,
            q75=q75,
            min=np.min(data),
            max=np.max(data),
            median=np.median(data),
            total_bytes=total_nbytes[address],
        )
    return aggregate


def aggregate_transfer_log_data(
    aggregator: Callable[[Any, Optional[int]], Any], ignore_size=None, dask_worker=None
) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Aggregate ``dask_worker.incoming_transfer_log`` on a single worker

    Parameters
    ----------
    aggregator: callable
        Function to massage raw data into aggregate form
    ignore_size: int, optional
        ignore contributions of a log entry to the aggregate data if
        the message was less than this many bytes in size (if not
        provided, then keep all messages).
    dask_worker:
        The dask ``Worker`` object.
    """
    return aggregator(dask_worker.incoming_transfer_log, ignore_size=ignore_size)


def peer_to_peer_bandwidths(
    aggregate_bandwidth_data: Mapping[str, Mapping[str, BandwidthStats]],
    address_to_index: Mapping[str, int],
) -> np.ndarray:
    """Flatten collective aggregated bandwidth data

    Parameters
    ----------
    aggregate_bandwidth_data
        Dict mapping worker addresses to per-worker bandwidth data

    name_worker
        Function mapping worker addresses to useful names

    Returns
    -------
    dict
        Flattened dict (keyed on pairs of massaged worker names)
        mapping to bandwidth data between that pair of workers.
    """
    nworker = len(aggregate_bandwidth_data)
    data = np.zeros((nworker, nworker, len(BandwidthStats._fields)), dtype=np.float32)
    for w1, per_worker in aggregate_bandwidth_data.items():
        for w2, stats in per_worker.items():
            # This loses type information on each entry, but we just
            # need indexing information which we can obtain from the
            # BandwidthStats._fields slot.
            data[address_to_index[w1], address_to_index[w2], :] = stats
    return data


def hmean(a):
    """Harmonic mean"""
    if len(a):
        return 1 / np.mean(1 / a)
    else:
        return 0


def hstd(a):
    """Harmonic standard deviation"""
    if len(a):
        rmean = np.mean(1 / a)
        rvar = np.var(1 / a)
        return np.sqrt(rvar / (len(a) * rmean**4))
    else:
        return 0
