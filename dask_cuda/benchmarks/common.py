from argparse import Namespace
from functools import partial
from typing import Any, Callable, List, Mapping, NamedTuple, Optional, Tuple
from warnings import filterwarnings

import numpy as np
import pandas as pd

import dask
from distributed import Client

from dask_cuda.benchmarks.utils import (
    address_to_index,
    aggregate_transfer_log_data,
    bandwidth_statistics,
    get_cluster_options,
    peer_to_peer_bandwidths,
    save_benchmark_data,
    setup_memory_pools,
    wait_for_cluster,
)
from dask_cuda.utils import all_to_all

__all__ = ("execute_benchmark", "Config")


class Config(NamedTuple):
    """Benchmark configuration"""

    args: Namespace
    """Parsed benchmark arguments"""
    bench_once: Callable[[Client, Namespace, Optional[str]], Any]
    """Callable to run a single benchmark iteration

    Parameters
    ----------
    client
        distributed Client object
    args
        Benchmark parsed arguments
    write_profile
        Should a profile be written?

    Returns
    -------
    Benchmark data to be interpreted by ``pretty_print_results`` and
    ``create_tidy_results``.
    """
    create_tidy_results: Callable[
        [Namespace, np.ndarray, List[Any]], Tuple[pd.DataFrame, np.ndarray]
    ]
    """Callable to create tidy results for saving to disk

    Parameters
    ----------
    args
        Benchmark parsed arguments
    p2p_bw
        Array of point-to-point bandwidths
    results: list
        List of results from running ``bench_once``
    Returns
    -------
    tuple
        two-tuple of a pandas dataframe and the point-to-point bandwidths
    """
    pretty_print_results: Callable[
        [Namespace, Mapping[str, int], np.ndarray, List[Any]], None
    ]
    """Callable to pretty-print results for human consumption

    Parameters
    ----------
    args
        Benchmark parsed arguments
    address_to_index
        Mapping from worker addresses to indices
    p2p_bw
        Array of point-to-point bandwidths
    results: list
        List of results from running ``bench_once``
    """


def run_benchmark(client: Client, args: Namespace, config: Config):
    """Run a benchmark a specified number of times

    If ``args.profile`` is set, the final run is profiled.
    """
    results = []
    for _ in range(max(1, args.runs) - 1):
        res = config.bench_once(client, args, write_profile=None)
        results.append(res)
    results.append(config.bench_once(client, args, write_profile=args.profile))
    return results


def gather_bench_results(client: Client, args: Namespace, config: Config):
    """Collect benchmark results from the workers"""
    address2index = address_to_index(client)
    if args.all_to_all:
        all_to_all(client)
    results = run_benchmark(client, args, config)
    # Collect aggregated peer-to-peer bandwidth
    message_data = client.run(
        partial(aggregate_transfer_log_data, bandwidth_statistics, args.ignore_size)
    )
    return address2index, results, message_data


def run(client: Client, args: Namespace, config: Config):
    """Run the full benchmark on the cluster

    Waits for the cluster, sets up memory pools, prints and saves results
    """

    wait_for_cluster(client, shutdown_on_failure=True)
    assert len(client.scheduler_info()["workers"]) > 0
    setup_memory_pools(
        client,
        args.type == "gpu",
        args.rmm_pool_size,
        args.disable_rmm_pool,
        args.enable_rmm_async,
        args.enable_rmm_managed,
        args.rmm_release_threshold,
        args.rmm_log_directory,
        args.enable_rmm_statistics,
        args.enable_rmm_track_allocations,
    )
    address_to_index, results, message_data = gather_bench_results(client, args, config)
    p2p_bw = peer_to_peer_bandwidths(message_data, address_to_index)
    config.pretty_print_results(args, address_to_index, p2p_bw, results)
    if args.output_basename:
        df, p2p_bw = config.create_tidy_results(args, p2p_bw, results)
        df["num_workers"] = len(address_to_index)
        save_benchmark_data(
            args.output_basename,
            address_to_index,
            df,
            p2p_bw,
        )


def run_client_from_existing_scheduler(args: Namespace, config: Config):
    """Set up a client by connecting to a scheduler

    Shuts down the cluster at the end of the benchmark conditional on
    ``args.shutdown_cluster``.
    """
    if args.scheduler_address is not None:
        kwargs = {"address": args.scheduler_address}
    elif args.scheduler_file is not None:
        kwargs = {"scheduler_file": args.scheduler_file}
    else:
        raise RuntimeError(
            "Need to specify either --scheduler-file " "or --scheduler-address"
        )
    with Client(**kwargs) as client:
        run(client, args, config)
        if args.shutdown_cluster:
            client.shutdown()


def run_create_client(args: Namespace, config: Config):
    """Create a client + cluster and run

    Shuts down the cluster at the end of the benchmark
    """
    cluster_options = get_cluster_options(args)
    Cluster = cluster_options["class"]
    cluster_args = cluster_options["args"]
    cluster_kwargs = cluster_options["kwargs"]
    scheduler_addr = cluster_options["scheduler_addr"]

    filterwarnings("ignore", message=".*NVLink.*rmm_pool_size.*", category=UserWarning)

    with Cluster(*cluster_args, **cluster_kwargs) as cluster:
        # Use the scheduler address with an SSHCluster rather than the cluster
        # object, otherwise we can't shut it down.
        with Client(scheduler_addr if args.multi_node else cluster) as client:
            run(client, args, config)
            # An SSHCluster will not automatically shut down, we have to
            # ensure it does.
            if args.multi_node:
                client.shutdown()


def execute_benchmark(config: Config):
    """Run complete benchmark given a configuration"""
    args = config.args
    if args.multiprocessing_method == "forkserver":
        import multiprocessing.forkserver as f

        f.ensure_running()
    with dask.config.set(
        {"distributed.worker.multiprocessing-method": args.multiprocessing_method}
    ):
        if args.scheduler_file is not None or args.scheduler_address is not None:
            run_client_from_existing_scheduler(args, config)
        else:
            run_create_client(args, config)
