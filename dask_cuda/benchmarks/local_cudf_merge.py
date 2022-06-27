import contextlib
import itertools
import math
import operator
from collections import ChainMap, defaultdict
from time import perf_counter
from warnings import filterwarnings

import numpy as np
import pandas as pd

import dask
from dask.base import tokenize
from dask.dataframe.core import new_dd_object
from dask.distributed import Client, performance_report, wait
from dask.utils import format_bytes, format_time, parse_bytes

from dask_cuda.benchmarks.utils import (
    get_cluster_options,
    get_scheduler_workers,
    hmean,
    hstd,
    parse_benchmark_args,
    peer_to_peer_bandwidths,
    plot_benchmark,
    print_key_value,
    print_separator,
    setup_memory_pools,
    wait_for_cluster,
    worker_renamer,
    write_benchmark_data_as_json,
)
from dask_cuda.utils import all_to_all

# Benchmarking cuDF merge operation based on
# <https://gist.github.com/rjzamora/0ffc35c19b5180ab04bbf7c793c45955>


def generate_chunk(i_chunk, local_size, num_chunks, chunk_type, frac_match, gpu):
    # Setting a seed that triggers max amount of comm in the two-GPU case.
    if gpu:
        import cupy as xp

        import cudf as xdf
    else:
        import numpy as xp
        import pandas as xdf

    xp.random.seed(2**32 - 1)

    chunk_type = chunk_type or "build"
    frac_match = frac_match or 1.0
    if chunk_type == "build":
        # Build dataframe
        #
        # "key" column is a unique sample within [0, local_size * num_chunks)
        #
        # "shuffle" column is a random selection of partitions (used for shuffle)
        #
        # "payload" column is a random permutation of the chunk_size

        start = local_size * i_chunk
        stop = start + local_size

        parts_array = xp.arange(num_chunks, dtype="int64")
        suffle_array = xp.repeat(parts_array, math.ceil(local_size / num_chunks))

        df = xdf.DataFrame(
            {
                "key": xp.arange(start, stop=stop, dtype="int64"),
                "shuffle": xp.random.permutation(suffle_array)[:local_size],
                "payload": xp.random.permutation(xp.arange(local_size, dtype="int64")),
            }
        )
    else:
        # Other dataframe
        #
        # "key" column matches values from the build dataframe
        # for a fraction (`frac_match`) of the entries. The matching
        # entries are perfectly balanced across each partition of the
        # "base" dataframe.
        #
        # "payload" column is a random permutation of the chunk_size

        # Step 1. Choose values that DO match
        sub_local_size = local_size // num_chunks
        sub_local_size_use = max(int(sub_local_size * frac_match), 1)
        arrays = []
        for i in range(num_chunks):
            bgn = (local_size * i) + (sub_local_size * i_chunk)
            end = bgn + sub_local_size
            ar = xp.arange(bgn, stop=end, dtype="int64")
            arrays.append(xp.random.permutation(ar)[:sub_local_size_use])
        key_array_match = xp.concatenate(tuple(arrays), axis=0)

        # Step 2. Add values that DON'T match
        missing_size = local_size - key_array_match.shape[0]
        start = local_size * num_chunks + local_size * i_chunk
        stop = start + missing_size
        key_array_no_match = xp.arange(start, stop=stop, dtype="int64")

        # Step 3. Combine and create the final dataframe chunk (dask_cudf partition)
        key_array_combine = xp.concatenate(
            (key_array_match, key_array_no_match), axis=0
        )
        df = xdf.DataFrame(
            {
                "key": xp.random.permutation(key_array_combine),
                "payload": xp.random.permutation(xp.arange(local_size, dtype="int64")),
            }
        )
    return df


def get_random_ddf(chunk_size, num_chunks, frac_match, chunk_type, args):

    parts = [chunk_size for _ in range(num_chunks)]
    device_type = True if args.type == "gpu" else False
    meta = generate_chunk(0, 4, 1, chunk_type, None, device_type)
    divisions = [None] * (len(parts) + 1)

    name = "generate-data-" + tokenize(chunk_size, num_chunks, frac_match, chunk_type)

    graph = {
        (name, i): (
            generate_chunk,
            i,
            part,
            len(parts),
            chunk_type,
            frac_match,
            device_type,
        )
        for i, part in enumerate(parts)
    }

    ddf = new_dd_object(graph, name, meta, divisions)

    if chunk_type == "build":
        if not args.no_shuffle:
            divisions = [i for i in range(num_chunks)] + [num_chunks]
            return ddf.set_index("shuffle", divisions=tuple(divisions))
        else:
            del ddf["shuffle"]

    return ddf


def merge(args, ddf1, ddf2):

    # Allow default broadcast behavior, unless
    # "--shuffle-join" or "--broadcast-join" was
    # specified (with "--shuffle-join" taking
    # precedence)
    broadcast = False if args.shuffle_join else (True if args.broadcast_join else None)

    # The merge/join operation
    ddf_join = ddf1.merge(ddf2, on=["key"], how="inner", broadcast=broadcast)
    if args.set_index:
        ddf_join = ddf_join.set_index("key")
    wait(ddf_join.persist())


def bench_once(client, args, write_profile=None):
    # Generate random Dask dataframes
    ddf_base = get_random_ddf(
        args.chunk_size, args.base_chunks, args.frac_match, "build", args
    ).persist()
    ddf_other = get_random_ddf(
        args.chunk_size, args.other_chunks, args.frac_match, "other", args
    ).persist()
    wait(ddf_base)
    wait(ddf_other)

    assert len(ddf_base.dtypes) == 2
    assert len(ddf_other.dtypes) == 2
    data_processed = len(ddf_base) * sum([t.itemsize for t in ddf_base.dtypes])
    data_processed += len(ddf_other) * sum([t.itemsize for t in ddf_other.dtypes])

    # Get contexts to use (defaults to null contexts that doesn't do anything)
    ctx1, ctx2 = contextlib.nullcontext(), contextlib.nullcontext()
    if args.backend == "explicit-comms":
        ctx1 = dask.config.set(explicit_comms=True)
    if write_profile is not None:
        ctx2 = performance_report(filename=args.profile)

    with ctx1:
        with ctx2:
            t1 = perf_counter()
            merge(args, ddf_base, ddf_other)
            t2 = perf_counter()

    return (data_processed, t2 - t1)


def pretty_print_results(args, incoming_logs, scheduler_workers, results):
    p2p_bw_dict = peer_to_peer_bandwidths(
        incoming_logs, scheduler_workers, args.ignore_size
    )
    bandwidths = p2p_bw_dict["bandwidths"]
    bandwidths_all = p2p_bw_dict["bandwidths_all"]
    total_nbytes = p2p_bw_dict["total_nbytes"]
    renamer = worker_renamer(
        scheduler_workers.values(),
        args.multi_node or args.sched_addr or args.scheduler_file,
    )
    bandwidths = {
        (renamer(scheduler_workers[w1].name), renamer(scheduler_workers[w2].name)): [
            "%s/s" % format_bytes(x) for x in np.quantile(v, [0.25, 0.50, 0.75])
        ]
        for (w1, w2), v in bandwidths.items()
    }
    total_nbytes = {
        (
            renamer(scheduler_workers[w1].name),
            renamer(scheduler_workers[w2].name),
        ): format_bytes(sum(nb))
        for (w1, w2), nb in total_nbytes.items()
    }

    broadcast = (
        False if args.shuffle_join else (True if args.broadcast_join else "default")
    )

    t_runs = np.empty(len(results))
    if args.markdown:
        print("```")
    print("Merge benchmark")
    print_separator(separator="-")
    print_key_value(key="Backend", value=f"{args.backend}")
    print_key_value(key="Merge type", value=f"{args.type}")
    print_key_value(key="Rows-per-chunk", value=f"{args.chunk_size}")
    print_key_value(key="Base-chunks", value=f"{args.base_chunks}")
    print_key_value(key="Other-chunks", value=f"{args.other_chunks}")
    print_key_value(key="Broadcast", value=f"{broadcast}")
    print_key_value(key="Protocol", value=f"{args.protocol}")
    print_key_value(key="Device(s)", value=f"{args.devs}")
    if args.device_memory_limit:
        print_key_value(
            key="Device memory limit", value=f"{format_bytes(args.device_memory_limit)}"
        )
    print_key_value(key="RMM Pool", value=f"{not args.disable_rmm_pool}")
    print_key_value(key="Frac-match", value=f"{args.frac_match}")
    if args.protocol == "ucx":
        print_key_value(key="TCP", value=f"{args.enable_tcp_over_ucx}")
        print_key_value(key="InfiniBand", value=f"{args.enable_infiniband}")
        print_key_value(key="NVLink", value=f"{args.enable_nvlink}")
    print_key_value(key="Worker thread(s)", value=f"{args.threads_per_worker}")
    print_key_value(key="Data processed", value=f"{format_bytes(results[0][0])}")
    print_separator(separator="=")
    print_key_value(key="Wall clock", value="Throughput")
    print_separator(separator="-")
    t_p = []
    times = []
    for idx, (data_processed, took) in enumerate(results):
        throughput = int(data_processed / took)
        m = format_time(took)
        times.append(took)
        t_p.append(throughput)
        print_key_value(key=f"{m}", value=f"{format_bytes(throughput)}/s")
        t_runs[idx] = float(format_bytes(throughput).split(" ")[0])
    t_p = np.asarray(t_p)
    times = np.asarray(times)
    bandwidths_all = np.asarray(bandwidths_all)
    print_separator(separator="=")
    print_key_value(
        key="Throughput",
        value=f"{format_bytes(hmean(t_p))}/s +/- {format_bytes(hstd(t_p))}/s",
    )
    print_key_value(
        key="Bandwidth",
        value=f"{format_bytes(hmean(bandwidths_all))}/s +/- "
        f"{format_bytes(hstd(bandwidths_all))}/s",
    )
    print_key_value(
        key="Wall clock",
        value=f"{format_time(times.mean())} +/- {format_time(times.std()) }",
    )
    print_separator(separator="=")
    if args.markdown:
        print("\n```")

    if args.plot is not None:
        plot_benchmark(t_runs, args.plot, historical=True)

    if args.backend == "dask":
        if args.markdown:
            print("<details>\n<summary>Worker-Worker Transfer Rates</summary>\n\n```")
        print_key_value(key="(w1,w2)", value="25% 50% 75% (total nbytes)")
        print_separator(separator="-")
        for (d1, d2), bw in sorted(bandwidths.items()):
            n1 = f"{d1[0]}-{d1[1]}"
            n2 = f"{d2[0]}-{d2[1]}"
            key = f"({n1},{n2})"
            print_key_value(
                key=key, value=f"{bw[0]} {bw[1]} {bw[2]} ({total_nbytes[(d1, d2)]})"
            )
        if args.markdown:
            print("```\n</details>\n")


def create_tidy_results(args, incoming_logs, scheduler_workers, results):
    result, *_ = results
    broadcast = (
        False if args.shuffle_join else (True if args.broadcast_join else "default")
    )
    configuration = {
        "backend": args.backend,
        "merge_type": args.type,
        "base_chunks": args.base_chunks,
        "other_chunks": args.other_chunks,
        "broadcast": broadcast,
        "rows_per_chunk": args.chunk_size,
        "ignore_size": args.ignore_size,
        "frac_match": args.frac_match,
        "devices": args.devs,
        "device_memory_limit": args.device_memory_limit,
        "worker_threads": args.threads_per_worker,
        "rmm_pool": not args.disable_rmm_pool,
        "protocol": args.protocol,
        "tcp": args.enable_tcp_over_ucx,
        "ib": args.enable_infiniband,
        "nvlink": args.enable_nvlink,
        "nreps": args.runs,
        "data_processed": result[0],
    }
    series = []
    times = np.asarray([r[1] for r in results], dtype=float)
    q25, q75 = np.quantile(times, [0.25, 0.75])
    timing = {
        "time_mean": times.mean(),
        "time_median": np.median(times),
        "time_std": times.std(),
        "time_q25": q25,
        "time_q75": q75,
        "time_min": times.min(),
        "time_max": times.max(),
    }
    if args.backend == "dask":
        # worker-to-worker bandwidth available
        bandwidths = defaultdict(list)
        total_nbytes = defaultdict(list)

        renamer = worker_renamer(
            scheduler_workers.values(),
            args.multi_node or args.sched_addr or args.scheduler_file,
        )

        for k, L in incoming_logs.items():
            source = scheduler_workers[k].name
            for d in L:
                dest = scheduler_workers[d["who"]].name
                key = tuple(map(renamer, (source, dest)))
                bandwidths[key].append(d["bandwidth"])
                total_nbytes[key].append(d["total"])
        for (_, group) in itertools.groupby(
            sorted(bandwidths.keys()), key=operator.itemgetter(0)
        ):
            for key in group:
                ((source_node, source_device), (dest_node, dest_device)) = key
                bandwidth = np.asarray(bandwidths[key])
                nbytes = np.asarray(total_nbytes[key])
                q25, q75 = np.quantile(bandwidth, [0.25, 0.75])
                data = pd.Series(
                    data=ChainMap(
                        configuration,
                        timing,
                        {
                            "source_node": source_node,
                            "source_device": source_device,
                            "dest_node": dest_node,
                            "dest_device": dest_device,
                            "total_bytes": nbytes.sum(),
                            "bandwidth_mean": bandwidth.mean(),
                            "bandwidth_median": np.median(bandwidth),
                            "bandwidth_std": bandwidth.std(),
                            "bandwidth_q25": q25,
                            "bandwidth_q75": q75,
                            "bandwidth_min": bandwidth.min(),
                            "bandwidth_max": bandwidth.max(),
                        },
                    )
                )
                series.append(data)
        return pd.DataFrame(series)
    else:
        return pd.DataFrame([pd.Series(data=ChainMap(configuration, timing))])


def run_benchmark(client, args):
    results = []
    for _ in range(max(1, args.runs) - 1):
        results.append(bench_once(client, args, write_profile=None))
    # Only profile final run (if wanted)
    results.append(bench_once(client, args, write_profile=args.profile))
    return results


def gather_bench_results(client, args):
    scheduler_workers = client.run_on_scheduler(get_scheduler_workers)
    n_workers = len(scheduler_workers)
    client.wait_for_workers(n_workers)
    # Allow the number of chunks to vary between
    # the "base" and "other" DataFrames
    args.base_chunks = args.base_chunks or n_workers
    args.other_chunks = args.other_chunks or n_workers
    if args.all_to_all:
        all_to_all(client)
    results = run_benchmark(client, args)
    # Collect, aggregate, and print peer-to-peer bandwidths
    incoming_logs = client.run(lambda dask_worker: dask_worker.incoming_transfer_log)
    return scheduler_workers, results, incoming_logs


def run(client, args):
    wait_for_cluster(client, shutdown_on_failure=True)
    setup_memory_pools(
        client,
        args.type == "gpu",
        args.rmm_pool_size,
        args.disable_rmm_pool,
        args.rmm_log_directory,
    )
    scheduler_workers, results, incoming_logs = gather_bench_results(client, args)
    pretty_print_results(args, incoming_logs, scheduler_workers, results)
    if args.benchmark_json:
        write_benchmark_data_as_json(
            args.benchmark_json,
            create_tidy_results(args, incoming_logs, scheduler_workers, results),
        )


def run_client_from_file(args):
    scheduler_file = args.scheduler_file
    if scheduler_file is None:
        raise RuntimeError("Need scheduler file to be provided")
    with Client(scheduler_file=scheduler_file) as client:
        run(client, args)
        client.shutdown()


def run_create_client(args):
    cluster_options = get_cluster_options(args)
    Cluster = cluster_options["class"]
    cluster_args = cluster_options["args"]
    cluster_kwargs = cluster_options["kwargs"]
    scheduler_addr = cluster_options["scheduler_addr"]

    filterwarnings("ignore", message=".*NVLink.*rmm_pool_size.*", category=UserWarning)
    with Cluster(*cluster_args, **cluster_kwargs) as cluster:
        with Client(scheduler_addr if args.multi_node else cluster) as client:
            run(client, args)
            if args.multi_node:
                client.shutdown()


def parse_args():
    special_args = [
        {
            "name": [
                "-b",
                "--backend",
            ],
            "choices": ["dask", "explicit-comms"],
            "default": "dask",
            "type": str,
            "help": "The backend to use.",
        },
        {
            "name": [
                "-t",
                "--type",
            ],
            "choices": ["cpu", "gpu"],
            "default": "gpu",
            "type": str,
            "help": "Do merge with GPU or CPU dataframes",
        },
        {
            "name": [
                "-c",
                "--chunk-size",
            ],
            "default": 1_000_000,
            "metavar": "n",
            "type": int,
            "help": "Chunk size (default 1_000_000)",
        },
        {
            "name": "--base-chunks",
            "default": None,
            "type": int,
            "help": "Number of base-DataFrame partitions (default: n_workers)",
        },
        {
            "name": "--other-chunks",
            "default": None,
            "type": int,
            "help": "Number of other-DataFrame partitions (default: n_workers)",
        },
        {
            "name": "--broadcast-join",
            "action": "store_true",
            "help": "Use broadcast join when possible.",
        },
        {
            "name": "--shuffle-join",
            "action": "store_true",
            "help": "Use shuffle join (takes precedence over '--broadcast-join').",
        },
        {
            "name": "--ignore-size",
            "default": "1 MiB",
            "metavar": "nbytes",
            "type": parse_bytes,
            "help": "Ignore messages smaller than this (default '1 MB')",
        },
        {
            "name": "--frac-match",
            "default": 0.3,
            "type": float,
            "help": "Fraction of rows that matches (default 0.3)",
        },
        {
            "name": "--no-shuffle",
            "action": "store_true",
            "help": "Don't shuffle the keys of the left (base) dataframe.",
        },
        {
            "name": "--markdown",
            "action": "store_true",
            "help": "Write output as markdown",
        },
        {
            "name": "--runs",
            "default": 3,
            "type": int,
            "help": "Number of runs",
        },
        {
            "name": [
                "-s",
                "--set-index",
            ],
            "action": "store_true",
            "help": "Call set_index on the key column to sort the joined dataframe.",
        },
    ]

    return parse_benchmark_args(
        description="Distributed merge (dask/cudf) benchmark", args_list=special_args
    )


if __name__ == "__main__":
    args = parse_args()
    if args.scheduler_file is not None:
        run_client_from_file(args)
    else:
        run_create_client(args)
