import math
from collections import defaultdict
from time import perf_counter as clock
from warnings import filterwarnings

import numpy

import dask
from dask.base import tokenize
from dask.dataframe.core import new_dd_object
from dask.distributed import Client, performance_report, wait
from dask.utils import format_bytes, format_time, parse_bytes

from dask_cuda.benchmarks.utils import (
    get_cluster_options,
    get_scheduler_workers,
    parse_benchmark_args,
    plot_benchmark,
    setup_memory_pool,
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

    xp.random.seed(2 ** 32 - 1)

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

    parts = [chunk_size for i in range(num_chunks)]
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


def merge(args, ddf1, ddf2, write_profile):

    # Allow default broadcast behavior, unless
    # "--shuffle-join" or "--broadcast-join" was
    # specified (with "--shuffle-join" taking
    # precedence)
    broadcast = False if args.shuffle_join else (True if args.broadcast_join else None)

    # Lazy merge/join operation
    ddf_join = ddf1.merge(ddf2, on=["key"], how="inner", broadcast=broadcast,)
    if args.set_index:
        ddf_join = ddf_join.set_index("key")

    # Execute the operations to benchmark
    if write_profile is not None:
        with performance_report(filename=args.profile):
            t1 = clock()
            wait(ddf_join.persist())
            took = clock() - t1
    else:
        t1 = clock()
        wait(ddf_join.persist())
        took = clock() - t1
    return took


def run(client, args, n_workers, write_profile=None):
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

    if args.backend == "dask":
        took = merge(args, ddf_base, ddf_other, write_profile)
    else:
        with dask.config.set(explicit_comms=True):
            took = merge(args, ddf_base, ddf_other, write_profile)

    return (data_processed, took)


def main(args):
    cluster_options = get_cluster_options(args)
    Cluster = cluster_options["class"]
    cluster_args = cluster_options["args"]
    cluster_kwargs = cluster_options["kwargs"]
    scheduler_addr = cluster_options["scheduler_addr"]

    if args.sched_addr:
        client = Client(args.sched_addr)
    else:
        filterwarnings(
            "ignore", message=".*NVLink.*rmm_pool_size.*", category=UserWarning
        )

        cluster = Cluster(*cluster_args, **cluster_kwargs)
        if args.multi_node:
            import time

            # Allow some time for workers to start and connect to scheduler
            # TODO: make this a command-line argument?
            time.sleep(15)

        client = Client(scheduler_addr if args.multi_node else cluster)

    if args.type == "gpu":
        client.run(
            setup_memory_pool,
            pool_size=args.rmm_pool_size,
            disable_pool=args.disable_rmm_pool,
            log_directory=args.rmm_log_directory,
        )
        # Create an RMM pool on the scheduler due to occasional deserialization
        # of CUDA objects. May cause issues with InfiniBand otherwise.
        client.run_on_scheduler(
            setup_memory_pool,
            pool_size=1e9,
            disable_pool=args.disable_rmm_pool,
            log_directory=args.rmm_log_directory,
        )

    scheduler_workers = client.run_on_scheduler(get_scheduler_workers)
    n_workers = len(scheduler_workers)
    client.wait_for_workers(n_workers)

    # Allow the number of chunks to vary between
    # the "base" and "other" DataFrames
    args.base_chunks = args.base_chunks or n_workers
    args.other_chunks = args.other_chunks or n_workers

    if args.all_to_all:
        all_to_all(client)

    took_list = []
    for _ in range(args.runs - 1):
        took_list.append(run(client, args, n_workers, write_profile=None))
    took_list.append(
        run(client, args, n_workers, write_profile=args.profile)
    )  # Only profiling the last run

    # Collect, aggregate, and print peer-to-peer bandwidths
    incoming_logs = client.run(lambda dask_worker: dask_worker.incoming_transfer_log)
    bandwidths = defaultdict(list)
    total_nbytes = defaultdict(list)
    for k, L in incoming_logs.items():
        for d in L:
            if d["total"] >= args.ignore_size:
                bandwidths[k, d["who"]].append(d["bandwidth"])
                total_nbytes[k, d["who"]].append(d["total"])
    bandwidths = {
        (scheduler_workers[w1].name, scheduler_workers[w2].name): [
            "%s/s" % format_bytes(x) for x in numpy.quantile(v, [0.25, 0.50, 0.75])
        ]
        for (w1, w2), v in bandwidths.items()
    }
    total_nbytes = {
        (scheduler_workers[w1].name, scheduler_workers[w2].name,): format_bytes(sum(nb))
        for (w1, w2), nb in total_nbytes.items()
    }

    broadcast = (
        False if args.shuffle_join else (True if args.broadcast_join else "default")
    )

    t_runs = numpy.empty(len(took_list))
    if args.markdown:
        print("```")
    print("Merge benchmark")
    print("-------------------------------")
    print(f"backend        | {args.backend}")
    print(f"merge type     | {args.type}")
    print(f"rows-per-chunk | {args.chunk_size}")
    print(f"base-chunks    | {args.base_chunks}")
    print(f"other-chunks   | {args.other_chunks}")
    print(f"broadcast      | {broadcast}")
    print(f"protocol       | {args.protocol}")
    print(f"device(s)      | {args.devs}")
    print(f"rmm-pool       | {(not args.disable_rmm_pool)}")
    print(f"frac-match     | {args.frac_match}")
    if args.protocol == "ucx":
        print(f"tcp            | {args.enable_tcp_over_ucx}")
        print(f"ib             | {args.enable_infiniband}")
        print(f"nvlink         | {args.enable_nvlink}")
    print(f"data-processed | {format_bytes(took_list[0][0])}")
    print("===============================")
    print("Wall-clock     | Throughput")
    print("-------------------------------")
    for idx, (data_processed, took) in enumerate(took_list):
        throughput = int(data_processed / took)
        m = format_time(took)
        m += " " * (15 - len(m))
        print(f"{m}| {format_bytes(throughput)}/s")
        t_runs[idx] = float(format_bytes(throughput).split(" ")[0])
    print("===============================")
    if args.markdown:
        print("\n```")

    if args.plot is not None:
        plot_benchmark(t_runs, args.plot, historical=True)

    if args.backend == "dask":
        if args.markdown:
            print("<details>\n<summary>Worker-Worker Transfer Rates</summary>\n\n```")
        print("(w1,w2)     | 25% 50% 75% (total nbytes)")
        print("-------------------------------")
        for (d1, d2), bw in sorted(bandwidths.items()):
            fmt = (
                "(%s,%s)     | %s %s %s (%s)"
                if args.multi_node or args.sched_addr
                else "(%02d,%02d)     | %s %s %s (%s)"
            )
            print(fmt % (d1, d2, bw[0], bw[1], bw[2], total_nbytes[(d1, d2)]))
        if args.markdown:
            print("```\n</details>\n")

    if args.multi_node:
        client.shutdown()
        client.close()


def parse_args():
    special_args = [
        {
            "name": ["-b", "--backend",],
            "choices": ["dask", "explicit-comms"],
            "default": "dask",
            "type": str,
            "help": "The backend to use.",
        },
        {
            "name": ["-t", "--type",],
            "choices": ["cpu", "gpu"],
            "default": "gpu",
            "type": str,
            "help": "Do merge with GPU or CPU dataframes",
        },
        {
            "name": ["-c", "--chunk-size",],
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
        {"name": "--runs", "default": 3, "type": int, "help": "Number of runs",},
        {
            "name": ["-s", "--set-index",],
            "action": "store_true",
            "help": "Call set_index on the key column to sort the joined dataframe.",
        },
    ]

    return parse_benchmark_args(
        description="Distributed merge (dask/cudf) benchmark", args_list=special_args
    )


if __name__ == "__main__":
    main(parse_args())
