import math
from collections import defaultdict
from time import perf_counter as clock

from dask.base import tokenize
from dask.dataframe.core import new_dd_object
from dask.distributed import Client, performance_report, wait
from dask.utils import format_bytes, format_time, parse_bytes
from dask_cuda import explicit_comms
from dask_cuda.benchmarks.utils import (
    get_cluster_options,
    get_scheduler_workers,
    parse_benchmark_args,
    setup_memory_pool,
)

import cudf
import cupy
import numpy

# Benchmarking cuDF merge operation based on
# <https://gist.github.com/rjzamora/0ffc35c19b5180ab04bbf7c793c45955>


def generate_chunk(i_chunk, local_size, num_chunks, chunk_type, frac_match):
    # Setting a seed that triggers max amount of comm in the two-GPU case.
    cupy.random.seed(17561648246761420848)

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

        parts_array = cupy.arange(num_chunks, dtype="int64")
        suffle_array = cupy.repeat(parts_array, math.ceil(local_size / num_chunks))

        df = cudf.DataFrame(
            {
                "key": cupy.arange(start, stop=stop, dtype="int64"),
                "shuffle": cupy.random.permutation(suffle_array)[:local_size],
                "payload": cupy.random.permutation(
                    cupy.arange(local_size, dtype="int64")
                ),
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
            ar = cupy.arange(bgn, stop=end, dtype="int64")
            arrays.append(cupy.random.permutation(ar)[:sub_local_size_use])
        key_array_match = cupy.concatenate(tuple(arrays), axis=0)

        # Step 2. Add values that DON'T match
        missing_size = local_size - key_array_match.shape[0]
        start = local_size * num_chunks + local_size * i_chunk
        stop = start + missing_size
        key_array_no_match = cupy.arange(start, stop=stop, dtype="int64")

        # Step 3. Combine and create the final dataframe chunk (dask_cudf partition)
        key_array_combine = cupy.concatenate(
            (key_array_match, key_array_no_match), axis=0
        )
        df = cudf.DataFrame(
            {
                "key": cupy.random.permutation(key_array_combine),
                "payload": cupy.random.permutation(
                    cupy.arange(local_size, dtype="int64")
                ),
            }
        )
    return df


def get_random_ddf(chunk_size, num_chunks, frac_match, chunk_type, args):

    parts = [chunk_size for i in range(num_chunks)]
    meta = generate_chunk(0, 4, 1, chunk_type, None)
    divisions = [None] * (len(parts) + 1)

    name = "generate-data-" + tokenize(chunk_size, num_chunks, frac_match, chunk_type)

    graph = {
        (name, i): (generate_chunk, i, part, len(parts), chunk_type, frac_match)
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
    # Lazy merge/join operation
    ddf_join = ddf1.merge(ddf2, on=["key"], how="inner")
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


def merge_explicit_comms(args, ddf1, ddf2):
    t1 = clock()
    wait(explicit_comms.dataframe_merge(ddf1, ddf2, on="key").persist())
    took = clock() - t1
    return took


def run(client, args, n_workers, write_profile=None):
    # Generate random Dask dataframes
    ddf_base = get_random_ddf(
        args.chunk_size, n_workers, args.frac_match, "build", args
    ).persist()
    ddf_other = get_random_ddf(
        args.chunk_size, n_workers, args.frac_match, "other", args
    ).persist()
    wait(ddf_base)
    wait(ddf_other)
    client.wait_for_workers(n_workers)

    assert len(ddf_base.dtypes) == 2
    assert len(ddf_other.dtypes) == 2
    data_processed = len(ddf_base) * sum([t.itemsize for t in ddf_base.dtypes])
    data_processed += len(ddf_other) * sum([t.itemsize for t in ddf_other.dtypes])

    if args.backend == "dask":
        took = merge(args, ddf_base, ddf_other, write_profile)
    else:
        took = merge_explicit_comms(args, ddf_base, ddf_other)

    return (data_processed, took)


def main(args):
    cluster_options = get_cluster_options(args)
    Cluster = cluster_options["class"]
    cluster_args = cluster_options["args"]
    cluster_kwargs = cluster_options["kwargs"]
    scheduler_addr = cluster_options["scheduler_addr"]

    cluster = Cluster(*cluster_args, **cluster_kwargs)
    if args.multi_node:
        import time

        # Allow some time for workers to start and connect to scheduler
        # TODO: make this a command-line argument?
        time.sleep(15)

    client = Client(scheduler_addr if args.multi_node else cluster)

    client.run(setup_memory_pool, disable_pool=args.no_rmm_pool)
    # Create an RMM pool on the scheduler due to occasional deserialization
    # of CUDA objects. May cause issues with InfiniBand otherwise.
    client.run_on_scheduler(setup_memory_pool, 1e9, disable_pool=args.no_rmm_pool)

    scheduler_workers = client.run_on_scheduler(get_scheduler_workers)
    n_workers = len(scheduler_workers)

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

    if args.markdown:
        print("```")
    print("Merge benchmark")
    print("-------------------------------")
    print(f"backend        | {args.backend}")
    print(f"rows-per-chunk | {args.chunk_size}")
    print(f"protocol       | {args.protocol}")
    print(f"device(s)      | {args.devs}")
    print(f"rmm-pool       | {(not args.no_rmm_pool)}")
    print(f"frac-match     | {args.frac_match}")
    if args.protocol == "ucx":
        print(f"tcp            | {args.enable_tcp_over_ucx}")
        print(f"ib             | {args.enable_infiniband}")
        print(f"nvlink         | {args.enable_nvlink}")
    print(f"data-processed | {format_bytes(took_list[0][0])}")
    print("===============================")
    print("Wall-clock     | Throughput")
    print("-------------------------------")
    for data_processed, took in took_list:
        throughput = int(data_processed / took)
        m = format_time(took)
        m += " " * (15 - len(m))
        print(f"{m}| {format_bytes(throughput)}/s")
    print("===============================")
    if args.markdown:
        print("\n```")

    if args.backend == "dask":
        if args.markdown:
            print("<details>\n<summary>Worker-Worker Transfer Rates</summary>\n\n```")
        print("(w1,w2)     | 25% 50% 75% (total nbytes)")
        print("-------------------------------")
        for (d1, d2), bw in sorted(bandwidths.items()):
            fmt = (
                "(%s,%s)     | %s %s %s (%s)"
                if args.multi_node
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
            "name": ["-c", "--chunk-size",],
            "default": 1_000_000,
            "metavar": "n",
            "type": int,
            "help": "Chunk size (default 1_000_000)",
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
