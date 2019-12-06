import argparse
import math
from collections import defaultdict
from time import perf_counter as clock

from dask.base import tokenize
from dask.dataframe.core import new_dd_object
from dask.distributed import Client, performance_report, wait
from dask.utils import format_bytes, format_time, parse_bytes
from dask_cuda import LocalCUDACluster

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
        # "key" column is a unique range
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

        # Step 3. Combine and create the fineal dataframe chunk (dask_cudf partition)
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


def run(args, write_profile=None):
    # Generate random Dask dataframes
    ddf_base = get_random_ddf(
        args.chunk_size, args.n_workers, args.frac_match, "build", args
    ).persist()
    ddf_other = get_random_ddf(
        args.chunk_size, args.n_workers, args.frac_match, "other", args
    ).persist()
    wait(ddf_base)
    wait(ddf_other)

    # Lazy merge/join operation
    ddf_join = ddf_base.merge(ddf_other, on=["key"], how="inner")

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


def main(args):
    # Set up workers on the local machine
    cluster = LocalCUDACluster(
        protocol=args.protocol, n_workers=args.n_workers, CUDA_VISIBLE_DEVICES=args.devs
    )
    client = Client(cluster)

    if args.no_pool_allocator:
        client.run(cudf.set_allocator, "default", pool=False)
    else:
        client.run(cudf.set_allocator, "default", pool=True)

    took_list = []
    for _ in range(args.runs - 1):
        took_list.append(run(args, write_profile=None))
    took_list.append(
        run(args, write_profile=args.profile)
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
        (cluster.scheduler.workers[w1].name, cluster.scheduler.workers[w2].name): [
            "%s/s" % format_bytes(x) for x in numpy.quantile(v, [0.25, 0.50, 0.75])
        ]
        for (w1, w2), v in bandwidths.items()
    }
    total_nbytes = {
        (
            cluster.scheduler.workers[w1].name,
            cluster.scheduler.workers[w2].name,
        ): format_bytes(sum(nb))
        for (w1, w2), nb in total_nbytes.items()
    }

    if args.markdown:
        print("```")
    print("Merge benchmark")
    print("--------------------------")
    print(f"Chunk-size  | {args.chunk_size}")
    print(f"Frac-match  | {args.frac_match}")
    print(f"Ignore-size | {format_bytes(args.ignore_size)}")
    print(f"Protocol    | {args.protocol}")
    print(f"Device(s)   | {args.devs}")
    print("==========================")
    for took in took_list:
        print(f"Total time  | {format_time(took)}")
    print("==========================")
    if args.markdown:
        print(
            "\n```\n<details>\n<summary>Worker-Worker Transfer Rates</summary>\n\n```"
        )
    print("(w1,w2)     | 25% 50% 75% (total nbytes)")
    print("--------------------------")
    for (d1, d2), bw in sorted(bandwidths.items()):
        print(
            "(%02d,%02d)     | %s %s %s (%s)"
            % (d1, d2, bw[0], bw[1], bw[2], total_nbytes[(d1, d2)])
        )
    if args.markdown:
        print("```\n</details>\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge (dask/cudf) on LocalCUDACluster benchmark"
    )
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
        "-c",
        "--chunk-size",
        default=1_000_000,
        metavar="n",
        type=int,
        help="Chunk size (default 1_000_000)",
    )
    parser.add_argument(
        "--ignore-size",
        default="1 MiB",
        metavar="nbytes",
        type=parse_bytes,
        help='Ignore messages smaller than this (default "1 MB")',
    )
    parser.add_argument(
        "--frac-match",
        default=0.3,
        type=float,
        help="Fraction of rows that matches (default 0.3)",
    )
    parser.add_argument(
        "--no-pool-allocator", action="store_true", help="Disable the RMM memory pool"
    )
    parser.add_argument(
        "--profile",
        metavar="PATH",
        default=None,
        type=str,
        help="Write dask profile report (E.g. dask-report.html)",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Don't shuffle the keys of the left (base) dataframe.",
    )
    parser.add_argument(
        "--markdown", action="store_true", help="Write output as markdown"
    )
    parser.add_argument("--runs", default=3, type=int, help="Number of runs")
    args = parser.parse_args()
    args.n_workers = len(args.devs.split(","))
    return args


if __name__ == "__main__":
    main(parse_args())
