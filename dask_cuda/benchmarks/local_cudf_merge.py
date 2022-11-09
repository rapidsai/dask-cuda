import contextlib
import math
from collections import ChainMap
from time import perf_counter

import numpy as np
import pandas as pd

import dask
from dask.base import tokenize
from dask.dataframe.core import new_dd_object
from dask.distributed import performance_report, wait
from dask.utils import format_bytes, parse_bytes

from dask_cuda.benchmarks.common import Config, execute_benchmark
from dask_cuda.benchmarks.utils import (
    as_noop,
    parse_benchmark_args,
    print_key_value,
    print_separator,
    print_throughput_bandwidth,
)

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
    if args.backend == "dask-noop":
        t1 = perf_counter()
        ddf_join = as_noop(ddf_join)
        noopify_duration = perf_counter() - t1
    else:
        noopify_duration = 0
    wait(ddf_join.persist())
    return noopify_duration


def bench_once(client, args, write_profile=None):
    # Generate random Dask dataframes
    n_workers = len(client.scheduler_info()["workers"])
    # Allow the number of chunks to vary between
    # the "base" and "other" DataFrames
    args.base_chunks = args.base_chunks or n_workers
    args.other_chunks = args.other_chunks or n_workers
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
            noopify_duration = merge(args, ddf_base, ddf_other)
            duration = perf_counter() - t1 - noopify_duration

    return (data_processed, duration)


def pretty_print_results(args, address_to_index, p2p_bw, results):
    broadcast = (
        False if args.shuffle_join else (True if args.broadcast_join else "default")
    )

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
    if args.markdown:
        print("\n```")

    data_processed, durations = zip(*results)
    print_throughput_bandwidth(
        args, durations, data_processed, p2p_bw, address_to_index
    )


def create_tidy_results(
    args,
    p2p_bw: np.ndarray,
    results,
):
    broadcast = (
        False if args.shuffle_join else (True if args.broadcast_join else "default")
    )
    configuration = {
        "dataframe_type": "cudf" if args.type == "gpu" else "pandas",
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
    }
    timing_data = pd.DataFrame(
        [
            pd.Series(
                data=ChainMap(
                    configuration,
                    {"wallclock": duration, "data_processed": data_processed},
                )
            )
            for data_processed, duration in results
        ]
    )
    return timing_data, p2p_bw


def parse_args():
    special_args = [
        {
            "name": [
                "-b",
                "--backend",
            ],
            "choices": ["dask", "explicit-comms", "dask-noop"],
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
    execute_benchmark(
        Config(
            args=parse_args(),
            bench_once=bench_once,
            create_tidy_results=create_tidy_results,
            pretty_print_results=pretty_print_results,
        )
    )
