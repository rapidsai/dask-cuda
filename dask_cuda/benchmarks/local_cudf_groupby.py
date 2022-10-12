import contextlib
from collections import ChainMap
from time import perf_counter as clock

import pandas as pd

import dask
import dask.dataframe as dd
from dask.distributed import performance_report, wait
from dask.utils import format_bytes, parse_bytes

from dask_cuda.benchmarks.common import Config, execute_benchmark
from dask_cuda.benchmarks.utils import (
    parse_benchmark_args,
    print_key_value,
    print_separator,
    print_throughput_bandwidth,
)


def apply_groupby(
    df,
    sort=False,
    split_out=1,
    split_every=8,
    shuffle=None,
):
    # Handle special "explicit-comms" case
    config = {}
    if shuffle == "explicit-comms":
        shuffle = "tasks"
        config = {"explicit-comms": True}

    with dask.config.set(config):
        agg = df.groupby("key", sort=sort).agg(
            {"int64": ["max", "count"], "float64": "mean"},
            split_out=split_out,
            split_every=split_every,
            shuffle=shuffle,
        )

    wait(agg.persist())
    return agg


def generate_chunk(chunk_info, unique_size=1, gpu=True):
    # Setting a seed that triggers max amount of comm in the two-GPU case.
    if gpu:
        import cupy as xp

        import cudf as xdf
    else:
        import numpy as xp
        import pandas as xdf

    i_chunk, local_size = chunk_info
    xp.random.seed(i_chunk * 1_000)
    return xdf.DataFrame(
        {
            "key": xp.random.randint(0, unique_size, size=local_size, dtype="int64"),
            "int64": xp.random.permutation(xp.arange(local_size, dtype="int64")),
            "float64": xp.random.permutation(xp.arange(local_size, dtype="float64")),
        }
    )


def get_random_ddf(args):

    total_size = args.chunk_size * args.in_parts
    chunk_kwargs = {
        "unique_size": max(int(args.unique_ratio * total_size), 1),
        "gpu": True if args.type == "gpu" else False,
    }

    return dd.from_map(
        generate_chunk,
        [(i, args.chunk_size) for i in range(args.in_parts)],
        meta=generate_chunk((0, 1), **chunk_kwargs),
        enforce_metadata=False,
        **chunk_kwargs,
    )


def bench_once(client, args, write_profile=None):

    # Generate random Dask dataframe
    df = get_random_ddf(args)

    data_processed = len(df) * sum([t.itemsize for t in df.dtypes])
    shuffle = {
        "True": "tasks",
        "False": False,
    }.get(args.shuffle, args.shuffle)

    if write_profile is None:
        ctx = contextlib.nullcontext()
    else:
        ctx = performance_report(filename=args.profile)

    with ctx:
        t1 = clock()
        agg = apply_groupby(
            df,
            sort=args.sort,
            split_out=args.split_out,
            split_every=args.split_every,
            shuffle=shuffle,
        )
        t2 = clock()

    output_size = agg.memory_usage(index=True, deep=True).compute().sum()
    return (data_processed, output_size, t2 - t1)


def pretty_print_results(args, address_to_index, p2p_bw, results):
    if args.markdown:
        print("```")
    print("Groupby benchmark")
    print_separator(separator="-")
    print_key_value(key="Use shuffle", value=f"{args.shuffle}")
    print_key_value(key="Output partitions", value=f"{args.split_out}")
    print_key_value(key="Input partitions", value=f"{args.in_parts}")
    print_key_value(key="Sort Groups", value=f"{args.sort}")
    print_key_value(key="Rows-per-chunk", value=f"{args.chunk_size}")
    print_key_value(key="Unique-group ratio", value=f"{args.unique_ratio}")
    print_key_value(key="Protocol", value=f"{args.protocol}")
    print_key_value(key="Device(s)", value=f"{args.devs}")
    print_key_value(key="Tree-reduction width", value=f"{args.split_every}")
    if args.device_memory_limit:
        print_key_value(
            key="Device memory limit", value=f"{format_bytes(args.device_memory_limit)}"
        )
    print_key_value(key="RMM Pool", value=f"{not args.disable_rmm_pool}")
    if args.protocol == "ucx":
        print_key_value(key="TCP", value=f"{args.enable_tcp_over_ucx}")
        print_key_value(key="InfiniBand", value=f"{args.enable_infiniband}")
        print_key_value(key="NVLink", value=f"{args.enable_nvlink}")
    print_key_value(key="Worker thread(s)", value=f"{args.threads_per_worker}")
    print_key_value(key="Data processed", value=f"{format_bytes(results[0][0])}")
    print_key_value(key="Output size", value=f"{format_bytes(results[0][1])}")
    if args.markdown:
        print("\n```")
    data_processed, output_size, durations = zip(*results)
    print_throughput_bandwidth(
        args, durations, data_processed, p2p_bw, address_to_index
    )


def create_tidy_results(args, p2p_bw, results):
    configuration = {
        "dataframe_type": "cudf" if args.type == "gpu" else "pandas",
        "shuffle": args.shuffle,
        "sort": args.sort,
        "split_out": args.split_out,
        "split_every": args.split_every,
        "in_parts": args.in_parts,
        "rows_per_chunk": args.chunk_size,
        "unique_ratio": args.unique_ratio,
        "protocol": args.protocol,
        "devs": args.devs,
        "device_memory_limit": args.device_memory_limit,
        "rmm_pool": not args.disable_rmm_pool,
        "tcp": args.enable_tcp_over_ucx,
        "ib": args.enable_infiniband,
        "nvlink": args.enable_nvlink,
    }
    timing_data = pd.DataFrame(
        [
            pd.Series(
                data=ChainMap(
                    configuration,
                    {
                        "wallclock": duration,
                        "data_processed": data_processed,
                        "output_size": output_size,
                    },
                )
            )
            for data_processed, output_size, duration in results
        ]
    )
    return timing_data, p2p_bw


def parse_args():
    special_args = [
        {
            "name": "--in-parts",
            "default": 100,
            "metavar": "n",
            "type": int,
            "help": "Number of input partitions (default '100')",
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
            "name": "--unique-ratio",
            "default": 0.01,
            "type": float,
            "help": "Fraction of rows that are unique groups",
        },
        {
            "name": "--sort",
            "default": False,
            "action": "store_true",
            "help": "Whether to sort the output group order.",
        },
        {
            "name": "--split_out",
            "default": 1,
            "type": int,
            "help": "How many partitions to return.",
        },
        {
            "name": "--split_every",
            "default": 8,
            "type": int,
            "help": "Tree-reduction width.",
        },
        {
            "name": "--shuffle",
            "choices": ["False", "True", "tasks", "explicit-comms"],
            "default": "False",
            "type": str,
            "help": "Whether to use shuffle-based groupby.",
        },
        {
            "name": [
                "-t",
                "--type",
            ],
            "choices": ["cpu", "gpu"],
            "default": "gpu",
            "type": str,
            "help": "Do shuffle with GPU or CPU dataframes (default 'gpu')",
        },
        {
            "name": "--ignore-size",
            "default": "1 MiB",
            "metavar": "nbytes",
            "type": parse_bytes,
            "help": "Ignore messages smaller than this (default '1 MB')",
        },
        {
            "name": "--runs",
            "default": 3,
            "type": int,
            "help": "Number of runs",
        },
    ]

    return parse_benchmark_args(
        description="Distributed groupby (dask/cudf) benchmark", args_list=special_args
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
