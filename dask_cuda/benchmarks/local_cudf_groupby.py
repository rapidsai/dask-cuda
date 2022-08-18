import contextlib
from collections import ChainMap
from time import perf_counter as clock

import pandas as pd

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


def apply_groupby(df, shuffle=None):
    wait(
        # TODO: Add params for split_out and sort
        df.groupby("key", sort=False)
        .agg(
            {"mean": ["x", "y"], "count": "x"},
            split_out=1,
            split_every=8,
            # shuffle=shuffle,
        )
        .persist()
    )


def generate_chunk(chunk_info, unique_ratio=0.01, gpu=True):
    # Setting a seed that triggers max amount of comm in the two-GPU case.
    if gpu:
        import cupy as xp

        import cudf as xdf
    else:
        import numpy as xp
        import pandas as xdf

    i_chunk, local_size = chunk_info
    xp.random.seed(i_chunk * 1_000)
    low, high = 0, max(int(unique_ratio * local_size), 1)
    return xdf.DataFrame(
        {
            "key": xp.random.randint(low, high, size=local_size, dtype="int64"),
            "int64": xp.random.permutation(xp.arange(local_size, dtype="int64")),
            "float64": xp.random.permutation(xp.arange(local_size, dtype="float64")),
        }
    )


def get_random_ddf(args):

    chunk_kwargs = {
        "unique_ratio": args.unique_ratio,
        "gpu": True if args.type == "gpu" else False,
    }

    return dd.from_map(
        generate_chunk,
        [(i, args.chunk_size) for i in range(args.in_parts)],
        meta=generate_chunk(0, **chunk_kwargs),
        **chunk_kwargs,
    )


def bench_once(client, args, write_profile=None):

    # Generate random Dask dataframe
    df = get_random_ddf(args).persist()
    wait(df)

    data_processed = len(df) * sum([t.itemsize for t in df.dtypes])

    if write_profile is None:
        ctx = contextlib.nullcontext()
    else:
        ctx = performance_report(filename=args.profile)

    with ctx:
        t1 = clock()
        apply_groupby(df, shuffle=args.shuffle)
        t2 = clock()

    return (data_processed, t2 - t1)


def pretty_print_results(args, address_to_index, p2p_bw, results):
    if args.markdown:
        print("```")
    print("Groupby benchmark")
    print_separator(separator="-")
    print_key_value(key="Use shuffle", value=f"{args.shuffle}")
    print_key_value(key="Partition size", value=f"{format_bytes(args.partition_size)}")
    print_key_value(key="Input partitions", value=f"{args.in_parts}")
    print_key_value(key="Unique-group ratio", value=f"{args.unique_ratio}")
    print_key_value(key="Protocol", value=f"{args.protocol}")
    print_key_value(key="Device(s)", value=f"{args.devs}")
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
    if args.markdown:
        print("\n```")
    data_processed, durations = zip(*results)
    print_throughput_bandwidth(
        args, durations, data_processed, p2p_bw, address_to_index
    )


def create_tidy_results(args, p2p_bw, results):
    configuration = {
        "dataframe_type": "cudf" if args.type == "gpu" else "pandas",
        "shuffle": args.shuffle,
        "partition_size": args.partition_size,
        "in_parts": args.in_parts,
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
            "name": "--partition-size",
            "default": "1 MiB",
            "metavar": "nbytes",
            "type": parse_bytes,
            "help": "Size of each partition (default '1 MB')",
        },
        {
            "name": "--in-parts",
            "default": 100,
            "metavar": "n",
            "type": int,
            "help": "Number of input partitions (default '100')",
        },
        {
            "name": "--unique-ratio",
            "default": 0.01,
            "type": float,
            "help": "Fraction of rows that are unique groups",
        },
        {
            "name": "--shuffle",
            "choices": [True, False],
            "default": False,
            "type": bool,
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
        description="Distributed shuffle (dask/cudf) benchmark", args_list=special_args
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
