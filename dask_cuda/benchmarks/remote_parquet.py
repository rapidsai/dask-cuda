import contextlib
from collections import ChainMap
from time import perf_counter as clock

import pandas as pd

import dask
import dask.dataframe as dd
from dask.distributed import performance_report
from dask.utils import format_bytes, parse_bytes

from dask_cuda.benchmarks.common import Config, execute_benchmark
from dask_cuda.benchmarks.custom import custom_read_parquet
from dask_cuda.benchmarks.utils import (
    parse_benchmark_args,
    print_key_value,
    print_separator,
    print_throughput_bandwidth,
)

DEFAULT_DATASET_PATH = "s3://dask-cudf-parquet-testing/dedup_parquet"
DEFAULT_COLUMNS = ["text", "id"]
DEFAULT_STORAGE_SIZE = 2_843_373_145  # Compressed byte size


def read_data(
    backend,
    filesystem,
    aggregate_files,
    blocksize,
):
    path = DEFAULT_DATASET_PATH
    columns = DEFAULT_COLUMNS
    with dask.config.set({"dataframe.backend": backend}):
        if filesystem == "arrow" and backend == "cudf":
            df = custom_read_parquet(
                path,
                columns=columns,
                blocksize=blocksize,
            )
        else:
            if filesystem == "arrow":
                # TODO: Warn user that blocksize and aggregate_files
                # are ingored when `filesystem == "arrow"`
                _blocksize = {}
                _aggregate_files = {}
            else:
                _blocksize = {"blocksize": blocksize}
                _aggregate_files = {"aggregate_files": aggregate_files}

            df = dd.read_parquet(
                path,
                columns=columns,
                filesystem=filesystem,
                **_blocksize,
                **_aggregate_files,
            )
        return df.memory_usage().compute().sum()


def bench_once(client, args, write_profile=None):

    if write_profile is None:
        ctx = contextlib.nullcontext()
    else:
        ctx = performance_report(filename=args.profile)

    with ctx:
        t1 = clock()
        output_size = read_data(
            backend="cudf" if args.type == "gpu" else "pandas",
            filesystem=args.filesystem,
            aggregate_files=args.aggregate_files,
            blocksize=args.blocksize,
        )
        t2 = clock()

    return (DEFAULT_STORAGE_SIZE, output_size, t2 - t1)


def pretty_print_results(args, address_to_index, p2p_bw, results):
    if args.markdown:
        print("```")
    print("Remote Parquet benchmark")
    print_separator(separator="-")
    backend = "cudf" if args.type == "gpu" else "pandas"
    print_key_value(key="Backend", value=f"{backend}")
    print_key_value(key="Filesystem", value=f"{args.filesystem}")
    print_key_value(key="Blocksize", value=f"{args.blocksize}")
    print_key_value(key="Aggregate files", value=f"{args.aggregate_files}")
    print_key_value(key="Output size", value=f"{format_bytes(results[0][1])}")
    if args.markdown:
        print("\n```")
    data_processed, output_size, durations = zip(*results)
    print_throughput_bandwidth(
        args, durations, data_processed, p2p_bw, address_to_index
    )


def create_tidy_results(args, p2p_bw, results):
    configuration = {
        "backend": "cudf" if args.type == "gpu" else "pandas",
        "filesystem": args.filesystem,
        "blocksize": args.blocksize,
        "aggregate_files": args.aggregate_files,
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
            "name": "--blocksize",
            "default": "256MB",
            "type": str,
            "help": "How to set the blocksize option",
        },
        {
            "name": "--aggregate-files",
            "default": False,
            "action": "store_true",
            "help": "How to set the aggregate_files option",
        },
        {
            "name": [
                "-t",
                "--type",
            ],
            "choices": ["cpu", "gpu"],
            "default": "gpu",
            "type": str,
            "help": "Use GPU or CPU dataframes (default 'gpu')",
        },
        {
            "name": "--filesystem",
            "choices": ["arrow", "fsspec"],
            "default": "fsspec",
            "type": str,
            "help": "Filesystem backend",
        },
        {
            "name": "--runs",
            "default": 3,
            "type": int,
            "help": "Number of runs",
        },
        # NOTE: The following args are not relevant to this benchmark
        {
            "name": "--ignore-size",
            "default": "1 MiB",
            "metavar": "nbytes",
            "type": parse_bytes,
            "help": "Ignore messages smaller than this (default '1 MB')",
        },
    ]

    return parse_benchmark_args(
        description="Remote Parquet (dask/cudf) benchmark",
        args_list=special_args,
        check_explicit_comms=False,
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
