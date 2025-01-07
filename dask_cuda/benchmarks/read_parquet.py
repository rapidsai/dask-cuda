import contextlib
from collections import ChainMap
from time import perf_counter as clock

import fsspec
import pandas as pd

import dask
import dask.dataframe as dd
from dask.base import tokenize
from dask.distributed import performance_report
from dask.utils import format_bytes, parse_bytes

from dask_cuda.benchmarks.common import Config, execute_benchmark
from dask_cuda.benchmarks.utils import (
    parse_benchmark_args,
    print_key_value,
    print_separator,
    print_throughput_bandwidth,
)

DISK_SIZE_CACHE = {}
OPTIONS_CACHE = {}


def _noop(df):
    return df


def read_data(paths, columns, backend, **kwargs):
    with dask.config.set({"dataframe.backend": backend}):
        return dd.read_parquet(
            paths,
            columns=columns,
            **kwargs,
        )


def get_fs_paths_kwargs(args):
    kwargs = {}

    storage_options = {}
    if args.key:
        storage_options["key"] = args.key
    if args.secret:
        storage_options["secret"] = args.secret

    if args.filesystem == "arrow":
        import pyarrow.fs as pa_fs
        from fsspec.implementations.arrow import ArrowFSWrapper

        _mapping = {
            "key": "access_key",
            "secret": "secret_key",
        }  # See: pyarrow.fs.S3FileSystem docs
        s3_args = {}
        for k, v in storage_options.items():
            s3_args[_mapping[k]] = v

        fs = pa_fs.FileSystem.from_uri(args.path)[0]
        try:
            region = {"region": fs.region}
        except AttributeError:
            region = {}
        kwargs["filesystem"] = type(fs)(**region, **s3_args)
        fsspec_fs = ArrowFSWrapper(kwargs["filesystem"])

        if args.type == "gpu":
            kwargs["blocksize"] = args.blocksize
    else:
        fsspec_fs = fsspec.core.get_fs_token_paths(
            args.path, mode="rb", storage_options=storage_options
        )[0]
        kwargs["filesystem"] = fsspec_fs
        kwargs["blocksize"] = args.blocksize
        kwargs["aggregate_files"] = args.aggregate_files

    # Collect list of paths
    stripped_url_path = fsspec_fs._strip_protocol(args.path)
    if stripped_url_path.endswith("/"):
        stripped_url_path = stripped_url_path[:-1]
    paths = fsspec_fs.glob(f"{stripped_url_path}/*.parquet")
    if args.file_count:
        paths = paths[: args.file_count]

    return fsspec_fs, paths, kwargs


def bench_once(client, args, write_profile=None):
    global OPTIONS_CACHE
    global DISK_SIZE_CACHE

    # Construct kwargs
    token = tokenize(args)
    try:
        fsspec_fs, paths, kwargs = OPTIONS_CACHE[token]
    except KeyError:
        fsspec_fs, paths, kwargs = get_fs_paths_kwargs(args)
        OPTIONS_CACHE[token] = (fsspec_fs, paths, kwargs)

    if write_profile is None:
        ctx = contextlib.nullcontext()
    else:
        ctx = performance_report(filename=args.profile)

    with ctx:
        t1 = clock()
        df = read_data(
            paths,
            columns=args.columns,
            backend="cudf" if args.type == "gpu" else "pandas",
            **kwargs,
        )
        num_rows = len(
            # Use opaque `map_partitions` call to "block"
            # dask-expr from using pq metadata to get length
            df.map_partitions(
                _noop,
                meta=df._meta,
                enforce_metadata=False,
            )
        )
        t2 = clock()

    # Extract total size of files on disk
    token = tokenize(paths)
    try:
        disk_size = DISK_SIZE_CACHE[token]
    except KeyError:
        disk_size = sum(fsspec_fs.sizes(paths))
        DISK_SIZE_CACHE[token] = disk_size

    return (disk_size, num_rows, t2 - t1)


def pretty_print_results(args, address_to_index, p2p_bw, results):
    if args.markdown:
        print("```")
    print("Parquet read benchmark")
    data_processed, row_count, durations = zip(*results)
    print_separator(separator="-")
    backend = "cudf" if args.type == "gpu" else "pandas"
    print_key_value(key="Path", value=args.path)
    print_key_value(key="Columns", value=f"{args.columns}")
    print_key_value(key="Backend", value=f"{backend}")
    print_key_value(key="Filesystem", value=f"{args.filesystem}")
    print_key_value(key="Blocksize", value=f"{format_bytes(args.blocksize)}")
    print_key_value(key="Aggregate files", value=f"{args.aggregate_files}")
    print_key_value(key="Row count", value=f"{row_count[0]}")
    print_key_value(key="Size on disk", value=f"{format_bytes(data_processed[0])}")
    if args.markdown:
        print("\n```")
    args.no_show_p2p_bandwidth = True
    print_throughput_bandwidth(
        args, durations, data_processed, p2p_bw, address_to_index
    )
    print_separator(separator="=")


def create_tidy_results(args, p2p_bw, results):
    configuration = {
        "path": args.path,
        "columns": args.columns,
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
                        "num_rows": num_rows,
                    },
                )
            )
            for data_processed, num_rows, duration in results
        ]
    )
    return timing_data, p2p_bw


def parse_args():
    special_args = [
        {
            "name": "path",
            "type": str,
            "help": "Parquet directory to read from (must be a flat directory).",
        },
        {
            "name": "--blocksize",
            "default": "256MB",
            "type": parse_bytes,
            "help": "How to set the blocksize option",
        },
        {
            "name": "--aggregate-files",
            "default": False,
            "action": "store_true",
            "help": "How to set the aggregate_files option",
        },
        {
            "name": "--file-count",
            "type": int,
            "help": "Maximum number of files to read.",
        },
        {
            "name": "--columns",
            "type": str,
            "help": "Columns to read/select from data.",
        },
        {
            "name": "--key",
            "type": str,
            "help": "Public S3 key.",
        },
        {
            "name": "--secret",
            "type": str,
            "help": "Secret S3 key.",
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
    ]

    args = parse_benchmark_args(
        description="Parquet read benchmark",
        args_list=special_args,
    )
    args.no_show_p2p_bandwidth = True
    return args


if __name__ == "__main__":
    execute_benchmark(
        Config(
            args=parse_args(),
            bench_once=bench_once,
            create_tidy_results=create_tidy_results,
            pretty_print_results=pretty_print_results,
        )
    )
