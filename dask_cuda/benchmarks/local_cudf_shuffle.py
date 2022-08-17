import contextlib
from collections import ChainMap
from time import perf_counter as clock

import pandas as pd

import dask
from dask import array as da
from dask.dataframe.shuffle import shuffle
from dask.distributed import performance_report, wait
from dask.utils import format_bytes, parse_bytes

import dask_cuda.explicit_comms.dataframe.shuffle
from dask_cuda.benchmarks.common import Config, execute_benchmark
from dask_cuda.benchmarks.utils import (
    parse_benchmark_args,
    print_key_value,
    print_separator,
    print_throughput_bandwidth,
)


def shuffle_dask(df):
    wait(shuffle(df, index="data", shuffle="tasks").persist())


def shuffle_explicit_comms(df):
    wait(
        dask_cuda.explicit_comms.dataframe.shuffle.shuffle(
            df, column_names="data"
        ).persist()
    )


def bench_once(client, args, write_profile=None):
    # Generate random Dask dataframe
    chunksize = args.partition_size // 8  # Convert bytes to float64
    nchunks = args.in_parts
    totalsize = chunksize * nchunks
    x = da.random.random((totalsize,), chunks=(chunksize,))
    df = dask.dataframe.from_dask_array(x, columns="data").to_frame()

    if args.type == "gpu":
        import cudf

        df = df.map_partitions(cudf.from_pandas)

    df = df.persist()
    wait(df)
    data_processed = len(df) * sum([t.itemsize for t in df.dtypes])

    if write_profile is None:
        ctx = contextlib.nullcontext()
    else:
        ctx = performance_report(filename=args.profile)

    with ctx:
        t1 = clock()
        if args.backend == "dask":
            shuffle_dask(df)
        else:
            shuffle_explicit_comms(df)
        t2 = clock()

    return (data_processed, t2 - t1)


def pretty_print_results(args, address_to_index, p2p_bw, results):
    if args.markdown:
        print("```")
    print("Shuffle benchmark")
    print_separator(separator="-")
    print_key_value(key="Backend", value=f"{args.backend}")
    print_key_value(key="Partition size", value=f"{format_bytes(args.partition_size)}")
    print_key_value(key="Input partitions", value=f"{args.in_parts}")
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
        "backend": args.backend,
        "partition_size": args.partition_size,
        "in_parts": args.in_parts,
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
