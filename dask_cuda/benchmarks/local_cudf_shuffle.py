import contextlib
from collections import ChainMap
from time import perf_counter
from typing import Tuple

import numpy as np
import pandas as pd

import dask
import dask.dataframe
from dask.dataframe.core import new_dd_object
from dask.dataframe.shuffle import shuffle
from dask.distributed import Client, performance_report, wait
from dask.utils import format_bytes, parse_bytes

import dask_cuda.explicit_comms.dataframe.shuffle
from dask_cuda.benchmarks.common import Config, execute_benchmark
from dask_cuda.benchmarks.utils import (
    as_noop,
    parse_benchmark_args,
    print_key_value,
    print_separator,
    print_throughput_bandwidth,
)

try:
    import cupy

    import cudf
except ImportError:
    cupy = None
    cudf = None


def shuffle_dask(df, args):
    result = shuffle(df, index="data", shuffle="tasks", ignore_index=args.ignore_index)
    if args.backend == "dask-noop":
        result = as_noop(result)
    t1 = perf_counter()
    wait(result.persist())
    return perf_counter() - t1


def shuffle_explicit_comms(df, args):
    t1 = perf_counter()
    wait(
        dask_cuda.explicit_comms.dataframe.shuffle.shuffle(
            df, column_names=["data"], ignore_index=args.ignore_index
        ).persist()
    )
    return perf_counter() - t1


def create_df(nelem, df_type):
    if df_type == "cpu":
        return pd.DataFrame({"data": np.random.random(nelem)})
    elif df_type == "gpu":
        if cudf is None or cupy is None:
            raise RuntimeError("`--type=gpu` requires cudf and cupy ")
        return cudf.DataFrame({"data": cupy.random.random(nelem)})
    else:
        raise ValueError(f"Unknown type {df_type}")


def create_data(
    client: Client, args, name="balanced-df"
) -> Tuple[int, dask.dataframe.DataFrame]:
    """Create an evenly distributed dask dataframe

    The partitions are perfectly distributed across workers, if the number of
    requested partitions is evenly divisible by the number of workers.
    """
    chunksize = args.partition_size // np.float64().nbytes

    workers = list(client.scheduler_info()["workers"].keys())
    assert len(workers) > 0

    dist = args.partition_distribution
    if dist is None:
        # By default, we create a balanced distribution
        dist = [args.in_parts // len(workers)] * len(workers)
        for i in range(args.in_parts % len(workers)):
            dist[i] += 1

    if len(dist) != len(workers):
        raise ValueError(
            f"The length of `--devs`({len(dist)}) and "
            f"`--partition-distribution`({len(workers)}) doesn't match"
        )
    if sum(dist) != args.in_parts:
        raise ValueError(
            f"The sum of `--partition-distribution`({sum(dist)}) must match "
            f"the number of input partitions `--in-parts={args.in_parts}`"
        )

    # Create partition based to the specified partition distribution
    dsk = {}
    for i, part_size in enumerate(dist):
        for _ in range(part_size):
            # We use `client.submit` to control placement of the partition.
            dsk[(name, len(dsk))] = client.submit(
                create_df, chunksize, args.type, workers=[workers[i]], pure=False
            )
    wait(dsk.values())

    df_meta = create_df(0, args.type)
    divs = [None] * (len(dsk) + 1)
    ret = new_dd_object(dsk, name, df_meta, divs).persist()
    wait(ret)

    data_processed = args.in_parts * args.partition_size
    if not args.ignore_index:
        data_processed += args.in_parts * chunksize * df_meta.index.dtype.itemsize
    return data_processed, ret


def bench_once(client, args, write_profile=None):
    data_processed, df = create_data(client, args)

    if write_profile is None:
        ctx = contextlib.nullcontext()
    else:
        ctx = performance_report(filename=args.profile)

    with ctx:
        if args.backend in {"dask", "dask-noop"}:
            duration = shuffle_dask(df, args)
        else:
            duration = shuffle_explicit_comms(df, args)

    return (data_processed, duration)


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
        {
            "name": "--ignore-index",
            "action": "store_true",
            "help": "When shuffle, ignore the index",
        },
        {
            "name": "--partition-distribution",
            "default": None,
            "metavar": "PARTITION_SIZE_LIST",
            "type": lambda x: [int(y) for y in x.split(",")],
            "help": "Comma separated list defining the size of each partition, "
            "which must have the same length as `--devs`. "
            "If not set, a balanced distribution is used.",
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
