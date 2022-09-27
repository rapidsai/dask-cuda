import contextlib
from collections import ChainMap
from time import perf_counter as clock

import cupy as cp
import numpy as np
import pandas as pd
from cupyx.scipy.ndimage.filters import convolve as cp_convolve
from scipy.ndimage import convolve as sp_convolve

from dask import array as da
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


def mean_filter(a, shape):
    a_k = np.full_like(a, 1.0 / np.prod(shape), shape=shape)
    if isinstance(a, cp.ndarray):
        return cp_convolve(a, a_k)
    else:
        return sp_convolve(a, a_k)


def bench_once(client, args, write_profile=None):
    # Create a simple random array
    if args.type == "gpu":
        rs = da.random.RandomState(RandomState=cp.random.RandomState)
    else:
        rs = da.random.RandomState(RandomState=np.random.RandomState)
    x = rs.random((args.size, args.size), chunks=args.chunk_size).persist()
    ks = 2 * (2 * args.kernel_size + 1,)
    wait(x)

    data_processed = x.nbytes

    # Execute the operations to benchmark
    if args.profile is not None and write_profile is not None:
        ctx = performance_report(filename=args.profile)
    else:
        ctx = contextlib.nullcontext()

    with ctx:
        result = x.map_overlap(mean_filter, args.kernel_size, shape=ks)
        if args.backend == "dask-noop":
            result = as_noop(result)
        t1 = clock()
        wait(client.persist(result))
        took = clock() - t1

    return (data_processed, took)


def pretty_print_results(args, address_to_index, p2p_bw, results):
    if args.markdown:
        print("```")
    print("Cupy map overlap benchmark")
    print_separator(separator="-")
    print_key_value(key="Backend", value=f"{args.backend}")
    print_key_value(key="Array type", value="cupy" if args.type == "gpu" else "numpy")
    print_key_value(key="Size", value=f"{args.size}*{args.size}")
    print_key_value(key="Chunk size", value=f"{args.chunk_size}")
    print_key_value(key="Ignore size", value=f"{format_bytes(args.ignore_size)}")
    print_key_value(key="Kernel size", value=f"{args.kernel_size}")
    print_key_value(key="Device(s)", value=f"{args.devs}")
    if args.device_memory_limit:
        print_key_value(
            key="Device memory limit",
            value=f"{format_bytes(args.device_memory_limit)}",
        )
    print_key_value(key="RMM Pool", value=f"{not args.disable_rmm_pool}")
    print_key_value(key="Protocol", value=f"{args.protocol}")
    if args.protocol == "ucx":
        print_key_value(key="TCP", value=f"{args.enable_tcp_over_ucx}")
        print_key_value(key="InfiniBand", value=f"{args.enable_infiniband}")
        print_key_value(key="NVLink", value=f"{args.enable_nvlink}")
    print_key_value(key="Worker thread(s)", value=f"{args.threads_per_worker}")
    data_processed, durations = zip(*results)
    if args.markdown:
        print("\n```")
    print_throughput_bandwidth(
        args, durations, data_processed, p2p_bw, address_to_index
    )


def create_tidy_results(args, p2p_bw, results):
    configuration = {
        "array_type": "cupy" if args.type == "gpu" else "numpy",
        "backend": args.backend,
        "user_size": args.size,
        "chunk_size": args.chunk_size,
        "ignore_size": args.ignore_size,
        "devices": args.devs,
        "device_memory_limit": args.device_memory_limit,
        "worker_threads": args.threads_per_worker,
        "rmm_pool": not args.disable_rmm_pool,
        "protocol": args.protocol,
        "tcp": args.enable_tcp_over_ucx,
        "ib": args.enable_infiniband,
        "nvlink": args.enable_nvlink,
        "nreps": args.runs,
        "kernel_size": args.kernel_size,
    }
    timing_data = pd.DataFrame(
        [
            pd.Series(
                data=ChainMap(
                    configuration,
                    {
                        "wallclock": duration,
                        "data_processed": data_processed,
                    },
                )
            )
            for (data_processed, duration) in results
        ]
    )
    return timing_data, p2p_bw


def parse_args():
    special_args = [
        {
            "name": [
                "-s",
                "--size",
            ],
            "default": "10000",
            "metavar": "n",
            "type": int,
            "help": "The size n in n^2 (default 10000)",
        },
        {
            "name": [
                "-t",
                "--type",
            ],
            "choices": ["cpu", "gpu"],
            "default": "gpu",
            "type": str,
            "help": "Use GPU or CPU arrays",
        },
        {
            "name": [
                "-c",
                "--chunk-size",
            ],
            "default": "128 MiB",
            "metavar": "nbytes",
            "type": str,
            "help": "Chunk size (default '128 MiB')",
        },
        {
            "name": [
                "-k",
                "--kernel-size",
            ],
            "default": "1",
            "metavar": "k",
            "type": int,
            "help": "Kernel size, 2*k+1, in each dimension (default 1)",
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
            "name": [
                "-b",
                "--backend",
            ],
            "choices": ["dask", "dask-noop"],
            "default": "dask",
            "type": str,
            "help": "Compute backend to use.",
        },
    ]

    return parse_benchmark_args(
        description="Transpose on LocalCUDACluster benchmark", args_list=special_args
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
