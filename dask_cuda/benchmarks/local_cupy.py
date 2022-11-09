import contextlib
from collections import ChainMap
from time import perf_counter as clock

import numpy as np
import pandas as pd
from nvtx import end_range, start_range

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


def bench_once(client, args, write_profile=None):
    if args.type == "gpu":
        import cupy as xp
    else:
        import numpy as xp

    # Create a simple random array
    rs = da.random.RandomState(RandomState=xp.random.RandomState)

    if args.operation == "transpose_sum":
        rng = start_range(message="make array(s)", color="green")
        x = rs.random((args.size, args.size), chunks=args.chunk_size).persist()
        wait(x)
        end_range(rng)

        func_args = (x,)

        func = lambda x: (x + x.T).sum()
    elif args.operation == "dot":
        rng = start_range(message="make array(s)", color="green")
        x = rs.random((args.size, args.size), chunks=args.chunk_size).persist()
        y = rs.random((args.size, args.size), chunks=args.chunk_size).persist()
        wait(x)
        wait(y)
        end_range(rng)

        func_args = (x, y)

        func = lambda x, y: x.dot(y)
    elif args.operation == "svd":
        rng = start_range(message="make array(s)", color="green")
        x = rs.random(
            (args.size, args.second_size),
            chunks=(int(args.chunk_size), args.second_size),
        ).persist()
        wait(x)
        end_range(rng)

        func_args = (x,)

        func = lambda x: np.linalg.svd(x)
    elif args.operation == "fft":
        rng = start_range(message="make array(s)", color="green")
        x = rs.random(
            (args.size, args.size), chunks=(args.size, args.chunk_size)
        ).persist()
        wait(x)
        end_range(rng)

        func_args = (x,)

        func = lambda x: np.fft.fft(x, axis=0)
    elif args.operation == "sum":
        rng = start_range(message="make array(s)", color="green")
        x = rs.random((args.size, args.size), chunks=args.chunk_size).persist()
        wait(x)
        end_range(rng)

        func_args = (x,)

        func = lambda x: x.sum()
    elif args.operation == "mean":
        rng = start_range(message="make array(s)", color="green")
        x = rs.random((args.size, args.size), chunks=args.chunk_size).persist()
        wait(x)
        end_range(rng)

        func_args = (x,)

        func = lambda x: x.mean()
    elif args.operation == "slice":
        rng = start_range(message="make array(s)", color="green")
        x = rs.random((args.size, args.size), chunks=args.chunk_size).persist()
        wait(x)
        end_range(rng)

        func_args = (x,)

        func = lambda x: x[::3].copy()
    elif args.operation == "col_sum":
        rng = start_range(message="make array(s)", color="green")
        x = rs.normal(10, 1, (args.size,), chunks=args.chunk_size).persist()
        y = rs.normal(10, 1, (args.size,), chunks=args.chunk_size).persist()
        wait(x)
        wait(y)
        end_range(rng)

        func_args = (x, y)

        func = lambda x, y: x + y
    elif args.operation == "col_mask":
        rng = start_range(message="make array(s)", color="green")
        x = rs.normal(10, 1, (args.size,), chunks=args.chunk_size).persist()
        y = rs.normal(10, 1, (args.size,), chunks=args.chunk_size).persist()
        wait(x)
        wait(y)
        end_range(rng)

        func_args = (x, y)

        func = lambda x, y: x[y > 10]
    elif args.operation == "col_gather":
        rng = start_range(message="make array(s)", color="green")
        x = rs.normal(10, 1, (args.size,), chunks=args.chunk_size).persist()
        idx = rs.randint(
            0, len(x), (args.second_size,), chunks=args.chunk_size
        ).persist()
        wait(x)
        wait(idx)
        end_range(rng)

        func_args = (x, idx)

        func = lambda x, idx: x[idx]
    else:
        raise ValueError(f"Unknown operation type {args.operation}")

    shape = x.shape
    chunksize = x.chunksize
    data_processed = sum(arg.nbytes for arg in func_args)

    # Execute the operations to benchmark
    if args.profile is not None and write_profile is not None:
        ctx = performance_report(filename=args.profile)
    else:
        ctx = contextlib.nullcontext()

    with ctx:
        rng = start_range(message=args.operation, color="purple")
        result = func(*func_args)
        if args.backend == "dask-noop":
            result = as_noop(result)
        t1 = clock()
        wait(client.persist(result))
        if args.type == "gpu":
            client.run(lambda xp: xp.cuda.Device().synchronize(), xp)
        took = clock() - t1
        end_range(rng)

    return {
        "took": took,
        "data_processed": data_processed,
        "shape": shape,
        "chunksize": chunksize,
    }


def pretty_print_results(args, address_to_index, p2p_bw, results):
    result, *_ = results
    if args.markdown:
        print("```")
    print("Roundtrip benchmark")
    print_separator(separator="-")
    print_key_value(key="Backend", value=f"{args.backend}")
    print_key_value(key="Operation", value=f"{args.operation}")
    print_key_value(key="Array type", value="cupy" if args.type == "gpu" else "numpy")
    print_key_value(key="User size", value=f"{args.size}")
    print_key_value(key="User second size", value=f"{args.second_size}")
    print_key_value(key="User chunk size", value=f"{args.size}")
    print_key_value(key="Compute shape", value=f"{result['shape']}")
    print_key_value(key="Compute chunk size", value=f"{result['chunksize']}")
    print_key_value(key="Ignore size", value=f"{format_bytes(args.ignore_size)}")
    print_key_value(key="Device(s)", value=f"{args.devs}")
    print_key_value(
        key="Data processed", value=f"{format_bytes(result['data_processed'])}"
    )
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
    data_processed, durations = zip(
        *((result["data_processed"], result["took"]) for result in results)
    )
    if args.markdown:
        print("\n```")
    print_throughput_bandwidth(
        args, durations, data_processed, p2p_bw, address_to_index
    )


def create_tidy_results(args, p2p_bw, results):
    configuration = {
        "operation": args.operation,
        "backend": args.backend,
        "array_type": "cupy" if args.type == "gpu" else "numpy",
        "user_size": args.size,
        "user_second_size": args.second_size,
        "user_chunk_size": args.chunk_size,
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
    }
    timing_data = pd.DataFrame(
        [
            pd.Series(
                data=ChainMap(
                    configuration,
                    {
                        "wallclock": result["took"],
                        "compute_shape": result["shape"],
                        "compute_chunk_size": result["chunksize"],
                        "data_processed": result["data_processed"],
                    },
                )
            )
            for result in results
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
            "help": "The array size n in n^2 (default 10000). For 'svd' operation "
            "the second dimension is given by --second-size.",
        },
        {
            "name": [
                "-2",
                "--second-size",
            ],
            "default": "1000",
            "type": int,
            "help": "The second dimension size for 'svd' operation (default 1000).",
        },
        {
            "name": [
                "-t",
                "--type",
            ],
            "choices": ["cpu", "gpu"],
            "default": "gpu",
            "type": str,
            "help": "Do merge with GPU or CPU dataframes.",
        },
        {
            "name": [
                "-o",
                "--operation",
            ],
            "default": "transpose_sum",
            "type": str,
            "help": "The operation to run, valid options are: "
            "'transpose_sum' (default), 'dot', 'fft', 'svd', 'sum', 'mean', 'slice'.",
        },
        {
            "name": [
                "-c",
                "--chunk-size",
            ],
            "default": "2500",
            "type": int,
            "help": "Chunk size (default 2500).",
        },
        {
            "name": "--ignore-size",
            "default": "1 MiB",
            "metavar": "nbytes",
            "type": parse_bytes,
            "help": "Ignore messages smaller than this (default '1 MB').",
        },
        {
            "name": "--runs",
            "default": 3,
            "type": int,
            "help": "Number of runs (default 3).",
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
