import itertools
import operator
from collections import ChainMap, defaultdict
from time import perf_counter as clock
from warnings import filterwarnings

import numpy as np
import pandas as pd
from nvtx import end_range, start_range

from dask import array as da
from dask.distributed import Client, performance_report, wait
from dask.utils import format_bytes, format_time, parse_bytes

from dask_cuda.benchmarks.utils import (
    get_cluster_options,
    get_scheduler_workers,
    hmean,
    hstd,
    parse_benchmark_args,
    peer_to_peer_bandwidths,
    print_key_value,
    print_separator,
    setup_memory_pools,
    wait_for_cluster,
    worker_renamer,
    write_benchmark_data_as_json,
)
from dask_cuda.utils import all_to_all


def bench_once(client, args):
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
    if args.profile is not None:
        with performance_report(filename=args.profile):
            rng = start_range(message=args.operation, color="purple")
            t1 = clock()
            wait(client.persist(func(*func_args)))
            if args.type == "gpu":
                client.run(lambda xp: xp.cuda.Device().synchronize(), xp)
            took = clock() - t1
            end_range(rng)
    else:
        rng = start_range(message=args.operation, color="purple")
        t1 = clock()
        wait(client.persist(func(*func_args)))
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


def pretty_print_results(args, incoming_logs, scheduler_workers, results):
    p2p_bw_dict = peer_to_peer_bandwidths(
        incoming_logs, scheduler_workers, args.ignore_size
    )
    bandwidths = p2p_bw_dict["bandwidths"]
    bandwidths_all = p2p_bw_dict["bandwidths_all"]
    total_nbytes = p2p_bw_dict["total_nbytes"]
    renamer = worker_renamer(
        scheduler_workers.values(),
        args.multi_node or args.sched_addr or args.scheduler_file,
    )

    bandwidths = {
        (
            renamer(scheduler_workers[w1].name),
            renamer(scheduler_workers[w2].name),
        ): ["%s/s" % format_bytes(x) for x in np.quantile(v, [0.25, 0.50, 0.75])]
        for (w1, w2), v in bandwidths.items()
    }
    total_nbytes = {
        (
            renamer(scheduler_workers[w1].name),
            renamer(scheduler_workers[w2].name),
        ): format_bytes(sum(nb))
        for (w1, w2), nb in total_nbytes.items()
    }

    result, *_ = results

    print("Roundtrip benchmark")
    print_separator(separator="-")
    print_key_value(key="Operation", value=f"{args.operation}")
    print_key_value(key="Array type", value="cupy" if args.type == "gpu" else "numpy")
    print_key_value(key="User size", value=f"{args.size}")
    print_key_value(key="User second size", value=f"{args.second_size}")
    print_key_value(key="User chunk size", value=f"{args.size}")
    print_key_value(key="Compute shape", value=f"{result['shape']}")
    print_key_value(key="Compute chunk size", value=f"{result['chunksize']}")
    print_key_value(key="Ignore size", value=f"{format_bytes(args.ignore_size)}")
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
    print_separator(separator="=")
    print_key_value(key="Wall clock", value="Throughput")
    print_separator(separator="-")
    t_p = []
    times = []
    for result in results:
        took = result["took"]
        data_processed = result["data_processed"]
        throughput = data_processed / took
        times.append(took)
        t_p.append(throughput)
        print_key_value(
            key=f"{format_time(took)}", value=f"{format_bytes(throughput)}/s"
        )
    t_p = np.asarray(t_p)
    times = np.asarray(times)
    bandwidths_all = np.asarray(bandwidths_all)
    print_separator(separator="=")
    print_key_value(
        key="Throughput",
        value=f"{format_bytes(hmean(t_p))}/s +/- {format_bytes(hstd(t_p))}/s",
    )
    print_key_value(
        key="Bandwidth",
        value=f"{format_bytes(hmean(bandwidths_all))}/s +/- "
        f"{format_bytes(hstd(bandwidths_all))}/s",
    )
    print_key_value(
        key="Wall clock",
        value=f"{format_time(times.mean())} +/- {format_time(times.std()) }",
    )
    print_separator(separator="=")
    print_key_value(key="(w1,w2)", value="25% 50% 75% (total nbytes)")
    print_separator(separator="-")
    for (d1, d2), bw in sorted(bandwidths.items()):
        # hostid-deviceid
        n1 = f"{d1[0]}-{d1[1]}"
        n2 = f"{d2[0]}-{d2[1]}"
        print_key_value(
            key=f"({n1},{n2})",
            value=f"{bw[0]} {bw[1]} {bw[2]} ({total_nbytes[(d1, d2)]})",
        )


def create_tidy_results(args, incoming_logs, scheduler_workers, results):
    result, *_ = results
    configuration = {
        "operation": args.operation,
        "array_type": "cupy" if args.type == "gpu" else "numpy",
        "user_size": args.size,
        "user_second_size": args.second_size,
        "user_chunk_size": args.chunk_size,
        "compute_shape": result["shape"],
        "compute_chunk_size": result["chunksize"],
        "npartitions": result["npartitions"],
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
    series = []
    times = np.asarray([r["took"] for r in results], dtype=float)
    q25, q75 = np.quantile(times, [0.25, 0.75])
    timing = {
        "time_mean": times.mean(),
        "time_median": np.median(times),
        "time_std": times.std(),
        "time_q25": q25,
        "time_q75": q75,
        "time_min": times.min(),
        "time_max": times.max(),
    }
    bandwidths = defaultdict(list)
    total_nbytes = defaultdict(list)

    renamer = worker_renamer(
        scheduler_workers.values(),
        args.multi_node or args.sched_addr or args.scheduler_file,
    )

    for k, L in incoming_logs.items():
        source = scheduler_workers[k].name
        for d in L:
            dest = scheduler_workers[d["who"]].name
            key = tuple(map(renamer, (source, dest)))
            bandwidths[key].append(d["bandwidth"])
            total_nbytes[key].append(d["total"])
    for (_, group) in itertools.groupby(
        sorted(bandwidths.keys()), key=operator.itemgetter(0)
    ):
        for key in group:
            ((source_node, source_device), (dest_node, dest_device)) = key
            bandwidth = np.asarray(bandwidths[key])
            nbytes = np.asarray(total_nbytes[key])
            q25, q75 = np.quantile(bandwidth, [0.25, 0.75])
            data = pd.Series(
                data=ChainMap(
                    configuration,
                    timing,
                    {
                        "source_node": source_node,
                        "source_device": source_device,
                        "dest_node": dest_node,
                        "dest_device": dest_device,
                        "total_bytes": nbytes.sum(),
                        "bandwidth_mean": bandwidth.mean(),
                        "bandwidth_median": np.median(bandwidth),
                        "bandwidth_std": bandwidth.std(),
                        "bandwidth_q25": q25,
                        "bandwidth_q75": q75,
                        "bandwidth_min": bandwidth.min(),
                        "bandwidth_max": bandwidth.max(),
                    },
                )
            )
            series.append(data)
    return pd.DataFrame(series)


def run_benchmark(client, args):
    results = []
    for _ in range(max(1, args.runs)):
        res = bench_once(client, args)
        results.append(res)
    return results


def gather_bench_results(client, args):
    scheduler_workers = client.run_on_scheduler(get_scheduler_workers)
    if args.all_to_all:
        all_to_all(client)
    results = run_benchmark(client, args)
    # Collect, aggregate, and print peer-to-peer bandwidths
    incoming_logs = client.run(lambda dask_worker: dask_worker.incoming_transfer_log)
    return scheduler_workers, results, incoming_logs


def run(client, args):
    wait_for_cluster(client, shutdown_on_failure=True)
    setup_memory_pools(
        client,
        args.type == "gpu",
        args.rmm_pool_size,
        args.disable_rmm_pool,
        args.rmm_log_directory,
    )
    scheduler_workers, results, incoming_logs = gather_bench_results(client, args)
    pretty_print_results(args, incoming_logs, scheduler_workers, results)
    if args.benchmark_json:
        write_benchmark_data_as_json(
            args.benchmark_json,
            create_tidy_results(args, incoming_logs, scheduler_workers, results),
        )


def run_client_from_file(args):
    scheduler_file = args.scheduler_file
    if scheduler_file is None:
        raise RuntimeError("Need scheduler file to be provided")
    with Client(scheduler_file=scheduler_file) as client:
        run(client, args)
        client.shutdown()


def run_create_client(args):
    cluster_options = get_cluster_options(args)
    Cluster = cluster_options["class"]
    cluster_args = cluster_options["args"]
    cluster_kwargs = cluster_options["kwargs"]
    scheduler_addr = cluster_options["scheduler_addr"]

    filterwarnings("ignore", message=".*NVLink.*rmm_pool_size.*", category=UserWarning)

    with Cluster(*cluster_args, **cluster_kwargs) as cluster:
        # Use the scheduler address with an SSHCluster rather than the cluster
        # object, otherwise we can't shut it down.
        with Client(scheduler_addr if args.multi_node else cluster) as client:
            run(client, args)
            # An SSHCluster will not automatically shut down, we have to
            # ensure it does.
            if args.multi_node:
                client.shutdown()


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
    ]

    return parse_benchmark_args(
        description="Transpose on LocalCUDACluster benchmark", args_list=special_args
    )


if __name__ == "__main__":
    args = parse_args()
    if args.scheduler_file is not None:
        run_client_from_file(args)
    else:
        run_create_client(args)
