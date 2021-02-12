import asyncio
from collections import defaultdict
from json import dump
from time import perf_counter as clock
from warnings import filterwarnings

import numpy as np

from dask import array as da
from dask.distributed import Client, performance_report, wait
from dask.utils import format_bytes, format_time, parse_bytes

from dask_cuda.benchmarks.utils import (
    get_cluster_options,
    get_scheduler_workers,
    parse_benchmark_args,
    setup_memory_pool,
)


async def _run(client, args):
    if args.type == "gpu":
        import cupy as xp
    else:
        import numpy as xp

    # Create a simple random array
    rs = da.random.RandomState(RandomState=xp.random.RandomState)

    if args.operation == "transpose_sum":
        x = rs.random((args.size, args.size), chunks=args.chunk_size).persist()
        await wait(x)
        func_args = (x,)

        func = lambda x: (x + x.T).sum()
    elif args.operation == "dot":
        x = rs.random((args.size, args.size), chunks=args.chunk_size).persist()
        y = rs.random((args.size, args.size), chunks=args.chunk_size).persist()
        await wait(x)
        await wait(y)

        func_args = (x, y)

        func = lambda x, y: x.dot(y)
    elif args.operation == "svd":
        x = rs.random(
            (args.size, args.second_size),
            chunks=(int(args.chunk_size), args.second_size),
        ).persist()
        await wait(x)

        func_args = (x,)

        func = lambda x: np.linalg.svd(x)
    elif args.operation == "fft":
        x = rs.random(
            (args.size, args.size), chunks=(args.size, args.chunk_size)
        ).persist()
        await wait(x)

        func_args = (x,)

        func = lambda x: np.fft.fft(x, axis=0)
    elif args.operation == "sum":
        x = rs.random((args.size, args.size), chunks=args.chunk_size).persist()
        await wait(x)
        func_args = (x,)

        func = lambda x: x.sum()
    elif args.operation == "mean":
        x = rs.random((args.size, args.size), chunks=args.chunk_size).persist()
        await wait(x)
        func_args = (x,)

        func = lambda x: x.mean()
    elif args.operation == "slice":
        x = rs.random((args.size, args.size), chunks=args.chunk_size).persist()
        await wait(x)
        func_args = (x,)

        func = lambda x: x[::3].copy()

    shape = x.shape
    chunksize = x.chunksize

    # Execute the operations to benchmark
    if args.profile is not None:
        async with performance_report(filename=args.profile):
            t1 = clock()
            await client.compute(func(*func_args))
            took = clock() - t1
    else:
        t1 = clock()
        res = client.compute(func(*func_args))
        await client.gather(res)
        if args.type == "gpu":
            await client.run(lambda xp: xp.cuda.Device().synchronize(), xp)
        took = clock() - t1

    return {
        "took": took,
        "npartitions": x.npartitions,
        "shape": shape,
        "chunksize": chunksize,
    }


async def run(args):
    cluster_options = get_cluster_options(args)
    Cluster = cluster_options["class"]
    cluster_args = cluster_options["args"]
    cluster_kwargs = cluster_options["kwargs"]
    scheduler_addr = cluster_options["scheduler_addr"]

    filterwarnings("ignore", message=".*NVLink.*rmm_pool_size.*", category=UserWarning)

    async with Cluster(*cluster_args, **cluster_kwargs, asynchronous=True) as cluster:
        if args.multi_node:
            import time

            # Allow some time for workers to start and connect to scheduler
            # TODO: make this a command-line argument?
            time.sleep(15)

        # Use the scheduler address with an SSHCluster rather than the cluster
        # object, otherwise we can't shut it down.
        async with Client(
            scheduler_addr if args.multi_node else cluster, asynchronous=True
        ) as client:
            scheduler_workers = await client.run_on_scheduler(get_scheduler_workers)

            await client.run(setup_memory_pool, disable_pool=args.disable_rmm_pool)
            # Create an RMM pool on the scheduler due to occasional deserialization
            # of CUDA objects. May cause issues with InfiniBand otherwise.
            await client.run_on_scheduler(
                setup_memory_pool, 1e9, disable_pool=args.disable_rmm_pool
            )

            took_list = []
            for i in range(args.runs):
                res = await _run(client, args)
                took_list.append((res["took"], res["npartitions"]))
                size = res["shape"]
                chunksize = res["chunksize"]

            # Collect, aggregate, and print peer-to-peer bandwidths
            incoming_logs = await client.run(
                lambda dask_worker: dask_worker.incoming_transfer_log
            )
            bandwidths = defaultdict(list)
            total_nbytes = defaultdict(list)
            for k, L in incoming_logs.items():
                for d in L:
                    if d["total"] >= args.ignore_size:
                        bandwidths[k, d["who"]].append(d["bandwidth"])
                        total_nbytes[k, d["who"]].append(d["total"])

            bandwidths = {
                (scheduler_workers[w1].name, scheduler_workers[w2].name,): [
                    "%s/s" % format_bytes(x) for x in np.quantile(v, [0.25, 0.50, 0.75])
                ]
                for (w1, w2), v in bandwidths.items()
            }
            total_nbytes = {
                (scheduler_workers[w1].name, scheduler_workers[w2].name,): format_bytes(
                    sum(nb)
                )
                for (w1, w2), nb in total_nbytes.items()
            }

            print("Roundtrip benchmark")
            print("--------------------------")
            print(f"Operation          | {args.operation}")
            print(f"User size          | {args.size}")
            print(f"User second size   | {args.second_size}")
            print(f"User chunk-size    | {args.chunk_size}")
            print(f"Compute shape      | {size}")
            print(f"Compute chunk-size | {chunksize}")
            print(f"Ignore-size        | {format_bytes(args.ignore_size)}")
            print(f"Protocol           | {args.protocol}")
            print(f"Device(s)          | {args.devs}")
            print(f"Worker Thread(s)   | {args.threads_per_worker}")
            print("==========================")
            print("Wall-clock         | npartitions")
            print("--------------------------")
            for (took, npartitions) in took_list:
                t = format_time(took)
                t += " " * (11 - len(t))
                print(f"{t}        | {npartitions}")
            print("==========================")
            print("(w1,w2)            | 25% 50% 75% (total nbytes)")
            print("--------------------------")
            for (d1, d2), bw in sorted(bandwidths.items()):
                fmt = (
                    "(%s,%s)            | %s %s %s (%s)"
                    if args.multi_node or args.sched_addr
                    else "(%02d,%02d)            | %s %s %s (%s)"
                )
                print(fmt % (d1, d2, bw[0], bw[1], bw[2], total_nbytes[(d1, d2)]))

            if args.benchmark_json:

                d = {
                    "operation": args.operation,
                    "size": args.size,
                    "second_size": args.second_size,
                    "chunk_size": args.chunk_size,
                    "compute_size": size,
                    "compute_chunk_size": chunksize,
                    "ignore_size": format_bytes(args.ignore_size),
                    "protocol": args.protocol,
                    "devs": args.devs,
                    "threads_per_worker": args.threads_per_worker,
                    "times": [
                        {"wall_clock": took, "npartitions": npartitions}
                        for (took, npartitions) in took_list
                    ],
                    "bandwidths": {
                        f"({d1},{d2})"
                        if args.multi_node or args.sched_addr
                        else "(%02d,%02d)"
                        % (d1, d2): {
                            "25%": bw[0],
                            "50%": bw[1],
                            "75%": bw[2],
                            "total_nbytes": total_nbytes[(d1, d2)],
                        }
                        for (d1, d2), bw in sorted(bandwidths.items())
                    },
                }
                with open(args.benchmark_json, "w") as fp:
                    dump(d, fp, indent=2)

            # An SSHCluster will not automatically shut down, we have to
            # ensure it does.
            if args.multi_node:
                await client.shutdown()


def parse_args():
    special_args = [
        {
            "name": ["-s", "--size",],
            "default": "10000",
            "metavar": "n",
            "type": int,
            "help": "The array size n in n^2 (default 10000). For 'svd' operation "
            "the second dimension is given by --second-size.",
        },
        {
            "name": ["-2", "--second-size",],
            "default": "1000",
            "type": int,
            "help": "The second dimension size for 'svd' operation (default 1000).",
        },
        {
            "name": ["-t", "--type",],
            "choices": ["cpu", "gpu"],
            "default": "gpu",
            "type": str,
            "help": "Do merge with GPU or CPU dataframes.",
        },
        {
            "name": ["-o", "--operation",],
            "default": "transpose_sum",
            "type": str,
            "help": "The operation to run, valid options are: "
            "'transpose_sum' (default), 'dot', 'fft', 'svd', 'sum', 'mean', 'slice'.",
        },
        {
            "name": ["-c", "--chunk-size",],
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
            "name": "--benchmark-json",
            "default": None,
            "type": str,
            "help": "Dump a JSON report of benchmarks (optional).",
        },
    ]

    return parse_benchmark_args(
        description="Transpose on LocalCUDACluster benchmark", args_list=special_args
    )


def main():
    args = parse_args()
    asyncio.get_event_loop().run_until_complete(run(args))


if __name__ == "__main__":
    main()
