import asyncio
from collections import defaultdict
from time import perf_counter as clock

import dask.array as da
from dask.distributed import Client, performance_report, wait
from dask.utils import format_bytes, format_time, parse_bytes
from dask_cuda.benchmarks.utils import (
    get_cluster_options,
    get_scheduler_workers,
    parse_benchmark_args,
    setup_memory_pool,
)

import cupy
import numpy as np


async def _run(client, args):
    # Create a simple random array
    rs = da.random.RandomState(RandomState=cupy.random.RandomState)
    x = rs.random((args.size, args.size), chunks=args.chunk_size).persist()
    await wait(x)

    # Execute the operations to benchmark
    if args.profile is not None:
        async with performance_report(filename=args.profile):
            t1 = clock()
            await client.compute((x + x.T).sum())
            took = clock() - t1
    else:
        t1 = clock()
        await client.compute((x + x.T).sum())
        took = clock() - t1

    return (took, x.npartitions)


async def run(args):
    cluster_options = get_cluster_options(args)
    Cluster = cluster_options["class"]
    cluster_args = cluster_options["args"]
    cluster_kwargs = cluster_options["kwargs"]
    scheduler_addr = cluster_options["scheduler_addr"]

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

            await client.run(setup_memory_pool, disable_pool=args.no_rmm_pool)
            # Create an RMM pool on the scheduler due to occasional deserialization
            # of CUDA objects. May cause issues with InfiniBand otherwise.
            await client.run_on_scheduler(
                setup_memory_pool, 1e9, disable_pool=args.no_rmm_pool
            )

            took_list = []
            for i in range(args.runs):
                took_list.append(await _run(client, args))

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
            print(f"Size        | {args.size}*{args.size}")
            print(f"Chunk-size  | {args.chunk_size}")
            print(f"Ignore-size | {format_bytes(args.ignore_size)}")
            print(f"Protocol    | {args.protocol}")
            print(f"Device(s)   | {args.devs}")
            print("==========================")
            print("Wall-clock  | npartitions")
            print("--------------------------")
            for (took, npartitions) in took_list:
                t = format_time(took)
                t += " " * (11 - len(t))
                print(f"{t} | {npartitions}")
            print("==========================")
            print("(w1,w2)     | 25% 50% 75% (total nbytes)")
            print("--------------------------")
            for (d1, d2), bw in sorted(bandwidths.items()):
                fmt = (
                    "(%s,%s)     | %s %s %s (%s)"
                    if args.multi_node
                    else "(%02d,%02d)     | %s %s %s (%s)"
                )
                print(fmt % (d1, d2, bw[0], bw[1], bw[2], total_nbytes[(d1, d2)]))

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
            "help": "The size n in n^2 (default 10000)",
        },
        {
            "name": ["-c", "--chunk-size",],
            "default": "128 MiB",
            "metavar": "nbytes",
            "type": str,
            "help": "Chunk size (default '128 MiB')",
        },
        {
            "name": "--ignore-size",
            "default": "1 MiB",
            "metavar": "nbytes",
            "type": parse_bytes,
            "help": "Ignore messages smaller than this (default '1 MB')",
        },
        {"name": "--runs", "default": 3, "type": int, "help": "Number of runs",},
    ]

    return parse_benchmark_args(
        description="Transpose on LocalCUDACluster benchmark", args_list=special_args
    )


def main():
    args = parse_args()
    asyncio.get_event_loop().run_until_complete(run(args))


if __name__ == "__main__":
    main()
