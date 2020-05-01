import argparse
import asyncio
from collections import defaultdict
from time import perf_counter as clock

import dask.array as da
from dask.distributed import Client, performance_report, wait
from dask.utils import format_bytes, format_time, parse_bytes
from dask_cuda.local_cuda_cluster import LocalCUDACluster

import cupy
import numpy as np


async def run(args):

    # Set up workers on the local machine
    async with LocalCUDACluster(
        protocol=args.protocol,
        n_workers=len(args.devs.split(",")),
        CUDA_VISIBLE_DEVICES=args.devs,
        ucx_net_devices=args.ucx_net_devices,
        enable_tcp_over_ucx=args.enable_tcp_over_ucx,
        enable_infiniband=args.enable_infiniband,
        enable_nvlink=args.enable_nvlink,
        asynchronous=True,
    ) as cluster:
        async with Client(cluster, asynchronous=True) as client:

            def _worker_setup(size=None):
                import rmm

                rmm.reinitialize(
                    pool_allocator=not args.no_rmm_pool,
                    devices=0,
                    initial_pool_size=size,
                )
                cupy.cuda.set_allocator(rmm.rmm_cupy_allocator)

            await client.run(_worker_setup)
            # Create an RMM pool on the scheduler due to occasional deserialization
            # of CUDA objects. May cause issues with InfiniBand otherwise.
            await client.run_on_scheduler(_worker_setup, 1e9)

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
                (
                    cluster.scheduler.workers[w1].name,
                    cluster.scheduler.workers[w2].name,
                ): [
                    "%s/s" % format_bytes(x) for x in np.quantile(v, [0.25, 0.50, 0.75])
                ]
                for (w1, w2), v in bandwidths.items()
            }
            total_nbytes = {
                (
                    cluster.scheduler.workers[w1].name,
                    cluster.scheduler.workers[w2].name,
                ): format_bytes(sum(nb))
                for (w1, w2), nb in total_nbytes.items()
            }

            print("Roundtrip benchmark")
            print("--------------------------")
            print(f"Size        | {args.size}*{args.size}")
            print(f"Chunk-size  | {args.chunk_size}")
            print(f"Ignore-size | {format_bytes(args.ignore_size)}")
            print(f"Protocol    | {args.protocol}")
            print(f"Device(s)   | {args.devs}")
            print(f"npartitions | {x.npartitions}")
            print("==========================")
            print(f"Total time  | {format_time(took)}")
            print("==========================")
            print("(w1,w2)     | 25% 50% 75% (total nbytes)")
            print("--------------------------")
            for (d1, d2), bw in sorted(bandwidths.items()):
                print(
                    "(%02d,%02d)     | %s %s %s (%s)"
                    % (d1, d2, bw[0], bw[1], bw[2], total_nbytes[(d1, d2)])
                )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transpose on LocalCUDACluster benchmark"
    )
    parser.add_argument(
        "-d", "--devs", default="0", type=str, help='GPU devices to use (default "0").'
    )
    parser.add_argument(
        "-p",
        "--protocol",
        choices=["tcp", "ucx"],
        default="tcp",
        type=str,
        help="The communication protocol to use.",
    )
    parser.add_argument(
        "-s",
        "--size",
        default="10000",
        metavar="n",
        type=int,
        help="The size n in n^2 (default 10000)",
    )
    parser.add_argument(
        "-c",
        "--chunk-size",
        default="128 MiB",
        metavar="nbytes",
        type=str,
        help='Chunk size (default "128 MiB")',
    )
    parser.add_argument(
        "--ignore-size",
        default="1 MiB",
        metavar="nbytes",
        type=parse_bytes,
        help='Ignore messages smaller than this (default "1 MB")',
    )
    parser.add_argument(
        "--profile",
        metavar="PATH",
        default=None,
        type=str,
        help="Write dask profile report (E.g. dask-report.html)",
    )
    parser.add_argument(
        "--no-rmm-pool", action="store_true", help="Disable the RMM memory pool"
    )
    parser.add_argument(
        "--enable-tcp-over-ucx",
        action="store_true",
        dest="enable_tcp_over_ucx",
        help="Enable tcp over ucx.",
    )
    parser.add_argument(
        "--enable-infiniband",
        action="store_true",
        dest="enable_infiniband",
        help="Enable infiniband over ucx.",
    )
    parser.add_argument(
        "--enable-nvlink",
        action="store_true",
        dest="enable_nvlink",
        help="Enable NVLink over ucx.",
    )
    parser.add_argument(
        "--disable-tcp-over-ucx",
        action="store_false",
        dest="enable_tcp_over_ucx",
        help="Disable tcp over ucx.",
    )
    parser.add_argument(
        "--disable-infiniband",
        action="store_false",
        dest="enable_infiniband",
        help="Disable infiniband over ucx.",
    )
    parser.add_argument(
        "--disable-nvlink",
        action="store_false",
        dest="enable_nvlink",
        help="Disable NVLink over ucx.",
    )
    parser.add_argument(
        "--ucx-net-devices",
        default=None,
        type=str,
        help="The device to be used for UCX communication, or 'auto'. "
        "Ignored if protocol is 'tcp'",
    )
    parser.set_defaults(
        enable_tcp_over_ucx=True, enable_infiniband=True, enable_nvlink=True
    )
    args = parser.parse_args()
    if args.protocol == "tcp":
        args.enable_tcp_over_ucx = False
        args.enable_infinibank = False
        args.enable_nvlink = False
    return args


def main():
    args = parse_args()
    asyncio.get_event_loop().run_until_complete(run(args))


if __name__ == "__main__":
    main()
