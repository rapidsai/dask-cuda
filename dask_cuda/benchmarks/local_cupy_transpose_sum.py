from collections import defaultdict
import argparse
import asyncio
from time import perf_counter as clock
import numpy as np
from pprint import pprint
import cupy

import dask.array as da
from dask.distributed import Client, wait, performance_report
from dask.utils import format_time, format_bytes, parse_bytes
from dask_cuda.local_cuda_cluster import LocalCUDACluster

try:
    import cudf
    from rmm._lib.device_buffer import DeviceBuffer

    class RMMemory(cupy.cuda.memory.BaseMemory):
        def __init__(self, size):
            self.size = size
            self.device_id = cupy.cuda.device.get_device_id()
            if size > 0:
                self.rmm_array = DeviceBuffer(size=size)
                self.ptr = self.rmm_array.ptr
            else:
                self.rmm_array = None
                self.ptr = 0

    def rmm_mem_allocator(bsize):
        return cupy.cuda.memory.MemoryPointer(RMMemory(bsize), 0)


except ImportError:
    pass


async def run(args):

    # Set up workers on the local machine
    async with LocalCUDACluster(
        protocol=args.protocol,
        n_workers=len(args.devs.split(",")),
        CUDA_VISIBLE_DEVICES=args.devs,
        asynchronous=True,
    ) as cluster:
        async with Client(cluster, asynchronous=True) as client:

            if args.rmm:
                await client.run(cudf.set_allocator, "default", pool=True)
                await client.run(cupy.cuda.set_allocator, rmm_mem_allocator)

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
            print(f"rmm mpool   | {args.rmm}")
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
        "--rmm", action="store_true", help="Enable RMM memory pool for cupy",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    asyncio.get_event_loop().run_until_complete(run(args))


if __name__ == "__main__":
    main()
