import asyncio
from json import dumps
from time import perf_counter as clock
from warnings import filterwarnings

import cupy as cp
import numpy as np
from cupyx.scipy.ndimage.filters import convolve as cp_convolve
from scipy.ndimage import convolve as sp_convolve

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
    setup_memory_pool,
)


def mean_filter(a, shape):
    a_k = np.full_like(a, 1.0 / np.prod(shape), shape=shape)
    if isinstance(a, cp.ndarray):
        return cp_convolve(a, a_k)
    else:
        return sp_convolve(a, a_k)


async def _run(client, args):
    # Create a simple random array
    if args.type == "gpu":
        rs = da.random.RandomState(RandomState=cp.random.RandomState)
    else:
        rs = da.random.RandomState(RandomState=np.random.RandomState)
    x = rs.random((args.size, args.size), chunks=args.chunk_size).persist()
    ks = 2 * (2 * args.kernel_size + 1,)
    await wait(x)

    data_processed = x.nbytes

    # Execute the operations to benchmark
    if args.profile is not None:
        async with performance_report(filename=args.profile):
            t1 = clock()
            await wait(
                client.persist(x.map_overlap(mean_filter, args.kernel_size, shape=ks))
            )
            took = clock() - t1
    else:
        t1 = clock()
        await wait(
            client.persist(x.map_overlap(mean_filter, args.kernel_size, shape=ks))
        )
        took = clock() - t1

    return (took, data_processed)


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

            await client.run(
                setup_memory_pool,
                disable_pool=args.disable_rmm_pool,
                log_directory=args.rmm_log_directory,
            )
            # Create an RMM pool on the scheduler due to occasional deserialization
            # of CUDA objects. May cause issues with InfiniBand otherwise.
            await client.run_on_scheduler(
                setup_memory_pool,
                pool_size=1e9,
                disable_pool=args.disable_rmm_pool,
                log_directory=args.rmm_log_directory,
            )

            took_list = []
            for i in range(args.runs):
                took_list.append(await _run(client, args))

            incoming_logs = await client.run(
                lambda dask_worker: dask_worker.incoming_transfer_log
            )
            p2p_bw_dict = peer_to_peer_bandwidths(
                incoming_logs, scheduler_workers, args.ignore_size
            )
            bandwidths = p2p_bw_dict["bandwidths"]
            bandwidths_all = p2p_bw_dict["bandwidths_all"]
            total_nbytes = p2p_bw_dict["total_nbytes"]

            print("Roundtrip benchmark")
            print_separator(separator="-")
            print_key_value(key="Size", value=f"{args.size}*{args.size}")
            print_key_value(key="Chunk size", value=f"{args.chunk_size}")
            print_key_value(
                key="Ignore size", value=f"{format_bytes(args.ignore_size)}"
            )
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
            print_key_value(key="Wall clock", value="Partitions")
            print_separator(separator="-")
            t_p = []
            times = []
            for (took, data_processed) in took_list:
                throughput = int(data_processed / took)
                m = format_time(took)
                times.append(took)
                t_p.append(throughput)
                print_key_value(key=f"{m}", value=f"{format_bytes(throughput)}/s")
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
                key = (
                    f"({d1},{d2})"
                    if args.multi_node or args.sched_addr
                    else f"({d1:02d},{d2:02d})"
                )
                print_key_value(
                    key=key, value=f"{bw[0]} {bw[1]} {bw[2]} ({total_nbytes[(d1, d2)]})"
                )

            if args.benchmark_json:
                bandwidths_json = {
                    "bandwidth_({d1},{d2})_{i}"
                    if args.multi_node or args.sched_addr
                    else "(%02d,%02d)_%s" % (d1, d2, i): parse_bytes(v.rstrip("/s"))
                    for (d1, d2), bw in sorted(bandwidths.items())
                    for i, v in zip(
                        ["25%", "50%", "75%", "total_nbytes"],
                        [bw[0], bw[1], bw[2], total_nbytes[(d1, d2)]],
                    )
                }

                with open(args.benchmark_json, "a") as fp:
                    for took, data_processed in took_list:
                        fp.write(
                            dumps(
                                dict(
                                    {
                                        "size": args.size * args.size,
                                        "chunk_size": args.chunk_size,
                                        "ignore_size": args.ignore_size,
                                        "protocol": args.protocol,
                                        "devs": args.devs,
                                        "device_memory_limit": args.device_memory_limit,
                                        "worker_threads": args.threads_per_worker,
                                        "rmm_pool": not args.disable_rmm_pool,
                                        "tcp": args.enable_tcp_over_ucx,
                                        "ib": args.enable_infiniband,
                                        "nvlink": args.enable_nvlink,
                                        "wall_clock": took,
                                        "npartitions": data_processed / took,
                                    },
                                    **bandwidths_json,
                                )
                            )
                            + "\n"
                        )

            # An SSHCluster will not automatically shut down, we have to
            # ensure it does.
            if args.multi_node:
                await client.shutdown()


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
    ]

    return parse_benchmark_args(
        description="Transpose on LocalCUDACluster benchmark", args_list=special_args
    )


def main():
    args = parse_args()
    asyncio.get_event_loop().run_until_complete(run(args))


if __name__ == "__main__":
    main()
