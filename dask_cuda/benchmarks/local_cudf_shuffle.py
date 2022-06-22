import contextlib
from json import dumps
from time import perf_counter as clock
from warnings import filterwarnings

import numpy as np

import dask
from dask import array as da
from dask.dataframe.shuffle import shuffle
from dask.distributed import Client, performance_report, wait
from dask.utils import format_bytes, format_time, parse_bytes

import dask_cuda.explicit_comms.dataframe.shuffle
from dask_cuda.benchmarks.utils import (
    get_cluster_options,
    get_scheduler_workers,
    hmean,
    hstd,
    parse_benchmark_args,
    peer_to_peer_bandwidths,
    plot_benchmark,
    print_key_value,
    print_separator,
    setup_memory_pool,
)
from dask_cuda.utils import all_to_all


def shuffle_dask(df):
    wait(shuffle(df, index="data", shuffle="tasks").persist())


def shuffle_explicit_comms(df):
    wait(
        dask_cuda.explicit_comms.dataframe.shuffle.shuffle(
            df, column_names="data"
        ).persist()
    )


def run(client, args, n_workers, write_profile=None):
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


def main(args):
    cluster_options = get_cluster_options(args)
    Cluster = cluster_options["class"]
    cluster_args = cluster_options["args"]
    cluster_kwargs = cluster_options["kwargs"]
    scheduler_addr = cluster_options["scheduler_addr"]

    if args.sched_addr:
        client = Client(args.sched_addr)
    else:
        filterwarnings(
            "ignore", message=".*NVLink.*rmm_pool_size.*", category=UserWarning
        )

        cluster = Cluster(*cluster_args, **cluster_kwargs)
        if args.multi_node:
            import time

            # Allow some time for workers to start and connect to scheduler
            # TODO: make this a command-line argument?
            time.sleep(15)

        client = Client(scheduler_addr if args.multi_node else cluster)

    if args.type == "gpu":
        client.run(
            setup_memory_pool,
            pool_size=args.rmm_pool_size,
            disable_pool=args.disable_rmm_pool,
            log_directory=args.rmm_log_directory,
        )
        # Create an RMM pool on the scheduler due to occasional deserialization
        # of CUDA objects. May cause issues with InfiniBand otherwise.
        client.run_on_scheduler(
            setup_memory_pool,
            pool_size=1e9,
            disable_pool=args.disable_rmm_pool,
            log_directory=args.rmm_log_directory,
        )

    scheduler_workers = client.run_on_scheduler(get_scheduler_workers)
    n_workers = len(scheduler_workers)
    client.wait_for_workers(n_workers)

    if args.all_to_all:
        all_to_all(client)

    took_list = []
    for _ in range(args.runs - 1):
        took_list.append(run(client, args, n_workers, write_profile=None))
    took_list.append(
        run(client, args, n_workers, write_profile=args.profile)
    )  # Only profiling the last run

    incoming_logs = client.run(lambda dask_worker: dask_worker.incoming_transfer_log)
    p2p_bw_dict = peer_to_peer_bandwidths(
        incoming_logs, scheduler_workers, args.ignore_size
    )
    bandwidths = p2p_bw_dict["bandwidths"]
    bandwidths_all = p2p_bw_dict["bandwidths_all"]
    total_nbytes = p2p_bw_dict["total_nbytes"]

    t_runs = np.empty(len(took_list))
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
    print_key_value(key="Data processed", value=f"{format_bytes(took_list[0][0])}")
    print_separator(separator="=")
    print_key_value(key="Wall clock", value="Throughput")
    print_separator(separator="-")
    t_p = []
    times = []
    for idx, (data_processed, took) in enumerate(took_list):
        throughput = int(data_processed / took)
        m = format_time(took)
        times.append(took)
        t_p.append(throughput)
        print_key_value(key=f"{m}", value=f"{format_bytes(throughput)}/s")
        t_runs[idx] = float(format_bytes(throughput).split(" ")[0])
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

    if args.markdown:
        print("\n```")

    if args.plot is not None:
        plot_benchmark(t_runs, args.plot, historical=True)

    if args.backend == "dask":
        if args.markdown:
            print("<details>\n<summary>Worker-Worker Transfer Rates</summary>\n\n```")
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
        if args.markdown:
            print("```\n</details>\n")

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
            for data_processed, took in took_list:
                fp.write(
                    dumps(
                        dict(
                            {
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
                                "data_processed": data_processed,
                                "wall_clock": took,
                                "throughput": data_processed / took,
                            },
                            **bandwidths_json,
                        )
                    )
                    + "\n"
                )

    if args.multi_node:
        client.shutdown()
        client.close()


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
            "name": "--markdown",
            "action": "store_true",
            "help": "Write output as markdown",
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
    main(parse_args())
