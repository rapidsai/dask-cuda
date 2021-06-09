import contextlib
from collections import defaultdict
from time import perf_counter as clock
from warnings import filterwarnings

import numpy

import dask
from dask import array as da
from dask.dataframe.shuffle import shuffle
from dask.distributed import Client, performance_report, wait
from dask.utils import format_bytes, format_time, parse_bytes

import dask_cuda.explicit_comms.dataframe.shuffle
from dask_cuda.benchmarks.utils import (
    get_cluster_options,
    get_scheduler_workers,
    parse_benchmark_args,
    plot_benchmark,
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

    # Collect, aggregate, and print peer-to-peer bandwidths
    incoming_logs = client.run(lambda dask_worker: dask_worker.incoming_transfer_log)
    bandwidths = defaultdict(list)
    total_nbytes = defaultdict(list)
    for k, L in incoming_logs.items():
        for d in L:
            if d["total"] >= args.ignore_size:
                bandwidths[k, d["who"]].append(d["bandwidth"])
                total_nbytes[k, d["who"]].append(d["total"])
    bandwidths = {
        (scheduler_workers[w1].name, scheduler_workers[w2].name): [
            "%s/s" % format_bytes(x) for x in numpy.quantile(v, [0.25, 0.50, 0.75])
        ]
        for (w1, w2), v in bandwidths.items()
    }
    total_nbytes = {
        (scheduler_workers[w1].name, scheduler_workers[w2].name,): format_bytes(sum(nb))
        for (w1, w2), nb in total_nbytes.items()
    }

    t_runs = numpy.empty(len(took_list))
    if args.markdown:
        print("```")
    print("Shuffle benchmark")
    print("-------------------------------")
    print(f"backend        | {args.backend}")
    print(f"partition-size | {format_bytes(args.partition_size)}")
    print(f"in-parts       | {args.in_parts}")
    print(f"protocol       | {args.protocol}")
    print(f"device(s)      | {args.devs}")
    print(f"rmm-pool       | {(not args.disable_rmm_pool)}")
    if args.protocol == "ucx":
        print(f"tcp            | {args.enable_tcp_over_ucx}")
        print(f"ib             | {args.enable_infiniband}")
        print(f"nvlink         | {args.enable_nvlink}")
    print(f"data-processed | {format_bytes(took_list[0][0])}")
    print("===============================")
    print("Wall-clock     | Throughput")
    print("-------------------------------")
    for idx, (data_processed, took) in enumerate(took_list):
        throughput = int(data_processed / took)
        m = format_time(took)
        m += " " * (15 - len(m))
        print(f"{m}| {format_bytes(throughput)}/s")
        t_runs[idx] = float(format_bytes(throughput).split(" ")[0])
    print("===============================")
    if args.markdown:
        print("\n```")

    if args.plot is not None:
        plot_benchmark(t_runs, args.plot, historical=True)

    if args.backend == "dask":
        if args.markdown:
            print("<details>\n<summary>Worker-Worker Transfer Rates</summary>\n\n```")
        print("(w1,w2)     | 25% 50% 75% (total nbytes)")
        print("-------------------------------")
        for (d1, d2), bw in sorted(bandwidths.items()):
            fmt = (
                "(%s,%s)     | %s %s %s (%s)"
                if args.multi_node or args.sched_addr
                else "(%02d,%02d)     | %s %s %s (%s)"
            )
            print(fmt % (d1, d2, bw[0], bw[1], bw[2], total_nbytes[(d1, d2)]))
        if args.markdown:
            print("```\n</details>\n")

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
            "name": ["-b", "--backend",],
            "choices": ["dask", "explicit-comms"],
            "default": "dask",
            "type": str,
            "help": "The backend to use.",
        },
        {
            "name": ["-t", "--type",],
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
        {"name": "--runs", "default": 3, "type": int, "help": "Number of runs",},
    ]

    return parse_benchmark_args(
        description="Distributed shuffle (dask/cudf) benchmark", args_list=special_args
    )


if __name__ == "__main__":
    main(parse_args())
