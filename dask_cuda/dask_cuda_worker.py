from __future__ import print_function, division, absolute_import

import atexit
import logging
import multiprocessing
import os

import click
from distributed import Nanny
from distributed.config import config
from distributed.utils import get_ip_interface, parse_bytes
from distributed.worker import parse_memory_limit
from distributed.security import Security
from distributed.cli.utils import check_python_3, install_signal_handlers
from distributed.preloading import validate_preload_argv
from distributed.proctitle import (
    enable_proctitle_on_children,
    enable_proctitle_on_current,
)

from .initialize import initialize
from .device_host_file import DeviceHostFile
from .local_cuda_cluster import cuda_visible_devices
from .utils import CPUAffinity, get_cpu_affinity, get_n_gpus, get_device_total_memory

from toolz import valmap
from tornado.ioloop import IOLoop, TimeoutError
from tornado import gen

logger = logging.getLogger(__name__)


pem_file_option_type = click.Path(exists=True, resolve_path=True)


@click.command(context_settings=dict(ignore_unknown_options=True))
@click.argument("scheduler", type=str, required=False)
@click.option(
    "--tls-ca-file",
    type=pem_file_option_type,
    default=None,
    help="CA cert(s) file for TLS (in PEM format)",
)
@click.option(
    "--tls-cert",
    type=pem_file_option_type,
    default=None,
    help="certificate file for TLS (in PEM format)",
)
@click.option(
    "--tls-key",
    type=pem_file_option_type,
    default=None,
    help="private key file for TLS (in PEM format)",
)
@click.option("--dashboard-address", type=str, default=":0", help="dashboard address")
@click.option(
    "--dashboard/--no-dashboard",
    "dashboard",
    default=True,
    show_default=True,
    required=False,
    help="Launch dashboard",
)
@click.option(
    "--host",
    type=str,
    default=None,
    help="Serving host. Should be an ip address that is"
    " visible to the scheduler and other workers. "
    "See --listen-address and --contact-address if you "
    "need different listen and contact addresses. "
    "See --interface.",
)
@click.option(
    "--interface", type=str, default=None, help="Network interface like 'eth0' or 'ib0'"
)
@click.option("--nthreads", type=int, default=0, help="Number of threads per process.")
@click.option(
    "--name",
    type=str,
    default=None,
    help="A unique name for this worker like 'worker-1'. "
    "If used with --nprocs then the process number "
    "will be appended like name-0, name-1, name-2, ...",
)
@click.option(
    "--memory-limit",
    default="auto",
    help="Bytes of memory per process that the worker can use. "
    "This can be an integer (bytes), "
    "float (fraction of total system memory), "
    "string (like 5GB or 5000M), "
    "'auto', or zero for no memory management",
)
@click.option(
    "--device-memory-limit",
    default="auto",
    help="Bytes of memory per CUDA device that the worker can use. "
    "This can be an integer (bytes), "
    "float (fraction of total device memory), "
    "string (like 5GB or 5000M), "
    "'auto', or zero for no memory management "
    "(i.e., allow full device memory usage).",
)
@click.option(
    "--reconnect/--no-reconnect",
    default=True,
    help="Reconnect to scheduler if disconnected",
)
@click.option("--pid-file", type=str, default="", help="File to write the process PID")
@click.option(
    "--local-directory", default="", type=str, help="Directory to place worker files"
)
@click.option(
    "--resources",
    type=str,
    default="",
    help='Resources for task constraints like "GPU=2 MEM=10e9". '
    "Resources are applied separately to each worker process "
    "(only relevant when starting multiple worker processes with '--nprocs').",
)
@click.option(
    "--scheduler-file",
    type=str,
    default="",
    help="Filename to JSON encoded scheduler information. "
    "Use with dask-scheduler --scheduler-file",
)
@click.option(
    "--death-timeout",
    type=str,
    default=None,
    help="Seconds to wait for a scheduler before closing",
)
@click.option(
    "--dashboard-prefix", type=str, default=None, help="Prefix for the Dashboard"
)
@click.option(
    "--preload",
    type=str,
    multiple=True,
    is_eager=True,
    help="Module that should be loaded by each worker process "
    'like "foo.bar" or "/path/to/foo.py"',
)
@click.argument(
    "preload_argv", nargs=-1, type=click.UNPROCESSED, callback=validate_preload_argv
)
@click.option(
    "--enable-tcp-over-ucx/--disable-tcp-over-ucx",
    default=False,
    help="Enable TCP communication over UCX",
)
@click.option(
    "--enable-infiniband/--disable-infiniband",
    default=False,
    help="Enable InfiniBand communication",
)
@click.option(
    "--enable-nvlink/--disable-nvlink",
    default=False,
    help="Enable NVLink communication",
)
@click.option(
    "--net-devices",
    type=str,
    default=None,
    help="Network interface to establish UCX connection, "
    "usually the Ethernet interface, like 'eth0' or 'enp1s0f0'",
)
def main(
    scheduler,
    host,
    nthreads,
    name,
    memory_limit,
    device_memory_limit,
    pid_file,
    resources,
    dashboard,
    dashboard_address,
    local_directory,
    scheduler_file,
    interface,
    death_timeout,
    preload,
    dashboard_prefix,
    tls_ca_file,
    tls_cert,
    tls_key,
    enable_tcp_over_ucx,
    enable_infiniband,
    enable_nvlink,
    net_devices,
    **kwargs,
):
    enable_proctitle_on_current()
    enable_proctitle_on_children()

    sec = Security(
        tls_ca_file=tls_ca_file, tls_worker_cert=tls_cert, tls_worker_key=tls_key
    )

    try:
        nprocs = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    except KeyError:
        nprocs = get_n_gpus()

    if not nthreads:
        nthreads = min(1, multiprocessing.cpu_count() // nprocs)

    if pid_file:
        with open(pid_file, "w") as f:
            f.write(str(os.getpid()))

        def del_pid_file():
            if os.path.exists(pid_file):
                os.remove(pid_file)

        atexit.register(del_pid_file)

    services = {}

    if dashboard:
        try:
            from distributed.dashboard import BokehWorker
        except ImportError:
            pass
        else:
            if dashboard_prefix:
                result = (BokehWorker, {"prefix": dashboard_prefix})
            else:
                result = BokehWorker
            services[("dashboard", dashboard_address)] = result

    if resources:
        resources = resources.replace(",", " ").split()
        resources = dict(pair.split("=") for pair in resources)
        resources = valmap(float, resources)
    else:
        resources = None

    loop = IOLoop.current()

    preload_argv = kwargs.get("preload_argv", [])
    kwargs = {"worker_port": None, "listen_address": None}
    t = Nanny

    if not scheduler and not scheduler_file and "scheduler-address" not in config:
        raise ValueError(
            "Need to provide scheduler address like\n"
            "dask-worker SCHEDULER_ADDRESS:8786"
        )

    if interface:
        if host:
            raise ValueError("Can not specify both interface and host")
        else:
            host = get_ip_interface(interface)

    initialize(
        create_cuda_context=True,
        enable_tcp_over_ucx=enable_tcp_over_ucx,
        enable_infiniband=enable_infiniband,
        enable_nvlink=enable_nvlink,
        net_devices=net_devices,
    )

    nannies = [
        t(
            scheduler,
            scheduler_file=scheduler_file,
            nthreads=nthreads,
            services=services,
            loop=loop,
            resources=resources,
            memory_limit=memory_limit,
            host=host,
            preload=(list(preload) or []) + ["dask_cuda.initialize"],
            preload_argv=(list(preload_argv) or []) + ["--create-cuda-context"],
            security=sec,
            env={"CUDA_VISIBLE_DEVICES": cuda_visible_devices(i)},
            plugins={CPUAffinity(get_cpu_affinity(i))},
            name=name if nprocs == 1 or not name else name + "-" + str(i),
            local_directory=local_directory,
            data=(
                DeviceHostFile,
                {
                    "device_memory_limit": get_device_total_memory(index=i)
                    if (device_memory_limit == "auto" or device_memory_limit == int(0))
                    else parse_bytes(device_memory_limit),
                    "memory_limit": parse_memory_limit(
                        memory_limit, nthreads, total_cores=nprocs
                    ),
                    "local_directory": local_directory,
                },
            ),
            **kwargs,
        )
        for i in range(nprocs)
    ]

    @gen.coroutine
    def close_all():
        # Unregister all workers from scheduler
        yield [n._close(timeout=2) for n in nannies]

    def on_signal(signum):
        logger.info("Exiting on signal %d", signum)
        close_all()

    @gen.coroutine
    def run():
        yield nannies
        yield [n.finished() for n in nannies]

    install_signal_handlers(loop, cleanup=on_signal)

    try:
        loop.run_sync(run)
    except (KeyboardInterrupt, TimeoutError):
        pass
    finally:
        logger.info("End worker")


def go():
    check_python_3()
    main()


if __name__ == "__main__":
    go()
