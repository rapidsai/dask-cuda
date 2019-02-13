from __future__ import print_function, division, absolute_import

import atexit
import logging
import os
from sys import exit

import click
from distributed import Nanny, Worker
from distributed.config import config
from distributed.utils import get_ip_interface, parse_timedelta
from distributed.worker import _ncores
from distributed.security import Security
from distributed.cli.utils import (
    check_python_3,
    uri_from_host_port,
    install_signal_handlers,
)
from distributed.comm import get_address_host_port
from distributed.preloading import validate_preload_argv
from distributed.proctitle import (
    enable_proctitle_on_children,
    enable_proctitle_on_current,
)

from .local_cuda_cluster import cuda_visible_devices
from .utils import get_n_gpus

from toolz import valmap
from tornado.ioloop import IOLoop, TimeoutError
from tornado import gen

logger = logging.getLogger("distributed.dask_worker")


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
@click.option(
    "--bokeh-port", type=int, default=0, help="Bokeh port, defaults to random port"
)
@click.option(
    "--bokeh/--no-bokeh",
    "bokeh",
    default=True,
    show_default=True,
    required=False,
    help="Launch Bokeh Web UI",
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
    default="",
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
@click.option("--bokeh-prefix", type=str, default=None, help="Prefix for the bokeh app")
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
def main(
    scheduler,
    host,
    nthreads,
    name,
    memory_limit,
    pid_file,
    reconnect,
    resources,
    bokeh,
    bokeh_port,
    local_directory,
    scheduler_file,
    interface,
    death_timeout,
    preload,
    preload_argv,
    bokeh_prefix,
    tls_ca_file,
    tls_cert,
    tls_key,
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
        nthreads = min(1, _ncores // nprocs )

    if pid_file:
        with open(pid_file, "w") as f:
            f.write(str(os.getpid()))

        def del_pid_file():
            if os.path.exists(pid_file):
                os.remove(pid_file)

        atexit.register(del_pid_file)

    services = {}

    if bokeh:
        try:
            from distributed.bokeh.worker import BokehWorker
        except ImportError:
            pass
        else:
            if bokeh_prefix:
                result = (BokehWorker, {"prefix": bokeh_prefix})
            else:
                result = BokehWorker
            services[("bokeh", bokeh_port)] = result

    if resources:
        resources = resources.replace(",", " ").split()
        resources = dict(pair.split("=") for pair in resources)
        resources = valmap(float, resources)
    else:
        resources = None

    loop = IOLoop.current()

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

    if host:
        addr = uri_from_host_port(host, 0, 0)
    else:
        # Choose appropriate address for scheduler
        addr = None

    if death_timeout is not None:
        death_timeout = parse_timedelta(death_timeout, "s")

    nannies = [
        t(
            scheduler,
            scheduler_file=scheduler_file,
            ncores=nthreads,
            services=services,
            loop=loop,
            resources=resources,
            memory_limit=memory_limit,
            reconnect=reconnect,
            local_dir=local_directory,
            death_timeout=death_timeout,
            preload=preload,
            preload_argv=preload_argv,
            security=sec,
            contact_address=None,
            env={"CUDA_VISIBLE_DEVICES": cuda_visible_devices(i)},
            name=name if nprocs == 1 or not name else name + "-" + str(i),
            **kwargs
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
        yield [n._start(addr) for n in nannies]
        while all(n.status != "closed" for n in nannies):
            yield gen.sleep(0.2)

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
