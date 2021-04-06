from __future__ import absolute_import, division, print_function

import logging

import click
from tornado.ioloop import IOLoop, TimeoutError

from distributed.cli.utils import check_python_3, install_signal_handlers
from distributed.preloading import validate_preload_argv
from distributed.security import Security

from ..cuda_worker import CUDAWorker

logger = logging.getLogger(__name__)


pem_file_option_type = click.Path(exists=True, resolve_path=True)


@click.command(context_settings=dict(ignore_unknown_options=True))
@click.argument("scheduler", type=str, required=False)
@click.argument(
    "preload_argv", nargs=-1, type=click.UNPROCESSED, callback=validate_preload_argv
)
@click.option(
    "--host",
    type=str,
    default=None,
    help="""Serving host; should be an IP address visible to the scheduler and other
    workers""",
)
@click.option(
    "--nthreads",
    type=int,
    default=1,
    show_default=True,
    help="Number of threads per process",
)
@click.option(
    "--name",
    type=str,
    default=None,
    help="""A unique name for this worker like ``worker-1``; if used with ``--nprocs``,
    then the process number will be appended to the worker name, e.g. ``worker-1-0``,
    ``worker-1-1``, ``worker-1-2``, ...""",
)
@click.option(
    "--memory-limit",
    default="auto",
    show_default=True,
    help="""Bytes of memory per process that the worker can use; can be an integer
    (bytes), float (fraction of total system memory), string (like ``5GB`` or
    ``5000M``), ``auto``, or 0 for no memory management""",
)
@click.option(
    "--device-memory-limit",
    default="0.8",
    show_default=True,
    help="""Specifies the size of the CUDA device LRU cache, which is used to determine
    when the worker starts spilling to host memory; can be an integer (bytes), float
    (fraction of total device memory), string (like ``5GB`` or ``5000M``), ``auto``,
    or 0 to disable spilling to host (i.e., allow full device memory usage)""",
)
@click.option(
    "--rmm-pool-size",
    default=None,
    help="""If specified, initialize each worker with an RMM pool of the given size; can
    be an integer (bytes) or string (like ``5GB`` or ``5000M``); this size is per-worker
    (GPU) and not cluster-wide

    .. note::
        This size is a per-worker (GPU) configuration, and not cluster-wide""",
)
@click.option(
    "--rmm-managed-memory/--no-rmm-managed-memory",
    default=False,
    show_default=True,
    help="""If enabled, initialize each worker with RMM and set it to use managed
    memory; if disabled, RMM may still be used by specifying ``--rmm-pool-size``

    .. warning::
        Managed memory is currently incompatible with NVLink; trying to enable both will
        result in an exception.""",
)
@click.option(
    "--rmm-log-directory",
    default=None,
    help="""Directory to write per-worker RMM log files to; the client
    and scheduler are not logged here.

    .. note::
        Logging will only be enabled if ``--rmm-pool-size`` or ``--rmm-managed-memory``
        are specified.""",
)
@click.option("--pid-file", type=str, default="", help="File to write the process PID")
@click.option(
    "--resources",
    type=str,
    default="",
    help="""Resources for task constraints like ``GPU=2 MEM=10e9``; resources are
    applied separately to each worker process (only relevant when starting multiple
    worker processes with ``--nprocs``)""",
)
@click.option(
    "--dashboard/--no-dashboard",
    "dashboard",
    default=True,
    show_default=True,
    required=False,
    help="Launch the dashboard",
)
@click.option(
    "--dashboard-address",
    type=str,
    default=":0",
    show_default=True,
    help="Relative address to serve dashboard (if enabled)",
)
@click.option(
    "--local-directory", default=None, type=str, help="Directory to place worker files"
)
@click.option(
    "--scheduler-file",
    type=str,
    default="",
    help="""Filename to JSON encoded scheduler information; use with ``dask-scheduler``
    ``--scheduler-file``""",
)
@click.option(
    "--interface",
    type=str,
    default=None,
    help="""External interface used to connect to the scheduler; usually an ethernet
    interface is used for connection, and not an InfiniBand interface (if one is
    available)""",
)
@click.option(
    "--death-timeout",
    type=str,
    default=None,
    help="Seconds to wait for a scheduler before closing",
)
@click.option(
    "--preload",
    type=str,
    multiple=True,
    is_eager=True,
    help="""Module that should be loaded by each worker process like ``foo.bar`` or
    ``/path/to/foo.py``""",
)
@click.option(
    "--dashboard-prefix", type=str, default=None, help="Prefix for the dashboard"
)
@click.option(
    "--tls-ca-file",
    type=pem_file_option_type,
    default=None,
    help="CA certificate(s) file for TLS (in PEM format)",
)
@click.option(
    "--tls-cert",
    type=pem_file_option_type,
    default=None,
    help="Certificate file for TLS (in PEM format)",
)
@click.option(
    "--tls-key",
    type=pem_file_option_type,
    default=None,
    help="Private key file for TLS (in PEM format)",
)
@click.option(
    "--enable-tcp-over-ucx/--disable-tcp-over-ucx",
    default=False,
    show_default=True,
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
    show_default=True,
    help="Enable NVLink communication",
)
@click.option(
    "--enable-rdmacm/--disable-rdmacm",
    default=False,
    show_default=True,
    help="Enable RDMA connection manager, currently requires InfiniBand enabled.",
)
@click.option(
    "--net-devices",
    type=str,
    default=None,
    help="""If specified, workers will use this interface for UCX communication; can be
    a string (like ``eth0``), or ``auto`` to pick the optimal interface based on
    the system's topology; typically only used with ``--enable-infiniband``

    .. warning::
        ``auto`` requires UCX-Py to be installed and compiled with hwloc support;
        unexpected errors can occur when using ``auto`` if any interfaces are
        disconnected or improperly configured""",
)
@click.option(
    "--enable-jit-unspill/--disable-jit-unspill",
    default=None,
    help="""Enable just-in-time unspilling; this is experimental and doesn't support
    memory spilling to disk; see ``proxy_object.ProxyObject`` and
    ``proxify_host_file.ProxifyHostFile`` for more info""",
)
def main(
    scheduler,
    host,
    nthreads,
    name,
    memory_limit,
    device_memory_limit,
    rmm_pool_size,
    rmm_managed_memory,
    rmm_log_directory,
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
    enable_rdmacm,
    net_devices,
    enable_jit_unspill,
    **kwargs,
):
    if tls_ca_file and tls_cert and tls_key:
        security = Security(
            tls_ca_file=tls_ca_file, tls_worker_cert=tls_cert, tls_worker_key=tls_key,
        )
    else:
        security = None

    if isinstance(scheduler, str) and scheduler.startswith("-"):
        raise ValueError(
            "The scheduler address can't start with '-'. Please check "
            "your command line arguments, you probably attempted to use "
            "unsupported one. Scheduler address: %s" % scheduler
        )

    worker = CUDAWorker(
        scheduler,
        host,
        nthreads,
        name,
        memory_limit,
        device_memory_limit,
        rmm_pool_size,
        rmm_managed_memory,
        rmm_log_directory,
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
        security,
        enable_tcp_over_ucx,
        enable_infiniband,
        enable_nvlink,
        enable_rdmacm,
        net_devices,
        enable_jit_unspill,
        **kwargs,
    )

    async def on_signal(signum):
        logger.info("Exiting on signal %d", signum)
        await worker.close()

    async def run():
        await worker
        await worker.finished()

    loop = IOLoop.current()

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
