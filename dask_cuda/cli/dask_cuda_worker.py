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
    "--interface",
    type=str,
    default=None,
    help="The external interface used to connect to the scheduler, usually "
    "an ethernet interface is used for connection, and not an InfiniBand "
    "interface (if one is available).",
)
@click.option("--nthreads", type=int, default=1, help="Number of threads per process.")
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
    default="0.8",
    help="Specifies the size of the CUDA device LRU cache, which "
    "is used to determine when the worker starts spilling to host "
    "memory.  This can be a float (fraction of total device "
    "memory), an integer (bytes), a string (like 5GB or 5000M), "
    "and 'auto' or 0 to disable spilling to host (i.e., allow "
    "full device memory usage). Default is 0.8, 80% of the "
    "worker's total device memory.",
)
@click.option(
    "--rmm-pool-size",
    default=None,
    help="If specified, initialize each worker with an RMM pool of "
    "the given size, otherwise no RMM pool is created. This can be "
    "an integer (bytes) or string (like 5GB or 5000M)."
    "NOTE: This size is a per worker (i.e., per GPU) configuration, "
    "and not cluster-wide!",
)
@click.option(
    "--rmm-managed-memory/--no-rmm-managed-memory",
    default=False,
    help="If enabled, initialize each worker with RMM and set it to "
    "use managed memory. If disabled, RMM may still be used if "
    "--rmm-pool-size is specified, but in that case with default "
    "(non-managed) memory type."
    "WARNING: managed memory is currently incompatible with NVLink, "
    "trying to enable both will result in an exception.",
)
@click.option(
    "--rmm-log-directory",
    default=None,
    help="Directory to write per-worker RMM log files to; the client "
    "and scheduler are not logged here."
    "NOTE: Logging will only be enabled if --rmm-pool-size or "
    "--rmm-managed-memory are specified.",
)
@click.option(
    "--reconnect/--no-reconnect",
    default=True,
    help="Reconnect to scheduler if disconnected",
)
@click.option("--pid-file", type=str, default="", help="File to write the process PID")
@click.option(
    "--local-directory", default=None, type=str, help="Directory to place worker files"
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
    "--enable-rdmacm/--disable-rdmacm",
    default=False,
    help="Enable RDMA connection manager, currently requires InfiniBand enabled.",
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
    help="When None (default), 'UCX_NET_DEVICES' will be left to its default. "
    "Otherwise, it must be a non-empty string with the interface name, such as "
    "such as 'eth0' or 'auto' to allow for automatically choosing the closest "
    "interface based on the system's topology. Normally used only with "
    "--enable-infiniband to specify the interface to be used by the worker, "
    "such as 'mlx5_0:1' or 'ib0'. "
    "WARNING: 'auto' requires UCX-Py to be installed and compiled with hwloc "
    "support. Additionally that will always use the closest interface, and "
    "that may cause unexpected errors if that interface is not properly "
    "configured or is disconnected, for that reason it's limited to "
    "InfiniBand only and will still cause unpredictable errors if not _ALL_ "
    "interfaces are connected and properly configured.",
)
@click.option(
    "--enable-jit-unspill/--disable-jit-unspill",
    default=None,
    help="Enable just-in-time unspilling. This is experimental and doesn't "
    "support memory spilling to disk Please see proxy_object.ProxyObject "
    "and proxify_host_file.ProxifyHostFile.",
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
