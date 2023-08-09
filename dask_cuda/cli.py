from __future__ import absolute_import, division, print_function

import logging

import click
from tornado.ioloop import IOLoop, TimeoutError

from dask import config as dask_config
from distributed import Client
from distributed.cli.utils import install_signal_handlers
from distributed.preloading import validate_preload_argv
from distributed.security import Security
from distributed.utils import import_term

from .cuda_worker import CUDAWorker
from .utils import print_cluster_config

logger = logging.getLogger(__name__)


pem_file_option_type = click.Path(exists=True, resolve_path=True)
scheduler = click.argument("scheduler", type=str, required=False)
preload_argv = click.argument(
    "preload_argv", nargs=-1, type=click.UNPROCESSED, callback=validate_preload_argv
)
scheduler_file = click.option(
    "--scheduler-file",
    type=str,
    default=None,
    help="""Filename to JSON encoded scheduler information. To be used in conjunction
    with the equivalent ``dask scheduler`` option.""",
)
tls_ca_file = click.option(
    "--tls-ca-file",
    type=pem_file_option_type,
    default=None,
    help="""CA certificate(s) file for TLS (in PEM format). Can be a string (like
    ``"path/to/certs"``), or ``None`` for no certificate(s).""",
)
tls_cert = click.option(
    "--tls-cert",
    type=pem_file_option_type,
    default=None,
    help="""Certificate file for TLS (in PEM format). Can be a string (like
    ``"path/to/certs"``), or ``None`` for no certificate(s).""",
)
tls_key = click.option(
    "--tls-key",
    type=pem_file_option_type,
    default=None,
    help="""Private key file for TLS (in PEM format). Can be a string (like
    ``"path/to/certs"``), or ``None`` for no private key.""",
)


@click.group
def cuda():
    """Subcommands to launch or query distributed workers with GPUs."""


@cuda.command(name="worker", context_settings=dict(ignore_unknown_options=True))
@scheduler
@preload_argv
@click.option(
    "--host",
    type=str,
    default=None,
    help="""IP address of serving host; should be visible to the scheduler and other
    workers. Can be a string (like ``"127.0.0.1"``) or ``None`` to fall back on the
    address of the interface specified by ``--interface`` or the default interface.""",
)
@click.option(
    "--nthreads",
    type=int,
    default=1,
    show_default=True,
    help="Number of threads to be used for each Dask worker process.",
)
@click.option(
    "--name",
    type=str,
    default=None,
    help="""A unique name for the worker. Can be a string (like ``"worker-1"``) or
    ``None`` for a nameless worker.""",
)
@click.option(
    "--memory-limit",
    default="auto",
    show_default=True,
    help="""Size of the host LRU cache, which is used to determine when the worker
    starts spilling to disk (not available if JIT-Unspill is enabled). Can be an
    integer (bytes), float (fraction of total system memory), string (like ``"5GB"``
    or ``"5000M"``), or ``"auto"``, 0, or ``None`` for no memory management.""",
)
@click.option(
    "--device-memory-limit",
    default="0.8",
    show_default=True,
    help="""Size of the CUDA device LRU cache, which is used to determine when the
    worker starts spilling to host memory. Can be an integer (bytes), float (fraction of
    total device memory), string (like ``"5GB"`` or ``"5000M"``), or ``"auto"`` or 0 to
    disable spilling to host (i.e. allow full device memory usage).""",
)
@click.option(
    "--rmm-pool-size",
    default=None,
    help="""RMM pool size to initialize each worker with. Can be an integer (bytes),
    float (fraction of total device memory), string (like ``"5GB"`` or ``"5000M"``), or
    ``None`` to disable RMM pools.

    .. note::
        This size is a per-worker configuration, and not cluster-wide.""",
)
@click.option(
    "--rmm-maximum-pool-size",
    default=None,
    help="""When ``--rmm-pool-size`` is specified, this argument indicates the maximum
    pool size.  Can be an integer (bytes), float (fraction of total device memory),
    string (like ``"5GB"`` or ``"5000M"``) or ``None``. By default, the total available
    memory on the GPU is used. ``rmm_pool_size`` must be specified to use RMM pool and
    to set the maximum pool size.

    .. note::
        This size is a per-worker configuration, and not cluster-wide.""",
)
@click.option(
    "--rmm-managed-memory/--no-rmm-managed-memory",
    default=False,
    show_default=True,
    help="""Initialize each worker with RMM and set it to use managed memory. If
    disabled, RMM may still be used by specifying ``--rmm-pool-size``.

    .. warning::
        Managed memory is currently incompatible with NVLink. Trying to enable both will
        result in failure.""",
)
@click.option(
    "--rmm-async/--no-rmm-async",
    default=False,
    show_default=True,
    help="""Initialize each worker with RMM and set it to use RMM's asynchronous
    allocator. See ``rmm.mr.CudaAsyncMemoryResource`` for more info.

    .. warning::
        The asynchronous allocator requires CUDA Toolkit 11.2 or newer. It is also
        incompatible with RMM pools and managed memory, trying to enable both will
        result in failure.""",
)
@click.option(
    "--rmm-release-threshold",
    default=None,
    help="""When ``rmm.async`` is ``True`` and the pool size grows beyond this value, unused
    memory held by the pool will be released at the next synchronization point. Can be
    an integer (bytes), float (fraction of total device memory), string (like ``"5GB"``
    or ``"5000M"``) or ``None``. By default, this feature is disabled.

    .. note::
        This size is a per-worker configuration, and not cluster-wide.""",
)
@click.option(
    "--rmm-log-directory",
    default=None,
    help="""Directory to write per-worker RMM log files to. The client and scheduler are
    not logged here. Can be a string (like ``"/path/to/logs/"``) or ``None`` to disable
    logging.

    .. note::
        Logging will only be enabled if ``--rmm-pool-size`` or ``--rmm-managed-memory``
        are specified.""",
)
@click.option(
    "--rmm-track-allocations/--no-rmm-track-allocations",
    default=False,
    show_default=True,
    help="""Track memory allocations made by RMM. If ``True``, wraps the memory
    resource of each worker with a ``rmm.mr.TrackingResourceAdaptor`` that
    allows querying the amount of memory allocated by RMM.""",
)
@click.option(
    "--pid-file",
    type=str,
    default="",
    help="File to write the process PID.",
)
@click.option(
    "--resources",
    type=str,
    default="",
    help="""Resources for task constraints like ``"GPU=2 MEM=10e9"``.""",
)
@click.option(
    "--dashboard/--no-dashboard",
    "dashboard",
    default=True,
    show_default=True,
    required=False,
    help="Launch the dashboard.",
)
@click.option(
    "--dashboard-address",
    type=str,
    default=":0",
    show_default=True,
    help="Relative address to serve the dashboard (if enabled).",
)
@click.option(
    "--local-directory",
    default=None,
    type=str,
    help="""Path on local machine to store temporary files. Can be a string (like
    ``"path/to/files"``) or ``None`` to fall back on the value of
    ``dask.temporary-directory`` in the local Dask configuration, using the current
    working directory if this is not set.""",
)
@click.option(
    "--shared-filesystem/--no-shared-filesystem",
    default=None,
    type=bool,
    help="""If `--shared-filesystem` is specified, inform JIT-Unspill that
    `local_directory` is a shared filesystem available for all workers, whereas
    `--no-shared-filesystem` informs it may not assume it's a shared filesystem.
    If neither is specified, JIT-Unspill will decide based on the Dask config value
    specified by `"jit-unspill-shared-fs"`.
    Notice, a shared filesystem must support the `os.link()` operation.""",
)
@scheduler_file
@click.option(
    "--protocol", type=str, default=None, help="Protocol like tcp, tls, or ucx"
)
@click.option(
    "--interface",
    type=str,
    default=None,
    help="""External interface used to connect to the scheduler. Usually an ethernet
    interface is used for connection, and not an InfiniBand interface (if one is
    available). Can be a string (like ``"eth0"`` for NVLink or ``"ib0"`` for
    InfiniBand) or ``None`` to fall back on the default interface.""",
)
@click.option(
    "--preload",
    type=str,
    multiple=True,
    is_eager=True,
    help="""Module that should be loaded by each worker process like ``"foo.bar"`` or
    ``"/path/to/foo.py"``.""",
)
@click.option(
    "--death-timeout",
    type=str,
    default=None,
    help="Seconds to wait for a scheduler before closing",
)
@click.option(
    "--dashboard-prefix",
    type=str,
    default=None,
    help="""Prefix for the dashboard. Can be a string (like ...) or ``None`` for no
    prefix.""",
)
@tls_ca_file
@tls_cert
@tls_key
@click.option(
    "--enable-tcp-over-ucx/--disable-tcp-over-ucx",
    default=None,
    show_default=True,
    help="""Set environment variables to enable TCP over UCX, even if InfiniBand and
    NVLink are not supported or disabled.""",
)
@click.option(
    "--enable-infiniband/--disable-infiniband",
    default=None,
    show_default=True,
    help="""Set environment variables to enable UCX over InfiniBand, implies
    ``--enable-tcp-over-ucx`` when enabled.""",
)
@click.option(
    "--enable-nvlink/--disable-nvlink",
    default=None,
    show_default=True,
    help="""Set environment variables to enable UCX over NVLink, implies
    ``--enable-tcp-over-ucx`` when enabled.""",
)
@click.option(
    "--enable-rdmacm/--disable-rdmacm",
    default=None,
    show_default=True,
    help="""Set environment variables to enable UCX RDMA connection manager support,
    requires ``--enable-infiniband``.""",
)
@click.option(
    "--enable-jit-unspill/--disable-jit-unspill",
    default=None,
    help="""Enable just-in-time unspilling. Can be a boolean or ``None`` to fall back on
    the value of ``dask.jit-unspill`` in the local Dask configuration, disabling
    unspilling if this is not set.

    .. note::
        This is experimental and doesn't support memory spilling to disk. See
        ``proxy_object.ProxyObject`` and ``proxify_host_file.ProxifyHostFile`` for more
        info.""",
)
@click.option(
    "--worker-class",
    default=None,
    help="""Use a different class than Distributed's default (``distributed.Worker``)
    to spawn ``distributed.Nanny``.""",
)
@click.option(
    "--pre-import",
    default=None,
    help="""Pre-import libraries as a Worker plugin to prevent long import times
    bleeding through later Dask operations. Should be a list of comma-separated names,
    such as "cudf,rmm".""",
)
@click.option(
    "--multiprocessing-method",
    default="spawn",
    type=click.Choice(["spawn", "fork", "forkserver"]),
    help="""Method used to start new processes with multiprocessing""",
)
def worker(
    scheduler,
    host,
    nthreads,
    name,
    memory_limit,
    device_memory_limit,
    rmm_pool_size,
    rmm_maximum_pool_size,
    rmm_managed_memory,
    rmm_async,
    rmm_release_threshold,
    rmm_log_directory,
    rmm_track_allocations,
    pid_file,
    resources,
    dashboard,
    dashboard_address,
    local_directory,
    shared_filesystem,
    scheduler_file,
    interface,
    preload,
    dashboard_prefix,
    tls_ca_file,
    tls_cert,
    tls_key,
    enable_tcp_over_ucx,
    enable_infiniband,
    enable_nvlink,
    enable_rdmacm,
    enable_jit_unspill,
    worker_class,
    pre_import,
    multiprocessing_method,
    **kwargs,
):
    """Launch a distributed worker with GPUs attached to an existing scheduler.

    A scheduler can be specified either through a URI passed through the ``SCHEDULER``
    argument or a scheduler file passed through the ``--scheduler-file`` option.

    See
    https://docs.rapids.ai/api/dask-cuda/stable/quickstart.html#dask-cuda-worker
    for info.
    """
    if multiprocessing_method == "forkserver":
        import multiprocessing.forkserver as f

        f.ensure_running()
    if tls_ca_file and tls_cert and tls_key:
        security = Security(
            tls_ca_file=tls_ca_file,
            tls_worker_cert=tls_cert,
            tls_worker_key=tls_key,
        )
    else:
        security = None

    if isinstance(scheduler, str) and scheduler.startswith("-"):
        raise ValueError(
            "The scheduler address can't start with '-'. Please check "
            "your command line arguments, you probably attempted to use "
            "unsupported one. Scheduler address: %s" % scheduler
        )

    if worker_class is not None:
        worker_class = import_term(worker_class)

    with dask_config.set(
        {"distributed.worker.multiprocessing-method": multiprocessing_method}
    ):
        worker = CUDAWorker(
            scheduler,
            host,
            nthreads,
            name,
            memory_limit,
            device_memory_limit,
            rmm_pool_size,
            rmm_maximum_pool_size,
            rmm_managed_memory,
            rmm_async,
            rmm_release_threshold,
            rmm_log_directory,
            rmm_track_allocations,
            pid_file,
            resources,
            dashboard,
            dashboard_address,
            local_directory,
            shared_filesystem,
            scheduler_file,
            interface,
            preload,
            dashboard_prefix,
            security,
            enable_tcp_over_ucx,
            enable_infiniband,
            enable_nvlink,
            enable_rdmacm,
            enable_jit_unspill,
            worker_class,
            pre_import,
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


@cuda.command(name="config", context_settings=dict(ignore_unknown_options=True))
@scheduler
@preload_argv
@scheduler_file
@tls_ca_file
@tls_cert
@tls_key
def config(
    scheduler,
    scheduler_file,
    tls_ca_file,
    tls_cert,
    tls_key,
    **kwargs,
):
    """Query an existing GPU cluster's configuration.

    A cluster can be specified either through a URI passed through the ``SCHEDULER``
    argument or a scheduler file passed through the ``--scheduler-file`` option.
    """
    if (
        scheduler is None
        and scheduler_file is None
        and dask_config.get("scheduler-address", None) is None
    ):
        raise ValueError(
            "No scheduler specified. A scheduler can be specified by "
            "passing an address through the SCHEDULER argument or "
            "'dask.scheduler-address' config option, or by passing the "
            "location of a scheduler file through the --scheduler-file "
            "option"
        )

    if isinstance(scheduler, str) and scheduler.startswith("-"):
        raise ValueError(
            "The scheduler address can't start with '-'. Please check "
            "your command line arguments, you probably attempted to use "
            "unsupported one. Scheduler address: %s" % scheduler
        )

    if tls_ca_file and tls_cert and tls_key:
        security = Security(
            tls_ca_file=tls_ca_file,
            tls_worker_cert=tls_cert,
            tls_worker_key=tls_key,
        )
    else:
        security = None

    if scheduler_file is not None:
        client = Client(scheduler_file=scheduler_file, security=security)
    else:
        client = Client(scheduler, security=security)
    print_cluster_config(client)


if __name__ == "__main__":
    worker()
