from __future__ import absolute_import, division, print_function

import logging

import click

from distributed import Client
from distributed.preloading import validate_preload_argv
from distributed.security import Security

from ..utils import print_cluster_config

logger = logging.getLogger(__name__)


pem_file_option_type = click.Path(exists=True, resolve_path=True)


@click.command(context_settings=dict(ignore_unknown_options=True))
@click.argument("scheduler", type=str, required=False)
@click.argument(
    "preload_argv", nargs=-1, type=click.UNPROCESSED, callback=validate_preload_argv
)
@click.option(
    "--scheduler-file",
    type=str,
    default=None,
    help="""Filename to JSON encoded scheduler information. To be used in conjunction
    with the equivalent ``dask-scheduler`` option.""",
)
@click.option(
    "--get-cluster-configuration",
    "get_cluster_conf",
    default=False,
    is_flag=True,
    required=False,
    show_default=True,
    help="""Print a table of the current cluster configuration""",
)
@click.option(
    "--tls-ca-file",
    type=pem_file_option_type,
    default=None,
    help="""CA certificate(s) file for TLS (in PEM format). Can be a string (like
    ``"path/to/certs"``), or ``None`` for no certificate(s).""",
)
@click.option(
    "--tls-cert",
    type=pem_file_option_type,
    default=None,
    help="""Certificate file for TLS (in PEM format). Can be a string (like
    ``"path/to/certs"``), or ``None`` for no certificate(s).""",
)
@click.option(
    "--tls-key",
    type=pem_file_option_type,
    default=None,
    help="""Private key file for TLS (in PEM format). Can be a string (like
    ``"path/to/certs"``), or ``None`` for no private key.""",
)
def main(
    scheduler,
    scheduler_file,
    get_cluster_conf,
    tls_ca_file,
    tls_cert,
    tls_key,
    **kwargs,
):
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

    if get_cluster_conf:
        if scheduler_file is not None:
            client = Client(scheduler_file=scheduler_file, security=security)
        else:
            client = Client(scheduler, security=security)
        print_cluster_config(client)


if __name__ == "__main__":
    main()
