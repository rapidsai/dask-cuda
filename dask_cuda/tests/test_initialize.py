import pytest

import os

from dask_cuda.initialize import initialize


def test_initialize_cuda_context():
    initialize(create_cuda_context=True)

def test_initialize_ucx_tcp():
    ucp = pytest.importorskip("ucp")

    initialize(enable_tcp_over_ucx=True)

    conf = ucp.get_config()
    env = os.environ

    assert "TLS" in conf
    assert "UCX_TLS" in env

    assert "tcp" in conf["TLS"] and "tcp" in env["UCX_TLS"]
    assert "sockcm" in conf["TLS"] and "sockcm" in env["UCX_TLS"]
    assert "cuda_copy" in conf["TLS"] and "cuda_copy" in env["UCX_TLS"]

    assert "sockcm" in conf["SOCKADDR_TLS_PRIORITY"] and "sockcm" in env["UCX_SOCKADDR_TLS_PRIORITY"]


def test_initialize_ucx_infiniband():
    ucp = pytest.importorskip("ucp")

    initialize(enable_infiniband=True, net_devices="ib0")

    conf = ucp.get_config()
    env = os.environ

    assert "TLS" in conf
    assert "UCX_TLS" in env

    assert "rc" in conf["TLS"] and "rc" in env["UCX_TLS"]
    assert "tcp" in conf["TLS"] and "tcp" in env["UCX_TLS"]
    assert "sockcm" in conf["TLS"] and "sockcm" in env["UCX_TLS"]
    assert "cuda_copy" in conf["TLS"] and "cuda_copy" in env["UCX_TLS"]

    assert "sockcm" in conf["SOCKADDR_TLS_PRIORITY"] and "sockcm" in env["UCX_SOCKADDR_TLS_PRIORITY"]

    assert conf["NET_DEVICES"] == "ib0" and env["UCX_NET_DEVICES"] == "ib0"


def test_initialize_ucx_nvlink():
    ucp = pytest.importorskip("ucp")

    initialize(enable_nvlink=True)

    conf = ucp.get_config()
    env = os.environ

    assert "TLS" in conf
    assert "UCX_TLS" in env

    assert "cuda_ipc" in conf["TLS"] and "cuda_ipc" in env["UCX_TLS"]
    assert "tcp" in conf["TLS"] and "tcp" in env["UCX_TLS"]
    assert "sockcm" in conf["TLS"] and "sockcm" in env["UCX_TLS"]
    assert "cuda_copy" in conf["TLS"] and "cuda_copy" in env["UCX_TLS"]

    assert "sockcm" in conf["SOCKADDR_TLS_PRIORITY"] and "sockcm" in env["UCX_SOCKADDR_TLS_PRIORITY"]
