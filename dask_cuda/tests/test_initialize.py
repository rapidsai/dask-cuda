import os
import psutil
import multiprocessing as mp
import pytest
import numpy
import dask.array as da

from dask_cuda.initialize import initialize
from distributed.deploy.local import LocalCluster
from distributed import Client

mp = mp.get_context("spawn")
ucp = pytest.importorskip("ucp")

# Notice, all of the following tests is executed in a new process such
# that UCX options of the different tests doesn't conflict.
# Furthermore, all tests do some computation to trigger initialization
# of UCX before retrieving the current config.


def _test_initialize_ucx_tcp():
    initialize(enable_tcp_over_ucx=True)
    with LocalCluster(
        protocol="ucx",
        dashboard_address=None,
        n_workers=1,
        threads_per_worker=1,
        processes=True,
    ) as cluster:
        with Client(cluster) as client:
            res = da.from_array(numpy.arange(10000), chunks=(1000,))
            res = res.sum().compute()
            assert res == 49995000

            def check_ucx_options():
                conf = ucp.get_config()
                assert "TLS" in conf
                assert "tcp" in conf["TLS"]
                assert "sockcm" in conf["TLS"]
                assert "cuda_copy" in conf["TLS"]
                assert "sockcm" in conf["SOCKADDR_TLS_PRIORITY"]
                return True

            assert all(client.run(check_ucx_options).values())


def test_initialize_ucx_tcp():
    p = mp.Process(target=_test_initialize_ucx_tcp)
    p.start()
    p.join()
    assert not p.exitcode


def _test_initialize_ucx_nvlink():
    initialize(enable_nvlink=True)
    with LocalCluster(
        protocol="ucx",
        dashboard_address=None,
        n_workers=1,
        threads_per_worker=1,
        processes=True,
    ) as cluster:
        with Client(cluster) as client:
            res = da.from_array(numpy.arange(10000), chunks=(1000,))
            res = res.sum().compute()
            assert res == 49995000

            def check_ucx_options():
                conf = ucp.get_config()
                assert "TLS" in conf
                assert "cuda_ipc" in conf["TLS"]
                assert "tcp" in conf["TLS"]
                assert "sockcm" in conf["TLS"]
                assert "cuda_copy" in conf["TLS"]
                assert "sockcm" in conf["SOCKADDR_TLS_PRIORITY"]
                return True

            assert all(client.run(check_ucx_options).values())


def test_initialize_ucx_nvlink():
    p = mp.Process(target=_test_initialize_ucx_nvlink)
    p.start()
    p.join()
    assert not p.exitcode


def _test_initialize_ucx_infiniband():
    initialize(enable_infiniband=True, net_devices="ib0")
    with LocalCluster(
        protocol="ucx",
        dashboard_address=None,
        n_workers=1,
        threads_per_worker=1,
        processes=True,
    ) as cluster:
        with Client(cluster) as client:
            res = da.from_array(numpy.arange(10000), chunks=(1000,))
            res = res.sum().compute()
            assert res == 49995000

            def check_ucx_options():
                conf = ucp.get_config()
                assert "TLS" in conf
                assert "rc" in conf["TLS"]
                assert "tcp" in conf["TLS"]
                assert "sockcm" in conf["TLS"]
                assert "cuda_copy" in conf["TLS"]
                assert "sockcm" in conf["SOCKADDR_TLS_PRIORITY"]
                assert conf["NET_DEVICES"] == "ib0"
                return True

            assert all(client.run(check_ucx_options).values())


@pytest.mark.skipif(
    "ib0" not in psutil.net_if_addrs(), reason="Infiniband interface ib0 not found"
)
def test_initialize_ucx_infiniband():
    p = mp.Process(target=_test_initialize_ucx_infiniband)
    p.start()
    p.join()
    assert not p.exitcode
