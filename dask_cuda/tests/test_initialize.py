import os
import psutil
import pytest
import dask.array as da

from dask_cuda.initialize import initialize
from distributed.deploy.local import LocalCluster
from distributed import Client
from multiprocessing import Process

ucp = pytest.importorskip("ucp")
cupy = pytest.importorskip("cupy")


@pytest.mark.xfail(reason="https://github.com/rapidsai/dask-cuda/pull/168")
def test_initialize_ucx_tcp():
    def test():
        initialize(enable_tcp_over_ucx=True)
        with LocalCluster(
            protocol="ucx",
            dashboard_address=None,
            n_workers=1,
            threads_per_worker=1,
            processes=True,
        ) as cluster:
            with Client(cluster):
                res = da.from_array(cupy.arange(10000), chunks=(1000,), asarray=False)
                res = res.sum().compute()
                assert res == 49995000

                conf = ucp.get_config()
                assert "TLS" in conf
                assert "tcp" in conf["TLS"]
                assert "sockcm" in conf["TLS"]
                assert "cuda_copy" in conf["TLS"]
                assert "sockcm" in conf["SOCKADDR_TLS_PRIORITY"]

    # We run in process to avoid using the UCX options from the other tests
    p = Process(target=test)
    p.start()
    p.join()
    assert not p.exitcode


@pytest.mark.xfail(reason="https://github.com/rapidsai/dask-cuda/pull/168")
def test_initialize_ucx_nvlink():
    def test():
        initialize(enable_nvlink=True)
        with LocalCluster(
            protocol="ucx",
            dashboard_address=None,
            n_workers=1,
            threads_per_worker=1,
            processes=True,
        ) as cluster:
            with Client(cluster):
                res = da.from_array(cupy.arange(10000), chunks=(1000,), asarray=False)
                res = res.sum().compute()
                assert res == 49995000
                conf = ucp.get_config()
                assert "TLS" in conf
                assert "cuda_ipc" in conf["TLS"]
                assert "tcp" in conf["TLS"]
                assert "sockcm" in conf["TLS"]
                assert "cuda_copy" in conf["TLS"]
                assert "sockcm" in conf["SOCKADDR_TLS_PRIORITY"]

    # We run in process to avoid using the UCX options from the other tests
    p = Process(target=test)
    p.start()
    p.join()
    assert not p.exitcode


@pytest.mark.xfail(reason="https://github.com/rapidsai/dask-cuda/pull/168")
@pytest.mark.skipif(
    "ib0" not in psutil.net_if_addrs(), reason="Infiniband interface ib0 not found"
)
def test_initialize_ucx_infiniband():
    initialize(enable_infiniband=True, net_devices="ib0")

    def test():
        with LocalCluster(
            protocol="ucx",
            dashboard_address=None,
            n_workers=1,
            threads_per_worker=1,
            processes=True,
        ) as cluster:
            with Client(cluster):
                res = da.from_array(cupy.arange(10000), chunks=(1000,), asarray=False)
                res = res.sum().compute()
                assert res == 49995000
                conf = ucp.get_config()
                assert "TLS" in conf
                assert "rc" in conf["TLS"]
                assert "tcp" in conf["TLS"]
                assert "sockcm" in conf["TLS"]
                assert "cuda_copy" in conf["TLS"]
                assert "sockcm" in conf["SOCKADDR_TLS_PRIORITY"]
                assert conf["NET_DEVICES"] == "ib0"

    # We run in process to avoid using the UCX options from the other tests
    p = Process(target=test)
    p.start()
    p.join()
    assert not p.exitcode
