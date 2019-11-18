import os
import multiprocessing as mp
import pytest
import numpy
import dask.array as da

from dask_cuda import DGX
from distributed import Client

mp = mp.get_context("spawn")
ucp = pytest.importorskip("ucp")
psutil = pytest.importorskip("psutil")

if "ib0" not in psutil.net_if_addrs():
    pytest.skip("Infiniband interface ib0 not found", allow_module_level=True)


# Notice, all of the following tests is executed in a new process such
# that UCX options of the different tests doesn't conflict.
# Furthermore, all tests do some computation to trigger initialization
# of UCX before retrieving the current config.


def _test_default():
    with DGX() as cluster:
        with Client(cluster):
            res = da.from_array(numpy.arange(10000), chunks=(1000,))
            res = res.sum().compute()
            assert res == 49995000


def test_default():
    p = mp.Process(target=_test_default)
    p.start()
    p.join()
    assert not p.exitcode


def _test_tcp_over_ucx():
    with DGX(enable_tcp_over_ucx=True) as cluster:
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


def test_tcp_over_ucx():
    p = mp.Process(target=_test_tcp_over_ucx)
    p.start()
    p.join()
    assert not p.exitcode


def _test_tcp_only():
    with DGX(protocol="tcp") as cluster:
        with Client(cluster):
            res = da.from_array(numpy.arange(10000), chunks=(1000,))
            res = res.sum().compute()
            assert res == 49995000


def test_tcp_only():
    p = mp.Process(target=_test_tcp_only)
    p.start()
    p.join()
    assert not p.exitcode


def _test_ucx_infiniband_nvlink(enable_infiniband, enable_nvlink):
    cupy = pytest.importorskip("cupy")
    with DGX(
        enable_tcp_over_ucx=True,
        enable_infiniband=enable_infiniband,
        enable_nvlink=enable_nvlink,
    ) as cluster:
        with Client(cluster) as client:
            res = da.from_array(cupy.arange(10000), chunks=(1000,), asarray=False)
            res = res.sum().compute()
            assert res == 49995000

            def check_ucx_options():
                conf = ucp.get_config()
                assert "TLS" in conf
                assert "tcp" in conf["TLS"]
                assert "sockcm" in conf["TLS"]
                assert "cuda_copy" in conf["TLS"]
                assert "sockcm" in conf["SOCKADDR_TLS_PRIORITY"]
                if enable_nvlink:
                    assert "cuda_ipc" in conf["TLS"]
                if enable_infiniband:
                    assert "rc" in conf["TLS"]
                return True

            assert all(client.run(check_ucx_options).values())


@pytest.mark.parametrize(
    "params",
    [
        {"enable_infiniband": False, "enable_nvlink": False},
        {"enable_infiniband": True, "enable_nvlink": True},
    ],
)
def test_ucx_infiniband_nvlink(params):
    p = mp.Process(
        target=_test_ucx_infiniband_nvlink,
        args=(params["enable_infiniband"], params["enable_nvlink"]),
    )
    p.start()
    p.join()
    assert not p.exitcode
