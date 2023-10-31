import multiprocessing as mp

import numpy
import psutil
import pytest

from dask import array as da
from distributed import Client
from distributed.deploy.local import LocalCluster

from dask_cuda.initialize import initialize
from dask_cuda.utils import get_ucx_config
from dask_cuda.utils_test import IncreasedCloseTimeoutNanny

mp = mp.get_context("spawn")  # type: ignore

# Notice, all of the following tests is executed in a new process such
# that UCX options of the different tests doesn't conflict.
# Furthermore, all tests do some computation to trigger initialization
# of UCX before retrieving the current config.


def _test_initialize_ucx_tcp(protocol):
    if protocol == "ucx":
        ucp = pytest.importorskip("ucp")
    elif protocol == "ucxx":
        ucp = pytest.importorskip("ucxx")

    kwargs = {"enable_tcp_over_ucx": True}
    initialize(protocol=protocol, **kwargs)
    with LocalCluster(
        protocol=protocol,
        dashboard_address=None,
        n_workers=1,
        threads_per_worker=1,
        processes=True,
        worker_class=IncreasedCloseTimeoutNanny,
        config={"distributed.comm.ucx": get_ucx_config(**kwargs)},
    ) as cluster:
        with Client(cluster) as client:
            res = da.from_array(numpy.arange(10000), chunks=(1000,))
            res = res.sum().compute()
            assert res == 49995000

            def check_ucx_options():
                conf = ucp.get_config()
                assert "TLS" in conf
                assert "tcp" in conf["TLS"]
                assert "cuda_copy" in conf["TLS"]
                assert "tcp" in conf["SOCKADDR_TLS_PRIORITY"]
                return True

            assert client.run_on_scheduler(check_ucx_options) is True
            assert all(client.run(check_ucx_options).values())


@pytest.mark.parametrize("protocol", ["ucx", "ucxx"])
def test_initialize_ucx_tcp(protocol):
    if protocol == "ucx":
        pytest.importorskip("ucp")
    elif protocol == "ucxx":
        pytest.importorskip("ucxx")

    p = mp.Process(target=_test_initialize_ucx_tcp, args=(protocol,))
    p.start()
    p.join()
    assert not p.exitcode


def _test_initialize_ucx_nvlink(protocol):
    if protocol == "ucx":
        ucp = pytest.importorskip("ucp")
    elif protocol == "ucxx":
        ucp = pytest.importorskip("ucxx")

    kwargs = {"enable_nvlink": True}
    initialize(protocol=protocol, **kwargs)
    with LocalCluster(
        protocol=protocol,
        dashboard_address=None,
        n_workers=1,
        threads_per_worker=1,
        processes=True,
        worker_class=IncreasedCloseTimeoutNanny,
        config={"distributed.comm.ucx": get_ucx_config(**kwargs)},
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
                assert "cuda_copy" in conf["TLS"]
                assert "tcp" in conf["SOCKADDR_TLS_PRIORITY"]
                return True

            assert client.run_on_scheduler(check_ucx_options) is True
            assert all(client.run(check_ucx_options).values())


@pytest.mark.parametrize("protocol", ["ucx", "ucxx"])
def test_initialize_ucx_nvlink(protocol):
    if protocol == "ucx":
        pytest.importorskip("ucp")
    elif protocol == "ucxx":
        pytest.importorskip("ucxx")

    p = mp.Process(target=_test_initialize_ucx_nvlink, args=(protocol,))
    p.start()
    p.join()
    assert not p.exitcode


def _test_initialize_ucx_infiniband(protocol):
    if protocol == "ucx":
        ucp = pytest.importorskip("ucp")
    elif protocol == "ucxx":
        ucp = pytest.importorskip("ucxx")

    kwargs = {"enable_infiniband": True}
    initialize(protocol=protocol, **kwargs)
    with LocalCluster(
        protocol=protocol,
        dashboard_address=None,
        n_workers=1,
        threads_per_worker=1,
        processes=True,
        worker_class=IncreasedCloseTimeoutNanny,
        config={"distributed.comm.ucx": get_ucx_config(**kwargs)},
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
                assert "cuda_copy" in conf["TLS"]
                assert "tcp" in conf["SOCKADDR_TLS_PRIORITY"]
                return True

            assert client.run_on_scheduler(check_ucx_options) is True
            assert all(client.run(check_ucx_options).values())


@pytest.mark.skipif(
    "ib0" not in psutil.net_if_addrs(), reason="Infiniband interface ib0 not found"
)
@pytest.mark.parametrize("protocol", ["ucx", "ucxx"])
def test_initialize_ucx_infiniband(protocol):
    if protocol == "ucx":
        pytest.importorskip("ucp")
    elif protocol == "ucxx":
        pytest.importorskip("ucxx")

    p = mp.Process(target=_test_initialize_ucx_infiniband, args=(protocol,))
    p.start()
    p.join()
    assert not p.exitcode


def _test_initialize_ucx_all(protocol):
    if protocol == "ucx":
        ucp = pytest.importorskip("ucp")
    elif protocol == "ucxx":
        ucp = pytest.importorskip("ucxx")

    initialize(protocol=protocol)
    with LocalCluster(
        protocol=protocol,
        dashboard_address=None,
        n_workers=1,
        threads_per_worker=1,
        processes=True,
        worker_class=IncreasedCloseTimeoutNanny,
        config={"distributed.comm.ucx": get_ucx_config()},
    ) as cluster:
        with Client(cluster) as client:
            res = da.from_array(numpy.arange(10000), chunks=(1000,))
            res = res.sum().compute()
            assert res == 49995000

            def check_ucx_options():
                conf = ucp.get_config()
                assert "TLS" in conf
                assert conf["TLS"] == "all"
                assert all(
                    [
                        p in conf["SOCKADDR_TLS_PRIORITY"]
                        for p in ["rdmacm", "tcp", "sockcm"]
                    ]
                )
                return True

            assert client.run_on_scheduler(check_ucx_options) is True
            assert all(client.run(check_ucx_options).values())


@pytest.mark.parametrize("protocol", ["ucx", "ucxx"])
def test_initialize_ucx_all(protocol):
    if protocol == "ucx":
        pytest.importorskip("ucp")
    elif protocol == "ucxx":
        pytest.importorskip("ucxx")

    p = mp.Process(target=_test_initialize_ucx_all, args=(protocol,))
    p.start()
    p.join()
    assert not p.exitcode
