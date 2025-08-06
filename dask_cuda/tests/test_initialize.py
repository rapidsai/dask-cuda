# SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp
import sys

import numpy
import psutil
import pytest

from dask import array as da
from distributed import Client
from distributed.deploy.local import LocalCluster

from dask_cuda.initialize import initialize
from dask_cuda.utils import get_ucx_config
from dask_cuda.utils_test import IncreasedCloseTimeoutNanny, get_ucx_implementation

mp = mp.get_context("spawn")  # type: ignore

# Notice, all of the following tests is executed in a new process such
# that UCX options of the different tests doesn't conflict.
# Furthermore, all tests do some computation to trigger initialization
# of UCX before retrieving the current config.


def _test_initialize_ucx_tcp(protocol):
    ucp = get_ucx_implementation(protocol)

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


@pytest.mark.parametrize("protocol", ["ucx", "ucx-old"])
def test_initialize_ucx_tcp(protocol):
    get_ucx_implementation(protocol)

    p = mp.Process(target=_test_initialize_ucx_tcp, args=(protocol,))
    p.start()
    p.join()
    assert not p.exitcode


def _test_initialize_ucx_nvlink(protocol):
    ucp = get_ucx_implementation(protocol)

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


@pytest.mark.parametrize("protocol", ["ucx", "ucx-old"])
def test_initialize_ucx_nvlink(protocol):
    get_ucx_implementation(protocol)

    p = mp.Process(target=_test_initialize_ucx_nvlink, args=(protocol,))
    p.start()
    p.join()
    assert not p.exitcode


def _test_initialize_ucx_infiniband(protocol):
    ucp = get_ucx_implementation(protocol)

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
@pytest.mark.parametrize("protocol", ["ucx", "ucx-old"])
def test_initialize_ucx_infiniband(protocol):
    get_ucx_implementation(protocol)

    p = mp.Process(target=_test_initialize_ucx_infiniband, args=(protocol,))
    p.start()
    p.join()
    assert not p.exitcode


def _test_initialize_ucx_all(protocol):
    ucp = get_ucx_implementation(protocol)

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


@pytest.mark.parametrize("protocol", ["ucx", "ucx-old"])
def test_initialize_ucx_all(protocol):
    get_ucx_implementation(protocol)

    p = mp.Process(target=_test_initialize_ucx_all, args=(protocol,))
    p.start()
    p.join()
    assert not p.exitcode


def _test_dask_cuda_import():
    # Check that importing `dask_cuda` does NOT
    # require `dask.dataframe` or `dask.array`.

    # Patch sys.modules so that `dask.dataframe`
    # and `dask.array` cannot be found.
    with pytest.MonkeyPatch.context() as monkeypatch:
        for k in list(sys.modules):
            if k.startswith("dask.dataframe") or k.startswith("dask.array"):
                monkeypatch.setitem(sys.modules, k, None)
        monkeypatch.delitem(sys.modules, "dask_cuda")

        # Check that top-level imports still succeed.
        import dask_cuda  # noqa: F401
        from dask_cuda import CUDAWorker  # noqa: F401
        from dask_cuda import LocalCUDACluster

        with LocalCUDACluster(
            dashboard_address=None,
            n_workers=1,
            threads_per_worker=1,
            processes=True,
            worker_class=IncreasedCloseTimeoutNanny,
        ) as cluster:
            with Client(cluster) as client:
                client.run(lambda *args: None)


def test_dask_cuda_import():
    p = mp.Process(target=_test_dask_cuda_import)
    p.start()
    p.join()
    assert not p.exitcode
