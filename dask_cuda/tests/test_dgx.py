import multiprocessing as mp
import os
from enum import Enum, auto

import numpy
import pytest

from dask import array as da
from distributed import Client

from dask_cuda import LocalCUDACluster
from dask_cuda.initialize import initialize

mp = mp.get_context("spawn")  # type: ignore
psutil = pytest.importorskip("psutil")


class DGXVersion(Enum):
    DGX_1 = auto()
    DGX_2 = auto()
    DGX_A100 = auto()


def _get_dgx_name():
    product_name_file = "/sys/class/dmi/id/product_name"
    dgx_release_file = "/etc/dgx-release"

    # We verify `product_name_file` to check it's a DGX, and check
    # if `dgx_release_file` exists to confirm it's not a container.
    if not os.path.isfile(product_name_file) or not os.path.isfile(dgx_release_file):
        return None

    with open(product_name_file) as f:
        for line in f:
            return line


def _get_dgx_version():
    dgx_name = _get_dgx_name()

    if dgx_name is None:
        return None
    elif "DGX-1" in dgx_name:
        return DGXVersion.DGX_1
    elif "DGX-2" in dgx_name:
        return DGXVersion.DGX_2
    elif "DGXA100" in dgx_name:
        return DGXVersion.DGX_A100


if _get_dgx_version() is None:
    pytest.skip("Not a DGX server", allow_module_level=True)


# Notice, all of the following tests is executed in a new process such
# that UCX options of the different tests doesn't conflict.
# Furthermore, all tests do some computation to trigger initialization
# of UCX before retrieving the current config.


def _test_default():
    with LocalCUDACluster() as cluster:
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
    ucp = pytest.importorskip("ucp")

    with LocalCUDACluster(enable_tcp_over_ucx=True) as cluster:
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

            assert all(client.run(check_ucx_options).values())


def test_tcp_over_ucx():
    ucp = pytest.importorskip("ucp")  # NOQA: F841

    p = mp.Process(target=_test_tcp_over_ucx)
    p.start()
    p.join()
    assert not p.exitcode


def _test_tcp_only():
    with LocalCUDACluster(protocol="tcp") as cluster:
        with Client(cluster):
            res = da.from_array(numpy.arange(10000), chunks=(1000,))
            res = res.sum().compute()
            assert res == 49995000


def test_tcp_only():
    p = mp.Process(target=_test_tcp_only)
    p.start()
    p.join()
    assert not p.exitcode


def _test_ucx_infiniband_nvlink(enable_infiniband, enable_nvlink, enable_rdmacm):
    cupy = pytest.importorskip("cupy")
    ucp = pytest.importorskip("ucp")

    if enable_infiniband is None and enable_nvlink is None and enable_rdmacm is None:
        enable_tcp_over_ucx = None
        cm_tls = ["all"]
        cm_tls_priority = ["rdmacm", "tcp", "sockcm"]
    else:
        enable_tcp_over_ucx = True

        cm_tls = ["tcp"]
        if enable_rdmacm is True:
            cm_tls_priority = ["rdmacm"]
        else:
            cm_tls_priority = ["tcp"]

    initialize(
        enable_tcp_over_ucx=enable_tcp_over_ucx,
        enable_infiniband=enable_infiniband,
        enable_nvlink=enable_nvlink,
        enable_rdmacm=enable_rdmacm,
    )

    with LocalCUDACluster(
        interface="ib0",
        enable_tcp_over_ucx=enable_tcp_over_ucx,
        enable_infiniband=enable_infiniband,
        enable_nvlink=enable_nvlink,
        enable_rdmacm=enable_rdmacm,
        rmm_pool_size="1 GiB",
    ) as cluster:
        with Client(cluster) as client:
            res = da.from_array(cupy.arange(10000), chunks=(1000,), asarray=False)
            res = res.sum().compute()
            assert res == 49995000

            def check_ucx_options():
                conf = ucp.get_config()
                assert "TLS" in conf
                assert all(t in conf["TLS"] for t in cm_tls)
                assert all(p in conf["SOCKADDR_TLS_PRIORITY"] for p in cm_tls_priority)
                if cm_tls != ["all"]:
                    assert "tcp" in conf["TLS"]
                    assert "cuda_copy" in conf["TLS"]
                    if enable_nvlink:
                        assert "cuda_ipc" in conf["TLS"]
                    if enable_infiniband:
                        assert "rc" in conf["TLS"]
                return True

            assert all(client.run(check_ucx_options).values())


@pytest.mark.parametrize(
    "params",
    [
        {"enable_infiniband": False, "enable_nvlink": False, "enable_rdmacm": False},
        {"enable_infiniband": True, "enable_nvlink": True, "enable_rdmacm": False},
        {"enable_infiniband": True, "enable_nvlink": False, "enable_rdmacm": True},
        {"enable_infiniband": True, "enable_nvlink": True, "enable_rdmacm": True},
        {"enable_infiniband": None, "enable_nvlink": None, "enable_rdmacm": None},
    ],
)
@pytest.mark.skipif(
    _get_dgx_version() == DGXVersion.DGX_A100,
    reason="Automatic InfiniBand device detection Unsupported for %s" % _get_dgx_name(),
)
def test_ucx_infiniband_nvlink(params):
    ucp = pytest.importorskip("ucp")  # NOQA: F841

    if params["enable_infiniband"]:
        if not any([at.startswith("rc") for at in ucp.get_active_transports()]):
            pytest.skip("No support available for 'rc' transport in UCX")

    p = mp.Process(
        target=_test_ucx_infiniband_nvlink,
        args=(
            params["enable_infiniband"],
            params["enable_nvlink"],
            params["enable_rdmacm"],
        ),
    )
    p.start()
    p.join()

    # Starting a new cluster on the same pytest process after an rdmacm cluster
    # has been used may cause UCX-Py to complain about being already initialized.
    if params["enable_rdmacm"] is True:
        ucp.reset()

    assert not p.exitcode
