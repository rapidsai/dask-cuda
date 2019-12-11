import multiprocessing as mp
import os

import dask.array as da
from dask_cuda import DGX
from distributed import Client

import numpy
import pytest

mp = mp.get_context("spawn")
ucp = pytest.importorskip("ucp")
psutil = pytest.importorskip("psutil")


def _check_dgx_version():
    dgx_server = None

    if not os.path.isfile("/etc/dgx-release"):
        return dgx_server

    for line in open("/etc/dgx-release"):
        if line.startswith("DGX_PLATFORM"):
            if "DGX Server for DGX-1" in line:
                dgx_server = 1
            elif "DGX Server for DGX-2" in line:
                dgx_server = 2
            break

    return dgx_server


if _check_dgx_version() is None:
    pytest.skip("Not a DGX server", allow_module_level=True)


# Notice, all of the following tests is executed in a new process such
# that UCX options of the different tests doesn't conflict.
# Furthermore, all tests do some computation to trigger initialization
# of UCX before retrieving the current config.


def _test_default():
    with pytest.warns(DeprecationWarning):
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
    with pytest.warns(DeprecationWarning):
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
    with pytest.warns(DeprecationWarning):
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

    if _check_dgx_version() == 1:
        net_devices = [
            "mlx5_0:1,ib0",
            "mlx5_0:1,ib0",
            "mlx5_1:1,ib1",
            "mlx5_1:1,ib1",
            "mlx5_2:1,ib2",
            "mlx5_2:1,ib2",
            "mlx5_3:1,ib3",
            "mlx5_3:1,ib3",
        ]
    elif _check_dgx_version() == 2:
        net_devices = [
            "mlx5_0:1,ib0",
            "mlx5_0:1,ib0",
            "mlx5_1:1,ib1",
            "mlx5_1:1,ib1",
            "mlx5_2:1,ib2",
            "mlx5_2:1,ib2",
            "mlx5_3:1,ib3",
            "mlx5_3:1,ib3",
            "mlx5_6:1,ib4",
            "mlx5_6:1,ib4",
            "mlx5_7:1,ib5",
            "mlx5_7:1,ib5",
            "mlx5_8:1,ib6",
            "mlx5_8:1,ib6",
            "mlx5_9:1,ib7",
            "mlx5_9:1,ib7",
        ]

    ucx_net_devices = "auto" if enable_infiniband else None

    with pytest.warns(DeprecationWarning):
        with DGX(
            enable_tcp_over_ucx=True,
            enable_infiniband=enable_infiniband,
            enable_nvlink=enable_nvlink,
            ucx_net_devices=ucx_net_devices,
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

                if enable_infiniband:
                    assert all(
                        [
                            cluster.worker_spec[k]["options"]["env"]["UCX_NET_DEVICES"]
                            == net_devices[k]
                            for k in cluster.worker_spec.keys()
                        ]
                    )

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
