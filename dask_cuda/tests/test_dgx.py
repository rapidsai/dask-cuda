import multiprocessing as mp
import os
import subprocess
from enum import Enum, auto
from time import sleep

import numpy
import pytest
from tornado.ioloop import IOLoop

from dask import array as da
from distributed import Client
from distributed.utils import get_ip_interface

from dask_cuda import LocalCUDACluster
from dask_cuda.initialize import initialize
from dask_cuda.utils import _ucx_110, wait_workers

mp = mp.get_context("spawn")
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

    for line in open(product_name_file):
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


def _get_dgx_net_devices():
    if _get_dgx_version() == DGXVersion.DGX_1:
        return [
            "mlx5_0:1,ib0",
            "mlx5_0:1,ib0",
            "mlx5_1:1,ib1",
            "mlx5_1:1,ib1",
            "mlx5_2:1,ib2",
            "mlx5_2:1,ib2",
            "mlx5_3:1,ib3",
            "mlx5_3:1,ib3",
        ]
    elif _get_dgx_version() == DGXVersion.DGX_2:
        return [
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
    else:
        return None


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
                if _ucx_110:
                    assert "tcp" in conf["SOCKADDR_TLS_PRIORITY"]
                else:
                    assert "sockcm" in conf["TLS"]
                    assert "sockcm" in conf["SOCKADDR_TLS_PRIORITY"]
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

    net_devices = _get_dgx_net_devices()
    openfabrics_devices = [d.split(",")[0] for d in net_devices]

    ucx_net_devices = None
    if enable_infiniband and not _ucx_110:
        ucx_net_devices = "auto"

    if _ucx_110 is True:
        cm_tls = ["tcp"]
        if enable_rdmacm is True:
            cm_tls_priority = "rdmacm"
        else:
            cm_tls_priority = "tcp"
    else:
        cm_tls = ["tcp"]
        if enable_rdmacm is True:
            cm_tls.append(["rdmacm"])
            cm_tls_priority = "rdmacm"
        else:
            cm_tls.append(["sockcm"])
            cm_tls_priority = "sockcm"

    initialize(
        enable_tcp_over_ucx=True,
        enable_infiniband=enable_infiniband,
        enable_nvlink=enable_nvlink,
        enable_rdmacm=enable_rdmacm,
    )

    with LocalCUDACluster(
        interface="ib0",
        enable_tcp_over_ucx=True,
        enable_infiniband=enable_infiniband,
        enable_nvlink=enable_nvlink,
        enable_rdmacm=enable_rdmacm,
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
                assert "cuda_copy" in conf["TLS"]
                assert all(t in conf["TLS"] for t in cm_tls)
                assert cm_tls_priority in conf["SOCKADDR_TLS_PRIORITY"]
                if enable_nvlink:
                    assert "cuda_ipc" in conf["TLS"]
                if enable_infiniband:
                    assert "rc" in conf["TLS"]
                return True

            if ucx_net_devices == "auto":
                assert all(
                    [
                        cluster.worker_spec[k]["options"]["env"]["UCX_NET_DEVICES"]
                        == openfabrics_devices[k].split(",")[0]
                        for k in cluster.worker_spec.keys()
                    ]
                )

            assert all(client.run(check_ucx_options).values())


@pytest.mark.parametrize(
    "params",
    [
        {"enable_infiniband": False, "enable_nvlink": False, "enable_rdmacm": False},
        {"enable_infiniband": True, "enable_nvlink": True, "enable_rdmacm": False},
        {"enable_infiniband": True, "enable_nvlink": False, "enable_rdmacm": True},
        {"enable_infiniband": True, "enable_nvlink": True, "enable_rdmacm": True},
    ],
)
@pytest.mark.skipif(
    _get_dgx_version() == DGXVersion.DGX_A100,
    reason="Automatic InfiniBand device detection Unsupported for %s" % _get_dgx_name(),
)
def test_ucx_infiniband_nvlink(params):
    ucp = pytest.importorskip("ucp")  # NOQA: F841

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
    assert not p.exitcode


def _test_dask_cuda_worker_ucx_net_devices(enable_rdmacm):
    loop = IOLoop.current()
    ucp = pytest.importorskip("ucp")

    cm_protocol = "rdmacm" if enable_rdmacm else "sockcm"
    net_devices = _get_dgx_net_devices()
    openfabrics_devices = [d.split(",")[0] for d in net_devices]

    sched_addr = "127.0.0.1"

    # Enable proper variables for scheduler
    sched_env = os.environ.copy()
    sched_env["DASK_UCX__INFINIBAND"] = "True"
    sched_env["DASK_UCX__TCP"] = "True"
    sched_env["DASK_UCX__CUDA_COPY"] = "True"
    sched_env["DASK_UCX__NET_DEVICES"] = openfabrics_devices[0]

    if enable_rdmacm:
        sched_env["DASK_UCX__RDMACM"] = "True"
        sched_addr = get_ip_interface("ib0")

    sched_url = "ucx://" + sched_addr + ":9379"

    # Enable proper variables for workers
    worker_ucx_opts = [
        "--enable-infiniband",
        "--net-devices",
        "auto",
    ]
    if enable_rdmacm:
        worker_ucx_opts.append("--enable-rdmacm")

    # Enable proper variables for client
    initialize(
        enable_tcp_over_ucx=True,
        enable_infiniband=True,
        enable_rdmacm=enable_rdmacm,
        net_devices=openfabrics_devices[0],
    )

    with subprocess.Popen(
        [
            "dask-scheduler",
            "--protocol",
            "ucx",
            "--host",
            sched_addr,
            "--port",
            "9379",
            "--no-dashboard",
        ],
        env=sched_env,
    ) as sched_proc:
        # Scheduler with UCX will take a few seconds to fully start
        sleep(5)

        with subprocess.Popen(
            ["dask-cuda-worker", sched_url, "--no-dashboard",] + worker_ucx_opts
        ) as worker_proc:
            with Client(sched_url, loop=loop) as client:

                def _timeout_callback():
                    # We must ensure processes are terminated to avoid hangs
                    # if a timeout occurs
                    worker_proc.kill()
                    sched_proc.kill()

                assert wait_workers(client, timeout_callback=_timeout_callback)

                workers_tls = client.run(lambda: ucp.get_config()["TLS"])
                workers_tls_priority = client.run(
                    lambda: ucp.get_config()["SOCKADDR_TLS_PRIORITY"]
                )
                for tls, tls_priority in zip(
                    workers_tls.values(), workers_tls_priority.values()
                ):
                    assert cm_protocol in tls
                    assert cm_protocol in tls_priority
                worker_net_devices = client.run(lambda: ucp.get_config()["NET_DEVICES"])
                cuda_visible_devices = client.run(
                    lambda: os.environ["CUDA_VISIBLE_DEVICES"]
                )

                for i, v in enumerate(
                    zip(worker_net_devices.values(), cuda_visible_devices.values())
                ):
                    net_dev = v[0]
                    dev_idx = int(v[1].split(",")[0])
                    assert net_dev == openfabrics_devices[dev_idx]

            # A dask-worker with UCX protocol will not close until some work
            # is dispatched, therefore we kill the worker and scheduler to
            # ensure timely closing.
            worker_proc.kill()
            sched_proc.kill()


@pytest.mark.parametrize("enable_rdmacm", [False, True])
@pytest.mark.skipif(
    _get_dgx_version() == DGXVersion.DGX_A100,
    reason="Automatic InfiniBand device detection Unsupported for %s" % _get_dgx_name(),
)
def test_dask_cuda_worker_ucx_net_devices(enable_rdmacm):
    ucp = pytest.importorskip("ucp")  # NOQA: F841

    if ucp.get_ucx_version() >= (1, 10, 0):
        pytest.skip("UCX 1.10 and higher should rely on default UCX_NET_DEVICES")

    p = mp.Process(
        target=_test_dask_cuda_worker_ucx_net_devices, args=(enable_rdmacm,),
    )
    p.start()
    p.join()
    assert not p.exitcode
