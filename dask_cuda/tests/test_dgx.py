import multiprocessing as mp
import os
import subprocess

import dask.array as da
from dask_cuda import LocalCUDACluster
from distributed import Client

import numpy
import pytest

from time import sleep
from dask_cuda.utils import get_gpu_count
from dask_cuda.initialize import initialize
from distributed.metrics import time
from distributed.utils import get_ip_interface
from distributed.utils_test import popen
from distributed.worker import get_worker
from tornado.ioloop import IOLoop

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


def _get_dgx_net_devices():
    if _check_dgx_version() == 1:
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
    elif _check_dgx_version() == 2:
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


if _check_dgx_version() is None:
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
    with LocalCUDACluster(enable_tcp_over_ucx=True) as cluster:
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


def _test_ucx_infiniband_nvlink(enable_infiniband, enable_nvlink):
    cupy = pytest.importorskip("cupy")

    net_devices = _get_dgx_net_devices()

    ucx_net_devices = "auto" if enable_infiniband else None

    with LocalCUDACluster(
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


def _test_dask_cuda_worker_ucx_net_devices(enable_rdmacm):
    loop = IOLoop.current()

    cm_protocol = "rdmacm" if enable_rdmacm else "sockcm"
    net_devices = _get_dgx_net_devices()

    sched_addr = "127.0.0.1"

    # Enable proper variables for scheduler
    sched_env = os.environ.copy()
    sched_env["DASK_UCX__INFINIBAND"] = "True"
    sched_env["DASK_UCX__TCP"] = "True"

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
        enable_tcp_over_ucx=True, enable_infiniband=True, enable_rdmacm=enable_rdmacm
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

                start = time()
                while True:
                    if len(client.scheduler_info()["workers"]) == get_gpu_count():
                        break
                    else:
                        assert time() - start < 10
                        sleep(0.1)

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
                    assert net_dev == net_devices[dev_idx]

            # A dask-worker with UCX protocol will not close until some work
            # is dispatched, therefore we kill the worker and scheduler to
            # ensure timely closing.
            worker_proc.kill()
            sched_proc.kill()


@pytest.mark.parametrize("enable_rdmacm", [False, True])
def test_dask_cuda_worker_ucx_net_devices(enable_rdmacm):
    p = mp.Process(
        target=_test_dask_cuda_worker_ucx_net_devices, args=(enable_rdmacm,),
    )
    p.start()
    p.join()
    assert not p.exitcode
