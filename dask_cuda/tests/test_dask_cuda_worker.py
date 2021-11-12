from __future__ import absolute_import, division, print_function

import os
import subprocess

import pytest

from distributed import Client, wait
from distributed.system import MEMORY_LIMIT
from distributed.utils_test import loop  # noqa: F401
from distributed.utils_test import popen

import rmm

from dask_cuda.utils import (
    get_gpu_count_mig,
    get_gpu_uuid_from_index,
    get_n_gpus,
    wait_workers,
)

_driver_version = rmm._cuda.gpu.driverGetVersion()
_runtime_version = rmm._cuda.gpu.runtimeGetVersion()


def test_cuda_visible_devices_and_memory_limit_and_nthreads(loop):  # noqa: F811
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,3,7,8"
    nthreads = 4
    try:
        with popen(["dask-scheduler", "--port", "9359", "--no-dashboard"]):
            with popen(
                [
                    "dask-cuda-worker",
                    "127.0.0.1:9359",
                    "--host",
                    "127.0.0.1",
                    "--device-memory-limit",
                    "1 MB",
                    "--nthreads",
                    str(nthreads),
                    "--no-dashboard",
                    "--worker-class",
                    "dask_cuda.utils.MockWorker",
                ]
            ):
                with Client("127.0.0.1:9359", loop=loop) as client:
                    assert wait_workers(client, n_gpus=4)

                    def get_visible_devices():
                        return os.environ["CUDA_VISIBLE_DEVICES"]

                    # verify 4 workers with the 4 expected CUDA_VISIBLE_DEVICES
                    result = client.run(get_visible_devices)
                    expected = {"0,3,7,8": 1, "3,7,8,0": 1, "7,8,0,3": 1, "8,0,3,7": 1}
                    for v in result.values():
                        del expected[v]

                    workers = client.scheduler_info()["workers"]
                    for w in workers.values():
                        assert (
                            w["memory_limit"] == MEMORY_LIMIT // len(workers) * nthreads
                        )

                    assert len(expected) == 0
    finally:
        del os.environ["CUDA_VISIBLE_DEVICES"]


def test_rmm_pool(loop):  # noqa: F811
    rmm = pytest.importorskip("rmm")
    with popen(["dask-scheduler", "--port", "9369", "--no-dashboard"]):
        with popen(
            [
                "dask-cuda-worker",
                "127.0.0.1:9369",
                "--host",
                "127.0.0.1",
                "--rmm-pool-size",
                "2 GB",
                "--no-dashboard",
            ]
        ):
            with Client("127.0.0.1:9369", loop=loop) as client:
                assert wait_workers(client, n_gpus=get_n_gpus())

                memory_resource_type = client.run(
                    rmm.mr.get_current_device_resource_type
                )
                for v in memory_resource_type.values():
                    assert v is rmm.mr.PoolMemoryResource


def test_rmm_managed(loop):  # noqa: F811
    rmm = pytest.importorskip("rmm")
    with popen(["dask-scheduler", "--port", "9369", "--no-dashboard"]):
        with popen(
            [
                "dask-cuda-worker",
                "127.0.0.1:9369",
                "--host",
                "127.0.0.1",
                "--rmm-managed-memory",
                "--no-dashboard",
            ]
        ):
            with Client("127.0.0.1:9369", loop=loop) as client:
                assert wait_workers(client, n_gpus=get_n_gpus())

                memory_resource_type = client.run(
                    rmm.mr.get_current_device_resource_type
                )
                for v in memory_resource_type.values():
                    assert v is rmm.mr.ManagedMemoryResource


@pytest.mark.skipif(
    _driver_version < 11020 or _runtime_version < 11020,
    reason="cudaMallocAsync not supported",
)
def test_rmm_async(loop):  # noqa: F811
    rmm = pytest.importorskip("rmm")
    with popen(["dask-scheduler", "--port", "9369", "--no-dashboard"]):
        with popen(
            [
                "dask-cuda-worker",
                "127.0.0.1:9369",
                "--host",
                "127.0.0.1",
                "--rmm-async",
                "--no-dashboard",
            ]
        ):
            with Client("127.0.0.1:9369", loop=loop) as client:
                assert wait_workers(client, n_gpus=get_n_gpus())

                memory_resource_type = client.run(
                    rmm.mr.get_current_device_resource_type
                )
                for v in memory_resource_type.values():
                    assert v is rmm.mr.CudaAsyncMemoryResource


def test_rmm_logging(loop):  # noqa: F811
    rmm = pytest.importorskip("rmm")
    with popen(["dask-scheduler", "--port", "9369", "--no-dashboard"]):
        with popen(
            [
                "dask-cuda-worker",
                "127.0.0.1:9369",
                "--host",
                "127.0.0.1",
                "--rmm-pool-size",
                "2 GB",
                "--rmm-log-directory",
                ".",
                "--no-dashboard",
            ]
        ):
            with Client("127.0.0.1:9369", loop=loop) as client:
                assert wait_workers(client, n_gpus=get_n_gpus())

                memory_resource_type = client.run(
                    rmm.mr.get_current_device_resource_type
                )
                for v in memory_resource_type.values():
                    assert v is rmm.mr.LoggingResourceAdaptor


def test_dashboard_address(loop):  # noqa: F811
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    with popen(["dask-scheduler", "--port", "9369", "--no-dashboard"]):
        with popen(
            [
                "dask-cuda-worker",
                "127.0.0.1:9369",
                "--dashboard-address",
                "127.0.0.1:9370",
            ]
        ):
            with Client("127.0.0.1:9369", loop=loop) as client:
                assert wait_workers(client, n_gpus=get_n_gpus())

                dashboard_addresses = client.run(
                    lambda dask_worker: dask_worker._dashboard_address
                )
                for v in dashboard_addresses.values():
                    assert v == "127.0.0.1:9370"


def test_unknown_argument():
    ret = subprocess.run(["dask-cuda-worker", "--my-argument"], capture_output=True)
    assert ret.returncode != 0
    assert b"Scheduler address: --my-argument" in ret.stderr


def test_cuda_mig_visible_devices_and_memory_limit_and_nthreads(loop):  # noqa: F811
    init_nvmlstatus = os.environ.get("DASK_DISTRIBUTED__DIAGNOSTICS__NVML")
    try:
        os.environ["DASK_DISTRIBUTED__DIAGNOSTICS__NVML"] = "False"
        uuids = get_gpu_count_mig(return_uuids=True)[1]
        # test only with some MIG Instances assuming the test bed
        # does not have a huge number of mig instances
        if len(uuids) > 0:
            uuids = [i.decode("utf-8") for i in uuids]
        else:
            pytest.skip("No MIG devices found")
        CUDA_VISIBLE_DEVICES = ",".join(uuids)
        os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
        nthreads = len(CUDA_VISIBLE_DEVICES)
        with popen(["dask-scheduler", "--port", "9359", "--no-dashboard"]):
            with popen(
                [
                    "dask-cuda-worker",
                    "127.0.0.1:9359",
                    "--host",
                    "127.0.0.1",
                    "--nthreads",
                    str(nthreads),
                    "--no-dashboard",
                    "--worker-class",
                    "dask_cuda.utils.MockWorker",
                ]
            ):
                with Client("127.0.0.1:9359", loop=loop) as client:
                    assert wait_workers(client, n_gpus=len(uuids))
                    # Check to see if all workers are up and
                    # CUDA_VISIBLE_DEVICES cycles properly

                    def get_visible_devices():
                        return os.environ["CUDA_VISIBLE_DEVICES"]

                    result = client.run(get_visible_devices)
                    wait(result)
                    assert all(len(v.split(",")) == len(uuids) for v in result.values())
                    for i in range(len(uuids)):
                        assert set(v.split(",")[i] for v in result.values()) == set(
                            uuids
                        )
    finally:
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]
        if init_nvmlstatus:
            os.environ["DASK_DISTRIBUTED__DIAGNOSTICS__NVML"] = init_nvmlstatus


def test_cuda_visible_devices_uuid(loop):  # noqa: F811
    gpu_uuid = get_gpu_uuid_from_index(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    try:
        with popen(["dask-scheduler", "--port", "9359", "--no-dashboard"]):
            with popen(
                [
                    "dask-cuda-worker",
                    "127.0.0.1:9359",
                    "--host",
                    "127.0.0.1",
                    "--no-dashboard",
                    "--worker-class",
                    "dask_cuda.utils.MockWorker",
                ]
            ):
                with Client("127.0.0.1:9359", loop=loop) as client:
                    assert wait_workers(client, n_gpus=1)

                    result = client.run(lambda: os.environ["CUDA_VISIBLE_DEVICES"])
                    assert list(result.values())[0] == gpu_uuid
    finally:
        del os.environ["CUDA_VISIBLE_DEVICES"]
