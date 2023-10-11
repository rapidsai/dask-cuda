from __future__ import absolute_import, division, print_function

import os
import pkgutil
import subprocess
import sys
from unittest.mock import patch

import pytest

from distributed import Client, wait
from distributed.system import MEMORY_LIMIT
from distributed.utils_test import cleanup, loop, loop_in_thread, popen  # noqa: F401

from dask_cuda.utils import (
    get_cluster_configuration,
    get_device_total_memory,
    get_gpu_count_mig,
    get_gpu_uuid_from_index,
    get_n_gpus,
    wait_workers,
)


@patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,3,7,8"})
def test_cuda_visible_devices_and_memory_limit_and_nthreads(loop):  # noqa: F811
    nthreads = 4
    with popen(["dask", "scheduler", "--port", "9359", "--no-dashboard"]):
        with popen(
            [
                "dask",
                "cuda",
                "worker",
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
                    assert w["memory_limit"] == MEMORY_LIMIT // len(workers)

                assert len(expected) == 0


def test_rmm_pool(loop):  # noqa: F811
    rmm = pytest.importorskip("rmm")
    with popen(["dask", "scheduler", "--port", "9369", "--no-dashboard"]):
        with popen(
            [
                "dask",
                "cuda",
                "worker",
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
    with popen(["dask", "scheduler", "--port", "9369", "--no-dashboard"]):
        with popen(
            [
                "dask",
                "cuda",
                "worker",
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


def test_rmm_async(loop):  # noqa: F811
    rmm = pytest.importorskip("rmm")

    driver_version = rmm._cuda.gpu.driverGetVersion()
    runtime_version = rmm._cuda.gpu.runtimeGetVersion()
    if driver_version < 11020 or runtime_version < 11020:
        pytest.skip("cudaMallocAsync not supported")

    with popen(["dask", "scheduler", "--port", "9369", "--no-dashboard"]):
        with popen(
            [
                "dask",
                "cuda",
                "worker",
                "127.0.0.1:9369",
                "--host",
                "127.0.0.1",
                "--rmm-async",
                "--rmm-pool-size",
                "2 GB",
                "--rmm-release-threshold",
                "3 GB",
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

                ret = get_cluster_configuration(client)
                wait(ret)
                assert ret["[plugin] RMMSetup"]["initial_pool_size"] == 2000000000
                assert ret["[plugin] RMMSetup"]["release_threshold"] == 3000000000


def test_rmm_async_with_maximum_pool_size(loop):  # noqa: F811
    rmm = pytest.importorskip("rmm")

    driver_version = rmm._cuda.gpu.driverGetVersion()
    runtime_version = rmm._cuda.gpu.runtimeGetVersion()
    if driver_version < 11020 or runtime_version < 11020:
        pytest.skip("cudaMallocAsync not supported")

    with popen(["dask", "scheduler", "--port", "9369", "--no-dashboard"]):
        with popen(
            [
                "dask",
                "cuda",
                "worker",
                "127.0.0.1:9369",
                "--host",
                "127.0.0.1",
                "--rmm-async",
                "--rmm-pool-size",
                "2 GB",
                "--rmm-release-threshold",
                "3 GB",
                "--rmm-maximum-pool-size",
                "4 GB",
                "--no-dashboard",
            ]
        ):
            with Client("127.0.0.1:9369", loop=loop) as client:
                assert wait_workers(client, n_gpus=get_n_gpus())

                memory_resource_types = client.run(
                    lambda: (
                        rmm.mr.get_current_device_resource_type(),
                        type(rmm.mr.get_current_device_resource().get_upstream()),
                    )
                )
                for v in memory_resource_types.values():
                    memory_resource_type, upstream_memory_resource_type = v
                    assert memory_resource_type is rmm.mr.LimitingResourceAdaptor
                    assert (
                        upstream_memory_resource_type is rmm.mr.CudaAsyncMemoryResource
                    )

                ret = get_cluster_configuration(client)
                wait(ret)
                assert ret["[plugin] RMMSetup"]["initial_pool_size"] == 2000000000
                assert ret["[plugin] RMMSetup"]["release_threshold"] == 3000000000
                assert ret["[plugin] RMMSetup"]["maximum_pool_size"] == 4000000000


def test_rmm_logging(loop):  # noqa: F811
    rmm = pytest.importorskip("rmm")
    with popen(["dask", "scheduler", "--port", "9369", "--no-dashboard"]):
        with popen(
            [
                "dask",
                "cuda",
                "worker",
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


@patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"})
def test_dashboard_address(loop):  # noqa: F811
    with popen(["dask", "scheduler", "--port", "9369", "--no-dashboard"]):
        with popen(
            [
                "dask",
                "cuda",
                "worker",
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
    ret = subprocess.run(
        ["dask", "cuda", "worker", "--my-argument"], capture_output=True
    )
    assert ret.returncode != 0
    assert b"Scheduler address: --my-argument" in ret.stderr


@patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"})
def test_pre_import(loop):  # noqa: F811
    module = None

    # Pick a module that isn't currently loaded
    for m in pkgutil.iter_modules():
        if m.ispkg and m.name not in sys.modules.keys():
            module = m.name
            break

    if module is None:
        pytest.skip("No module found that isn't already loaded")

    with popen(["dask", "scheduler", "--port", "9369", "--no-dashboard"]):
        with popen(
            [
                "dask",
                "cuda",
                "worker",
                "127.0.0.1:9369",
                "--pre-import",
                module,
            ]
        ):
            with Client("127.0.0.1:9369", loop=loop) as client:
                assert wait_workers(client, n_gpus=get_n_gpus())

                imported = client.run(lambda: module in sys.modules)
                assert all(imported)


@pytest.mark.timeout(20)
@patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"})
def test_pre_import_not_found():
    with popen(["dask", "scheduler", "--port", "9369", "--no-dashboard"]):
        ret = subprocess.run(
            ["dask", "cuda", "worker", "127.0.0.1:9369", "--pre-import", "my_module"],
            capture_output=True,
        )
        assert ret.returncode != 0
        assert b"ModuleNotFoundError: No module named 'my_module'" in ret.stderr


def test_cuda_mig_visible_devices_and_memory_limit_and_nthreads(loop):  # noqa: F811
    uuids = get_gpu_count_mig(return_uuids=True)[1]
    # test only with some MIG Instances assuming the test bed
    # does not have a huge number of mig instances
    if len(uuids) > 0:
        cuda_visible_devices = ",".join([i.decode("utf-8") for i in uuids])
    else:
        pytest.skip("No MIG devices found")

    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": cuda_visible_devices}):
        nthreads = len(cuda_visible_devices)
        with popen(["dask", "scheduler", "--port", "9359", "--no-dashboard"]):
            with popen(
                [
                    "dask",
                    "cuda",
                    "worker",
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
                        assert set(
                            bytes(v.split(",")[i], "utf-8") for v in result.values()
                        ) == set(uuids)


def test_cuda_visible_devices_uuid(loop):  # noqa: F811
    gpu_uuid = get_gpu_uuid_from_index(0)

    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": gpu_uuid}):
        with popen(["dask", "scheduler", "--port", "9359", "--no-dashboard"]):
            with popen(
                [
                    "dask",
                    "cuda",
                    "worker",
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


def test_rmm_track_allocations(loop):  # noqa: F811
    rmm = pytest.importorskip("rmm")
    with popen(["dask", "scheduler", "--port", "9369", "--no-dashboard"]):
        with popen(
            [
                "dask",
                "cuda",
                "worker",
                "127.0.0.1:9369",
                "--host",
                "127.0.0.1",
                "--rmm-pool-size",
                "2 GB",
                "--no-dashboard",
                "--rmm-track-allocations",
            ]
        ):
            with Client("127.0.0.1:9369", loop=loop) as client:
                assert wait_workers(client, n_gpus=get_n_gpus())

                memory_resource_type = client.run(
                    rmm.mr.get_current_device_resource_type
                )
                for v in memory_resource_type.values():
                    assert v is rmm.mr.TrackingResourceAdaptor

                memory_resource_upstream_type = client.run(
                    lambda: type(rmm.mr.get_current_device_resource().upstream_mr)
                )
                for v in memory_resource_upstream_type.values():
                    assert v is rmm.mr.PoolMemoryResource


@patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"})
def test_get_cluster_configuration(loop):  # noqa: F811
    pytest.importorskip("rmm")
    with popen(["dask", "scheduler", "--port", "9369", "--no-dashboard"]):
        with popen(
            [
                "dask",
                "cuda",
                "worker",
                "127.0.0.1:9369",
                "--host",
                "127.0.0.1",
                "--device-memory-limit",
                "30 B",
                "--rmm-pool-size",
                "2 GB",
                "--rmm-maximum-pool-size",
                "3 GB",
                "--no-dashboard",
                "--rmm-track-allocations",
            ]
        ):
            with Client("127.0.0.1:9369", loop=loop) as client:
                assert wait_workers(client, n_gpus=get_n_gpus())

                ret = get_cluster_configuration(client)
                wait(ret)
                assert ret["[plugin] RMMSetup"]["initial_pool_size"] == 2000000000
                assert ret["[plugin] RMMSetup"]["maximum_pool_size"] == 3000000000
                assert ret["jit-unspill"] is False
                assert ret["device-memory-limit"] == 30


@patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"})
def test_worker_fraction_limits(loop):  # noqa: F811
    pytest.importorskip("rmm")
    with popen(["dask", "scheduler", "--port", "9369", "--no-dashboard"]):
        with popen(
            [
                "dask",
                "cuda",
                "worker",
                "127.0.0.1:9369",
                "--host",
                "127.0.0.1",
                "--device-memory-limit",
                "0.1",
                "--rmm-pool-size",
                "0.2",
                "--rmm-maximum-pool-size",
                "0.3",
                "--no-dashboard",
                "--rmm-track-allocations",
            ]
        ):
            with Client("127.0.0.1:9369", loop=loop) as client:
                assert wait_workers(client, n_gpus=get_n_gpus())

                device_total_memory = client.run(get_device_total_memory)
                wait(device_total_memory)
                _, device_total_memory = device_total_memory.popitem()

                ret = get_cluster_configuration(client)
                wait(ret)

                assert ret["device-memory-limit"] == int(device_total_memory * 0.1)
                assert (
                    ret["[plugin] RMMSetup"]["initial_pool_size"]
                    == (device_total_memory * 0.2) // 256 * 256
                )
                assert (
                    ret["[plugin] RMMSetup"]["maximum_pool_size"]
                    == (device_total_memory * 0.3) // 256 * 256
                )


@patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"})
def test_worker_timeout():
    ret = subprocess.run(
        [
            "dask",
            "cuda",
            "worker",
            "192.168.1.100:7777",
            "--death-timeout",
            "1",
        ],
        text=True,
        encoding="utf8",
        capture_output=True,
    )

    assert "closing nanny at" in ret.stderr.lower()

    # Depending on the environment, the error raised may be different
    try:
        assert "reason: failure-to-start-" in ret.stderr.lower()
        assert "timeouterror" in ret.stderr.lower()
    except AssertionError:
        assert "reason: nanny-close" in ret.stderr.lower()

    assert ret.returncode == 0
