# SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import os
import pkgutil
import sys
from unittest.mock import patch

import pytest

from dask.distributed import Client
from distributed.system import MEMORY_LIMIT
from distributed.utils_test import gen_test, raises_with_cause

from dask_cuda import CUDAWorker, LocalCUDACluster, utils
from dask_cuda.initialize import initialize
from dask_cuda.utils import (
    get_cluster_configuration,
    get_device_total_memory,
    get_gpu_count_mig,
    get_gpu_uuid,
    has_device_memory_resource,
    print_cluster_config,
)
from dask_cuda.utils_test import MockWorker


@gen_test(timeout=20)
async def test_local_cuda_cluster():
    async with LocalCUDACluster(
        scheduler_port=0,
        asynchronous=True,
    ) as cluster:
        async with Client(cluster, asynchronous=True) as client:
            assert len(cluster.workers) == utils.get_n_gpus()

            # CUDA_VISIBLE_DEVICES cycles properly
            def get_visible_devices():
                return os.environ["CUDA_VISIBLE_DEVICES"]

            result = await client.run(get_visible_devices)

            assert all(len(v.split(",")) == utils.get_n_gpus() for v in result.values())
            for i in range(utils.get_n_gpus()):
                assert {int(v.split(",")[i]) for v in result.values()} == set(
                    range(utils.get_n_gpus())
                )

            # Use full memory, checked with some buffer to ignore rounding difference
            full_mem = sum(
                w.memory_manager.memory_limit for w in cluster.workers.values()
            )
            assert full_mem >= MEMORY_LIMIT - 1024 and full_mem < MEMORY_LIMIT + 1024

            for w, devices in result.items():
                ident = devices.split(",")[0]
                assert int(ident) == cluster.scheduler.workers[w].name

            with pytest.raises(ValueError):
                cluster.scale(1000)


# Notice, this test might raise errors when the number of available GPUs is less
# than 8 but as long as the test passes the errors can be ignored.
@pytest.mark.filterwarnings("ignore:Cannot get CPU affinity")
@patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,3,6,8"})
@gen_test(timeout=20)
async def test_with_subset_of_cuda_visible_devices():
    async with LocalCUDACluster(
        scheduler_port=0,
        asynchronous=True,
        worker_class=MockWorker,
        data=dict,
    ) as cluster:
        async with Client(cluster, asynchronous=True) as client:
            assert len(cluster.workers) == 4

            # CUDA_VISIBLE_DEVICES cycles properly
            def get_visible_devices():
                return os.environ["CUDA_VISIBLE_DEVICES"]

            result = await client.run(get_visible_devices)

            assert all(len(v.split(",")) == 4 for v in result.values())
            for i in range(4):
                assert {int(v.split(",")[i]) for v in result.values()} == {
                    0,
                    3,
                    6,
                    8,
                }


@gen_test(timeout=20)
async def test_ucx_protocol():
    pytest.importorskip("distributed_ucxx")

    async with LocalCUDACluster(
        protocol="ucx", asynchronous=True, data=dict
    ) as cluster:
        assert all(
            ws.address.startswith("ucx://") for ws in cluster.scheduler.workers.values()
        )


@gen_test(timeout=20)
async def test_explicit_ucx_with_protocol_none():
    pytest.importorskip("distributed_ucxx")

    initialize(protocol="ucx", enable_tcp_over_ucx=True)
    async with LocalCUDACluster(
        protocol=None,
        enable_tcp_over_ucx=True,
        asynchronous=True,
    ) as cluster:
        assert all(
            ws.address.startswith("ucx://") for ws in cluster.scheduler.workers.values()
        )


@pytest.mark.filterwarnings("ignore:Exception ignored in")
@gen_test(timeout=20)
async def test_ucx_protocol_type_error():
    pytest.importorskip("distributed_ucxx")

    initialize(protocol="ucx", enable_tcp_over_ucx=True)
    with pytest.raises(TypeError):
        async with LocalCUDACluster(
            protocol="tcp", enable_tcp_over_ucx=True, asynchronous=True, data=dict
        ):
            pass


@gen_test(timeout=20)
async def test_n_workers():
    async with LocalCUDACluster(
        CUDA_VISIBLE_DEVICES="0,1",
        worker_class=MockWorker,
        asynchronous=True,
        data=dict,
    ) as cluster:
        assert len(cluster.workers) == 2
        assert len(cluster.worker_spec) == 2


@gen_test(timeout=20)
async def test_threads_per_worker_and_memory_limit():
    async with LocalCUDACluster(threads_per_worker=4, asynchronous=True) as cluster:
        assert all(ws.nthreads == 4 for ws in cluster.scheduler.workers.values())
        full_mem = sum(w.memory_manager.memory_limit for w in cluster.workers.values())
        assert full_mem >= MEMORY_LIMIT - 1024 and full_mem < MEMORY_LIMIT + 1024


@gen_test(timeout=20)
async def test_no_memory_limits_cluster():

    async with LocalCUDACluster(
        asynchronous=True, memory_limit=None, device_memory_limit=None
    ) as cluster:
        async with Client(cluster, asynchronous=True) as client:
            # Check that all workers use a regular dict as their "data store".
            res = await client.run(
                lambda dask_worker: isinstance(dask_worker.data, dict)
            )
            assert all(res.values())


@gen_test(timeout=20)
async def test_no_memory_limits_cudaworker():

    async with LocalCUDACluster(
        asynchronous=True,
        memory_limit=None,
        device_memory_limit=None,
        n_workers=1,
    ) as cluster:
        assert len(cluster.workers) == 1
        async with Client(cluster, asynchronous=True) as client:
            new_worker = CUDAWorker(
                cluster, memory_limit=None, device_memory_limit=None
            )
            await new_worker
            await client.wait_for_workers(2)
            # Check that all workers use a regular dict as their "data store".
            res = await client.run(
                lambda dask_worker: isinstance(dask_worker.data, dict)
            )
            assert all(res.values())
            await new_worker.close()


@gen_test(timeout=20)
async def test_all_to_all():
    async with LocalCUDACluster(
        CUDA_VISIBLE_DEVICES="0,1",
        worker_class=MockWorker,
        asynchronous=True,
        data=dict,
    ) as cluster:
        async with Client(cluster, asynchronous=True) as client:
            workers = list(client.scheduler_info(n_workers=-1)["workers"])
            n_workers = len(workers)
            await utils.all_to_all(client)
            # assert all to all has resulted in all data on every worker
            data = await client.has_what()
            all_data = [v for w in data.values() for v in w if "lambda" in v]
            assert all(all_data.count(i) == n_workers for i in all_data)


@gen_test(timeout=20)
async def test_rmm_pool():
    rmm = pytest.importorskip("rmm")

    async with LocalCUDACluster(
        rmm_pool_size="2GB",
        asynchronous=True,
    ) as cluster:
        async with Client(cluster, asynchronous=True) as client:
            memory_resource_type = await client.run(
                rmm.mr.get_current_device_resource_type
            )
            for v in memory_resource_type.values():
                assert v is rmm.mr.PoolMemoryResource


@gen_test(timeout=20)
async def test_rmm_maximum_poolsize_without_poolsize_error():
    pytest.importorskip("rmm")
    with pytest.raises(ValueError):
        await LocalCUDACluster(rmm_maximum_pool_size="2GB", asynchronous=True)


@gen_test(timeout=20)
async def test_rmm_managed():
    rmm = pytest.importorskip("rmm")

    async with LocalCUDACluster(
        rmm_managed_memory=True,
        asynchronous=True,
    ) as cluster:
        async with Client(cluster, asynchronous=True) as client:
            memory_resource_type = await client.run(
                rmm.mr.get_current_device_resource_type
            )
            for v in memory_resource_type.values():
                assert v is rmm.mr.ManagedMemoryResource


@gen_test(timeout=20)
async def test_rmm_async():
    rmm = pytest.importorskip("rmm")

    async with LocalCUDACluster(
        rmm_async=True,
        rmm_pool_size="2GB",
        rmm_release_threshold="3GB",
        asynchronous=True,
    ) as cluster:
        async with Client(cluster, asynchronous=True) as client:
            memory_resource_type = await client.run(
                rmm.mr.get_current_device_resource_type
            )
            for v in memory_resource_type.values():
                assert v is rmm.mr.CudaAsyncMemoryResource

            ret = await get_cluster_configuration(client)
            assert ret["[plugin] RMMSetup"]["initial_pool_size"] == 2000000000
            assert ret["[plugin] RMMSetup"]["release_threshold"] == 3000000000


@gen_test(timeout=20)
async def test_rmm_async_with_maximum_pool_size():
    rmm = pytest.importorskip("rmm")

    async with LocalCUDACluster(
        rmm_async=True,
        rmm_pool_size="2GB",
        rmm_release_threshold="3GB",
        rmm_maximum_pool_size="4GB",
        asynchronous=True,
    ) as cluster:
        async with Client(cluster, asynchronous=True) as client:
            memory_resource_types = await client.run(
                lambda: (
                    rmm.mr.get_current_device_resource_type(),
                    type(rmm.mr.get_current_device_resource().get_upstream()),
                )
            )
            for v in memory_resource_types.values():
                memory_resource_type, upstream_memory_resource_type = v
                assert memory_resource_type is rmm.mr.LimitingResourceAdaptor
                assert upstream_memory_resource_type is rmm.mr.CudaAsyncMemoryResource

            ret = await get_cluster_configuration(client)
            assert ret["[plugin] RMMSetup"]["initial_pool_size"] == 2000000000
            assert ret["[plugin] RMMSetup"]["release_threshold"] == 3000000000
            assert ret["[plugin] RMMSetup"]["maximum_pool_size"] == 4000000000


@gen_test(timeout=20)
async def test_rmm_logging():
    rmm = pytest.importorskip("rmm")

    async with LocalCUDACluster(
        rmm_pool_size="2GB",
        rmm_log_directory=".",
        asynchronous=True,
    ) as cluster:
        async with Client(cluster, asynchronous=True) as client:
            memory_resource_type = await client.run(
                rmm.mr.get_current_device_resource_type
            )
            for v in memory_resource_type.values():
                assert v is rmm.mr.LoggingResourceAdaptor


@gen_test(timeout=20)
async def test_pre_import():
    module = None

    # Pick a module that isn't currently loaded
    for m in pkgutil.iter_modules():
        if m.ispkg and m.name not in sys.modules.keys():
            module = m.name
            break

    if module is None:
        pytest.skip("No module found that isn't already loaded")

    async with LocalCUDACluster(
        n_workers=1,
        pre_import=module,
        asynchronous=True,
    ) as cluster:
        async with Client(cluster, asynchronous=True) as client:
            imported = await client.run(lambda: module in sys.modules)

            assert all(imported.values())


# Intentionally not using @gen_test to skip cleanup checks
@pytest.mark.xfail(reason="https://github.com/rapidsai/dask-cuda/issues/1265")
def test_pre_import_not_found():
    async def _test_pre_import_not_found():
        with raises_with_cause(RuntimeError, None, ImportError, None):
            await LocalCUDACluster(
                n_workers=1,
                pre_import="my_module",
                asynchronous=True,
                silence_logs=True,
            )

    asyncio.run(_test_pre_import_not_found())


@gen_test(timeout=20)
async def test_cluster_worker():
    async with LocalCUDACluster(
        scheduler_port=0,
        asynchronous=True,
        n_workers=1,
    ) as cluster:
        assert len(cluster.workers) == 1
        async with Client(cluster, asynchronous=True) as client:
            new_worker = CUDAWorker(cluster)
            await new_worker
            await client.wait_for_workers(2)
            await new_worker.close()


@gen_test(timeout=20)
async def test_available_mig_workers():
    uuids = get_gpu_count_mig(return_uuids=True)[1]
    if len(uuids) > 0:
        cuda_visible_devices = ",".join([i.decode("utf-8") for i in uuids])
    else:
        pytest.skip("No MIG devices found")

    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": cuda_visible_devices}):
        async with LocalCUDACluster(
            CUDA_VISIBLE_DEVICES=cuda_visible_devices, asynchronous=True
        ) as cluster:
            async with Client(cluster, asynchronous=True) as client:
                len(cluster.workers) == len(uuids)

                # Check to see if CUDA_VISIBLE_DEVICES cycles properly
                def get_visible_devices():
                    return os.environ["CUDA_VISIBLE_DEVICES"]

                result = await client.run(get_visible_devices)

                assert all(len(v.split(",")) == len(uuids) for v in result.values())
                for i in range(len(cluster.workers)):
                    assert set(v.split(",")[i] for v in result.values()) == set(
                        uuid.decode("utf-8") for uuid in uuids
                    )


@gen_test(timeout=20)
async def test_gpu_uuid():
    gpu_uuid = get_gpu_uuid(0)

    async with LocalCUDACluster(
        CUDA_VISIBLE_DEVICES=gpu_uuid,
        scheduler_port=0,
        asynchronous=True,
    ) as cluster:
        assert len(cluster.workers) == 1
        async with Client(cluster, asynchronous=True) as client:
            await client.wait_for_workers(1)

            result = await client.run(lambda: os.environ["CUDA_VISIBLE_DEVICES"])
            assert list(result.values())[0] == gpu_uuid


@gen_test(timeout=20)
async def test_rmm_track_allocations():
    rmm = pytest.importorskip("rmm")
    async with LocalCUDACluster(
        rmm_pool_size="2GB",
        asynchronous=True,
        rmm_track_allocations=True,
    ) as cluster:
        async with Client(cluster, asynchronous=True) as client:
            memory_resource_type = await client.run(
                rmm.mr.get_current_device_resource_type
            )
            for v in memory_resource_type.values():
                assert v is rmm.mr.TrackingResourceAdaptor

            memory_resource_upstream_type = await client.run(
                lambda: type(rmm.mr.get_current_device_resource().upstream_mr)
            )
            for v in memory_resource_upstream_type.values():
                assert v is rmm.mr.PoolMemoryResource


@gen_test(timeout=20)
async def test_get_cluster_configuration():
    async with LocalCUDACluster(
        rmm_pool_size="2GB",
        rmm_maximum_pool_size="3GB",
        device_memory_limit="30B" if has_device_memory_resource() else None,
        CUDA_VISIBLE_DEVICES="0",
        scheduler_port=0,
        asynchronous=True,
    ) as cluster:
        async with Client(cluster, asynchronous=True) as client:
            ret = await get_cluster_configuration(client)
            assert ret["[plugin] RMMSetup"]["initial_pool_size"] == 2000000000
            assert ret["[plugin] RMMSetup"]["maximum_pool_size"] == 3000000000
            assert ret["jit-unspill"] is False
            if has_device_memory_resource():
                assert ret["device-memory-limit"] == 30


@gen_test(timeout=20)
@pytest.mark.skip_if_no_device_memory(
    "Devices without dedicated memory resources do not support fractional limits"
)
async def test_worker_fraction_limits():
    async with LocalCUDACluster(
        dashboard_address=None,
        device_memory_limit=0.1,
        rmm_pool_size=0.2,
        rmm_maximum_pool_size=0.3,
        CUDA_VISIBLE_DEVICES="0",
        scheduler_port=0,
        asynchronous=True,
    ) as cluster:
        async with Client(cluster, asynchronous=True) as client:
            device_total_memory = await client.run(get_device_total_memory)
            _, device_total_memory = device_total_memory.popitem()
            ret = await get_cluster_configuration(client)
            assert ret["device-memory-limit"] == int(device_total_memory * 0.1)
            assert (
                ret["[plugin] RMMSetup"]["initial_pool_size"]
                == (device_total_memory * 0.2) // 256 * 256
            )
            assert (
                ret["[plugin] RMMSetup"]["maximum_pool_size"]
                == (device_total_memory * 0.3) // 256 * 256
            )


# Intentionally not using @gen_test to skip cleanup checks
@pytest.mark.parametrize(
    "argument", ["pool_size", "maximum_pool_size", "release_threshold"]
)
@pytest.mark.xfail(reason="https://github.com/rapidsai/dask-cuda/issues/1265")
@pytest.mark.skip_if_device_memory(
    "Devices with dedicated memory resources cannot test error"
)
def test_worker_fraction_limits_no_dedicated_memory(argument):
    async def _test_worker_fraction_limits_no_dedicated_memory():
        if argument == "pool_size":
            kwargs = {"rmm_pool_size": "0.1"}
        elif argument == "maximum_pool_size":
            kwargs = {"rmm_pool_size": "1 GiB", "rmm_maximum_pool_size": "0.1"}
        else:
            kwargs = {"rmm_async": True, "rmm_release_threshold": "0.1"}

        with raises_with_cause(
            RuntimeError,
            "Nanny failed to start",
            RuntimeError,
            "Worker failed to start",
            ValueError,
            "Fractional of total device memory not supported in devices without a "
            "dedicated memory resource",
        ):
            await LocalCUDACluster(
                asynchronous=True,
                **kwargs,
            )

    asyncio.run(_test_worker_fraction_limits_no_dedicated_memory())


@gen_test(timeout=20)
async def test_cudf_spill_disabled():
    cudf = pytest.importorskip("cudf")

    async with LocalCUDACluster(
        asynchronous=True,
    ) as cluster:
        async with Client(cluster, asynchronous=True) as client:
            cudf_spill = await client.run(
                cudf.get_option,
                "spill",
            )
            for v in cudf_spill.values():
                assert v is False

            cudf_spill_stats = await client.run(
                cudf.get_option,
                "spill_stats",
            )
            for v in cudf_spill_stats.values():
                assert v == 0


@gen_test(timeout=20)
@pytest.mark.skip_if_no_device_memory(
    "Devices without dedicated memory resources cannot enable cuDF spill"
)
async def test_cudf_spill():
    cudf = pytest.importorskip("cudf")

    async with LocalCUDACluster(
        enable_cudf_spill=True,
        cudf_spill_stats=2,
        asynchronous=True,
    ) as cluster:
        async with Client(cluster, asynchronous=True) as client:
            cudf_spill = await client.run(
                cudf.get_option,
                "spill",
            )
            for v in cudf_spill.values():
                assert v is True

            cudf_spill_stats = await client.run(
                cudf.get_option,
                "spill_stats",
            )
            for v in cudf_spill_stats.values():
                assert v == 2


@pytest.mark.skip_if_device_memory(
    "Devices with dedicated memory resources cannot test error"
)
@gen_test(timeout=20)
async def test_cudf_spill_no_dedicated_memory():
    cudf = pytest.importorskip("cudf")  # noqa: F841

    with pytest.raises(
        ValueError,
        match="cuDF spilling is not supported on devices without dedicated memory",
    ):
        await LocalCUDACluster(
            enable_cudf_spill=True,
            cudf_spill_stats=2,
            asynchronous=True,
        )


@pytest.mark.parametrize(
    "jit_unspill",
    [False, True],
)
@pytest.mark.parametrize(
    "device_memory_limit",
    [None, "1B"],
)
def test_print_cluster_config(capsys, jit_unspill, device_memory_limit):
    pytest.importorskip("distributed_ucxx")

    pytest.importorskip("rich")

    ctx = contextlib.nullcontext()
    if not has_device_memory_resource():
        if device_memory_limit:
            ctx = pytest.raises(
                ValueError,
                match="device_memory_limit is set but device has no dedicated memory.",
            )
        if jit_unspill:
            # JIT-Unspill exception has precedence, thus overwrite ctx if both are
            # enabled
            ctx = pytest.raises(
                ValueError,
                match="JIT-Unspill is not supported on devices without dedicated "
                "memory",
            )

    with ctx:
        with LocalCUDACluster(
            n_workers=1,
            device_memory_limit=device_memory_limit,
            jit_unspill=jit_unspill,
            protocol="ucx",
        ) as cluster:
            with Client(cluster) as client:
                print_cluster_config(client)
                captured = capsys.readouterr()
                assert "Dask Cluster Configuration" in captured.out
                assert "ucx" in captured.out
                if device_memory_limit == "1B":
                    assert "1 B" in captured.out
                assert "[plugin]" in captured.out
                client.shutdown()


@pytest.mark.xfail(reason="https://github.com/rapidsai/dask-cuda/issues/1265")
def test_death_timeout_raises():
    with pytest.raises(asyncio.exceptions.TimeoutError):
        with LocalCUDACluster(
            silence_logs=False,
            death_timeout=1e-10,
            dashboard_address=":0",
        ):
            pass
