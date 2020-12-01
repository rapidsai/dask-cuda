import os

import pytest

from dask.distributed import Client
from distributed.system import MEMORY_LIMIT
from distributed.utils_test import gen_test

from dask_cuda import LocalCUDACluster, CUDAWorker, utils
from dask_cuda.initialize import initialize


@gen_test(timeout=20)
async def test_local_cuda_cluster():
    async with LocalCUDACluster(
        scheduler_port=0, asynchronous=True, device_memory_limit=1
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
            full_mem = sum(w.memory_limit for w in cluster.workers.values())
            assert full_mem >= MEMORY_LIMIT - 1024 and full_mem < MEMORY_LIMIT + 1024

            for w, devices in result.items():
                ident = devices.split(",")[0]
                assert int(ident) == cluster.scheduler.workers[w].name

            with pytest.raises(ValueError):
                cluster.scale(1000)


# Notice, this test might raise errors when the number of available GPUs is less
# than 8 but as long as the test passes the errors can be ignored.
@gen_test(timeout=20)
async def test_with_subset_of_cuda_visible_devices():
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,6,7"
    try:
        async with LocalCUDACluster(
            scheduler_port=0, asynchronous=True, device_memory_limit=1
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
                        2,
                        3,
                        6,
                        7,
                    }
    finally:
        del os.environ["CUDA_VISIBLE_DEVICES"]


@pytest.mark.parametrize("protocol", ["ucx", None])
@pytest.mark.asyncio
async def test_ucx_protocol(protocol):
    pytest.importorskip("ucp")

    initialize(enable_tcp_over_ucx=True)
    async with LocalCUDACluster(
        protocol=protocol, enable_tcp_over_ucx=True, asynchronous=True, data=dict
    ) as cluster:
        assert all(
            ws.address.startswith("ucx://") for ws in cluster.scheduler.workers.values()
        )


@pytest.mark.asyncio
async def test_ucx_protocol_type_error():
    pytest.importorskip("ucp")

    initialize(enable_tcp_over_ucx=True)
    with pytest.raises(TypeError):
        async with LocalCUDACluster(
            protocol="tcp", enable_tcp_over_ucx=True, asynchronous=True, data=dict
        ):
            pass


@gen_test(timeout=20)
async def test_n_workers():
    async with LocalCUDACluster(
        CUDA_VISIBLE_DEVICES="0,1", asynchronous=True
    ) as cluster:
        assert len(cluster.workers) == 2
        assert len(cluster.worker_spec) == 2


@gen_test(timeout=20)
async def test_all_to_all():
    async with LocalCUDACluster(
        CUDA_VISIBLE_DEVICES="0,1", asynchronous=True
    ) as cluster:
        async with Client(cluster, asynchronous=True) as client:
            workers = list(client.scheduler_info()["workers"])
            n_workers = len(workers)
            await utils.all_to_all(client)
            # assert all to all has resulted in all data on every worker
            data = await client.has_what()
            all_data = [v for w in data.values() for v in w if "lambda" in v]
            assert all(all_data.count(i) == n_workers for i in all_data)


@gen_test(timeout=20)
async def test_rmm_pool():
    rmm = pytest.importorskip("rmm")

    async with LocalCUDACluster(rmm_pool_size="2GB", asynchronous=True) as cluster:
        async with Client(cluster, asynchronous=True) as client:
            memory_resource_type = await client.run(
                rmm.mr.get_current_device_resource_type
            )
            for v in memory_resource_type.values():
                assert v is rmm.mr.PoolMemoryResource


@gen_test(timeout=20)
async def test_rmm_managed():
    rmm = pytest.importorskip("rmm")

    async with LocalCUDACluster(rmm_managed_memory=True, asynchronous=True) as cluster:
        async with Client(cluster, asynchronous=True) as client:
            memory_resource_type = await client.run(
                rmm.mr.get_current_device_resource_type
            )
            for v in memory_resource_type.values():
                assert v is rmm.mr.ManagedMemoryResource


@gen_test(timeout=20)
async def test_cluster_worker():
    async with LocalCUDACluster(
        scheduler_port=0, asynchronous=True, device_memory_limit=1, n_workers=1,
    ) as cluster:
        assert len(cluster.workers) == 1
        async with Client(cluster, asynchronous=True) as client:
            new_worker = CUDAWorker(cluster)
            await client.wait_for_workers(2)
            await new_worker.close()
