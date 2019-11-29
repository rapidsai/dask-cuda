import os

from dask.distributed import Client
from dask_cuda import LocalCUDACluster, utils
from dask_cuda.initialize import initialize
from distributed.system import MEMORY_LIMIT
from distributed.utils_test import gen_test

import pytest


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
                ident = devices[0]
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


@gen_test(timeout=20)
async def test_ucx_protocol():
    pytest.importorskip("distributed.comm.ucx")

    initialize(enable_tcp_over_ucx=True)
    async with LocalCUDACluster(
        protocol="ucx", asynchronous=True, data=dict
    ) as cluster:
        assert all(
            ws.address.startswith("ucx://") for ws in cluster.scheduler.workers.values()
        )


@gen_test(timeout=20)
async def test_n_workers():
    async with LocalCUDACluster(
        CUDA_VISIBLE_DEVICES="0,1", asynchronous=True
    ) as cluster:
        assert len(cluster.workers) == 2
        assert len(cluster.worker_spec) == 2
