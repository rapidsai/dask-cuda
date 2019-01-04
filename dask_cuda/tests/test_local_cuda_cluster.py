from distributed.utils_test import gen_test

import os

from dask.distributed import Client
from distributed.worker import TOTAL_MEMORY
from dask_cuda import LocalCUDACluster
from dask_cuda import utils


@gen_test(timeout=20)
async def test_local_cuda_cluster():
    async with LocalCUDACluster(
        scheduler_port=0, asynchronous=True, diagnostics_port=None
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

            # Use full memory
            assert sum(w.memory_limit for w in cluster.workers) == TOTAL_MEMORY


@gen_test(timeout=20)
async def test_with_subset_of_cuda_visible_devices():
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,7,8"
    try:
        async with LocalCUDACluster(
            scheduler_port=0, asynchronous=True, diagnostics_port=None
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
                        7,
                        8,
                    }
    finally:
        del os.environ["CUDA_VISIBLE_DEVICES"]
