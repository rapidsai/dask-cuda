import pytest

from distributed.utils_test import gen_test

import os

from dask.distributed import Client
from distributed.worker import TOTAL_MEMORY
from dask_cuda import LocalCUDACluster
from dask_cuda import utils
import pytest


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

            for w, devices in result.items():
                ident = devices[0]
                assert ident in cluster.scheduler.workers[w].name


@gen_test(timeout=20)
async def test_with_subset_of_cuda_visible_devices():
    n_gpus = utils.get_n_gpus()
    if n_gpus < 2:
        pytest.skip("More than 1 GPU required for test")
    test_gpus = n_gpus // 2
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in
        range(test_gpus)])
    try:
        async with LocalCUDACluster(
            scheduler_port=0, asynchronous=True, diagnostics_port=None
        ) as cluster:
            async with Client(cluster, asynchronous=True) as client:
                assert len(cluster.workers) == test_gpus

                # CUDA_VISIBLE_DEVICES cycles properly
                def get_visible_devices():
                    return os.environ["CUDA_VISIBLE_DEVICES"]

                result = await client.run(get_visible_devices)

                assert all(len(v.split(",")) == test_gpus for v in
                        result.values())
                for i in range(test_gpus):
                    assert {int(v.split(",")[i]) for v in
                            result.values()} == set(list(range(test_gpus)))
    finally:
        del os.environ["CUDA_VISIBLE_DEVICES"]


@gen_test(timeout=20)
async def test_ucx_protocol():
    pytest.importorskip("distributed.comm.ucx")
    async with LocalCUDACluster(
        protocol="ucx",
        interface="ib0",
        scheduler_port=0,
        asynchronous=True,
        dashboard_address=None,
        data=dict,
    ) as cluster:
        assert all(
            ws.address.startswith("ucx://") for ws in cluster.scheduler.workers.values()
        )
