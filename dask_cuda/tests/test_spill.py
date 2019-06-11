import pytest

import os
import asyncio

from distributed.utils_test import gen_cluster, loop, gen_test
from distributed.worker import Worker
from distributed import Client, get_worker, wait
from dask_cuda import LocalCUDACluster
from dask_cuda.device_host_file import DeviceHostFile
from zict.file import _safe_key as safe_key

import dask.array as da
import cupy
import cudf
import dask_cudf


def assert_device_host_file_size(
    dhf, total_bytes, device_chunk_overhead=0, serialized_chunk_overhead=1024
):
    byte_sum = dhf.device_buffer.fast.total_weight + dhf.host_buffer.fast.total_weight
    file_path = [os.path.join(dhf.disk.directory, safe_key(k)) for k in dhf.disk.keys()]
    file_size = [os.path.getsize(f) for f in file_path]
    byte_sum += sum(file_size)

    # Allow up to chunk_overhead bytes overhead per chunk on disk
    device_overhead = len(dhf.device) * device_chunk_overhead
    host_overhead = len(dhf.host) * serialized_chunk_overhead
    disk_overhead = len(dhf.disk) * serialized_chunk_overhead
    assert (
        byte_sum >= total_bytes
        and byte_sum <= total_bytes + device_overhead + host_overhead + disk_overhead
    )


def worker_assert(total_size, device_chunk_overhead, serialized_chunk_overhead):
    assert_device_host_file_size(
        get_worker().data, total_size, device_chunk_overhead, serialized_chunk_overhead
    )


@pytest.mark.parametrize(
    "params",
    [
        {
            "device_memory_limit": 2e9,
            "memory_limit": 4e9,
            "host_target": 0.6,
            "host_spill": 0.7,
            "spills_to_disk": False,
        },
        {
            "device_memory_limit": 1e9,
            "memory_limit": 1e9,
            "host_target": 0.3,
            "host_spill": 0.3,
            "spills_to_disk": True,
        },
    ],
)
def test_device_spill(params):
    @gen_cluster(
        client=True,
        ncores=[("127.0.0.1", 1)],
        Worker=Worker,
        worker_kwargs={
            "memory_limit": params["memory_limit"],
            "data": DeviceHostFile(
                device_memory_limit=params["device_memory_limit"],
                memory_limit=params["memory_limit"],
            ),
        },
        config={
            "distributed.worker.memory.target": params["host_target"],
            "distributed.worker.memory.spill": params["host_spill"],
        },
    )
    def test_device_spill(client, scheduler, worker):
        rs = da.random.RandomState(RandomState=cupy.random.RandomState)
        x = rs.random(int(250e6), chunks=10e6)
        yield wait(x)

        xx = x.persist()
        yield wait(xx)

        # Allow up to 1024 bytes overhead per chunk serialized
        assert_device_host_file_size(worker.data, x.nbytes, 1024)

        y = client.compute(x.sum())
        res = yield y

        assert (abs(res / x.size) - 0.5) < 1e-3

        assert_device_host_file_size(worker.data, x.nbytes, 1024)
        if params["spills_to_disk"]:
            assert len(worker.data.disk) > 0
        else:
            assert len(worker.data.disk) == 0

    test_device_spill()


@pytest.mark.parametrize(
    "params",
    [
        {
            "device_memory_limit": 2e9,
            "memory_limit": 4e9,
            "host_target": 0.6,
            "host_spill": 0.7,
            "spills_to_disk": False,
        },
        {
            "device_memory_limit": 1e9,
            "memory_limit": 1e9,
            "host_target": 0.3,
            "host_spill": 0.3,
            "spills_to_disk": True,
        },
    ],
)
@pytest.mark.asyncio
async def test_cluster_device_spill(loop, params):
    async with LocalCUDACluster(
        1,
        scheduler_port=0,
        processes=True,
        silence_logs=False,
        dashboard_address=None,
        asynchronous=True,
        death_timeout=10,
        device_memory_limit=params["device_memory_limit"],
        memory_limit=params["memory_limit"],
        memory_target_fraction=params["host_target"],
        memory_spill_fraction=params["host_spill"],
    ) as cluster:
        async with Client(cluster, asynchronous=True) as client:

            rs = da.random.RandomState(RandomState=cupy.random.RandomState)
            x = rs.random(int(250e6), chunks=10e6)
            await wait(x)

            xx = x.persist()
            await wait(xx)

            def get_data(worker, total_size):
                assert_device_host_file_size(get_worker().data, total_size)

            # Allow up to 1024 bytes overhead per chunk serialized
            await client.run(get_data, cluster.workers[0].Worker, x.nbytes)

            y = client.compute(x.sum())
            res = await y

            assert (abs(res / x.size) - 0.5) < 1e-3

            await client.run(get_data, cluster.workers[0].Worker, x.nbytes)
            disk_chunks = await client.run(lambda: len(get_worker().data.disk))
            for dc in disk_chunks.values():
                if params["spills_to_disk"]:
                    assert dc > 0
                else:
                    assert dc == 0


@pytest.mark.parametrize(
    "params",
    [
        {
            "device_memory_limit": 1e9,
            "memory_limit": 1e9,
            "host_target": 0.6,
            "host_spill": 0.7,
            "spills_to_disk": False,
        },
        {
            "device_memory_limit": 250e6,
            "memory_limit": 250e6,
            "host_target": 0.3,
            "host_spill": 0.3,
            "spills_to_disk": True,
        },
    ],
)
def test_cudf_cluster_device_spill(loop, params):
    async def test():
        async with LocalCUDACluster(
            1,
            device_memory_limit=params["device_memory_limit"],
            memory_limit=params["memory_limit"],
            memory_target_fraction=params["host_target"],
            memory_spill_fraction=params["host_spill"],
            death_timeout=300,
            asynchronous=True,
        ) as cluster:
            async with Client(cluster, asynchronous=True) as client:

                rows = int(20e6)

                df = cudf.DataFrame([("A", [8] * rows), ("B", [32] * rows)])

                cdf = dask_cudf.from_cudf(df, npartitions=20)

                tasks = await asyncio.gather(
                    *[client.compute(p) for p in cdf.partitions]
                )
                nbytes = sum(t.__sizeof__() for t in tasks)
                part_index_nbytes = tasks[0]._index.__sizeof__()

                cdf2 = cdf.persist()
                await wait(cdf2)

                del df
                del cdf

                await client.run(worker_assert, nbytes, 32, 2048 + part_index_nbytes)
                disk_chunks = await client.run(lambda: len(get_worker().data.disk))
                for dc in disk_chunks.values():
                    if params["spills_to_disk"]:
                        assert dc > 0
                    else:
                        assert dc == 0

                del cdf2

                await client.run(worker_assert, 0, 0, 0)

    loop.run_sync(test)
