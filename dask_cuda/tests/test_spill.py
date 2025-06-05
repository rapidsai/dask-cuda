# Copyright (c) 2025, NVIDIA CORPORATION.

import gc
import os
from time import sleep
from typing import TypedDict

import pytest

import dask
from dask import array as da
from distributed import Client, Worker, wait
from distributed.metrics import time
from distributed.sizeof import sizeof
from distributed.utils import Deadline
from distributed.utils_test import gen_cluster, gen_test, loop  # noqa: F401

import dask_cudf

from dask_cuda import LocalCUDACluster, utils
from dask_cuda.utils_test import IncreasedCloseTimeoutNanny

if utils.get_device_total_memory() < 1e10:
    pytest.skip("Not enough GPU memory", allow_module_level=True)


def _set_cudf_device_limit():
    """Ensure spilling for objects of all sizes"""
    import cudf

    cudf.set_option("spill_device_limit", 0)


def _assert_cudf_spill_stats(enable_cudf_spill, dask_worker=None):
    """Ensure cuDF has spilled data with its internal mechanism"""
    import cudf

    global_manager = cudf.core.buffer.spill_manager.get_global_manager()

    if enable_cudf_spill:
        stats = global_manager.statistics
        buffers = global_manager.buffers()
        assert stats.spill_totals[("gpu", "cpu")][0] > 1000
        assert stats.spill_totals[("cpu", "gpu")][0] > 1000
        assert len(buffers) > 0
    else:
        assert global_manager is None


@pytest.fixture(params=[False, True])
def cudf_spill(request):
    """Fixture to enable and clear cuDF spill manager in client process"""
    cudf = pytest.importorskip("cudf")

    enable_cudf_spill = request.param

    if enable_cudf_spill:
        # If the global spill manager was previously set, fail.
        assert cudf.core.buffer.spill_manager._global_manager is None

        cudf.set_option("spill", True)
        cudf.set_option("spill_stats", True)

        # This change is to prevent changing RMM resource stack in cuDF,
        # workers do not need this because they are spawned as new
        # processes for every new test that runs.
        cudf.set_option("spill_on_demand", False)

        _set_cudf_device_limit()

    yield enable_cudf_spill

    cudf.set_option("spill", False)
    cudf.core.buffer.spill_manager._global_manager_uninitialized = True
    cudf.core.buffer.spill_manager._global_manager = None


def device_host_file_size_matches(
    dask_worker: Worker,
    total_bytes,
    device_chunk_overhead=0,
    serialized_chunk_overhead=1024,
):
    worker_data_sizes = collect_device_host_file_size(
        dask_worker,
        device_chunk_overhead=device_chunk_overhead,
        serialized_chunk_overhead=serialized_chunk_overhead,
    )
    byte_sum = (
        worker_data_sizes["device_fast"]
        + worker_data_sizes["host_fast"]
        + worker_data_sizes["host_buffer"]
        + worker_data_sizes["disk"]
    )
    return (
        byte_sum >= total_bytes
        and byte_sum
        <= total_bytes
        + worker_data_sizes["device_overhead"]
        + worker_data_sizes["host_overhead"]
        + worker_data_sizes["disk_overhead"]
    )


class WorkerDataSizes(TypedDict):
    device_fast: int
    host_fast: int
    host_buffer: int
    disk: int
    device_overhead: int
    host_overhead: int
    disk_overhead: int


def collect_device_host_file_size(
    dask_worker: Worker,
    device_chunk_overhead: int,
    serialized_chunk_overhead: int,
) -> WorkerDataSizes:
    dhf = dask_worker.data

    device_fast = dhf.device_buffer.fast.total_weight or 0
    if hasattr(dhf.host_buffer, "fast"):
        host_fast = dhf.host_buffer.fast.total_weight or 0
        host_buffer = 0
    else:
        host_buffer = sum([sizeof(b) for b in dhf.host_buffer.values()])
        host_fast = 0

    if dhf.disk is not None:
        file_path = [
            os.path.join(dhf.disk.directory, fname)
            for fname in dhf.disk.filenames.values()
        ]
        file_size = [os.path.getsize(f) for f in file_path]
        disk = sum(file_size)
    else:
        disk = 0

    # Allow up to chunk_overhead bytes overhead per chunk
    device_overhead = len(dhf.device) * device_chunk_overhead
    host_overhead = len(dhf.host) * serialized_chunk_overhead
    disk_overhead = (
        len(dhf.disk) * serialized_chunk_overhead if dhf.disk is not None else 0
    )

    return WorkerDataSizes(
        device_fast=device_fast,
        host_fast=host_fast,
        host_buffer=host_buffer,
        disk=disk,
        device_overhead=device_overhead,
        host_overhead=host_overhead,
        disk_overhead=disk_overhead,
    )


def assert_device_host_file_size(
    dask_worker: Worker,
    total_bytes,
    device_chunk_overhead=0,
    serialized_chunk_overhead=1024,
):
    assert device_host_file_size_matches(
        dask_worker, total_bytes, device_chunk_overhead, serialized_chunk_overhead
    )


def worker_assert(
    total_size,
    device_chunk_overhead,
    serialized_chunk_overhead,
    dask_worker=None,
):
    assert_device_host_file_size(
        dask_worker, total_size, device_chunk_overhead, serialized_chunk_overhead
    )


def delayed_worker_assert(
    total_size,
    device_chunk_overhead,
    serialized_chunk_overhead,
    dask_worker=None,
):
    start = time()
    while not device_host_file_size_matches(
        dask_worker, total_size, device_chunk_overhead, serialized_chunk_overhead
    ):
        sleep(0.01)
        if time() < start + 3:
            assert_device_host_file_size(
                dask_worker,
                total_size,
                device_chunk_overhead,
                serialized_chunk_overhead,
            )


def assert_host_chunks(spills_to_disk, dask_worker=None):
    if spills_to_disk is False:
        assert len(dask_worker.data.host)


def assert_disk_chunks(spills_to_disk, dask_worker=None):
    if spills_to_disk is True:
        assert len(dask_worker.data.disk or list()) > 0
    else:
        assert len(dask_worker.data.disk or list()) == 0


@pytest.mark.parametrize(
    "params",
    [
        {
            "device_memory_limit": int(200e6),
            "memory_limit": int(2000e6),
            "host_target": False,
            "host_spill": False,
            "host_pause": False,
            "spills_to_disk": False,
        },
        {
            "device_memory_limit": int(200e6),
            "memory_limit": int(200e6),
            "host_target": False,
            "host_spill": False,
            "host_pause": False,
            "spills_to_disk": True,
        },
        {
            # This test setup differs from the one above as Distributed worker
            # spilling fraction is very low and thus forcefully triggers
            # `DeviceHostFile.evict()`
            "device_memory_limit": int(200e6),
            "memory_limit": int(200e6),
            "host_target": False,
            "host_spill": 0.01,
            "host_pause": False,
            "spills_to_disk": True,
        },
        {
            "device_memory_limit": int(200e6),
            "memory_limit": None,
            "host_target": False,
            "host_spill": False,
            "host_pause": False,
            "spills_to_disk": False,
        },
    ],
)
@gen_test(timeout=30)
async def test_cupy_cluster_device_spill(params):
    cupy = pytest.importorskip("cupy")
    with dask.config.set(
        {
            "distributed.worker.memory.terminate": False,
            "distributed.worker.memory.pause": params["host_pause"],
            "distributed.worker.memory.spill": params["host_spill"],
            "distributed.worker.memory.target": params["host_target"],
        }
    ):
        async with LocalCUDACluster(
            n_workers=1,
            scheduler_port=0,
            silence_logs=False,
            dashboard_address=None,
            asynchronous=True,
            device_memory_limit=params["device_memory_limit"],
            memory_limit=params["memory_limit"],
            worker_class=IncreasedCloseTimeoutNanny,
        ) as cluster:
            async with Client(cluster, asynchronous=True) as client:

                await client.wait_for_workers(1)

                rs = da.random.RandomState(RandomState=cupy.random.RandomState)
                x = rs.random(int(50e6), chunks=2e6)
                await wait(x)

                [xx] = client.persist([x])
                await xx

                # Allow up to 1024 bytes overhead per chunk serialized
                await client.run(
                    worker_assert,
                    x.nbytes,
                    1024,
                    1024,
                )

                y = client.compute(x.sum())
                res = await y

                assert (abs(res / x.size) - 0.5) < 1e-3

                await client.run(
                    worker_assert,
                    x.nbytes,
                    1024,
                    1024,
                )
                await client.run(
                    assert_host_chunks,
                    params["spills_to_disk"],
                )
                await client.run(
                    assert_disk_chunks,
                    params["spills_to_disk"],
                )


@pytest.mark.parametrize(
    "params",
    [
        {
            "device_memory_limit": int(50e6),
            "memory_limit": int(1000e6),
            "host_target": False,
            "host_spill": False,
            "host_pause": False,
            "spills_to_disk": False,
        },
        {
            "device_memory_limit": int(50e6),
            "memory_limit": int(50e6),
            "host_target": False,
            "host_spill": False,
            "host_pause": False,
            "spills_to_disk": True,
        },
        {
            # This test setup differs from the one above as Distributed worker
            # spilling fraction is very low and thus forcefully triggers
            # `DeviceHostFile.evict()`
            "device_memory_limit": int(50e6),
            "memory_limit": int(50e6),
            "host_target": False,
            "host_spill": 0.01,
            "host_pause": False,
            "spills_to_disk": True,
        },
        {
            "device_memory_limit": int(50e6),
            "memory_limit": None,
            "host_target": False,
            "host_spill": False,
            "host_pause": False,
            "spills_to_disk": False,
        },
    ],
)
@gen_test(timeout=30)
async def test_cudf_cluster_device_spill(params, cudf_spill):
    cudf = pytest.importorskip("cudf")

    enable_cudf_spill = cudf_spill

    with dask.config.set(
        {
            "distributed.comm.compression": False,
            "distributed.worker.memory.terminate": False,
            "distributed.worker.memory.spill-compression": False,
            "distributed.worker.memory.pause": params["host_pause"],
            "distributed.worker.memory.spill": params["host_spill"],
            "distributed.worker.memory.target": params["host_target"],
        }
    ):
        async with LocalCUDACluster(
            n_workers=1,
            scheduler_port=0,
            silence_logs=False,
            dashboard_address=None,
            asynchronous=True,
            device_memory_limit=params["device_memory_limit"],
            memory_limit=params["memory_limit"],
            worker_class=IncreasedCloseTimeoutNanny,
            enable_cudf_spill=enable_cudf_spill,
        ) as cluster:
            async with Client(cluster, asynchronous=True) as client:

                await client.wait_for_workers(1)

                # There's a known issue with datetime64:
                # https://github.com/numpy/numpy/issues/4983#issuecomment-441332940
                # The same error above happens when spilling datetime64 to disk
                cdf = (
                    dask.datasets.timeseries(
                        dtypes={"x": int, "y": float}, freq="400ms"
                    )
                    .reset_index(drop=True)
                    .map_partitions(cudf.from_pandas)
                )

                sizes = await client.compute(
                    cdf.map_partitions(lambda df: df.memory_usage())
                )
                sizes = sizes.to_arrow().to_pylist()
                nbytes = sum(sizes)

                [cdf2] = client.persist([cdf])
                await cdf2

                del cdf
                gc.collect()

                if enable_cudf_spill:
                    expected_data = WorkerDataSizes(
                        device_fast=0,
                        host_fast=0,
                        host_buffer=0,
                        disk=0,
                        device_overhead=0,
                        host_overhead=0,
                        disk_overhead=0,
                    )

                    deadline = Deadline.after(duration=3)
                    while not deadline.expired:
                        data = await client.run(
                            collect_device_host_file_size,
                            device_chunk_overhead=0,
                            serialized_chunk_overhead=0,
                        )
                        expected = {k: expected_data for k in data}
                        if data == expected:
                            break
                        sleep(0.01)

                    # final assertion for pytest to reraise with a nice traceback
                    assert data == expected

                else:
                    await client.run(
                        assert_host_chunks,
                        params["spills_to_disk"],
                    )
                    await client.run(
                        assert_disk_chunks,
                        params["spills_to_disk"],
                    )
                    await client.run(
                        worker_assert,
                        nbytes,
                        32,
                        2048,
                    )

                del cdf2

                while True:
                    try:
                        await client.run(
                            delayed_worker_assert,
                            0,
                            0,
                            0,
                        )
                    except AssertionError:
                        gc.collect()
                    else:
                        break


@gen_test(timeout=30)
async def test_cudf_spill_cluster(cudf_spill):
    cudf = pytest.importorskip("cudf")
    enable_cudf_spill = cudf_spill

    async with LocalCUDACluster(
        n_workers=1,
        scheduler_port=0,
        silence_logs=False,
        dashboard_address=None,
        asynchronous=True,
        device_memory_limit=None,
        memory_limit=None,
        worker_class=IncreasedCloseTimeoutNanny,
        enable_cudf_spill=enable_cudf_spill,
        cudf_spill_stats=enable_cudf_spill,
    ) as cluster:
        async with Client(cluster, asynchronous=True) as client:

            await client.wait_for_workers(1)
            await client.run(_set_cudf_device_limit)

            cdf = cudf.DataFrame(
                {
                    "a": list(range(200)),
                    "b": list(reversed(range(200))),
                    "c": list(range(200)),
                }
            )

            [ddf] = client.persist([dask_cudf.from_cudf(cdf, npartitions=2).sum()])
            await ddf

            await client.run(_assert_cudf_spill_stats, enable_cudf_spill)
            _assert_cudf_spill_stats(enable_cudf_spill)
