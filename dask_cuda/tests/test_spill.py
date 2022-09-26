import os
from time import sleep

import pytest
from zict.file import _safe_key as safe_key

import dask
from dask import array as da
from distributed import Client, get_worker, wait
from distributed.metrics import time
from distributed.sizeof import sizeof
from distributed.utils_test import gen_cluster, gen_test, loop  # noqa: F401

from dask_cuda import LocalCUDACluster, utils

if utils.get_device_total_memory() < 1e10:
    pytest.skip("Not enough GPU memory", allow_module_level=True)


def device_host_file_size_matches(
    dhf, total_bytes, device_chunk_overhead=0, serialized_chunk_overhead=1024
):
    byte_sum = dhf.device_buffer.fast.total_weight

    # `dhf.host_buffer.fast` is only available when Worker's `memory_limit != 0`
    if hasattr(dhf.host_buffer, "fast"):
        byte_sum += dhf.host_buffer.fast.total_weight
    else:
        byte_sum += sum([sizeof(b) for b in dhf.host_buffer.values()])

    # `dhf.disk` is only available when Worker's `memory_limit != 0`
    if dhf.disk is not None:
        file_path = [
            os.path.join(dhf.disk.directory, safe_key(k)) for k in dhf.disk.keys()
        ]
        file_size = [os.path.getsize(f) for f in file_path]
        byte_sum += sum(file_size)

    # Allow up to chunk_overhead bytes overhead per chunk
    device_overhead = len(dhf.device) * device_chunk_overhead
    host_overhead = len(dhf.host) * serialized_chunk_overhead
    disk_overhead = (
        len(dhf.disk) * serialized_chunk_overhead if dhf.disk is not None else 0
    )

    return (
        byte_sum >= total_bytes
        and byte_sum <= total_bytes + device_overhead + host_overhead + disk_overhead
    )


def assert_device_host_file_size(
    dhf, total_bytes, device_chunk_overhead=0, serialized_chunk_overhead=1024
):
    assert device_host_file_size_matches(
        dhf, total_bytes, device_chunk_overhead, serialized_chunk_overhead
    )


def worker_assert(total_size, device_chunk_overhead, serialized_chunk_overhead):
    assert_device_host_file_size(
        get_worker().data, total_size, device_chunk_overhead, serialized_chunk_overhead
    )


def delayed_worker_assert(total_size, device_chunk_overhead, serialized_chunk_overhead):
    start = time()
    while not device_host_file_size_matches(
        get_worker().data, total_size, device_chunk_overhead, serialized_chunk_overhead
    ):
        sleep(0.01)
        if time() < start + 3:
            assert_device_host_file_size(
                get_worker().data,
                total_size,
                device_chunk_overhead,
                serialized_chunk_overhead,
            )


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
            # pausing is enabled and thus triggers `DeviceHostFile.evict()`
            "device_memory_limit": int(200e6),
            "memory_limit": int(200e6),
            "host_target": None,
            "host_spill": None,
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
    with dask.config.set({"distributed.worker.memory.terminate": False}):
        async with LocalCUDACluster(
            n_workers=1,
            scheduler_port=0,
            silence_logs=False,
            dashboard_address=None,
            asynchronous=True,
            device_memory_limit=params["device_memory_limit"],
            memory_limit=params["memory_limit"],
            memory_target_fraction=params["host_target"],
            memory_spill_fraction=params["host_spill"],
            memory_pause_fraction=params["host_pause"],
        ) as cluster:
            async with Client(cluster, asynchronous=True) as client:

                rs = da.random.RandomState(RandomState=cupy.random.RandomState)
                x = rs.random(int(50e6), chunks=2e6)
                await wait(x)

                xx = x.persist()
                await wait(xx)

                # Allow up to 1024 bytes overhead per chunk serialized
                await client.run(worker_assert, x.nbytes, 1024, 1024)

                y = client.compute(x.sum())
                res = await y

                assert (abs(res / x.size) - 0.5) < 1e-3

                await client.run(worker_assert, x.nbytes, 1024, 1024)
                host_chunks = await client.run(lambda: len(get_worker().data.host))
                disk_chunks = await client.run(
                    lambda: len(get_worker().data.disk or list())
                )
                for hc, dc in zip(host_chunks.values(), disk_chunks.values()):
                    if params["spills_to_disk"]:
                        assert dc > 0
                    else:
                        assert hc > 0
                        assert dc == 0


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
            # pausing is enabled and thus triggers `DeviceHostFile.evict()`
            "device_memory_limit": int(50e6),
            "memory_limit": int(50e6),
            "host_target": None,
            "host_spill": None,
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
async def test_cudf_cluster_device_spill(params):
    cudf = pytest.importorskip("cudf")

    with dask.config.set(
        {
            "distributed.comm.compression": False,
            "distributed.worker.memory.terminate": False,
        }
    ):
        async with LocalCUDACluster(
            n_workers=1,
            device_memory_limit=params["device_memory_limit"],
            memory_limit=params["memory_limit"],
            memory_target_fraction=params["host_target"],
            memory_spill_fraction=params["host_spill"],
            memory_pause_fraction=params["host_pause"],
            asynchronous=True,
        ) as cluster:
            async with Client(cluster, asynchronous=True) as client:

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

                cdf2 = cdf.persist()
                await wait(cdf2)

                del cdf

                host_chunks = await client.run(lambda: len(get_worker().data.host))
                disk_chunks = await client.run(
                    lambda: len(get_worker().data.disk or list())
                )
                for hc, dc in zip(host_chunks.values(), disk_chunks.values()):
                    if params["spills_to_disk"]:
                        assert dc > 0
                    else:
                        assert hc > 0
                        assert dc == 0

                await client.run(worker_assert, nbytes, 32, 2048)

                del cdf2

                await client.run(delayed_worker_assert, 0, 0, 0)
