from distributed.utils_test import gen_cluster
from distributed.worker import Worker
from distributed import wait
from dask_cuda.device_host_file import DeviceHostFile
import dask.array as da
import pytest
import os
from zict.file import _safe_key as safe_key


def assert_device_host_file_size(dhf, total_bytes, chunk_overhead=1024):
    byte_sum = dhf.device_buffer.fast.total_weight + dhf.host_buffer.fast.total_weight
    file_path = [os.path.join(dhf.disk.directory, safe_key(k)) for k in dhf.disk.keys()]
    file_size = [os.path.getsize(f) for f in file_path]
    byte_sum += sum(file_size)

    # Allow up to chunk_overhead bytes overhead per chunk on disk
    host_overhead = len(dhf.host) * chunk_overhead
    disk_overhead = len(dhf.disk) * chunk_overhead
    assert (
        byte_sum >= total_bytes
        and byte_sum <= total_bytes + host_overhead + disk_overhead
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
        import cupy

        cupy.cuda.set_allocator(None)
        cupy.cuda.set_pinned_memory_allocator(None)

        rs = da.random.RandomState(RandomState=cupy.random.RandomState)
        x = rs.random(int(250e6), chunks=10e6)
        yield wait(x)

        xx = x.persist()
        yield wait(xx)

        print(worker.data.device_buffer)

        # Allow up to 1024 bytes overhead per chunk serialized
        assert_device_host_file_size(worker.data, x.nbytes, 1024)

        y = client.compute(x.sum())
        res = yield y
        print(worker.data.device_buffer)

        assert (abs(res / x.size) - 0.5) < 1e-3

        assert_device_host_file_size(worker.data, x.nbytes, 1024)
        if params["spills_to_disk"]:
            assert len(worker.data.disk) > 0
        else:
            assert len(worker.data.disk) == 0

    test_device_spill()
