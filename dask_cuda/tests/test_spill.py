from distributed.utils_test import gen_cluster
from distributed import wait
from dask_cuda.worker import CUDAWorker
from dask_cuda.utils_test import assert_device_host_file_size as assert_dhf_size
import dask.array as da
import pytest


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
            "device_memory_limit": 2e9,
            "memory_limit": 1e9,
            "host_target": 0.3,
            "host_spill": 0.3,
            "spills_to_disk": True,
        },
    ],
)
@pytest.mark.parametrize("device_limit", [0.3, 0.5])
def test_device_spill(params, device_limit):
    @gen_cluster(
        client=True,
        ncores=[("127.0.0.1", 1)],
        Worker=CUDAWorker,
        worker_kwargs={
            "device_memory_limit": params["device_memory_limit"],
            "memory_limit": params["memory_limit"],
        },
        config={
            "distributed.worker.memory.target": params["host_target"],
            "distributed.worker.memory.spill": params["host_spill"],
            "distributed.worker.device-memory.target": device_limit,
            "distributed.worker.device-memory.spill": device_limit,
        },
    )
    def test_device_spill(client, scheduler, worker):
        import cupy

        cupy.cuda.set_allocator(None)
        cupy.cuda.set_pinned_memory_allocator(None)

        rs = da.random.RandomState(RandomState=cupy.random.RandomState)
        x = rs.random(int(200e6 * 0.7), chunks=10e6)
        yield wait(x)

        xx = x.persist()
        yield wait(xx)

        # Allow up to 1024 bytes overhead per chunk serialized
        assert_dhf_size(worker.data, x.nbytes, 1024)

        y = client.compute(x.sum())
        res = yield y

        assert (abs(res / x.size) - 0.5) < 1e-3

        assert_dhf_size(worker.data, x.nbytes, 1024)
        if params["spills_to_disk"]:
            assert len(worker.data.host.slow) > 0
        else:
            assert len(worker.data.host.slow) == 0

    test_device_spill()
