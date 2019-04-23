from distributed.utils_test import gen_cluster
from distributed import wait
from dask_cuda.cuda_worker import CUDAWorker
from dask_cuda.utils_test import assert_device_host_file_size as assert_dhf_size
import dask.array as da


@gen_cluster(
    client=True,
    ncores=[("127.0.0.1", 1)],
    Worker=CUDAWorker,
    worker_kwargs={"device_memory_limit": 2e9, "memory_limit": 1e9},
    config={
        "cuda.worker.device-memory.target": 0.2,
        "cuda.worker.device-memory.spill": 0.2
    },
)
def test_device_spill(client, scheduler, worker):
    import cupy
    cupy.cuda.set_allocator(None)
    cupy.cuda.set_pinned_memory_allocator(None)

    rs = da.random.RandomState(RandomState=cupy.random.RandomState)
    x = rs.random(int(200e6 * 0.7), chunks=5e6)
    yield wait(x)

    xx = x.persist()
    yield wait(xx)

    # Allow up to 1024 bytes overhead per chunk on disk
    assert_dhf_size(worker.data, x.nbytes, 1024)

    # At this point, some data must be on the host, but not necessarily
    # on device or disk
    assert worker.data.host.fast.total_weight > 0

    y = client.compute(x.sum())
    res = yield y

    assert (abs(res / x.size) - 0.5) < 1e-3

    assert_dhf_size(worker.data, x.nbytes, 1024)

    # Here no data should be on the device (may contain a small
    # 'finalize' task only, < 64 bytes), both host and disk should
    # contain some data
    assert worker.data.device.fast.total_weight < 64
    assert worker.data.host.fast.total_weight > 0
    assert len(worker.data.host.slow) > 0
