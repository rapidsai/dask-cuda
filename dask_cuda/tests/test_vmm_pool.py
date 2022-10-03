import numpy as np
import pytest
from cuda import cuda

from dask_cuda.vmm_pool import VmmPool, checkCudaErrors


@pytest.mark.parametrize(
    "size", [1, 1000, 100 * 1024**2, 100 * 1024**2 + 1, int(200e6)]
)
def test_allocate(size):
    pool = VmmPool()

    buf = pool.allocate(size)
    assert isinstance(buf, int)
    assert buf > 0
    pool.deallocate(buf, size)


@pytest.mark.parametrize(
    "size", [1, 1000, 100 * 1024**2, 100 * 1024**2 + 1, int(200e6)]
)
def test_mapping_content(size):
    pool = VmmPool()

    d_buf = pool.allocate(size)

    h_buf_in = np.arange(size, dtype="u1")
    checkCudaErrors(
        cuda.cuMemcpyHtoDAsync(
            cuda.CUdeviceptr(d_buf), h_buf_in.ctypes.data, size, cuda.CUstream(0)
        )
    )

    h_buf_out = np.empty(size, dtype="u1")
    checkCudaErrors(
        cuda.cuMemcpyDtoHAsync(
            h_buf_out.ctypes.data, cuda.CUdeviceptr(d_buf), size, cuda.CUstream(0)
        )
    )

    checkCudaErrors(cuda.cuStreamSynchronize(cuda.CUstream(0)))
    pool.deallocate(d_buf, size)

    np.testing.assert_equal(h_buf_out, h_buf_in)


@pytest.mark.parametrize(
    "size", [1, 1000, 64 * 1024**2, 64 * 1024**2 + 1, int(200e6)]
)
def test_block_content(size):
    pool = VmmPool()

    d_buf = pool.allocate(size)
    vmm_alloc = pool._allocs[d_buf]

    h_buf_in = np.arange(size, dtype="u1")
    offset = 0
    block_num = 0
    while offset < size:
        block = vmm_alloc.blocks[block_num]
        block_size = min(block.size, size - offset)
        checkCudaErrors(
            cuda.cuMemcpyHtoDAsync(
                cuda.CUdeviceptr(block._ptr),
                h_buf_in.ctypes.data + offset,
                block_size,
                cuda.CUstream(0),
            )
        )

        h_buf_out = np.empty(block_size, dtype="u1")
        checkCudaErrors(
            cuda.cuMemcpyDtoHAsync(
                h_buf_out.ctypes.data,
                cuda.CUdeviceptr(block._ptr),
                block_size,
                cuda.CUstream(0),
            )
        )

        checkCudaErrors(cuda.cuStreamSynchronize(cuda.CUstream(0)))
        np.testing.assert_equal(h_buf_out, h_buf_in[offset : offset + block_size])

        offset += block_size

    pool.deallocate(d_buf, size)
