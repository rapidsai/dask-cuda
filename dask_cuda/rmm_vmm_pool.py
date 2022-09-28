import traceback
from typing import Dict

from cuda import cuda, cudart, nvrtc

import rmm.mr

CU_VMM_SUPPORTED = (
    cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED
)
CU_VMM_GDR_SUPPORTED = (
    cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED
)
CU_VMM_GRANULARITY = (
    cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM
)


def _cudaGetErrorEnum(error):
    if isinstance(error, cuda.CUresult):
        err, name = cuda.cuGetErrorName(error)
        return name if err == cuda.CUresult.CUDA_SUCCESS else "<unknown>"
    elif isinstance(error, cudart.cudaError_t):
        return cudart.cudaGetErrorName(error)[1]
    elif isinstance(error, nvrtc.nvrtcResult):
        return nvrtc.nvrtcGetErrorString(error)[1]
    else:
        raise RuntimeError("Unknown error type: {}".format(error))


def checkCudaErrors(result):
    if not result[0]:
        err = result[0]
        msg = "CUDA error code={}({})".format(err.value, _cudaGetErrorEnum(err))
        if err == cuda.CUresult.CUDA_ERROR_OUT_OF_MEMORY:
            raise MemoryError(msg)
        raise RuntimeError(
            msg + "\n" + "\n".join(str(frame) for frame in traceback.extract_stack())
        )

    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]


def _check_support(dev: cuda.CUdevice, attr: cuda.CUdevice_attribute, msg: str) -> None:
    if not checkCudaErrors(
        cuda.cuDeviceGetAttribute(
            attr,
            dev,
        )
    ):
        raise ValueError(msg)


def check_vmm_support(dev: cuda.CUdevice) -> None:
    return _check_support(
        dev,
        CU_VMM_SUPPORTED,
        f"Device {dev} doesn't support VIRTUAL ADDRESS MANAGEMENT",
    )


def check_vmm_gdr_support(dev: cuda.CUdevice) -> None:
    return _check_support(
        dev,
        CU_VMM_GDR_SUPPORTED,
        f"Device {dev} doesn't support GPUDirectRDMA for VIRTUAL ADDRESS MANAGEMENT",
    )


def get_granularity(dev: cuda.CUdevice):
    prop = cuda.CUmemAllocationProp()
    prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = dev
    return checkCudaErrors(cuda.cuMemGetAllocationGranularity(prop, CU_VMM_GRANULARITY))


def to_aligned_size(size: int, granularity: int):
    rest = size % granularity
    if rest != 0:
        size = size + (granularity - rest)
    assert size % granularity == 0
    return size


class RegularMemAlloc:
    def allocate(self, size: int) -> int:
        checkCudaErrors(cudart.cudaGetDevice())  # TODO: avoid use of the Runtime API
        device = checkCudaErrors(cuda.cuCtxGetDevice())

        # Check that the selected device supports virtual address management
        check_vmm_support(device)

        alloc_size = to_aligned_size(size, get_granularity(device))
        return int(checkCudaErrors(cuda.cuMemAlloc(alloc_size)))

    def deallocate(self, ptr: int, size: int) -> None:
        checkCudaErrors(cuda.cuMemFree(ptr))


class VmmAlloc:
    def __init__(self) -> None:
        self._store = {}

    def allocate(self, size: int) -> int:
        checkCudaErrors(cudart.cudaGetDevice())  # TODO: avoid use of the Runtime API
        device = checkCudaErrors(cuda.cuCtxGetDevice())

        # Check that the selected device supports virtual address management
        check_vmm_support(device)

        alloc_size = to_aligned_size(size, get_granularity(device))

        prop = cuda.CUmemAllocationProp()
        prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = device

        # Enable IB/GDRCopy support if available
        try:
            check_vmm_gdr_support(device)
        except ValueError:
            pass
        else:
            prop.allocFlags.gpuDirectRDMACapable = 1

        mem_handle = checkCudaErrors(cuda.cuMemCreate(alloc_size, prop, 0))

        reserve_ptr = checkCudaErrors(
            cuda.cuMemAddressReserve(
                size=alloc_size, alignment=0, addr=cuda.CUdeviceptr(0), flags=0
            )
        )

        checkCudaErrors(
            cuda.cuMemMap(
                ptr=reserve_ptr, size=alloc_size, offset=0, handle=mem_handle, flags=0
            )
        )

        # Since we do not need to make any other mappings of this memory or export it,
        # we no longer need and can release the mem_alloc.
        # The allocation will be kept live until it is unmapped.
        checkCudaErrors(cuda.cuMemRelease(mem_handle))

        # Specify both read and write access.
        prop = cuda.CUmemAccessDesc()
        prop.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = device
        prop.flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        checkCudaErrors(
            cuda.cuMemSetAccess(ptr=reserve_ptr, size=alloc_size, desc=[prop], count=1)
        )

        self._store[int(reserve_ptr)] = alloc_size
        return int(reserve_ptr)

    def deallocate(self, ptr: int, size: int) -> None:
        alloc_size = self._store.pop(ptr)
        checkCudaErrors(cuda.cuStreamSynchronize(cuda.CUstream_flags.CU_STREAM_DEFAULT))
        checkCudaErrors(cuda.cuMemUnmap(ptr, alloc_size))
        checkCudaErrors(cuda.cuMemAddressFree(ptr, alloc_size))


class VmmHeap:
    def __init__(
        self, device: cuda.CUdevice, granularity: int, size: int = 2**40
    ) -> None:
        self.granularity = granularity
        self.device = device
        self._offset = 0
        self._size = size

        # Make a virtual memory reservation.
        result = cuda.cuMemAddressReserve(
            size=size, alignment=0, addr=cuda.CUdeviceptr(0), flags=0
        )
        # We check the result manually to avoid raise MemoryError, which would
        # trigger out-of-memory handling.
        if result[0] == cuda.CUresult.CUDA_ERROR_OUT_OF_MEMORY:
            raise RuntimeError("cuda.cuMemAddressReserve() - CUDA_ERROR_OUT_OF_MEMORY")
        self._heap = int(checkCudaErrors(result))

    def allocate(self, size: int) -> cuda.CUdeviceptr:
        assert size % self.granularity == 0
        ret = self._heap + self._offset
        self._offset += to_aligned_size(size, self.granularity)
        assert self._offset <= self._size
        assert ret % self.granularity == 0
        return cuda.CUdeviceptr(ret)


class VmmAllocPool:
    def __init__(self) -> None:
        self._store: Dict[int, int] = {}
        self._heaps: Dict[cuda.CUdevice, VmmHeap] = {}

    def get_device(self) -> cuda.CUdevice:
        checkCudaErrors(cudart.cudaGetDevice())  # TODO: avoid use of the Runtime API
        return checkCudaErrors(cuda.cuCtxGetDevice())

    def get_heap(self) -> VmmHeap:
        device = self.get_device()

        # Notice, `hash(cuda.CUdevice(0)) != hash(cuda.CUdevice(0))` thus the
        # explicit convertion to integer.
        if int(device) not in self._heaps:
            check_vmm_support(device)
            self._heaps[int(device)] = VmmHeap(device, get_granularity(device))
        return self._heaps[int(device)]

    def allocate(self, size: int) -> int:
        heap = self.get_heap()
        alloc_size = to_aligned_size(size, heap.granularity)

        # Allocate physical memory
        prop = cuda.CUmemAllocationProp()
        prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = heap.device

        # Enable IB/GDRCopy support if available
        try:
            check_vmm_gdr_support(self.get_device())
        except ValueError:
            pass
        else:
            prop.allocFlags.gpuDirectRDMACapable = 1

        mem_handle = checkCudaErrors(cuda.cuMemCreate(alloc_size, prop, 0))

        # Map physical memory to the heap
        reserve_ptr = heap.allocate(alloc_size)
        checkCudaErrors(
            cuda.cuMemMap(
                ptr=reserve_ptr,
                size=alloc_size,
                offset=0,
                handle=mem_handle,
                flags=0,
            )
        )

        # Since we do not need to make any other mappings of this memory or export it,
        # we no longer need and can release the mem_handle.
        # The allocation will be kept live until it is unmapped.
        checkCudaErrors(cuda.cuMemRelease(mem_handle))

        # Specify both read and write access.
        prop = cuda.CUmemAccessDesc()
        prop.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = heap.device
        prop.flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        checkCudaErrors(
            cuda.cuMemSetAccess(ptr=reserve_ptr, size=alloc_size, desc=[prop], count=1)
        )

        self._store[int(reserve_ptr)] = alloc_size
        # print(f"alloc({int(reserve_ptr)}) - size: {size}, alloc_size: {alloc_size}")
        return int(reserve_ptr)

    def deallocate(self, ptr: int, size: int) -> None:
        checkCudaErrors(cudart.cudaGetDevice())  # TODO: avoid use of the Runtime API
        alloc_size = self._store.pop(ptr)
        checkCudaErrors(cuda.cuStreamSynchronize(cuda.CUstream_flags.CU_STREAM_DEFAULT))
        checkCudaErrors(cuda.cuMemUnmap(ptr, alloc_size))
        # print(f"deloc({int(ptr)}) - size: {size}, alloc_size: {alloc_size}")


def set_vmm_pool():
    # store = RegularMemAlloc()
    # store = VmmAlloc()
    store = VmmAllocPool()

    def allocate(size: int):
        return store.allocate(size)

    def deallocate(ptr: int, size: int):
        store.deallocate(ptr, size)

    rmm.mr.set_current_device_resource(
        rmm.mr.CallbackMemoryResource(allocate, deallocate)
    )
