from cuda import cuda, cudart, nvrtc

import rmm.mr

CU_VMM_SUPPORTED = (
    cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED
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
    err = result[0]
    if err != cuda.CUresult.CUDA_SUCCESS:
        msg = "CUDA error code={}({})".format(err.value, _cudaGetErrorEnum(err))
        if err == cuda.CUresult.CUDA_ERROR_OUT_OF_MEMORY:
            raise MemoryError(msg)
        raise RuntimeError(msg)

    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]


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
        if not checkCudaErrors(
            cuda.cuDeviceGetAttribute(
                CU_VMM_SUPPORTED,
                device,
            )
        ):
            raise ValueError("Device doesn't support VIRTUAL ADDRESS MANAGEMENT")

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
        if not checkCudaErrors(
            cuda.cuDeviceGetAttribute(
                CU_VMM_SUPPORTED,
                device,
            )
        ):
            raise ValueError("Device doesn't support VIRTUAL ADDRESS MANAGEMENT")

        alloc_size = to_aligned_size(size, get_granularity(device))

        prop = cuda.CUmemAllocationProp()
        prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = device
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


def set_vmm_pool():
    #store = RegularMemAlloc()
    store = VmmAlloc()

    def allocate(size: int):
        return store.allocate(size)

    def deallocate(ptr: int, size: int):
        store.deallocate(ptr, size)

    rmm.mr.set_current_device_resource(
        rmm.mr.CallbackMemoryResource(allocate, deallocate)
    )
