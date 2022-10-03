import math
import traceback
from collections import defaultdict
from dataclasses import dataclass
from threading import Lock
from typing import DefaultDict, Dict
from weakref import WeakValueDictionary

from black import List
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


def virtual_memory_reserve(size: int) -> int:
    # Make a virtual memory reservation.
    result = cuda.cuMemAddressReserve(
        size=size, alignment=0, addr=cuda.CUdeviceptr(0), flags=0
    )
    # We check the result manually to avoid raise MemoryError, which would
    # trigger out-of-memory handling.
    if result[0] == cuda.CUresult.CUDA_ERROR_OUT_OF_MEMORY:
        raise RuntimeError("cuda.cuMemAddressReserve() - CUDA_ERROR_OUT_OF_MEMORY")
    return int(checkCudaErrors(result))


def virtual_memory_set_access(ptr: int, size: int, device: cuda.CUdevice):
    # Specify both read and write access.
    prop = cuda.CUmemAccessDesc()
    prop.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = device
    prop.flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    checkCudaErrors(
        cuda.cuMemSetAccess(ptr=cuda.CUdeviceptr(ptr), size=size, desc=[prop], count=1)
    )


class VmmBlock:
    size: int

    def __init__(self, device: cuda.CUdevice, size: int) -> None:
        self._device = device
        self.size = size

        # Allocate physical memory
        prop = cuda.CUmemAllocationProp()
        prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = device
        self.mem_handle = checkCudaErrors(cuda.cuMemCreate(size, prop, 0))

        # Make a virtual memory reservation.
        self._ptr = virtual_memory_reserve(size)

        # Map physical memory to the virtual memory
        checkCudaErrors(
            cuda.cuMemMap(
                ptr=cuda.CUdeviceptr(self._ptr),
                size=size,
                offset=0,
                handle=self.mem_handle,
                flags=0,
            )
        )

        # Specify both read and write access.
        virtual_memory_set_access(ptr=self._ptr, size=size, device=device)

    def __repr__(self) -> str:
        return f"<VmmBlock size={self.size}>"


@dataclass
class DeviceInfo:
    device: cuda.CUdevice
    granularity: int


@dataclass
class VmmAlloc:
    size: int
    blocks: List[VmmBlock]
    ptr: int


class VmmPool:
    def __init__(self) -> None:
        self._max_block_size = 2**29  # 512MiB
        self._pool: DefaultDict[int, List[VmmBlock]] = defaultdict(list)
        self._allocs: Dict[int, VmmAlloc] = {}
        self._lock = Lock()

    def clear(self) -> None:
        with self._lock:
            # Free all blocks in the pool
            for blocks in self._pool.values():
                for block in blocks:
                    checkCudaErrors(
                        cuda.cuMemUnmap(cuda.CUdeviceptr(block._ptr), block.size)
                    )
                    # checkCudaErrors(cuda.cuMemAddressFree(alloc.ptr, alloc.size))

    def __del__(self):
        self.clear()

    def get_device_info(self) -> DeviceInfo:
        checkCudaErrors(cudart.cudaGetDevice())  # TODO: avoid use of the Runtime API
        device = checkCudaErrors(cuda.cuCtxGetDevice())
        granularity = get_granularity(device)
        return DeviceInfo(device=device, granularity=granularity)

    def get_block(self, size: int, dev_info: DeviceInfo) -> VmmBlock:
        # The block size is the highest power of 2 smaller than `size`
        block_size = 2 ** int(math.log2(size))
        # but not larger than the maximum block size
        block_size = min(block_size, self._max_block_size)
        # and not smaller than the granularity
        block_size = max(block_size, dev_info.granularity)
        # print(f"get_block({size}) - block_size: {block_size}")

        blocks = self._pool.get(block_size, [])
        if blocks:
            return blocks.pop()
        return VmmBlock(device=dev_info.device, size=block_size)

    def get_blocks(self, min_size: int, dev_info: DeviceInfo) -> List[VmmBlock]:
        cur_size = min_size
        ret = []
        while cur_size > 0:
            block = self.get_block(size=cur_size, dev_info=dev_info)
            cur_size -= block.size
            ret.append(block)
        return ret

    def allocate(self, size: int) -> int:
        with self._lock:
            dev_info = self.get_device_info()
            alloc_size = to_aligned_size(size, granularity=dev_info.granularity)
            blocks = self.get_blocks(min_size=alloc_size, dev_info=dev_info)
            ptr = virtual_memory_reserve(alloc_size)
            print(
                f"allocate({hex(ptr)}) - size: {size}, alloc_size: {alloc_size}, "
                f"blocks: {blocks}"
            )

            # Map the physical memory of each block to the virtual memory
            cur_ptr = ptr
            for block in blocks:
                checkCudaErrors(
                    cuda.cuMemMap(
                        ptr=cuda.CUdeviceptr(cur_ptr),
                        size=block.size,
                        offset=0,
                        handle=block.mem_handle,
                        flags=0,
                    )
                )
                cur_ptr += block.size

            # Specify both read and write access.
            virtual_memory_set_access(ptr=ptr, size=alloc_size, device=dev_info.device)

            self._allocs[ptr] = VmmAlloc(blocks=blocks, size=alloc_size, ptr=ptr)
            return ptr

    def deallocate(self, ptr: int, size: int) -> None:
        with self._lock:
            checkCudaErrors(cudart.cudaGetDevice())
            checkCudaErrors(cuda.cuStreamSynchronize(cuda.CUstream(0)))
            alloc = self._allocs.pop(ptr)
            assert alloc.ptr == ptr

            print(
                f"delocate({hex(ptr)}) - size: {size}, alloc_size: {alloc.size}, "
                f"blocks: {alloc.blocks}"
            )

            # Move all blocks of the allocation to the pool
            for block in alloc.blocks:
                self._pool[block.size].append(block)
                checkCudaErrors(
                    cuda.cuMemUnmap(cuda.CUdeviceptr(block._ptr), block.size)
                )

            # # Free up the previously reserved virtual memory
            # checkCudaErrors(cuda.cuMemAddressFree(alloc.ptr, alloc.size))


_vmm_pools = WeakValueDictionary()


def rmm_get_current_vmm_pool() -> VmmPool:
    def get_stack(mr):
        if hasattr(mr, "upstream_mr"):
            return [mr] + get_stack(mr.upstream_mr)
        return [mr]

    print(
        "rmm_get_current_vmm_pool() - stack: ",
        get_stack(rmm.mr.get_current_device_resource()),
    )
    for mr in get_stack(rmm.mr.get_current_device_resource()):
        if id(mr) in _vmm_pools:
            return _vmm_pools[id(mr)]
    raise ValueError()


def rmm_set_current_vmm_pool(skip_if_exist=True) -> None:

    try:
        rmm_get_current_vmm_pool()
    except ValueError:
        pass
    else:
        if skip_if_exist:
            return
        raise ValueError("A VMM pool already set")

    vmm_pool = VmmPool()

    def allocate(size: int):
        return vmm_pool.allocate(size)

    def deallocate(ptr: int, size: int):
        vmm_pool.deallocate(ptr, size)

    mr = rmm.mr.CallbackMemoryResource(allocate, deallocate)
    _vmm_pools[id(mr)] = vmm_pool
    rmm.mr.set_current_device_resource(mr)
