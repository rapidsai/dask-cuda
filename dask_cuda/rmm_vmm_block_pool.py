from typing import Dict, List, Set, Tuple

from cuda import cuda, cudart

from dask_cuda.rmm_vmm_pool import (
    VmmHeap,
    check_vmm_gdr_support,
    check_vmm_support,
    checkCudaErrors,
    get_granularity,
    to_aligned_size,
)

CU_VMM_SUPPORTED = (
    cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED
)
CU_VMM_GDR_SUPPORTED = (
    cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED
)
CU_VMM_GRANULARITY = (
    cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM
)


class VmmBlockPool:
    def __init__(self) -> None:
        self._block_size: int = 0

        self._heaps: Dict[cuda.CUdevice, VmmHeap] = {}

        self._mem_handles: Dict[int, cuda.CUmemGenericAllocationHandle] = {}

        self._store_block: Dict[int, int] = {}
        self._block_to_mem_handle: Dict[int, int] = {}
        self._free_blocks: Set[int] = set()
        self._used_blocks: Set[int] = set()

        self._store_user: Dict[int, int] = {}
        self._store_user_blocks: Dict[int, List[Tuple[int]]] = {}

        self._allocate_blocks()

    def __del__(self) -> None:
        if len(self._store_user) > 0:
            print(f"WARN: {len(self._store_user)} user pointers still allocated")
        if len(self._used_blocks) > 0:
            print(f"WARN: {len(self._used_blocks)} blocks still in use")

        checkCudaErrors(cudart.cudaGetDevice())  # TODO: avoid use of the Runtime API
        checkCudaErrors(cuda.cuStreamSynchronize(cuda.CUstream(0)))

        while len(self._store_user) > 0:
            ptr, alloc_size = self._store_user.popitem()
            checkCudaErrors(cuda.cuMemUnmap(cuda.CUdeviceptr(ptr), alloc_size))

        while len(self._store_block) > 0:
            ptr, alloc_size = self._store_block.popitem()
            checkCudaErrors(cuda.cuMemUnmap(cuda.CUdeviceptr(ptr), alloc_size))

        self._free_blocks.clear()
        self._used_blocks.clear()
        self._block_to_mem_handle.clear()

        while len(self._mem_handles) > 0:
            _, mem_handle = self._mem_handles.popitem()
            checkCudaErrors(cuda.cuMemRelease(mem_handle))

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

    def _allocate_blocks(self, block_size: int = 134217728) -> None:
        heap = self.get_heap()
        block_size = to_aligned_size(block_size, heap.granularity)
        self._block_size = block_size

        # Allocate physical memory
        allocation_prop = cuda.CUmemAllocationProp()
        allocation_prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        allocation_prop.location.type = (
            cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        )
        allocation_prop.location.id = heap.device

        # Enable IB/GDRCopy support if available
        try:
            check_vmm_gdr_support(self.get_device())
        except ValueError:
            pass
        else:
            allocation_prop.allocFlags.gpuDirectRDMACapable = 1

        # Pre-allocate ~30 GiB
        # TODO: Replace by user-input factor based on GPU size.
        for i in range(240):
            mem_handle = checkCudaErrors(
                cuda.cuMemCreate(block_size, allocation_prop, 0)
            )

            # Map physical memory to the heap
            block_reserve_ptr = heap.allocate(block_size)
            checkCudaErrors(
                cuda.cuMemMap(
                    ptr=block_reserve_ptr,
                    size=block_size,
                    offset=0,
                    handle=mem_handle,
                    flags=0,
                )
            )
            # print(f"block_reserve_ptr: {hex(int(block_reserve_ptr))}")

            # Specify both read and write access.
            access_desc = cuda.CUmemAccessDesc()
            access_desc.location.type = (
                cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
            )
            access_desc.location.id = heap.device
            access_desc.flags = (
                cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
            )
            checkCudaErrors(
                cuda.cuMemSetAccess(
                    ptr=block_reserve_ptr, size=block_size, desc=[access_desc], count=1
                )
            )

            self._mem_handles[int(mem_handle)] = mem_handle
            self._block_to_mem_handle[int(block_reserve_ptr)] = int(mem_handle)
            self._store_block[int(block_reserve_ptr)] = block_size
            self._free_blocks.add(int(block_reserve_ptr))

    def _take_block(self) -> int:
        block = self._free_blocks.pop()
        self._used_blocks.add(block)
        return block

    def _release_block(self, ptr: int) -> None:
        self._used_blocks.remove(ptr)
        self._free_blocks.add(ptr)

    def _get_block_mem_handle(
        self, block_ptr: int
    ) -> cuda.CUmemGenericAllocationHandle:
        return self._mem_handles[self._block_to_mem_handle[block_ptr]]

    def allocate(self, size: int) -> int:
        heap = self.get_heap()
        alloc_size = to_aligned_size(size, heap.granularity)

        # Map physical memory to the heap
        reserve_ptr = heap.allocate(alloc_size)
        # print(f"user reserve_ptr: {hex(int(reserve_ptr))}")

        used_blocks: List[Tuple[int, int]] = []

        total_allocated_size = 0
        while total_allocated_size < alloc_size:
            offset = total_allocated_size

            block = self._take_block()
            block_size = self._store_block[block]
            block_size = min(block_size, alloc_size - total_allocated_size)
            used_blocks.append((block, block_size))
            mem_handle = self._get_block_mem_handle(block)

            total_allocated_size += block_size
            # print(total_allocated_size, alloc_size, block_size)

            checkCudaErrors(
                cuda.cuMemMap(
                    ptr=cuda.CUdeviceptr(int(reserve_ptr) + offset),
                    size=block_size,
                    offset=0,
                    handle=mem_handle,
                    flags=0,
                )
            )

        # Specify both read and write access.
        prop = cuda.CUmemAccessDesc()
        prop.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = heap.device
        prop.flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        checkCudaErrors(
            cuda.cuMemSetAccess(ptr=reserve_ptr, size=alloc_size, desc=[prop], count=1)
        )

        self._store_user[int(reserve_ptr)] = alloc_size
        self._store_user_blocks[int(reserve_ptr)] = used_blocks
        # print(f"alloc({int(reserve_ptr)}) - size: {size}, alloc_size: {alloc_size}")
        return int(reserve_ptr)

    def get_allocation_blocks(self, ptr: int) -> List[Tuple[int, int]]:
        return self._store_user_blocks[ptr]

    def deallocate(self, ptr: int, size: int) -> None:
        checkCudaErrors(cudart.cudaGetDevice())  # TODO: avoid use of the Runtime API
        alloc_size = self._store_user.pop(ptr)
        used_blocks = self._store_user_blocks.pop(ptr)
        # print(
        #     f"deallocating {len(used_blocks)} blocks from user allocation "
        #     f"{hex(int(ptr))}"
        # )

        checkCudaErrors(cuda.cuStreamSynchronize(cuda.CUstream(0)))
        for block in used_blocks:
            block_ptr, block_size = block
            # print(f"deallocating: {hex(int(ptr))}, {block_size}")

            checkCudaErrors(cuda.cuMemUnmap(cuda.CUdeviceptr(ptr), alloc_size))

            self._release_block(block_ptr)

            # print(f"deloc({int(ptr)}) - size: {size}, alloc_size: {alloc_size}")
