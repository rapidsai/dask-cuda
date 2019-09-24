import toolz
import os
import math
import numpy as np
from multiprocessing import cpu_count
from numba import cuda
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetCpuAffinity


class CPUAffinity:
    def __init__(self, cores):
        self.cores = cores

    def setup(self, worker=None):
        print("before", os.sched_getaffinity(0))
        os.sched_setaffinity(0, self.cores)
        print("after ", os.sched_getaffinity(0))


def bitmask_to_list(x, mask_bits=64):
    res = []

    for i, mask in enumerate(x):
        if not isinstance(mask, int):
            raise TypeError("All elements of the list `x` must be integers")

        cpu_offset = i * mask_bits

        bytestr = np.frombuffer(
            bytes(np.binary_repr(mask, width=mask_bits), "utf-8"), "u1"
        )
        mask = np.flip(bytestr - ord("0")).astype(np.bool)
        affinity = np.where(
            mask, np.arange(mask_bits) + cpu_offset, np.full(mask_bits, -1)
        )

        res += affinity[(affinity >= 0)].tolist()

    return res


@toolz.memoize
def get_cpu_count():
    return cpu_count()


def get_cpu_affinity_list(device_index):
    nvmlInit()

    # Result is a list of 64-bit integers, thus ceil(CPU_COUNT / 64)
    affinity = nvmlDeviceGetCpuAffinity(
        nvmlDeviceGetHandleByIndex(device_index), math.ceil(get_cpu_count() / 64)
    )
    return bitmask_to_list(affinity)


def get_n_gpus():
    try:
        return len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    except KeyError:
        return _n_gpus_from_nvidia_smi()


@toolz.memoize
def _n_gpus_from_nvidia_smi():
    return len(os.popen("nvidia-smi -L").read().strip().split("\n"))


def get_device_total_memory(index=0):
    """
    Return total memory of CUDA device with index
    """
    with cuda.gpus[index]:
        return cuda.current_context().get_memory_info()[1]
