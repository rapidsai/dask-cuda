import toolz
import os
from numba import cuda


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


def close_cuda_context():
    return cuda.close()
