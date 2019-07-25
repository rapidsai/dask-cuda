import toolz
import os
import numpy as np
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


def convert_frames_to_numba(frames):
    try:
        import cupy
    except ImportError:
        return frames

    return [cuda.as_cuda_array(f) if isinstance(f, cupy.ndarray) else f for f in frames]


def move_frames_to_host(frames):
    # Conversion of frames to Numba can/should be eventually done during the
    # serialization process in dask/distributed
    frames = convert_frames_to_numba(frames)

    return [f.copy_to_host() if isinstance(f, cuda.cudadrv.devicearray.DeviceNDArray)
            else f for f in frames]


def move_frames_to_device(frames):
    return [cuda.to_device(f) if isinstance(f, np.ndarray) else f for f in frames]
