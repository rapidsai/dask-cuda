import toolz
import os


def get_n_gpus():
    try:
        return len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    except KeyError:
        return _n_gpus_from_nvidia_smi()


@toolz.memoize
def _n_gpus_from_nvidia_smi():
    return len(os.popen("nvidia-smi -L").read().strip().split("\n"))
