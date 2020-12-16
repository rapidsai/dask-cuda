from ._version import get_versions
from .cuda_worker import CUDAWorker
from .local_cuda_cluster import LocalCUDACluster

__version__ = get_versions()["version"]
del get_versions
