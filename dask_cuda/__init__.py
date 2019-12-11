from ._version import get_versions
from .dgx import DGX
from .local_cuda_cluster import LocalCUDACluster

__version__ = get_versions()["version"]
del get_versions
