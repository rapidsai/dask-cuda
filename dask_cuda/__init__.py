from .local_cuda_cluster import LocalCUDACluster
from .dgx import DGX

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
