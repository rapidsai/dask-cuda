from .local_cuda_cluster import LocalCUDACluster
from .dgx import DGX
from .scheduler import Scheduler

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
