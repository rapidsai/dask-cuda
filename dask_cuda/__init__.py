import sys

if sys.platform != "linux":
    raise ImportError("Only Linux is supported by Dask-CUDA at this time")

import dask

try:
    import dask_cuda._dataframe
except ImportError:
    # Dataframe requirements not installed
    pass

try:
    import dask_cuda._bag
except ImportError:
    # Bag requirements not installed
    pass


from ._version import __git_commit__, __version__
from .cuda_worker import CUDAWorker
from .local_cuda_cluster import LocalCUDACluster
