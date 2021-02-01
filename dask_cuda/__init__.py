import dask
import dask.dataframe.shuffle
from .explicit_comms.dataframe_shuffle import get_rearrange_by_column_tasks_wrapper
from ._version import get_versions
from .cuda_worker import CUDAWorker
from .local_cuda_cluster import LocalCUDACluster

__version__ = get_versions()["version"]
del get_versions


# Monkey patching Dask to make use of explicit-comms when `DASK_EXPLICIT_COMMS=True`
dask.dataframe.shuffle.rearrange_by_column_tasks = get_rearrange_by_column_tasks_wrapper(
    dask.dataframe.shuffle.rearrange_by_column_tasks
)
