import dask
import dask.dataframe.core
import dask.dataframe.shuffle

from ._version import get_versions
from .cuda_worker import CUDAWorker
from .explicit_comms.dataframe.shuffle import get_rearrange_by_column_tasks_wrapper
from .proxify_device_objects import proxify_decorator, unproxify_decorator
from .local_cuda_cluster import LocalCUDACluster

__version__ = get_versions()["version"]
del get_versions


# Monkey patching Dask to make use of explicit-comms when `DASK_EXPLICIT_COMMS=True`
dask.dataframe.shuffle.rearrange_by_column_tasks = get_rearrange_by_column_tasks_wrapper(
    dask.dataframe.shuffle.rearrange_by_column_tasks
)


# Monkey patching Dask to make use of proxify and unproxify in compatibility mode
dask.dataframe.shuffle.shuffle_group = proxify_decorator(
    dask.dataframe.shuffle.shuffle_group
)
dask.dataframe.core._concat = unproxify_decorator(dask.dataframe.core._concat)
