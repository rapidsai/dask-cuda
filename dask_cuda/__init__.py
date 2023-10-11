import sys

if sys.platform != "linux":
    raise ImportError("Only Linux is supported by Dask-CUDA at this time")


import dask
import dask.utils
import dask.dataframe.core
import dask.dataframe.shuffle
import dask.dataframe.multi
import dask.bag.core

from .cuda_worker import CUDAWorker
from .explicit_comms.dataframe.shuffle import (
    get_rearrange_by_column_wrapper,
    get_default_shuffle_method,
)
from .local_cuda_cluster import LocalCUDACluster
from .proxify_device_objects import proxify_decorator, unproxify_decorator

__version__ = "23.10.00"

from . import compat

# Monkey patching Dask to make use of explicit-comms when `DASK_EXPLICIT_COMMS=True`
dask.dataframe.shuffle.rearrange_by_column = get_rearrange_by_column_wrapper(
    dask.dataframe.shuffle.rearrange_by_column
)
# We have to replace all modules that imports Dask's `get_default_shuffle_method()`
# TODO: introduce a shuffle-algorithm dispatcher in Dask so we don't need this hack
dask.dataframe.shuffle.get_default_shuffle_method = get_default_shuffle_method
dask.dataframe.multi.get_default_shuffle_method = get_default_shuffle_method
dask.bag.core.get_default_shuffle_method = get_default_shuffle_method


# Monkey patching Dask to make use of proxify and unproxify in compatibility mode
dask.dataframe.shuffle.shuffle_group = proxify_decorator(
    dask.dataframe.shuffle.shuffle_group
)
dask.dataframe.core._concat = unproxify_decorator(dask.dataframe.core._concat)
