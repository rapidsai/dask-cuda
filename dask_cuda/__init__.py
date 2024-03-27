import sys

if sys.platform != "linux":
    raise ImportError("Only Linux is supported by Dask-CUDA at this time")

import dask
import dask.utils
import dask.dataframe.core
import dask.dataframe.shuffle
import dask.dataframe.multi
import dask.bag.core

from ._version import __git_commit__, __version__
from .cuda_worker import CUDAWorker
from .explicit_comms.dataframe.shuffle import (
    get_rearrange_by_column_wrapper,
    get_default_shuffle_method,
)
from .local_cuda_cluster import LocalCUDACluster
from .proxify_device_objects import proxify_decorator, unproxify_decorator


if dask.config.get("dataframe.query-planning", None) is not False and dask.config.get(
    "explicit-comms", False
):
    raise NotImplementedError(
        "The 'explicit-comms' config is not yet supported when "
        "query-planning is enabled in dask. Please use the shuffle "
        "API directly, or use the legacy dask-dataframe API "
        "(set the 'dataframe.query-planning' config to `False`"
        "before importing `dask.dataframe`).",
    )


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
