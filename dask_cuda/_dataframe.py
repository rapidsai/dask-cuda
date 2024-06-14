import dask.dataframe as dd
from dask import config

from .explicit_comms.dataframe.shuffle import (
    get_default_shuffle_method,
    get_rearrange_by_column_wrapper,
)
from .proxify_device_objects import proxify_decorator, unproxify_decorator

if config.get("dataframe.query-planning", None) is not False and config.get(
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
dd.shuffle.rearrange_by_column = get_rearrange_by_column_wrapper(
    dd.shuffle.rearrange_by_column
)
# We have to replace all modules that imports Dask's `get_default_shuffle_method()`
# TODO: introduce a shuffle-algorithm dispatcher in Dask so we don't need this hack
dd.shuffle.get_default_shuffle_method = get_default_shuffle_method
dd.multi.get_default_shuffle_method = get_default_shuffle_method


# Monkey patching Dask to make use of proxify and unproxify in compatibility mode
dd.shuffle.shuffle_group = proxify_decorator(dd.shuffle.shuffle_group)
dd.core._concat = unproxify_decorator(dd.core._concat)
