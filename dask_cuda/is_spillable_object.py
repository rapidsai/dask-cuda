from __future__ import absolute_import, division, print_function

from typing import Optional

from dask.utils import Dispatch

is_spillable_object = Dispatch(name="is_spillable_object")


@is_spillable_object.register(list)
@is_spillable_object.register(tuple)
@is_spillable_object.register(set)
@is_spillable_object.register(frozenset)
def _(seq):
    return any([is_spillable_object(s) for s in seq])


@is_spillable_object.register(dict)
def _(seq):
    return any([is_spillable_object(s) for s in seq.items()])


@is_spillable_object.register(object)
def _(o):
    return False


@is_spillable_object.register_lazy("cudf")
def register_cudf():
    import cudf
    from cudf.core.frame import Frame

    @is_spillable_object.register(Frame)
    def is_device_object_cudf_dataframe(df):
        return cudf_spilling_status()

    @is_spillable_object.register(cudf.BaseIndex)
    def is_device_object_cudf_index(s):
        return cudf_spilling_status()


def cudf_spilling_status() -> Optional[bool]:
    """Check the status of cudf's built-in spilling

    Returns:
        - True if cudf's internal spilling is enabled, or
        - False if it is disabled, or
        - None if the current version of cudf doesn't support spilling, or
        - None if cudf isn't available.
    """
    try:
        from cudf.core.buffer.spill_manager import get_global_manager
    except ImportError:
        return None
    return get_global_manager() is not None
