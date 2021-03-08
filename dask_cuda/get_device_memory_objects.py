from typing import Any, Set

from dask.sizeof import sizeof
from dask.utils import Dispatch

dispatch = Dispatch(name="get_device_memory_objects")


def get_device_memory_objects(obj: Any) -> Set:
    """ Find all CUDA device objects in `obj`

    Search through `obj` and find all CUDA device objects, which are objects
    that either are known to `dispatch` or implement `__cuda_array_interface__`.

    Notice, the CUDA device objects must be hashable.

    Parameters
    ----------
    obj: Any
        Object to search through

    Returns
    -------
    ret: set
        Set of CUDA device memory objects
    """
    return set(dispatch(obj))


@dispatch.register(object)
def get_device_memory_objects_default(obj):
    if hasattr(obj, "_obj_pxy"):
        if obj._obj_pxy["serializers"] is None:
            return dispatch(obj._obj_pxy["obj"])
        else:
            return []
    if hasattr(obj, "data"):
        return dispatch(obj.data)
    if hasattr(obj, "_owner") and obj._owner is not None:
        return dispatch(obj._owner)
    if hasattr(obj, "__cuda_array_interface__"):
        return [obj]
    return []


@dispatch.register(list)
@dispatch.register(tuple)
@dispatch.register(set)
@dispatch.register(frozenset)
def get_device_memory_objects_python_sequence(seq):
    ret = []
    for s in seq:
        ret.extend(dispatch(s))
    return ret


@dispatch.register(dict)
def get_device_memory_objects_python_dict(seq):
    ret = []
    for s in seq.values():
        ret.extend(dispatch(s))
    return ret


@dispatch.register_lazy("cupy")
def get_device_memory_objects_register_cupy():
    from cupy.cuda.memory import MemoryPointer

    @dispatch.register(MemoryPointer)
    def get_device_memory_objects_cupy(obj):
        return [obj.mem]


@dispatch.register_lazy("cudf")
def get_device_memory_objects_register_cudf():
    import cudf.core.dataframe
    import cudf.core.index
    import cudf.core.multiindex
    import cudf.core.series

    @dispatch.register(cudf.core.dataframe.DataFrame)
    def get_device_memory_objects_cudf_dataframe(obj):

        ret = dispatch(obj._index)
        for col in obj._data.columns:
            ret += dispatch(col)
        return ret

    @dispatch.register(cudf.core.series.Series)
    def get_device_memory_objects_cudf_series(obj):
        return dispatch(obj._index) + dispatch(obj._column)

    @dispatch.register(cudf.core.index.RangeIndex)
    def get_device_memory_objects_cudf_range_index(obj):
        # Avoid materializing RangeIndex. This introduce some inaccuracies
        # in total device memory usage but we accept the memory use of
        # RangeIndexes are limited.
        return []

    @dispatch.register(cudf.core.index.Index)
    def get_device_memory_objects_cudf_index(obj):
        return dispatch(obj._values)

    @dispatch.register(cudf.core.multiindex.MultiIndex)
    def get_device_memory_objects_cudf_multiindex(obj):
        return dispatch(obj._columns)


@sizeof.register_lazy("cupy")
def register_cupy():  # NB: this overwrites dask.sizeof.register_cupy()
    import cupy.cuda.memory

    @sizeof.register(cupy.cuda.memory.BaseMemory)
    def sizeof_cupy_base_memory(x):
        return int(x.size)

    @sizeof.register(cupy.ndarray)
    def sizeof_cupy_ndarray(x):
        return int(x.nbytes)
