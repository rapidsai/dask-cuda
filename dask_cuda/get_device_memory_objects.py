from typing import Set

from dask.sizeof import sizeof
from dask.utils import Dispatch

dispatch = Dispatch(name="get_device_memory_objects")


class DeviceMemoryId:
    """ID and size of device memory objects

    Instead of keeping a reference to device memory objects this class
    only saves the id and size in order to avoid delayed freeing.
    """

    def __init__(self, obj: object):
        self.id = id(obj)
        self.nbytes = sizeof(obj)

    def __hash__(self) -> int:
        return self.id

    def __eq__(self, o) -> bool:
        return self.id == hash(o)


def get_device_memory_ids(obj) -> Set[DeviceMemoryId]:
    """Find all CUDA device objects in `obj`

    Search through `obj` and find all CUDA device objects, which are objects
    that either are known to `dispatch` or implement `__cuda_array_interface__`.

    Parameters
    ----------
    obj: Any
        Object to search through

    Returns
    -------
    ret: Set[DeviceMemoryId]
        Set of CUDA device memory IDs
    """
    return {DeviceMemoryId(o) for o in dispatch(obj)}


@dispatch.register(object)
def get_device_memory_objects_default(obj):
    from dask_cuda.proxy_object import ProxyObject

    if isinstance(obj, ProxyObject):
        return dispatch(obj._pxy_get().obj)
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
    import cudf.core.frame
    import cudf.core.index
    import cudf.core.multiindex
    import cudf.core.series

    @dispatch.register(cudf.core.frame.Frame)
    def get_device_memory_objects_cudf_frame(obj):
        ret = []
        for col in obj._data.columns:
            ret += dispatch(col)
        return ret

    @dispatch.register(cudf.core.indexed_frame.IndexedFrame)
    def get_device_memory_objects_cudf_indexed_frame(obj):
        return dispatch(obj._index) + get_device_memory_objects_cudf_frame(obj)

    @dispatch.register(cudf.core.series.Series)
    def get_device_memory_objects_cudf_series(obj):
        return dispatch(obj._index) + dispatch(obj._column)

    @dispatch.register(cudf.core.index.RangeIndex)
    def get_device_memory_objects_cudf_range_index(obj):
        # Avoid materializing RangeIndex. This introduce some inaccuracies
        # in total device memory usage, which we accept because the memory
        # use of RangeIndexes are limited.
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
