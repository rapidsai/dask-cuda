from dask.utils import Dispatch
from dask.sizeof import sizeof


get_device_memory_objects = Dispatch(name="get_device_memory_objects")


@get_device_memory_objects.register(object)
def get_device_memory_objects_default(obj):
    if hasattr(obj, "_obj_pxy"):
        if obj._obj_pxy["serializers"] is None:
            return get_device_memory_objects(obj._obj_pxy["obj"])
        else:
            return []
    if hasattr(obj, "data"):
        return get_device_memory_objects(obj.data)
    if hasattr(obj, "_owner") and obj._owner is not None:
        return get_device_memory_objects(obj._owner)
    if hasattr(obj, "__cuda_array_interface__"):
        return [obj]
    return []


@get_device_memory_objects.register(list)
@get_device_memory_objects.register(tuple)
@get_device_memory_objects.register(set)
@get_device_memory_objects.register(frozenset)
def get_device_memory_objects_python_sequence(seq):
    ret = []
    for s in seq:
        ret.extend(get_device_memory_objects(s))
    return ret


@get_device_memory_objects.register(dict)
def get_device_memory_objects_python_dict(seq):
    ret = []
    for s in seq.values():
        ret.extend(get_device_memory_objects(s))
    return ret


@get_device_memory_objects.register_lazy("cupy")
def get_device_memory_objects_register_cupy():
    from cupy.cuda.memory import MemoryPointer

    @get_device_memory_objects.register(MemoryPointer)
    def get_device_memory_objects_cupy(obj):
        return [obj.mem]


@get_device_memory_objects.register_lazy("cudf")
def get_device_memory_objects_register_cudf():
    import cudf.core.multiindex
    import cudf.core.index
    import cudf.core.dataframe
    import cudf.core.series

    @get_device_memory_objects.register(cudf.core.dataframe.DataFrame)
    def get_device_memory_objects_cudf_dataframe(obj):

        ret = get_device_memory_objects(obj._index)
        for col in obj._data.columns:
            ret += get_device_memory_objects(col)
        return ret

    @get_device_memory_objects.register(cudf.core.series.Series)
    def get_device_memory_objects_cudf_series(obj):
        return get_device_memory_objects(obj._index) + get_device_memory_objects(
            obj._column
        )

    @get_device_memory_objects.register(cudf.core.index.Index)
    def get_device_memory_objects_cudf_index(obj):
        return get_device_memory_objects(obj._values)

    @get_device_memory_objects.register(cudf.core.multiindex.MultiIndex)
    def get_device_memory_objects_cudf_multiindex(obj):
        return get_device_memory_objects(obj._columns)


@sizeof.register_lazy("cupy")
def register_cupy():  # NB: this overwrites dask.sizeof.register_cupy()
    import cupy.cuda.memory

    @sizeof.register(cupy.cuda.memory.BaseMemory)
    def sizeof_cupy_base_memory(x):
        return int(x.size)

    @sizeof.register(cupy.ndarray)
    def sizeof_cupy_ndarray(x):
        return int(x.nbytes)
