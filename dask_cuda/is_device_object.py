from __future__ import absolute_import, division, print_function

from dask.utils import Dispatch

is_device_object = Dispatch(name="is_device_object")


@is_device_object.register(object)
def is_device_object_default(o):
    return hasattr(o, "__cuda_array_interface__")


@is_device_object.register(list)
@is_device_object.register(tuple)
@is_device_object.register(set)
@is_device_object.register(frozenset)
def is_device_object_python_collection(seq):
    return any([is_device_object(s) for s in seq])


@is_device_object.register(dict)
def is_device_object_python_dict(seq):
    return any([is_device_object(s) for s in seq.items()])


@is_device_object.register_lazy("cudf")
def register_cudf():
    import cudf

    @is_device_object.register(cudf.DataFrame)
    def is_device_object_cudf_dataframe(df):
        return True

    @is_device_object.register(cudf.Series)
    def is_device_object_cudf_series(s):
        return True

    @is_device_object.register(cudf.BaseIndex)
    def is_device_object_cudf_index(s):
        return True
