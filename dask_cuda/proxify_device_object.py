from dask.utils import Dispatch
from .proxy_object import ProxyObject, asproxy

proxify_device_object = Dispatch(name="proxify_device_object")


def proxify(obj, proxied_id_to_proxy, found_proxies, subclass=None):
    _id = id(obj)
    if _id in proxied_id_to_proxy:
        ret = proxied_id_to_proxy[_id]
    else:
        proxied_id_to_proxy[_id] = ret = asproxy(obj, subclass=subclass)
    found_proxies.append(ret)
    return ret


@proxify_device_object.register(object)
def proxify_device_object_default(obj, proxied_id_to_proxy, found_proxies):
    if hasattr(obj, "__cuda_array_interface__"):
        return proxify(obj, proxied_id_to_proxy, found_proxies)
    return obj


@proxify_device_object.register(ProxyObject)
def proxify_device_object_proxy_object(obj, proxied_id_to_proxy, found_proxies):
    found_proxies.append(obj)
    if not obj._obj_pxy_serialized():
        return proxify(obj, proxied_id_to_proxy, found_proxies)
    return obj


@proxify_device_object.register(list)
@proxify_device_object.register(tuple)
@proxify_device_object.register(set)
@proxify_device_object.register(frozenset)
def proxify_device_object_python_collection(seq, proxied_id_to_proxy, found_proxies):
    return type(seq)(
        proxify_device_object(o, proxied_id_to_proxy, found_proxies) for o in seq
    )


@proxify_device_object.register(dict)
def proxify_device_object_python_dict(seq, proxied_id_to_proxy, found_proxies):
    for k, v in seq.items():
        assert v is not seq
    return {
        k: proxify_device_object(v, proxied_id_to_proxy, found_proxies)
        for k, v in seq.items()
    }


@proxify_device_object.register_lazy("cudf")
def register_cudf():
    import cudf

    @proxify_device_object.register(cudf.DataFrame)
    @proxify_device_object.register(cudf.Series)
    @proxify_device_object.register(cudf.Index)
    def proxify_device_object_cudf_dataframe(obj, proxied_id_to_proxy, found_proxies):
        return proxify(obj, proxied_id_to_proxy, found_proxies)
