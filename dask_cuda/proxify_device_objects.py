from typing import Any, List, MutableMapping

from dask.utils import Dispatch

from .proxy_object import ProxyObject, asproxy

dispatch = Dispatch(name="proxify_device_objects")


def proxify_device_objects(
    obj: Any,
    proxied_id_to_proxy: MutableMapping[int, ProxyObject],
    found_proxies: List[ProxyObject],
    excl_proxies: bool = False,
):
    """ Wrap device objects in ProxyObject

    Search through `obj` and wraps all CUDA device objects in ProxyObject.
    It uses `proxied_id_to_proxy` to make sure that identical CUDA device
    objects found in `obj` are wrapped by the same ProxyObject.

    Parameters
    ----------
    obj: Any
        Object to search through or wrap in a ProxyObject.
    proxied_id_to_proxy: MutableMapping[int, ProxyObject]
        Dict mapping the id() of proxied objects (CUDA device objects) to
        their proxy and is updated with all new proxied objects found in `obj`.
    found_proxies: List[ProxyObject]
        List of found proxies in `obj`. Notice, this includes all proxies found,
        including those already in `proxied_id_to_proxy`.
    excl_proxies: bool
        Don't add found objects that are already ProxyObject to found_proxies.

    Returns
    -------
    ret: Any
        A copy of `obj` where all CUDA device objects are wrapped in ProxyObject
    """
    return dispatch(obj, proxied_id_to_proxy, found_proxies, excl_proxies)


def proxify(obj, proxied_id_to_proxy, found_proxies, subclass=None):
    _id = id(obj)
    if _id in proxied_id_to_proxy:
        ret = proxied_id_to_proxy[_id]
        finalize = ret._obj_pxy.get("external_finalize", None)
        if finalize:
            finalize()
            proxied_id_to_proxy[_id] = ret = asproxy(obj, subclass=subclass)
    else:
        proxied_id_to_proxy[_id] = ret = asproxy(obj, subclass=subclass)
    found_proxies.append(ret)
    return ret


@dispatch.register(object)
def proxify_device_object_default(
    obj, proxied_id_to_proxy, found_proxies, excl_proxies
):
    if hasattr(obj, "__cuda_array_interface__"):
        return proxify(obj, proxied_id_to_proxy, found_proxies)
    return obj


@dispatch.register(ProxyObject)
def proxify_device_object_proxy_object(
    obj, proxied_id_to_proxy, found_proxies, excl_proxies
):
    # We deserialize CUDA-serialized objects since it is very cheap and
    # makes it easy to administrate device memory usage
    if obj._obj_pxy_is_serialized() and "cuda" in obj._obj_pxy["serializers"]:
        obj._obj_pxy_deserialize()

    # Check if `obj` is already known
    if not obj._obj_pxy_is_serialized():
        _id = id(obj._obj_pxy["obj"])
        if _id in proxied_id_to_proxy:
            obj = proxied_id_to_proxy[_id]
        else:
            proxied_id_to_proxy[_id] = obj

    finalize = obj._obj_pxy.get("external_finalize", None)
    if finalize:
        finalize()
        obj = obj._obj_pxy_copy()
        if not obj._obj_pxy_is_serialized():
            _id = id(obj._obj_pxy["obj"])
            proxied_id_to_proxy[_id] = obj

    if not excl_proxies:
        found_proxies.append(obj)
    return obj


@dispatch.register(list)
@dispatch.register(tuple)
@dispatch.register(set)
@dispatch.register(frozenset)
def proxify_device_object_python_collection(
    seq, proxied_id_to_proxy, found_proxies, excl_proxies
):
    return type(seq)(
        dispatch(o, proxied_id_to_proxy, found_proxies, excl_proxies) for o in seq
    )


@dispatch.register(dict)
def proxify_device_object_python_dict(
    seq, proxied_id_to_proxy, found_proxies, excl_proxies
):
    return {
        k: dispatch(v, proxied_id_to_proxy, found_proxies, excl_proxies)
        for k, v in seq.items()
    }


# Implement cuDF specific proxification
@dispatch.register_lazy("cudf")
def _register_cudf():
    import cudf
    import cudf._lib.table

    # In order to support the cuDF API implemented in Cython, we inherit from
    # `cudf._lib.table.Table`, which is the base class of Index, Series, and
    # Dataframes in cuDF.
    # Notice, the order of base classes matters. Since ProxyObject is the first
    # base class, ProxyObject.__init__() is called on creation, which doesn't
    # define the Table._data and Table._index attributes. Thus, accessing
    # FrameProxyObject._data and FrameProxyObject._index is pass-through to
    # ProxyObejct.__getattr__(), which is what we want.
    class FrameProxyObject(ProxyObject, cudf._lib.table.Table):
        pass

    @dispatch.register(cudf.DataFrame)
    @dispatch.register(cudf.Series)
    @dispatch.register(cudf.Index)
    def proxify_device_object_cudf_dataframe(
        obj, proxied_id_to_proxy, found_proxies, excl_proxies
    ):
        return proxify(
            obj, proxied_id_to_proxy, found_proxies, subclass=FrameProxyObject
        )
