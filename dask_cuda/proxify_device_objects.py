import functools
import pydoc
from collections import defaultdict
from functools import partial
from typing import Any, List, MutableMapping

import dask
from dask.utils import Dispatch

from .proxy_object import ProxyObject, asproxy

dispatch = Dispatch(name="proxify_device_objects")
ignore_types = None


def _register_ignore_types():
    """Lazy register types that shouldn't be proxified

    It reads the config key "jit-unspill-ignore" (DASK_JIT_UNSPILL_IGNORE),
    which should be a comma seperated list of types to ignore. The default
    value is:
        DASK_JIT_UNSPILL_IGNORE="cupy.ndarray"

    Notice, it is not possible to ignore types explicitly handled by this
    module such as `cudf.DataFrame`, `cudf.Series`, and `cudf.Index`.
    """
    global ignore_types
    if ignore_types is not None:
        return  # Only register once
    else:
        ignore_types = ()

    ignores = dask.config.get("jit-unspill-ignore", "cupy.ndarray")
    ignores = ignores.split(",")

    toplevels = defaultdict(set)
    for path in ignores:
        if path:
            toplevel = path.split(".", maxsplit=1)[0].strip()
            toplevels[toplevel].add(path.strip())

    for toplevel, ignores in toplevels.items():

        def f(paths):
            global ignore_types
            ignore_types = ignore_types + tuple(pydoc.locate(p) for p in paths)

        dispatch.register_lazy(toplevel, partial(f, ignores))


def proxify_device_objects(
    obj: Any,
    proxied_id_to_proxy: MutableMapping[int, ProxyObject] = None,
    found_proxies: List[ProxyObject] = None,
    excl_proxies: bool = False,
    mark_as_explicit_proxies: bool = False,
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
        If None, use an empty dict.
    found_proxies: List[ProxyObject]
        List of found proxies in `obj`. Notice, this includes all proxies found,
        including those already in `proxied_id_to_proxy`.
        If None, use an empty list.
    excl_proxies: bool
        Don't add found objects that are already ProxyObject to found_proxies.
    mark_as_explicit_proxies: bool
        Mark found proxies as "explicit", which means that the user allows them
        as input arguments to dask tasks even in compatibility-mode.

    Returns
    -------
    ret: Any
        A copy of `obj` where all CUDA device objects are wrapped in ProxyObject
    """
    _register_ignore_types()

    if proxied_id_to_proxy is None:
        proxied_id_to_proxy = {}
    if found_proxies is None:
        found_proxies = []
    ret = dispatch(obj, proxied_id_to_proxy, found_proxies, excl_proxies)
    if mark_as_explicit_proxies:
        for p in found_proxies:
            p._obj_pxy["explicit_proxy"] = True
    return ret


def unproxify_device_objects(obj: Any, skip_explicit_proxies: bool = False):
    """ Unproxify device objects

    Search through `obj` and un-wraps all CUDA device objects.

    Parameters
    ----------
    obj: Any
        Object to search through or unproxify.
    skip_explicit_proxies: bool
        When True, skipping proxy objects marked as explicit proxies.

    Returns
    -------
    ret: Any
        A copy of `obj` where all CUDA device objects are unproxify
    """
    if isinstance(obj, dict):
        return {
            k: unproxify_device_objects(v, skip_explicit_proxies)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple, set, frozenset)):
        return type(obj)(
            unproxify_device_objects(i, skip_explicit_proxies) for i in obj
        )

    if hasattr(obj, "_obj_pxy"):
        if not skip_explicit_proxies or not obj._obj_pxy["explicit_proxy"]:
            obj._obj_pxy["explicit_proxy"] = False
            obj = obj._obj_pxy_deserialize(maybe_evict=False)
    return obj


def proxify_decorator(func):
    """Returns a function wrapper that explicit proxify the output

    Notice, this function only has effect in compatibility mode.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        ret = func(*args, **kwargs)
        if dask.config.get("jit-unspill-compatibility-mode", default=False):
            ret = proxify_device_objects(ret, mark_as_explicit_proxies=True)
        return ret

    return wrapper


def unproxify_decorator(func):
    """Returns a function wrapper that unproxify output

    Notice, this function only has effect in compatibility mode.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        ret = func(*args, **kwargs)
        if dask.config.get("jit-unspill-compatibility-mode", default=False):
            ret = unproxify_device_objects(ret, skip_explicit_proxies=False)
        return ret

    return wrapper


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
    if hasattr(obj, "__cuda_array_interface__") and not isinstance(obj, ignore_types):
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
