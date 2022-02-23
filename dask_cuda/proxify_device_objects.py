import functools
import pydoc
from collections import defaultdict
from functools import partial
from typing import List, MutableMapping, Optional, Tuple, TypeVar

import dask
from dask.utils import Dispatch

from .proxy_object import ProxyObject, asproxy

dispatch = Dispatch(name="proxify_device_objects")
incompatible_types: Optional[Tuple[type]] = None

T = TypeVar("T")


def _register_incompatible_types():
    """Lazy register types that ProxifyHostFile should unproxify on retrieval.

    It reads the config key "jit-unspill-incompatible"
    (DASK_JIT_UNSPILL_INCOMPATIBLE), which should be a comma seperated
    list of types. The default value is:
        DASK_JIT_UNSPILL_INCOMPATIBLE="cupy.ndarray"
    """
    global incompatible_types
    if incompatible_types is not None:
        return  # Only register once
    else:
        incompatible_types = ()

    incompatibles = dask.config.get("jit-unspill-incompatible", "cupy.ndarray")
    incompatibles = incompatibles.split(",")

    toplevels = defaultdict(set)
    for path in incompatibles:
        if path:
            toplevel = path.split(".", maxsplit=1)[0].strip()
            toplevels[toplevel].add(path.strip())

    for toplevel, ignores in toplevels.items():

        def f(paths):
            global incompatible_types
            incompatible_types = incompatible_types + tuple(
                pydoc.locate(p) for p in paths
            )

        dispatch.register_lazy(toplevel, partial(f, ignores))


def proxify_device_objects(
    obj: T,
    proxied_id_to_proxy: MutableMapping[int, ProxyObject] = None,
    found_proxies: List[ProxyObject] = None,
    excl_proxies: bool = False,
    mark_as_explicit_proxies: bool = False,
) -> T:
    """Wrap device objects in ProxyObject

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
    _register_incompatible_types()

    if proxied_id_to_proxy is None:
        proxied_id_to_proxy = {}
    if found_proxies is None:
        found_proxies = []
    ret = dispatch(obj, proxied_id_to_proxy, found_proxies, excl_proxies)
    for p in found_proxies:
        p._pxy_get().explicit_proxy = mark_as_explicit_proxies
    return ret


def unproxify_device_objects(
    obj: T, skip_explicit_proxies: bool = False, only_incompatible_types: bool = False
) -> T:
    """Unproxify device objects

    Search through `obj` and un-wraps all CUDA device objects.

    Parameters
    ----------
    obj: Any
        Object to search through or unproxify.
    skip_explicit_proxies: bool
        When True, skipping proxy objects marked as explicit proxies.
    only_incompatible_types: bool
        When True, ONLY unproxify incompatible type. The skip_explicit_proxies
        argument is ignored.

    Returns
    -------
    ret: Any
        A copy of `obj` where all CUDA device objects are unproxify
    """
    if isinstance(obj, dict):
        return {
            k: unproxify_device_objects(
                v, skip_explicit_proxies, only_incompatible_types
            )
            for k, v in obj.items()
        }  # type: ignore
    if isinstance(obj, (list, tuple, set, frozenset)):
        return obj.__class__(
            unproxify_device_objects(i, skip_explicit_proxies, only_incompatible_types)
            for i in obj
        )  # type: ignore
    if isinstance(obj, ProxyObject):
        pxy = obj._pxy_get(copy=True)
        if only_incompatible_types:
            if incompatible_types and isinstance(obj, incompatible_types):
                obj = obj._pxy_deserialize(maybe_evict=False, proxy_detail=pxy)
        elif not skip_explicit_proxies or not pxy.explicit_proxy:
            pxy.explicit_proxy = False
            obj = obj._pxy_deserialize(maybe_evict=False, proxy_detail=pxy)
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
    else:
        ret = proxied_id_to_proxy[_id] = asproxy(obj, subclass=subclass)
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
    obj: ProxyObject, proxied_id_to_proxy, found_proxies, excl_proxies
):
    # Check if `obj` is already known
    pxy = obj._pxy_get()
    if not pxy.is_serialized():
        _id = id(pxy.obj)
        if _id in proxied_id_to_proxy:
            obj = proxied_id_to_proxy[_id]
        else:
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

    @dispatch.register(cudf.DataFrame)
    @dispatch.register(cudf.Series)
    @dispatch.register(cudf.BaseIndex)
    def proxify_device_object_cudf_dataframe(
        obj, proxied_id_to_proxy, found_proxies, excl_proxies
    ):
        return proxify(obj, proxied_id_to_proxy, found_proxies)

    try:
        from dask.array.dispatch import percentile_lookup

        from dask_cudf.backends import percentile_cudf

        percentile_lookup.register(ProxyObject, percentile_cudf)
    except ImportError:
        pass
