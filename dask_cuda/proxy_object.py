import copy
import functools
import operator
import pickle
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Set

import pandas

import dask
import dask.array.core
import dask.dataframe.methods
import dask.dataframe.utils
import distributed.protocol
import distributed.utils
from dask.sizeof import sizeof
from distributed.worker import dumps_function, loads_function

try:
    from dask.dataframe.backends import concat_pandas
except ImportError:
    from dask.dataframe.methods import concat_pandas

from .get_device_memory_objects import get_device_memory_objects
from .is_device_object import is_device_object

# List of attributes that should be copied to the proxy at creation, which makes
# them accessible without deserialization of the proxied object
_FIXED_ATTRS = ["name", "__len__"]


def asproxy(obj, serializers=None, subclass=None) -> "ProxyObject":
    """Wrap `obj` in a ProxyObject object if it isn't already.

    Parameters
    ----------
    obj: object
        Object to wrap in a ProxyObject object.
    serializers: list(str), optional
        List of serializers to use to serialize `obj`. If None,
        no serialization is done.
    subclass: class, optional
        Specify a subclass of ProxyObject to create instead of ProxyObject.
        `subclass` must be pickable.

    Returns
    -------
    The ProxyObject proxying `obj`
    """

    if hasattr(obj, "_obj_pxy"):  # Already a proxy object
        ret = obj
    else:
        fixed_attr = {}
        for attr in _FIXED_ATTRS:
            try:
                val = getattr(obj, attr)
                if callable(val):
                    val = val()
                fixed_attr[attr] = val
            except (AttributeError, TypeError):
                pass

        if subclass is None:
            subclass = ProxyObject
            subclass_serialized = None
        else:
            subclass_serialized = dumps_function(subclass)

        ret = subclass(
            obj=obj,
            fixed_attr=fixed_attr,
            type_serialized=pickle.dumps(type(obj)),
            typename=dask.utils.typename(type(obj)),
            is_cuda_object=is_device_object(obj),
            subclass=subclass_serialized,
            serializers=None,
            explicit_proxy=False,
        )
    if serializers is not None:
        ret._obj_pxy_serialize(serializers=serializers)
    return ret


def unproxy(obj):
    """Unwrap ProxyObject objects and pass-through anything else.

    Use this function to retrieve the proxied object. Notice, unproxy()
    search through list, tuples, sets, and frozensets.

    Parameters
    ----------
    obj: object
        Any kind of object

    Returns
    -------
    The proxied object or `obj` itself if it isn't a ProxyObject
    """
    try:
        obj = obj._obj_pxy_deserialize()
    except AttributeError:
        if type(obj) in (list, tuple, set, frozenset):
            return type(obj)(unproxy(o) for o in obj)
    return obj


def _obj_pxy_cache_wrapper(attr_name):
    """Caching the access of attr_name in ProxyObject._obj_pxy_cache"""

    def wrapper1(func):
        @functools.wraps(func)
        def wrapper2(self: "ProxyObject"):
            try:
                return self._obj_pxy_cache[attr_name]
            except KeyError:
                ret = func(self)
                self._obj_pxy_cache[attr_name] = ret
                return ret

        return wrapper2

    return wrapper1


class ProxyObject:
    """Object wrapper/proxy for serializable objects

    This is used by ProxifyHostFile to delay deserialization of returned objects.

    Objects proxied by an instance of this class will be JIT-deserialized when
    accessed. The instance behaves as the proxied object and can be accessed/used
    just like the proxied object.

    ProxyObject has some limitations and doesn't mimic the proxied object perfectly.
    Thus, if encountering problems remember that it is always possible to use unproxy()
    to access the proxied object directly or disable JIT deserialization completely
    with `jit_unspill=False`.

    Type checking using instance() works as expected but direct type checking
    doesn't:
    >>> import numpy as np
    >>> from dask_cuda.proxy_object import asproxy
    >>> x = np.arange(3)
    >>> isinstance(asproxy(x), type(x))
    True
    >>>  type(asproxy(x)) is type(x)
    False

    Attributes
    ----------
    _obj_pxy: dict
        Dictionary of all proxy information of the underlaying proxied object.
        Access to the dictionary is not pass-through to the proxied object,
        which is the case for most other access to the ProxyObject.

    _obj_pxy_lock: threading.RLock
        Threading lock for `self._obj_pxy` access

    _obj_pxy_cache: dict
        A dictionary used for caching attributes

    Parameters
    ----------
    obj: object
        Any kind of object to be proxied.
    fixed_attr: dict
        Dictionary of attributes that are accessible without deserializing
        the proxied object.
    type_serialized: bytes
        Pickled type of `obj`.
    typename: str
        Name of the type of `obj`.
    is_cuda_object: boolean
        Whether `obj` is a CUDA object or not.
    subclass: bytes
        Pickled type to use instead of ProxyObject when deserializing. The type
        must inherit from ProxyObject.
    serializers: list(str), optional
        List of serializers to use to serialize `obj`. If None, `obj`
        isn't serialized.
    explicit_proxy: bool
        Mark the proxy object as "explicit", which means that the user allows it
        as input argument to dask tasks even in compatibility-mode.
    """

    def __init__(
        self,
        obj: Any,
        fixed_attr: Dict[str, Any],
        type_serialized: bytes,
        typename: str,
        is_cuda_object: bool,
        subclass: bytes,
        serializers: Optional[List[str]],
        explicit_proxy: bool,
    ):
        self._obj_pxy = {
            "obj": obj,
            "fixed_attr": fixed_attr,
            "type_serialized": type_serialized,
            "typename": typename,
            "is_cuda_object": is_cuda_object,
            "subclass": subclass,
            "serializers": serializers,
            "explicit_proxy": explicit_proxy,
        }
        self._obj_pxy_lock = threading.RLock()
        self._obj_pxy_cache = {}

    def __del__(self):
        """In order to call `external_finalize()` ASAP, we call it here"""
        external_finalize = self._obj_pxy.get("external_finalize", None)
        if external_finalize is not None:
            external_finalize()

    def _obj_pxy_get_init_args(self, include_obj=True):
        """Return the attributes needed to initialize a ProxyObject

        Notice, the returned dictionary is ordered as the __init__() arguments

        Parameters
        ----------
        include_obj: bool
            Whether to include the "obj" argument or not

        Returns
        -------
        Dictionary of attributes
        """
        args = ["obj"] if include_obj else []
        args += [
            "fixed_attr",
            "type_serialized",
            "typename",
            "is_cuda_object",
            "subclass",
            "serializers",
            "explicit_proxy",
        ]
        return OrderedDict([(a, self._obj_pxy[a]) for a in args])

    def _obj_pxy_copy(self) -> "ProxyObject":
        """Return a deepcopy of the proxy meta data

        Use this to copy the proxy without copying the proxied object.

        Returns
        -------
        Copy of the proxy object that points to the same proxied object
        """
        args = copy.deepcopy(self._obj_pxy_get_init_args(include_obj=False))
        args["obj"] = self._obj_pxy["obj"]
        return type(self)(**args)

    def _obj_pxy_is_serialized(self):
        """Return whether the proxied object is serialized or not"""
        return self._obj_pxy["serializers"] is not None

    def _obj_pxy_serialize(self, serializers):
        """Inplace serialization of the proxied object using the `serializers`

        Parameters
        ----------
        serializers: tuple[str]
            Tuple of serializers to use to serialize the proxied object.

        Returns
        -------
        header: dict
            The header of the serialized frames
        frames: list[bytes]
            List of frames that make up the serialized object
        """
        if not serializers:
            raise ValueError("Please specify a list of serializers")

        if type(serializers) is not tuple:
            serializers = tuple(serializers)

        with self._obj_pxy_lock:
            if self._obj_pxy["serializers"] is not None:
                if self._obj_pxy["serializers"] == serializers:
                    return self._obj_pxy["obj"]  # Nothing to be done
                else:
                    # The proxied object is serialized with other serializers
                    self._obj_pxy_deserialize()

            if self._obj_pxy["serializers"] is None:
                self._obj_pxy["obj"] = distributed.protocol.serialize(
                    self._obj_pxy["obj"], serializers, on_error="raise"
                )
                self._obj_pxy["serializers"] = serializers
                hostfile = self._obj_pxy.get("hostfile", lambda: None)()
                if hostfile is not None:
                    external = self._obj_pxy.get("external", self)
                    hostfile.proxies_tally.spill_proxy(external)

            # Invalidate the (possible) cached "device_memory_objects"
            self._obj_pxy_cache.pop("device_memory_objects", None)
            return self._obj_pxy["obj"]

    def _obj_pxy_deserialize(self, maybe_evict: bool = True):
        """Inplace deserialization of the proxied object

        Parameters
        ----------
        maybe_evict: bool
            Before deserializing, call associated hostfile.maybe_evict()

        Returns
        -------
        object
            The proxied object (deserialized)
        """
        with self._obj_pxy_lock:
            if self._obj_pxy["serializers"] is not None:
                hostfile = self._obj_pxy.get("hostfile", lambda: None)()
                # When not deserializing a CUDA-serialized proxied, we might have
                # to evict because of the increased device memory usage.
                if maybe_evict and "cuda" not in self._obj_pxy["serializers"]:
                    if hostfile is not None:
                        # In order to avoid a potential deadlock, we skip the
                        # `maybe_evict()` call if another thread is also accessing
                        # the hostfile.
                        if hostfile.lock.acquire(blocking=False):
                            try:
                                hostfile.maybe_evict(self.__sizeof__())
                            finally:
                                hostfile.lock.release()

                header, frames = self._obj_pxy["obj"]
                self._obj_pxy["obj"] = distributed.protocol.deserialize(header, frames)
                self._obj_pxy["serializers"] = None
                if hostfile is not None:
                    external = self._obj_pxy.get("external", self)
                    hostfile.proxies_tally.unspill_proxy(external)

            self._obj_pxy["last_access"] = time.monotonic()
            return self._obj_pxy["obj"]

    def _obj_pxy_is_cuda_object(self) -> bool:
        """Return whether the proxied object is a CUDA or not

        Returns
        -------
        ret : boolean
            Is the proxied object a CUDA object?
        """
        with self._obj_pxy_lock:
            return self._obj_pxy["is_cuda_object"]

    @_obj_pxy_cache_wrapper("device_memory_objects")
    def _obj_pxy_get_device_memory_objects(self) -> Set:
        """Return all device memory objects within the proxied object.

        Calling this when the proxied object is serialized returns the
        empty list.

        Returns
        -------
        ret : set
            Set of device memory objects
        """
        return get_device_memory_objects(self._obj_pxy["obj"])

    def __reduce__(self):
        """Serialization of ProxyObject that uses pickle"""
        self._obj_pxy_serialize(serializers=("pickle",))
        args = self._obj_pxy_get_init_args()
        if args["subclass"]:
            subclass = loads_function(args["subclass"])
        else:
            subclass = ProxyObject

        # Make sure the frames are all bytes
        header, frames = args["obj"]
        args["obj"] = (header, [bytes(f) for f in frames])

        return (subclass, tuple(args.values()))

    def __getattr__(self, name):
        with self._obj_pxy_lock:
            typename = self._obj_pxy["typename"]
            if name in _FIXED_ATTRS:
                try:
                    return self._obj_pxy["fixed_attr"][name]
                except KeyError:
                    raise AttributeError(
                        f"type object '{typename}' has no attribute '{name}'"
                    )

            return getattr(self._obj_pxy_deserialize(), name)

    def __setattr__(self, name, val):
        if name in ("_obj_pxy", "_obj_pxy_lock", "_obj_pxy_cache"):
            return object.__setattr__(self, name, val)

        with self._obj_pxy_lock:
            if name in _FIXED_ATTRS:
                self._obj_pxy["fixed_attr"][name] = val
            else:
                object.__setattr__(self._obj_pxy_deserialize(), name, val)

    def __str__(self):
        return str(self._obj_pxy_deserialize())

    def __repr__(self):
        with self._obj_pxy_lock:
            typename = self._obj_pxy["typename"]
            ret = f"<{dask.utils.typename(type(self))} at {hex(id(self))} of {typename}"
            if self._obj_pxy["serializers"] is not None:
                ret += f" (serialized={repr(self._obj_pxy['serializers'])})>"
            else:
                ret += f" at {hex(id(self._obj_pxy['obj']))}>"
            return ret

    @property
    @_obj_pxy_cache_wrapper("type_serialized")
    def __class__(self):
        return pickle.loads(self._obj_pxy["type_serialized"])

    @_obj_pxy_cache_wrapper("sizeof")
    def __sizeof__(self):
        """Returns either the size of the proxied object

        Notice, we cache the result even though the size of proxied object
        when serialized or not serialized might slightly differ.
        """
        with self._obj_pxy_lock:
            if self._obj_pxy_is_serialized():
                frames = self._obj_pxy["obj"][1]
                return sum(map(distributed.utils.nbytes, frames))
            else:
                return sizeof(self._obj_pxy_deserialize())

    def __len__(self):
        with self._obj_pxy_lock:
            ret = self._obj_pxy["fixed_attr"].get("__len__", None)
            if ret is None:
                ret = len(self._obj_pxy_deserialize())
                self._obj_pxy["fixed_attr"]["__len__"] = ret
            return ret

    def __contains__(self, value):
        return value in self._obj_pxy_deserialize()

    def __getitem__(self, key):
        return self._obj_pxy_deserialize()[key]

    def __setitem__(self, key, value):
        self._obj_pxy_deserialize()[key] = value

    def __delitem__(self, key):
        del self._obj_pxy_deserialize()[key]

    def __getslice__(self, i, j):
        return self._obj_pxy_deserialize()[i:j]

    def __setslice__(self, i, j, value):
        self._obj_pxy_deserialize()[i:j] = value

    def __delslice__(self, i, j):
        del self._obj_pxy_deserialize()[i:j]

    def __iter__(self):
        return iter(self._obj_pxy_deserialize())

    def __array__(self, *args, **kwargs):
        return getattr(self._obj_pxy_deserialize(), "__array__")(*args, **kwargs)

    def __lt__(self, other):
        return self._obj_pxy_deserialize() < other

    def __le__(self, other):
        return self._obj_pxy_deserialize() <= other

    def __eq__(self, other):
        return self._obj_pxy_deserialize() == other

    def __ne__(self, other):
        return self._obj_pxy_deserialize() != other

    def __gt__(self, other):
        return self._obj_pxy_deserialize() > other

    def __ge__(self, other):
        return self._obj_pxy_deserialize() >= other

    def __add__(self, other):
        return self._obj_pxy_deserialize() + other

    def __sub__(self, other):
        return self._obj_pxy_deserialize() - other

    def __mul__(self, other):
        return self._obj_pxy_deserialize() * other

    def __truediv__(self, other):
        return operator.truediv(self._obj_pxy_deserialize(), other)

    def __floordiv__(self, other):
        return self._obj_pxy_deserialize() // other

    def __mod__(self, other):
        return self._obj_pxy_deserialize() % other

    def __divmod__(self, other):
        return divmod(self._obj_pxy_deserialize(), other)

    def __pow__(self, other, *args):
        return pow(self._obj_pxy_deserialize(), other, *args)

    def __lshift__(self, other):
        return self._obj_pxy_deserialize() << other

    def __rshift__(self, other):
        return self._obj_pxy_deserialize() >> other

    def __and__(self, other):
        return self._obj_pxy_deserialize() & other

    def __xor__(self, other):
        return self._obj_pxy_deserialize() ^ other

    def __or__(self, other):
        return self._obj_pxy_deserialize() | other

    def __radd__(self, other):
        return other + self._obj_pxy_deserialize()

    def __rsub__(self, other):
        return other - self._obj_pxy_deserialize()

    def __rmul__(self, other):
        return other * self._obj_pxy_deserialize()

    def __rtruediv__(self, other):
        return operator.truediv(other, self._obj_pxy_deserialize())

    def __rfloordiv__(self, other):
        return other // self._obj_pxy_deserialize()

    def __rmod__(self, other):
        return other % self._obj_pxy_deserialize()

    def __rdivmod__(self, other):
        return divmod(other, self._obj_pxy_deserialize())

    def __rpow__(self, other, *args):
        return pow(other, self._obj_pxy_deserialize(), *args)

    def __rlshift__(self, other):
        return other << self._obj_pxy_deserialize()

    def __rrshift__(self, other):
        return other >> self._obj_pxy_deserialize()

    def __rand__(self, other):
        return other & self._obj_pxy_deserialize()

    def __rxor__(self, other):
        return other ^ self._obj_pxy_deserialize()

    def __ror__(self, other):
        return other | self._obj_pxy_deserialize()

    def __iadd__(self, other):
        proxied = self._obj_pxy_deserialize()
        proxied += other
        return self

    def __isub__(self, other):
        proxied = self._obj_pxy_deserialize()
        proxied -= other
        return self

    def __imul__(self, other):
        proxied = self._obj_pxy_deserialize()
        proxied *= other
        return self

    def __itruediv__(self, other):
        with self._obj_pxy_lock:
            proxied = self._obj_pxy_deserialize()
            self._obj_pxy["obj"] = operator.itruediv(proxied, other)
        return self

    def __ifloordiv__(self, other):
        proxied = self._obj_pxy_deserialize()
        proxied //= other
        return self

    def __imod__(self, other):
        proxied = self._obj_pxy_deserialize()
        proxied %= other
        return self

    def __ipow__(self, other):
        proxied = self._obj_pxy_deserialize()
        proxied **= other
        return self

    def __ilshift__(self, other):
        proxied = self._obj_pxy_deserialize()
        proxied <<= other
        return self

    def __irshift__(self, other):
        proxied = self._obj_pxy_deserialize()
        proxied >>= other
        return self

    def __iand__(self, other):
        proxied = self._obj_pxy_deserialize()
        proxied &= other
        return self

    def __ixor__(self, other):
        proxied = self._obj_pxy_deserialize()
        proxied ^= other
        return self

    def __ior__(self, other):
        proxied = self._obj_pxy_deserialize()
        proxied |= other
        return self

    def __neg__(self):
        return -self._obj_pxy_deserialize()

    def __pos__(self):
        return +self._obj_pxy_deserialize()

    def __abs__(self):
        return abs(self._obj_pxy_deserialize())

    def __invert__(self):
        return ~self._obj_pxy_deserialize()

    def __int__(self):
        return int(self._obj_pxy_deserialize())

    def __float__(self):
        return float(self._obj_pxy_deserialize())

    def __complex__(self):
        return complex(self._obj_pxy_deserialize())

    def __index__(self):
        return operator.index(self._obj_pxy_deserialize())


@is_device_object.register(ProxyObject)
def obj_pxy_is_device_object(obj: ProxyObject):
    """
    In order to avoid de-serializing the proxied object, we call
    `_obj_pxy_is_cuda_object()` instead of the default
    `hasattr(o, "__cuda_array_interface__")` check.
    """
    return obj._obj_pxy_is_cuda_object()


@distributed.protocol.dask_serialize.register(ProxyObject)
def obj_pxy_dask_serialize(obj: ProxyObject):
    """
    The generic serialization of ProxyObject used by Dask when communicating
    ProxyObject. As serializers, it uses "dask" or "pickle", which means
    that proxied CUDA objects are spilled to main memory before communicated.
    """
    header, frames = obj._obj_pxy_serialize(serializers=("dask", "pickle"))
    meta = obj._obj_pxy_get_init_args(include_obj=False)
    return {"proxied-header": header, "obj-pxy-meta": meta}, frames


@distributed.protocol.cuda.cuda_serialize.register(ProxyObject)
def obj_pxy_cuda_serialize(obj: ProxyObject):
    """
    The CUDA serialization of ProxyObject used by Dask when communicating using UCX
    or another CUDA friendly communication library. As serializers, it uses "cuda",
    which means that proxied CUDA objects are _not_ spilled to main memory.
    """
    if obj._obj_pxy["serializers"] is not None:  # Already serialized
        header, frames = obj._obj_pxy["obj"]
    else:
        # Notice, since obj._obj_pxy_serialize() is a inplace operation, we make a
        # shallow copy of `obj` to avoid introducing a CUDA-serialized object in
        # the worker's data store.
        obj = obj._obj_pxy_copy()
        header, frames = obj._obj_pxy_serialize(serializers=("cuda",))
    meta = obj._obj_pxy_get_init_args(include_obj=False)
    return {"proxied-header": header, "obj-pxy-meta": meta}, frames


@distributed.protocol.dask_deserialize.register(ProxyObject)
@distributed.protocol.cuda.cuda_deserialize.register(ProxyObject)
def obj_pxy_dask_deserialize(header, frames):
    """
    The generic deserialization of ProxyObject. Notice, it doesn't deserialize
    the proxied object at this time. When accessed, the proxied object are
    deserialized using the same serializers that were used when the object was
    serialized.
    """
    meta = header["obj-pxy-meta"]
    if meta["subclass"] is None:
        subclass = ProxyObject
    else:
        subclass = loads_function(meta["subclass"])
    return subclass(obj=(header["proxied-header"], frames), **header["obj-pxy-meta"],)


@dask.dataframe.core.get_parallel_type.register(ProxyObject)
def get_parallel_type_proxy_object(obj: ProxyObject):
    # Notice, `get_parallel_type()` needs a instance not a type object
    return dask.dataframe.core.get_parallel_type(obj.__class__.__new__(obj.__class__))


def unproxify_input_wrapper(func):
    """Unproxify the input of `func`"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args = [unproxy(d) for d in args]
        kwargs = {k: unproxy(v) for k, v in kwargs.items()}
        return func(*args, **kwargs)

    return wrapper


# Register dispatch of ProxyObject on all known dispatch objects
for dispatch in (
    dask.dataframe.core.hash_object_dispatch,
    dask.dataframe.utils.make_meta,
    dask.dataframe.utils.make_scalar,
    dask.dataframe.core.group_split_dispatch,
    dask.array.core.tensordot_lookup,
    dask.array.core.einsum_lookup,
    dask.array.core.concatenate_lookup,
):
    dispatch.register(ProxyObject, unproxify_input_wrapper(dispatch))

dask.dataframe.methods.concat_dispatch.register(
    ProxyObject, unproxify_input_wrapper(dask.dataframe.methods.concat)
)


# We overwrite the Dask dispatch of Pandas objects in order to
# deserialize all ProxyObjects before concatenating
dask.dataframe.methods.concat_dispatch.register(
    (pandas.DataFrame, pandas.Series, pandas.Index),
    unproxify_input_wrapper(concat_pandas),
)
