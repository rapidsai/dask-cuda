import copy
import functools
import operator
import os
import pickle
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional, Type, Union

import pandas

import dask
import dask.array.core
import dask.dataframe.methods
import dask.dataframe.utils
import dask.utils
import distributed.protocol
import distributed.utils
from dask.sizeof import sizeof
from distributed.protocol.compression import decompress
from distributed.protocol.utils import unpack_frames
from distributed.worker import dumps_function, loads_function

try:
    from dask.dataframe.backends import concat_pandas
except ImportError:
    from dask.dataframe.methods import concat_pandas

try:
    from dask.dataframe.dispatch import make_meta_dispatch as make_meta_dispatch
except ImportError:
    from dask.dataframe.utils import make_meta as make_meta_dispatch

from .get_device_memory_objects import get_device_memory_objects
from .is_device_object import is_device_object

if TYPE_CHECKING:
    from .proxify_host_file import ProxyManager


# List of attributes that should be copied to the proxy at creation, which makes
# them accessible without deserialization of the proxied object
_FIXED_ATTRS = ["name", "__len__"]


def asproxy(
    obj: object, serializers: Iterable[str] = None, subclass: Type["ProxyObject"] = None
) -> "ProxyObject":
    """Wrap `obj` in a ProxyObject object if it isn't already.

    Parameters
    ----------
    obj: object
        Object to wrap in a ProxyObject object.
    serializers: Iterable[str], optional
        Serializers to use to serialize `obj`. If None, no serialization is done.
    subclass: class, optional
        Specify a subclass of ProxyObject to create instead of ProxyObject.
        `subclass` must be pickable.

    Returns
    -------
    The ProxyObject proxying `obj`
    """
    if isinstance(obj, ProxyObject):  # Already a proxy object
        ret = obj
    elif isinstance(obj, (list, set, tuple, dict)):
        raise ValueError(f"Cannot wrap a collection ({type(obj)}) in a proxy object")
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
            serializer=None,
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


def _obj_pxy_cache_wrapper(attr_name: str):
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


class ProxyManagerDummy:
    """Dummy of a ProxyManager that does nothing

    This is a dummy class returned by `ProxyObject._obj_pxy_get_manager()`
    when no manager has been registered the proxy object. It implements
    dummy methods that doesn't do anything it is purely for convenience.
    """

    def add(self, *args, **kwargs):
        pass

    def remove(self, *args, **kwargs):
        pass

    def move(self, *args, **kwargs):
        pass

    def maybe_evict(self, *args, **kwargs):
        pass

    @property
    def lock(self):
        return nullcontext()


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
    serializers: str, optional
        Serializers to use to serialize `obj`. If None, no serialization is done.
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
        subclass: Optional[bytes],
        serializer: Optional[str],
        explicit_proxy: bool,
    ):
        self._obj_pxy = {
            "obj": obj,
            "fixed_attr": fixed_attr,
            "type_serialized": type_serialized,
            "typename": typename,
            "is_cuda_object": is_cuda_object,
            "subclass": subclass,
            "serializer": serializer,
            "explicit_proxy": explicit_proxy,
        }
        self._obj_pxy_lock = threading.RLock()
        self._obj_pxy_cache: Dict[str, Any] = {}

    def __del__(self):
        """We have to unregister us from the manager if any"""
        self._obj_pxy_get_manager().remove(self)
        if self._obj_pxy["serializer"] == "disk":
            header, _ = self._obj_pxy["obj"]
            os.remove(header["path"])

    def _obj_pxy_get_init_args(self, include_obj=True) -> OrderedDict:
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
            "serializer",
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

    def _obj_pxy_register_manager(self, manager: "ProxyManager") -> None:
        """Register a manager

        The manager tallies the total memory usage of proxies and
        evicts/serialize proxy objects as needed.

        In order to prevent deadlocks, the proxy now use the lock of the
        manager.

        Parameters
        ----------
        manager: ProxyManager
            The manager to manage this proxy object
        """
        assert "manager" not in self._obj_pxy
        self._obj_pxy["manager"] = manager
        self._obj_pxy_lock = manager.lock

    def _obj_pxy_get_manager(self) -> Union["ProxyManager", ProxyManagerDummy]:
        """Get the registered manager or a dummy

        Parameters
        ----------
        manager: ProxyManager or ProxyManagerDummy
            The manager to manage this proxy object or a dummy
        """
        ret = self._obj_pxy.get("manager", None)
        if ret is None:
            ret = ProxyManagerDummy()
        return ret

    def _obj_pxy_is_serialized(self) -> bool:
        """Return whether the proxied object is serialized or not"""
        return self._obj_pxy["serializer"] is not None

    def _obj_pxy_serialize(self, serializers: Iterable[str]):
        """Inplace serialization of the proxied object using the `serializers`

        Parameters
        ----------
        serializers: Iterable[str]
            Serializers to use to serialize the proxied object.

        Returns
        -------
        header: dict
            The header of the serialized frames
        frames: list[bytes]
            List of frames that make up the serialized object
        """
        if not serializers:
            raise ValueError("Please specify a list of serializers")

        with self._obj_pxy_lock:
            if self._obj_pxy_is_serialized():
                if self._obj_pxy["serializer"] in serializers:
                    return self._obj_pxy["obj"]  # Nothing to be done
                else:
                    # The proxied object is serialized with other serializers
                    self._obj_pxy_deserialize()

            manager = self._obj_pxy_get_manager()
            with manager.lock:
                header, _ = self._obj_pxy["obj"] = distributed.protocol.serialize(
                    self._obj_pxy["obj"], serializers, on_error="raise"
                )
                assert "is-collection" not in header  # Collections not allowed
                org_ser, new_ser = self._obj_pxy["serializer"], header["serializer"]
                self._obj_pxy["serializer"] = new_ser

                # Tell the manager (if any) that this proxy has changed serializer
                manager.move(self, from_serializer=org_ser, to_serializer=new_ser)

                # Invalidate the (possible) cached "device_memory_objects"
                self._obj_pxy_cache.pop("device_memory_objects", None)
                return self._obj_pxy["obj"]

    def _obj_pxy_deserialize(self, maybe_evict: bool = True):
        """Inplace deserialization of the proxied object

        Parameters
        ----------
        maybe_evict: bool
            Before deserializing, maybe evict managered proxy objects

        Returns
        -------
        object
            The proxied object (deserialized)
        """
        with self._obj_pxy_lock:
            if self._obj_pxy_is_serialized():
                manager = self._obj_pxy_get_manager()
                with manager.lock:
                    # When not deserializing a CUDA-serialized proxied, tell the
                    # manager that it might have to evict because of the increased
                    # device memory usage.
                    if (
                        manager
                        and maybe_evict
                        and self._obj_pxy["serializer"] != "cuda"
                    ):
                        manager.maybe_evict(self.__sizeof__())

                    # Deserialize the proxied object
                    header, frames = self._obj_pxy["obj"]
                    self._obj_pxy["obj"] = distributed.protocol.deserialize(
                        header, frames
                    )

                    # Tell the manager (if any) that this proxy has changed serializer
                    manager.move(
                        self,
                        from_serializer=self._obj_pxy["serializer"],
                        to_serializer=None,
                    )
                    self._obj_pxy["serializer"] = None

            self._obj_pxy["last_access"] = time.monotonic()
            return self._obj_pxy["obj"]

    def _obj_pxy_is_cuda_object(self) -> bool:
        """Return whether the proxied object is a CUDA or not

        Returns
        -------
        ret : boolean
            Is the proxied object a CUDA object?
        """
        return self._obj_pxy["is_cuda_object"]

    @_obj_pxy_cache_wrapper("device_memory_objects")
    def _obj_pxy_get_device_memory_objects(self) -> set:
        """Return all device memory objects within the proxied object.

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

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        inputs = tuple(
            o._obj_pxy_deserialize() if isinstance(o, ProxyObject) else o
            for o in inputs
        )
        kwargs = {
            key: value._obj_pxy_deserialize()
            if isinstance(value, ProxyObject)
            else value
            for key, value in kwargs.items()
        }
        return self._obj_pxy_deserialize().__array_ufunc__(
            ufunc, method, *inputs, **kwargs
        )

    def __str__(self):
        return str(self._obj_pxy_deserialize())

    def __repr__(self):
        with self._obj_pxy_lock:
            typename = self._obj_pxy["typename"]
            ret = f"<{dask.utils.typename(type(self))} at {hex(id(self))} of {typename}"
            if self._obj_pxy_is_serialized():
                ret += f" (serialized={repr(self._obj_pxy['serializer'])})>"
            else:
                ret += f" at {hex(id(self._obj_pxy['obj']))}>"
            return ret

    @property  # type: ignore  # mypy doesn't support decorated property
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

    def __pow__(self, other):
        return pow(self._obj_pxy_deserialize(), other)

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


def handle_disk_serialized(obj: ProxyObject):
    """Handle serialization of an already disk serialized proxy

    On a shared filesystem, we do not have to deserialize instead we
    make a hard link of the file.

    On a non-shared filesystem, we deserialize the proxy to host memory.
    """

    header, frames = obj._obj_pxy["obj"]
    if header["shared-filesystem"]:
        old_path = header["path"]
        new_path = f"{old_path}-linked-{uuid.uuid4()}"
        os.link(old_path, new_path)
        header = copy.copy(header)
        header["path"] = new_path
    else:
        # When not on a shared filesystem, we deserialize to host memory
        assert frames == []
        with open(header["path"], "rb") as f:
            frames = unpack_frames(f.read())
        os.remove(header["path"])
        if "compression" in header["disk-sub-header"]:
            frames = decompress(header["disk-sub-header"], frames)
        header = header["disk-sub-header"]
        obj._obj_pxy["serializer"] = header["serializer"]
    return header, frames


@distributed.protocol.dask_serialize.register(ProxyObject)
def obj_pxy_dask_serialize(obj: ProxyObject):
    """The dask serialization of ProxyObject used by Dask when communicating using TCP

    As serializers, it uses "dask" or "pickle", which means that proxied CUDA objects
    are spilled to main memory before communicated. Deserialization is needed, unless
    obj is serialized to disk on a shared filesystem see `handle_disk_serialized()`.
    """
    if obj._obj_pxy["serializer"] == "disk":
        header, frames = handle_disk_serialized(obj)
    else:
        header, frames = obj._obj_pxy_serialize(serializers=("dask", "pickle"))
    meta = obj._obj_pxy_get_init_args(include_obj=False)
    return {"proxied-header": header, "obj-pxy-meta": meta}, frames


@distributed.protocol.cuda.cuda_serialize.register(ProxyObject)
def obj_pxy_cuda_serialize(obj: ProxyObject):
    """ The CUDA serialization of ProxyObject used by Dask when communicating using UCX

    As serializers, it uses "cuda", which means that proxied CUDA objects are _not_
    spilled to main memory before communicated. However, we still have to handle disk
    serialized proxied like in `obj_pxy_dask_serialize()`
    """
    if obj._obj_pxy["serializer"] in ("dask", "pickle"):
        header, frames = obj._obj_pxy["obj"]
    elif obj._obj_pxy["serializer"] == "disk":
        header, frames = handle_disk_serialized(obj)
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
    make_meta_dispatch,
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
