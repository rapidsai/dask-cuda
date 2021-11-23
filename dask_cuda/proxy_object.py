import copy as _copy
import functools
import operator
import os
import pickle
import time
import uuid
from collections import OrderedDict
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional, Tuple, Type, Union

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
from distributed.worker import dumps_function, loads_function

from dask_cuda.disk_io import disk_read

try:
    from dask.dataframe.backends import concat_pandas
except ImportError:
    from dask.dataframe.methods import concat_pandas

try:
    from dask.dataframe.dispatch import make_meta_dispatch as make_meta_dispatch
except ImportError:
    from dask.dataframe.utils import make_meta as make_meta_dispatch

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
            ProxyDetail(
                obj=obj,
                fixed_attr=fixed_attr,
                type_serialized=pickle.dumps(type(obj)),
                typename=dask.utils.typename(type(obj)),
                is_cuda_object=is_device_object(obj),
                subclass=subclass_serialized,
                serializer=None,
                explicit_proxy=False,
            )
        )
    if serializers is not None:
        ret._pxy_serialize(serializers=serializers)
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
        obj = obj._pxy_deserialize()
    except AttributeError:
        if type(obj) in (list, tuple, set, frozenset):
            return type(obj)(unproxy(o) for o in obj)
    return obj


def _pxy_cache_wrapper(attr_name: str):
    """Caching the access of attr_name in ProxyObject._pxy_cache"""

    def wrapper1(func):
        @functools.wraps(func)
        def wrapper2(self: "ProxyObject"):
            try:
                return self._pxy_cache[attr_name]
            except KeyError:
                ret = func(self)
                self._pxy_cache[attr_name] = ret
                return ret

        return wrapper2

    return wrapper1


class ProxyManagerDummy:
    """Dummy of a ProxyManager that does nothing

    This is a dummy class used as the manager when no manager has been
    registered the proxy object. It implements dummy methods that doesn't
    do anything it is purely for convenience.
    """

    def add(self, *args, **kwargs):
        pass

    def remove(self, *args, **kwargs):
        pass

    def maybe_evict(self, *args, **kwargs):
        pass

    @property
    def lock(self):
        return nullcontext()


class ProxyDetail:
    """Details of a ProxyObject

    In order to avoid having to use thread locks, a ProxyObject maintains
    its state in a ProxyDetail object. The idea is to first make a copy
    of the ProxyDetail object before modifying it and then assign the copy
    back to the ProxyObject in one atomic instruction.

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
    manager: ProxyManager or ProxyManagerDummy
        The manager to manage this proxy object or a dummy.
        The manager tallies the total memory usage of proxies and
        evicts/serialize proxy objects as needed.
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
        manager: Union["ProxyManager", ProxyManagerDummy] = ProxyManagerDummy(),
    ):
        self.obj = obj
        self.fixed_attr = fixed_attr
        self.type_serialized = type_serialized
        self.typename = typename
        self.is_cuda_object = is_cuda_object
        self.subclass = subclass
        self.serializer = serializer
        self.explicit_proxy = explicit_proxy
        self.manager = manager
        self.last_access: float = 0.0

    def get_init_args(self, include_obj=False) -> OrderedDict:
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
        return OrderedDict([(a, getattr(self, a)) for a in args])

    def is_serialized(self) -> bool:
        """Return whether the proxied object is serialized or not"""
        return self.serializer is not None

    def serialize(self, serializers: Iterable[str]) -> Tuple[dict, list]:
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

        if self.serializer is not None:
            if self.serializer in serializers:
                return self.obj  # Nothing to be done
            else:
                # The proxied object is serialized with other serializers
                self.deserialize()

        header, _ = self.obj = distributed.protocol.serialize(
            self.obj, serializers, on_error="raise"
        )
        assert "is-collection" not in header  # Collections not allowed
        self.serializer = header["serializer"]
        return self.obj

    def deserialize(self, maybe_evict: bool = True, nbytes=None):
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

        if self.is_serialized():
            # When not deserializing a CUDA-serialized proxied, tell the
            # manager that it might have to evict because of the increased
            # device memory usage.
            if maybe_evict and self.serializer != "cuda":
                if nbytes is None:
                    _, frames = self.obj
                    nbytes = sum(map(distributed.utils.nbytes, frames))
                self.manager.maybe_evict(nbytes)

            # Deserialize the proxied object
            header, frames = self.obj
            self.obj = distributed.protocol.deserialize(header, frames)
            self.serializer = None
            self.last_access = time.monotonic()
        return self.obj


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
    _pxy: ProxyDetail
        Details of all proxy information of the underlaying proxied object.
        Access to _pxy is not pass-through to the proxied object, which is
        the case for most other access to the ProxyObject.

    _pxy_cache: dict
        A dictionary used for caching attributes

    Parameters
    ----------
    detail: ProxyDetail
        The Any kind of object to be proxied.
    """

    def __init__(self, detail: ProxyDetail):
        self._pxy_detail = detail
        self._pxy_cache: Dict[str, Any] = {}

    def _pxy_get(self, copy=False) -> ProxyDetail:
        if copy:
            return _copy.copy(self._pxy_detail)
        else:
            return self._pxy_detail

    def _pxy_set(self, proxy_detail: ProxyDetail):
        with proxy_detail.manager.lock:
            self._pxy_detail = proxy_detail
            proxy_detail.manager.add(proxy=self, serializer=proxy_detail.serializer)

    def __del__(self):
        """We have to unregister us from the manager if any"""
        pxy = self._pxy_get()
        pxy.manager.remove(self)
        if pxy.serializer == "disk":
            header, _ = pxy.obj
            os.remove(header["disk-io-header"]["path"])

    def _pxy_serialize(
        self, serializers: Iterable[str], proxy_detail: ProxyDetail = None,
    ) -> None:
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

        pxy = self._pxy_get(copy=True) if not proxy_detail else proxy_detail
        if pxy.serializer is not None and pxy.serializer in serializers:
            return  # Nothing to be done

        pxy.serialize(serializers=serializers)
        self._pxy_set(pxy)

        # Invalidate the (possible) cached "device_memory_objects"
        self._pxy_cache.pop("device_memory_objects", None)

    def _pxy_deserialize(
        self, maybe_evict: bool = True, proxy_detail: ProxyDetail = None
    ):
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

        pxy = self._pxy_get(copy=True) if not proxy_detail else proxy_detail
        if not pxy.is_serialized():
            return pxy.obj

        ret = pxy.deserialize(maybe_evict=maybe_evict, nbytes=self.__sizeof__())
        self._pxy_set(pxy)
        return ret

    def __reduce__(self):
        """Serialization of ProxyObject that uses pickle"""
        pxy = self._pxy_get(copy=True)
        pxy.serialize(serializers=("pickle",))
        if pxy.subclass:
            subclass = loads_function(pxy.subclass)
        else:
            subclass = ProxyObject

        # Make sure the frames are all bytes
        header, frames = pxy.obj
        pxy.obj = (header, [bytes(f) for f in frames])
        self._pxy_set(pxy)
        return (subclass, (pxy,))

    def __getattr__(self, name):
        pxy = self._pxy_get()
        if name in _FIXED_ATTRS:
            try:
                return pxy.fixed_attr[name]
            except KeyError:
                raise AttributeError(
                    f"type object '{pxy.typename}' has no attribute '{name}'"
                )
        return getattr(self._pxy_deserialize(), name)

    def __setattr__(self, name: str, val):
        if name.startswith("_pxy_"):
            return object.__setattr__(self, name, val)

        pxy = self._pxy_get(copy=True)
        if name in _FIXED_ATTRS:
            pxy.fixed_attr[name] = val
        else:
            object.__setattr__(pxy.deserialize(nbytes=self.__sizeof__()), name, val)
        self._pxy_set(pxy)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        inputs = tuple(
            o._pxy_deserialize() if isinstance(o, ProxyObject) else o for o in inputs
        )
        kwargs = {
            key: value._pxy_deserialize() if isinstance(value, ProxyObject) else value
            for key, value in kwargs.items()
        }
        return self._pxy_deserialize().__array_ufunc__(ufunc, method, *inputs, **kwargs)

    def __str__(self):
        return str(self._pxy_deserialize())

    def __repr__(self):
        pxy = self._pxy_get()
        ret = f"<{dask.utils.typename(type(self))} "
        ret += f"at {hex(id(self))} of {pxy.typename}"
        if pxy.is_serialized():
            ret += f" (serialized={repr(pxy.serializer)})>"
        else:
            ret += f" at {hex(id(pxy.obj))}>"
        return ret

    @property  # type: ignore  # mypy doesn't support decorated property
    @_pxy_cache_wrapper("type_serialized")
    def __class__(self):
        return pickle.loads(self._pxy_get().type_serialized)

    @_pxy_cache_wrapper("sizeof")
    def __sizeof__(self):
        """Returns the size of the proxy object (serialized or not)

        Notice, we cache the result even though the size of proxied object
        when serialized or not serialized might slightly differ.
        """
        pxy = self._pxy_get()
        if pxy.is_serialized():
            _, frames = pxy.obj
            return sum(map(distributed.utils.nbytes, frames))
        else:
            return sizeof(pxy.obj)

    def __len__(self):
        pxy = self._pxy_get(copy=True)
        ret = pxy.fixed_attr.get("__len__", None)
        if ret is None:
            ret = len(pxy.deserialize(nbytes=self.__sizeof__()))
            pxy.fixed_attr["__len__"] = ret
            self._pxy_set(pxy)
        return ret

    def __contains__(self, value):
        return value in self._pxy_deserialize()

    def __getitem__(self, key):
        return self._pxy_deserialize()[key]

    def __setitem__(self, key, value):
        self._pxy_deserialize()[key] = value

    def __delitem__(self, key):
        del self._pxy_deserialize()[key]

    def __getslice__(self, i, j):
        return self._pxy_deserialize()[i:j]

    def __setslice__(self, i, j, value):
        self._pxy_deserialize()[i:j] = value

    def __delslice__(self, i, j):
        del self._pxy_deserialize()[i:j]

    def __iter__(self):
        return iter(self._pxy_deserialize())

    def __array__(self, *args, **kwargs):
        return getattr(self._pxy_deserialize(), "__array__")(*args, **kwargs)

    def __lt__(self, other):
        return self._pxy_deserialize() < other

    def __le__(self, other):
        return self._pxy_deserialize() <= other

    def __eq__(self, other):
        return self._pxy_deserialize() == other

    def __ne__(self, other):
        return self._pxy_deserialize() != other

    def __gt__(self, other):
        return self._pxy_deserialize() > other

    def __ge__(self, other):
        return self._pxy_deserialize() >= other

    def __add__(self, other):
        return self._pxy_deserialize() + other

    def __sub__(self, other):
        return self._pxy_deserialize() - other

    def __mul__(self, other):
        return self._pxy_deserialize() * other

    def __truediv__(self, other):
        return operator.truediv(self._pxy_deserialize(), other)

    def __floordiv__(self, other):
        return self._pxy_deserialize() // other

    def __mod__(self, other):
        return self._pxy_deserialize() % other

    def __divmod__(self, other):
        return divmod(self._pxy_deserialize(), other)

    def __pow__(self, other):
        return pow(self._pxy_deserialize(), other)

    def __lshift__(self, other):
        return self._pxy_deserialize() << other

    def __rshift__(self, other):
        return self._pxy_deserialize() >> other

    def __and__(self, other):
        return self._pxy_deserialize() & other

    def __xor__(self, other):
        return self._pxy_deserialize() ^ other

    def __or__(self, other):
        return self._pxy_deserialize() | other

    def __radd__(self, other):
        return other + self._pxy_deserialize()

    def __rsub__(self, other):
        return other - self._pxy_deserialize()

    def __rmul__(self, other):
        return other * self._pxy_deserialize()

    def __rtruediv__(self, other):
        return operator.truediv(other, self._pxy_deserialize())

    def __rfloordiv__(self, other):
        return other // self._pxy_deserialize()

    def __rmod__(self, other):
        return other % self._pxy_deserialize()

    def __rdivmod__(self, other):
        return divmod(other, self._pxy_deserialize())

    def __rpow__(self, other, *args):
        return pow(other, self._pxy_deserialize(), *args)

    def __rlshift__(self, other):
        return other << self._pxy_deserialize()

    def __rrshift__(self, other):
        return other >> self._pxy_deserialize()

    def __rand__(self, other):
        return other & self._pxy_deserialize()

    def __rxor__(self, other):
        return other ^ self._pxy_deserialize()

    def __ror__(self, other):
        return other | self._pxy_deserialize()

    def __iadd__(self, other):
        pxy = self._pxy_get(copy=True)
        proxied = pxy.deserialize(nbytes=self.__sizeof__())
        proxied += other
        self._pxy_set(pxy)
        return self

    def __isub__(self, other):
        pxy = self._pxy_get(copy=True)
        proxied = pxy.deserialize(nbytes=self.__sizeof__())
        proxied -= other
        self._pxy_set(pxy)
        return self

    def __imul__(self, other):
        pxy = self._pxy_get(copy=True)
        proxied = pxy.deserialize(nbytes=self.__sizeof__())
        proxied *= other
        self._pxy_set(pxy)
        return self

    def __itruediv__(self, other):
        pxy = self._pxy_get(copy=True)
        proxied = pxy.deserialize(nbytes=self.__sizeof__())
        pxy.obj = operator.itruediv(proxied, other)
        self._pxy_set(pxy)

    def __ifloordiv__(self, other):
        pxy = self._pxy_get(copy=True)
        proxied = pxy.deserialize(nbytes=self.__sizeof__())
        proxied //= other
        self._pxy_set(pxy)
        return self

    def __imod__(self, other):
        pxy = self._pxy_get(copy=True)
        proxied = pxy.deserialize(nbytes=self.__sizeof__())
        proxied %= other
        self._pxy_set(pxy)
        return self

    def __ipow__(self, other):
        pxy = self._pxy_get(copy=True)
        proxied = pxy.deserialize(nbytes=self.__sizeof__())
        proxied **= other
        self._pxy_set(pxy)
        return self

    def __ilshift__(self, other):
        pxy = self._pxy_get(copy=True)
        proxied = pxy.deserialize(nbytes=self.__sizeof__())
        proxied <<= other
        self._pxy_set(pxy)
        return self

    def __irshift__(self, other):
        pxy = self._pxy_get(copy=True)
        proxied = pxy.deserialize(nbytes=self.__sizeof__())
        proxied >>= other
        self._pxy_set(pxy)
        return self

    def __iand__(self, other):
        pxy = self._pxy_get(copy=True)
        proxied = pxy.deserialize(nbytes=self.__sizeof__())
        proxied &= other
        self._pxy_set(pxy)
        return self

    def __ixor__(self, other):
        pxy = self._pxy_get(copy=True)
        proxied = pxy.deserialize(nbytes=self.__sizeof__())
        proxied ^= other
        self._pxy_set(pxy)
        return self

    def __ior__(self, other):
        pxy = self._pxy_get(copy=True)
        proxied = pxy.deserialize(nbytes=self.__sizeof__())
        proxied |= other
        self._pxy_set(pxy)
        return self

    def __neg__(self):
        return -self._pxy_deserialize()

    def __pos__(self):
        return +self._pxy_deserialize()

    def __abs__(self):
        return abs(self._pxy_deserialize())

    def __invert__(self):
        return ~self._pxy_deserialize()

    def __int__(self):
        return int(self._pxy_deserialize())

    def __float__(self):
        return float(self._pxy_deserialize())

    def __complex__(self):
        return complex(self._pxy_deserialize())

    def __index__(self):
        return operator.index(self._pxy_deserialize())


@is_device_object.register(ProxyObject)
def obj_pxy_is_device_object(obj: ProxyObject):
    """
    In order to avoid de-serializing the proxied object,
    we check `is_cuda_object` instead of the default
    `hasattr(o, "__cuda_array_interface__")` check.
    """
    return obj._pxy_get().is_cuda_object


def handle_disk_serialized(pxy: ProxyDetail):
    """Handle serialization of an already disk serialized proxy

    On a shared filesystem, we do not have to deserialize instead we
    make a hard link of the file.

    On a non-shared filesystem, we deserialize the proxy to host memory.
    """
    header, frames = pxy.obj
    disk_io_header = header["disk-io-header"]
    if disk_io_header["shared-filesystem"]:
        old_path = disk_io_header["path"]
        new_path = f"{old_path}-linked-{uuid.uuid4()}"
        os.link(old_path, new_path)
        header = _copy.deepcopy(header)
        header["disk-io-header"]["path"] = new_path
    else:
        # When not on a shared filesystem, we deserialize to host memory
        assert frames == []
        frames = disk_read(disk_io_header)
        os.remove(disk_io_header["path"])
        if "compression" in header["serialize-header"]:
            frames = decompress(header["serialize-header"], frames)
        header = header["serialize-header"]
        pxy.serializer = header["serializer"]
    return header, frames


@distributed.protocol.dask_serialize.register(ProxyObject)
def obj_pxy_dask_serialize(obj: ProxyObject):
    """The dask serialization of ProxyObject used by Dask when communicating using TCP

    As serializers, it uses "dask" or "pickle", which means that proxied CUDA objects
    are spilled to main memory before communicated. Deserialization is needed, unless
    obj is serialized to disk on a shared filesystem see `handle_disk_serialized()`.
    """
    pxy = obj._pxy_get(copy=True)
    if pxy.serializer == "disk":
        header, frames = handle_disk_serialized(pxy)
    else:
        header, frames = pxy.serialize(serializers=("dask", "pickle"))
    obj._pxy_set(pxy)

    return {"proxied-header": header, "obj-pxy-detail": pxy.get_init_args()}, frames


@distributed.protocol.cuda.cuda_serialize.register(ProxyObject)
def obj_pxy_cuda_serialize(obj: ProxyObject):
    """ The CUDA serialization of ProxyObject used by Dask when communicating using UCX

    As serializers, it uses "cuda", which means that proxied CUDA objects are _not_
    spilled to main memory before communicated. However, we still have to handle disk
    serialized proxied like in `obj_pxy_dask_serialize()`
    """
    pxy = obj._pxy_get(copy=True)
    if pxy.serializer in ("dask", "pickle"):
        header, frames = pxy.obj
    elif pxy.serializer == "disk":
        header, frames = handle_disk_serialized(pxy)
        obj._pxy_set(pxy)

    else:
        # Notice, since obj._pxy_serialize() is a inplace operation, we make a
        # shallow copy of `obj` to avoid introducing a CUDA-serialized object in
        # the worker's data store.
        header, frames = pxy.serialize(serializers=("cuda",))

    return {"proxied-header": header, "obj-pxy-detail": pxy.get_init_args()}, frames


@distributed.protocol.dask_deserialize.register(ProxyObject)
@distributed.protocol.cuda.cuda_deserialize.register(ProxyObject)
def obj_pxy_dask_deserialize(header, frames):
    """
    The generic deserialization of ProxyObject. Notice, it doesn't deserialize
    the proxied object at this time. When accessed, the proxied object are
    deserialized using the same serializers that were used when the object was
    serialized.
    """
    args = header["obj-pxy-detail"]
    if args["subclass"] is None:
        subclass = ProxyObject
    else:
        subclass = loads_function(args["subclass"])
    return subclass(ProxyDetail(obj=(header["proxied-header"], frames), **args))


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
