import operator
import pickle
import threading
from collections import OrderedDict

import dask
import dask.dataframe.methods
import dask.dataframe.utils
import distributed.protocol
import distributed.utils
from dask.sizeof import sizeof

from .is_device_object import is_device_object

# List of attributes that should be copied to the proxy at creation, which makes
# them accessible without deserialization of the proxied object
_FIXED_ATTRS = ["name"]


def asproxy(obj, serializers=None, subclass=None):
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
                fixed_attr[attr] = getattr(obj, attr)
            except AttributeError:
                pass

        if subclass is None:
            subclass = ProxyObject
        ret = subclass(
            obj=obj,
            fixed_attr=fixed_attr,
            type_serialized=pickle.dumps(type(obj)),
            typename=dask.utils.typename(type(obj)),
            is_cuda_object=is_device_object(obj),
            subclass=pickle.dumps(subclass) if subclass else None,
            serializers=None,
        )
    if serializers is not None:
        ret._obj_pxy_serialize(serializers=serializers)
    return ret


def unproxy(obj):
    """Unwrap ProxyObject objects and pass-through anything else.

    Use this function to retrieve the proxied object.

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
        pass
    return obj


class ProxyObject:
    """Object wrapper/proxy for serializable objects

    This is used by DeviceHostFile to delay deserialization of returned objects.

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
    """

    __slots__ = [
        "_obj_pxy",  # A dict that holds the state of the proxy object
        "_obj_pxy_lock",  # Threading lock for all obj_pxy access
        "__obj_pxy_cache",  # A dict used for caching attributes
    ]

    def __init__(
        self,
        obj,
        fixed_attr,
        type_serialized,
        typename,
        is_cuda_object,
        subclass,
        serializers,
    ):
        self._obj_pxy = {
            "obj": obj,
            "fixed_attr": fixed_attr,
            "type_serialized": type_serialized,
            "typename": typename,
            "is_cuda_object": is_cuda_object,
            "subclass": subclass,
            "serializers": serializers,
        }
        self._obj_pxy_lock = threading.RLock()
        self.__obj_pxy_cache = {}

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
        ]
        return OrderedDict([(a, self._obj_pxy[a]) for a in args])

    def _obj_pxy_serialize(self, serializers):
        """Inplace serialization of the proxied object using the `serializers`

        Parameters
        ----------
        serializers: list(str)
            List of serializers to use to serialize the proxied object.

        Returns
        -------
        header: dict
            The header of the serialized frames
        frames: list(bytes)
            List of frames that make up the serialized object
        """
        if not serializers:
            raise ValueError("Please specify a list of serializers")

        with self._obj_pxy_lock:
            if self._obj_pxy["serializers"] is not None and tuple(
                self._obj_pxy["serializers"]
            ) != tuple(serializers):
                # The proxied object is serialized with other serializers
                self._obj_pxy_deserialize()

            if self._obj_pxy["serializers"] is None:
                self._obj_pxy["obj"] = distributed.protocol.serialize(
                    self._obj_pxy["obj"], serializers, on_error="raise"
                )
                self._obj_pxy["serializers"] = serializers

            return self._obj_pxy["obj"]

    def _obj_pxy_deserialize(self):
        """Inplace deserialization of the proxied object

        Returns
        -------
        object
            The proxied object (deserialized)
        """
        with self._obj_pxy_lock:
            if self._obj_pxy["serializers"] is not None:
                header, frames = self._obj_pxy["obj"]
                self._obj_pxy["obj"] = distributed.protocol.deserialize(header, frames)
                self._obj_pxy["serializers"] = None
            return self._obj_pxy["obj"]

    def _obj_pxy_is_cuda_object(self):
        """Return whether the proxied object is a CUDA or not

        Returns
        -------
        ret : boolean
            Is the proxied object a CUDA object?
        """
        with self._obj_pxy_lock:
            return self._obj_pxy["is_cuda_object"]

    def __reduce__(self):
        """Serialization of ProxyObject that uses pickle"""
        self._obj_pxy_serialize(serializers=["pickle"])
        args = self._obj_pxy_get_init_args()
        if args["subclass"]:
            subclass = pickle.loads(args["subclass"])
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

    def __str__(self):
        return str(self._obj_pxy_deserialize())

    def __repr__(self):
        with self._obj_pxy_lock:
            typename = self._obj_pxy["typename"]
            ret = (
                f"<{dask.utils.typename(type(self))} at {hex(id(self))} for {typename}"
            )
            if self._obj_pxy["serializers"] is not None:
                ret += f" (serialized={repr(self._obj_pxy['serializers'])})>"
            else:
                ret += f" at {hex(id(self._obj_pxy['obj']))}>"
            return ret

    @property
    def __class__(self):
        with self._obj_pxy_lock:
            try:
                return self.__obj_pxy_cache["type_serialized"]
            except KeyError:
                ret = pickle.loads(self._obj_pxy["type_serialized"])
                self.__obj_pxy_cache["type_serialized"] = ret
                return ret

    def __sizeof__(self):
        with self._obj_pxy_lock:
            if self._obj_pxy["serializers"] is not None:
                frames = self._obj_pxy["obj"][1]
                return sum(map(distributed.utils.nbytes, frames))
            else:
                return sizeof(self._obj_pxy_deserialize())

    def __len__(self):
        return len(self._obj_pxy_deserialize())

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

    def __array__(self):
        return getattr(self._obj_pxy_deserialize(), "__array__")()

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
    header, frames = obj._obj_pxy_serialize(serializers=["dask", "pickle"])
    meta = obj._obj_pxy_get_init_args(include_obj=False)
    return {"proxied-header": header, "obj-pxy-meta": meta}, frames


@distributed.protocol.cuda.cuda_serialize.register(ProxyObject)
def obj_pxy_cuda_serialize(obj: ProxyObject):
    """
    The CUDA serialization of ProxyObject used by Dask when communicating using UCX
    or another CUDA friendly communication library. As serializers, it uses "cuda",
    "dask" or "pickle", which means that proxied CUDA objects are _not_ spilled to
    main memory.
    """
    if obj._obj_pxy["serializers"] is not None:  # Already serialized
        header, frames = obj._obj_pxy["obj"]
    else:
        header, frames = obj._obj_pxy_serialize(serializers=["cuda", "dask", "pickle"])
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
        subclass = pickle.loads(meta["subclass"])
    return subclass(obj=(header["proxied-header"], frames), **header["obj-pxy-meta"],)


@dask.dataframe.utils.hash_object_dispatch.register(ProxyObject)
def obj_pxy_hash_object(obj: ProxyObject, index=True):
    return dask.dataframe.utils.hash_object_dispatch(obj._obj_pxy_deserialize(), index)


@dask.dataframe.utils.group_split_dispatch.register(ProxyObject)
def obj_pxy_group_split(obj: ProxyObject, c, k, ignore_index=False):
    return dask.dataframe.utils.group_split_dispatch(
        obj._obj_pxy_deserialize(), c, k, ignore_index
    )


@dask.dataframe.utils.make_scalar.register(ProxyObject)
def obj_pxy_make_scalar(obj: ProxyObject):
    return dask.dataframe.utils.make_scalar(obj._obj_pxy_deserialize())


@dask.dataframe.methods.concat_dispatch.register(ProxyObject)
def obj_pxy_concat(objs, *args, **kwargs):
    # Deserialize concat inputs (in-place)
    for i in range(len(objs)):
        try:
            objs[i] = objs[i]._obj_pxy_deserialize()
        except AttributeError:
            pass
    return dask.dataframe.methods.concat(objs, *args, **kwargs)
