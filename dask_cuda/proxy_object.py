import operator
import pickle
import sys

import dask
import dask.dataframe.methods
import dask.dataframe.utils
import distributed.protocol
import distributed.utils

from .is_device_object import is_device_object

# List of attributes that should be copied to the proxy at creation, which makes
# them accessible without deserialization of the proxied object
_FIXED_ATTRS = ["name"]


def asproxy(obj, serializers=None):
    """Wrap `obj` in a ObjectProxy object if it isn't already.

    Parameters
    ----------
    obj: object
        Object to wrap in a ObjectProxy object.
    serializers: List[Str], optional
        List of serializers to use to serialize `obj`. If None,
        no serialization is done.

    Returns
    -------
    ret: ObjectProxy
        The proxy object proxing `obj`
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

        ret = ObjectProxy(
            obj=obj,
            fixed_attr=fixed_attr,
            type_serialized=pickle.dumps(type(obj)),
            typename=dask.utils.typename(type(obj)),
            is_cuda_object=is_device_object(obj),
        )
    if serializers is not None:
        ret._obj_pxy_serialize(serializers=serializers)
    return ret


class ObjectProxy:
    __slots__ = [
        "_obj_pxy",  # A dict that holds the state of the proxy object
        "__obj_pxy_cache",  # A dict used for caching attributes
    ]

    def __init__(
        self,
        obj,
        fixed_attr,
        type_serialized,
        typename,
        is_cuda_object,
        serializers=None,
    ):
        self._obj_pxy = {
            "obj": obj,
            "fixed_attr": fixed_attr,
            "type_serialized": type_serialized,
            "typename": typename,
            "is_cuda_object": is_cuda_object,
            "serializers": serializers,
        }
        self.__obj_pxy_cache = {}

    def _obj_pxy_get_meta(self):
        """Return the metadata of the proxy object.

        Returns
        -------
        ret: dict
            Dictionary of metadata
        """
        return {k: self._obj_pxy[k] for k in self._obj_pxy.keys() if k != "obj"}

    def _obj_pxy_serialize(self, serializers):
        """Inplace serialization of the proxied object using the `serializers`

        Parameters
        ----------
        serializers: List[Str]
            List of serializers to use to serialize the proxied object.

        Returns
        -------
        header: dict
            The header of the serialized frames
        frames: List[Bytes]
            List of frames that makes up the serialized object
        """
        assert serializers is not None
        if (
            self._obj_pxy["serializers"] is not None
            and self._obj_pxy["serializers"] != serializers
        ):
            # The proxied object is serialized with other serializers
            self._obj_pxy_deserialize()

        if self._obj_pxy["serializers"] is None:
            self._obj_pxy["obj"] = distributed.protocol.serialize(
                self._obj_pxy["obj"], serializers
            )
            self._obj_pxy["serializers"] = serializers

        assert serializers == self._obj_pxy["serializers"]
        return self._obj_pxy["obj"]

    def _obj_pxy_deserialize(self):
        """Inplace deserialization of the proxied object

        Returns
        -------
        ret : object
            The proxied object (deserialized)
        """
        if self._obj_pxy["serializers"] is not None:
            header, frames = self._obj_pxy["obj"]
            self._obj_pxy["obj"] = distributed.protocol.deserialize(header, frames)
            self._obj_pxy["serializers"] = None
        return self._obj_pxy["obj"]

    def _obj_pxy_is_cuda_object(self):
        """Inplace deserialization of the proxied object

        Returns
        -------
        ret : object
            The proxied object (deserialized)
        """
        return self._obj_pxy["is_cuda_object"]

    def __getattr__(self, name):
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
        typename = self._obj_pxy["typename"]
        ret = f"<{dask.utils.typename(type(self))} at {hex(id(self))} for {typename}"
        if self._obj_pxy["serializers"] is not None:
            ret += f" (serialized={repr(self._obj_pxy['serializers'])})>"
        else:
            ret += f" at {hex(id(self._obj_pxy['obj']))}>"
        return ret

    @property
    def __class__(self):
        try:
            return self.__obj_pxy_cache["type_serialized"]
        except KeyError:
            ret = pickle.loads(self._obj_pxy["type_serialized"])
            self.__obj_pxy_cache["type_serialized"] = ret
            return ret

    def __sizeof__(self):
        if self._obj_pxy["serializers"] is not None:
            frames = self._obj_pxy["obj"][1]
            return sum(map(distributed.utils.nbytes, frames))
        else:
            return sys.getsizeof(self._obj_pxy_deserialize())

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
        ret = getattr(self._obj_pxy_deserialize(), "__array__")()
        return ret

    def __add__(self, other):
        return self._obj_pxy_deserialize() + other

    def __sub__(self, other):
        return self._obj_pxy_deserialize() - other

    def __mul__(self, other):
        return self._obj_pxy_deserialize() * other

    def __div__(self, other):
        return operator.div(self._obj_pxy_deserialize(), other)

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

    def __rdiv__(self, other):
        return operator.div(other, self._obj_pxy_deserialize())

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

    def __idiv__(self, other):
        proxied = self._obj_pxy_deserialize()
        self._obj_pxy["obj"] = operator.idiv(proxied, other)
        return self

    def __itruediv__(self, other):
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


@is_device_object.register(ObjectProxy)
def obj_pxy_is_device_object(obj: ObjectProxy):
    """
    In order to avoid de-serializing the proxied object, we call
    `_obj_pxy_is_cuda_object()` instead of the default
    `hasattr(o, "__cuda_array_interface__")` check.
    """
    return obj._obj_pxy_is_cuda_object()


@distributed.protocol.dask_serialize.register(ObjectProxy)
def obj_pxy_dask_serialize(obj: ObjectProxy):
    """
    The generic serialization of ObjectProxy used by Dask when communicating
    ObjectProxy. As serializers, it uses "dask" or "pickle", which means
    that proxied CUDA objects are spilled to main memory before communicated.
    """
    header, frames = obj._obj_pxy_serialize(serializers=["dask", "pickle"])
    return {"proxied-header": header, "obj-pxy-meta": obj._obj_pxy_get_meta()}, frames


@distributed.protocol.cuda.cuda_serialize.register(ObjectProxy)
def obj_pxy_cuda_serialize(obj: ObjectProxy):
    """
    The CUDA serialization of ObjectProxy used by Dask when communicating using UCX
    or another CUDA friendly communicantion library. As serializers, it uses "cuda",
    "dask" or "pickle", which means that proxied CUDA objects are _not_ spilled to
    main memory.
    """
    if obj._obj_pxy["serializers"] is not None:  # Already serialized
        header, frames = obj._obj_pxy["obj"]
    else:
        header, frames = obj._obj_pxy_serialize(serializers=["cuda", "dask", "pickle"])
    return {"proxied-header": header, "obj-pxy-meta": obj._obj_pxy_get_meta()}, frames


@distributed.protocol.dask_deserialize.register(ObjectProxy)
@distributed.protocol.cuda.cuda_deserialize.register(ObjectProxy)
def obj_pxy_dask_deserialize(header, frames):
    """
    The generic deserialization of ObjectProxy. Notice, it doesn't deserialize
    the proxied object at this time. When accessed, the proxied object are
    deserialized using the same serializers that were used when the object was
    serialized.
    """
    return ObjectProxy(
        obj=(header["proxied-header"], frames),
        **header["obj-pxy-meta"],
    )


@dask.dataframe.utils.hash_object_dispatch.register(ObjectProxy)
def obj_pxy_hash_object(obj: ObjectProxy, index=True):
    return dask.dataframe.utils.hash_object_dispatch(obj._obj_pxy_deserialize(), index)


@dask.dataframe.utils.group_split_dispatch.register(ObjectProxy)
def obj_pxy_group_split(obj: ObjectProxy, c, k, ignore_index=False):
    return dask.dataframe.utils.group_split_dispatch(
        obj._obj_pxy_deserialize(), c, k, ignore_index
    )


@dask.dataframe.utils.make_scalar.register(ObjectProxy)
def obj_pxy_make_scalar(obj: ObjectProxy):
    return dask.dataframe.utils.make_scalar(obj._obj_pxy_deserialize())


@dask.dataframe.methods.concat_dispatch.register(ObjectProxy)
def obj_pxy_concat(objs, *args, **kwargs):
    # Deserialize concat inputs (in-place)
    for i in range(len(objs)):
        try:
            objs[i] = objs[i]._obj_pxy_deserialize()
        except AttributeError:
            pass
    return dask.dataframe.methods.concat(objs, *args, **kwargs)
