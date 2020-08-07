import pickle
import sys

import dask
import distributed.protocol
import distributed.utils

# List of attributes that should be copied to the proxy at creation, which makes
# them accessible without deserialization of the proxied object
_FIXED_ATTRS = ["name"]


def asproxy(obj, serialize_obj=True, serializers=["dask", "pickle"]):
    if hasattr(obj, "_obj_pxy"):
        return obj  # Already a proxy object

    fixed_attr = {}
    for attr in _FIXED_ATTRS:
        try:
            fixed_attr[attr] = getattr(obj, attr)
        except AttributeError:
            pass

    orig_obj = obj
    if serialize_obj:
        obj = distributed.protocol.serialize(obj, serializers=serializers)

    return ObjectProxy(
        obj=obj,
        is_serialized=serialize_obj,
        fixed_attr=fixed_attr,
        type_serialized=pickle.dumps(type(orig_obj)),
        typename=dask.utils.typename(type(orig_obj)),
        serializers=serializers,
    )


class ObjectProxy:
    __slots__ = [
        "_obj_pxy",  # A dict that holds the state of the proxy object
        "__obj_pxy_cache",  # A dict used for caching attributes
    ]

    def __init__(
        self, obj, is_serialized, fixed_attr, type_serialized, typename, serializers
    ):
        self._obj_pxy = {
            "obj": obj,
            "is_serialized": is_serialized,
            "fixed_attr": fixed_attr,
            "type_serialized": type_serialized,
            "typename": typename,
            "serializers": serializers,
        }
        self.__obj_pxy_cache = {}

    def _obj_pxy_get_meta(self):
        return {k: self._obj_pxy[k] for k in self._obj_pxy.keys() if k != "obj"}

    def _obj_pxy_serialize(self):
        if not self._obj_pxy["is_serialized"]:
            self._obj_pxy["obj"] = distributed.protocol.serialize(
                self._obj_pxy["obj"], self._obj_pxy["serializers"]
            )
            self._obj_pxy["is_serialized"] = True
        return self._obj_pxy["obj"]

    def _obj_pxy_deserialize(self):
        if self._obj_pxy["is_serialized"]:
            header, frames = self._obj_pxy["obj"]
            self._obj_pxy["obj"] = distributed.protocol.deserialize(header, frames)
            self._obj_pxy["is_serialized"] = False
        return self._obj_pxy["obj"]

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
        if self._obj_pxy["is_serialized"]:
            ret += " (serialized)>"
        else:
            ret += f" at {hex(id(self._obj_pxy['obj']))}>"
        return ret

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

    @property
    def __class__(self):
        try:
            return self.__obj_pxy_cache["type_serialized"]
        except KeyError:
            ret = pickle.loads(self._obj_pxy["type_serialized"])
            self.__obj_pxy_cache["type_serialized"] = ret
            return ret

    def __sizeof__(self):
        if self._obj_pxy["is_serialized"]:
            frames = self._obj_pxy["obj"][1]
            return sum(map(distributed.utils.nbytes, frames))
        else:
            return sys.getsizeof(self._obj_pxy_deserialize())


@distributed.protocol.dask_serialize.register(ObjectProxy)
def obj_pxy_dask_serialize(obj: ObjectProxy):
    header, frames = obj._obj_pxy_serialize()
    return {"proxied-header": header, "obj-pxy-meta": obj._obj_pxy_get_meta()}, frames


@distributed.protocol.dask_deserialize.register(ObjectProxy)
def obj_pxy_dask_deserialize(header, frames):
    return ObjectProxy(
        obj=(header["proxied-header"], frames), **header["obj-pxy-meta"],
    )
