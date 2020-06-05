import functools
import os

import dask
from distributed.protocol import (
    dask_deserialize,
    dask_serialize,
    deserialize,
    deserialize_bytes,
    serialize,
    serialize_bytelist,
)
from distributed.utils import nbytes
from distributed.worker import weight

import numpy
from zict import Buffer, File, Func
from zict.common import ZictBase

from .is_device_object import is_device_object
from .utils import nvtx_annotate


class DeviceSerialized:
    """ Store device object on the host

    This stores a device-side object as

    1.  A msgpack encodable header
    2.  A list of `bytes`-like objects (like NumPy arrays)
        that are in host memory
    """

    def __init__(self, header, parts):
        self.header = header
        self.parts = parts

    def __sizeof__(self):
        return sum(map(nbytes, self.parts))


@dask_serialize.register(DeviceSerialized)
def device_serialize(obj):
    headers, frames = serialize(obj.parts)
    header = {"sub-headers": headers, "main-header": obj.header}
    return header, frames


@dask_deserialize.register(DeviceSerialized)
def device_deserialize(header, frames):
    parts = deserialize(header["sub-headers"], frames)
    return DeviceSerialized(header["main-header"], parts)


@nvtx_annotate("SPILL_D2H", color="red", domain="dask_cuda")
def device_to_host(obj: object) -> DeviceSerialized:
    header, frames = serialize(obj, serializers=["dask", "pickle"])
    frames = [numpy.asarray(f) for f in frames]
    return DeviceSerialized(header, frames)


@nvtx_annotate("SPILL_H2D", color="green", domain="dask_cuda")
def host_to_device(s: DeviceSerialized) -> object:
    return deserialize(s.header, s.parts)


class DeviceHostFile(ZictBase):
    """ Manages serialization/deserialization of objects.

    Three LRU cache levels are controlled, for device, host and disk.
    Each level takes care of serializing objects once its limit has been
    reached and pass it to the subsequent level. Similarly, each cache
    may deserialize the object, but storing it back in the appropriate
    cache, depending on the type of object being deserialized.

    Parameters
    ----------
    device_memory_limit: int
        Number of bytes of CUDA device memory for device LRU cache,
        spills to host cache once filled.
    memory_limit: int
        Number of bytes of host memory for host LRU cache, spills to
        disk once filled. Setting this to 0 means unlimited host memory,
        implies no spilling to disk.
    local_directory: path
        Path where to store serialized objects on disk
    """

    def __init__(
        self, device_memory_limit=None, memory_limit=None, local_directory=None,
    ):
        if local_directory is None:
            local_directory = dask.config.get("temporary-directory") or os.getcwd()
            os.makedirs(local_directory, exist_ok=True)
            local_directory = os.path.join(local_directory, "dask-worker-space")

        self.disk_func_path = os.path.join(local_directory, "storage")

        self.host_func = dict()
        self.disk_func = Func(
            functools.partial(serialize_bytelist, on_error="raise"),
            deserialize_bytes,
            File(self.disk_func_path),
        )
        if memory_limit == 0:
            self.host_buffer = self.host_func
        else:
            self.host_buffer = Buffer(
                self.host_func, self.disk_func, memory_limit, weight=weight
            )

        self.device_keys = set()
        self.device_func = dict()
        self.device_host_func = Func(device_to_host, host_to_device, self.host_buffer)
        self.device_buffer = Buffer(
            self.device_func, self.device_host_func, device_memory_limit, weight=weight
        )

        self.device = self.device_buffer.fast.d
        self.host = self.host_buffer if memory_limit == 0 else self.host_buffer.fast.d
        self.disk = None if memory_limit == 0 else self.host_buffer.slow.d

        # For Worker compatibility only, where `fast` is host memory buffer
        self.fast = self.host_buffer if memory_limit == 0 else self.host_buffer.fast

    def __setitem__(self, key, value):
        if is_device_object(value):
            self.device_keys.add(key)
            self.device_buffer[key] = value
        else:
            self.host_buffer[key] = value

    def __getitem__(self, key):
        if key in self.device_keys:
            return self.device_buffer[key]
        elif key in self.host_buffer:
            return self.host_buffer[key]
        else:
            raise KeyError(key)

    def __len__(self):
        return len(self.device_buffer)

    def __iter__(self):
        return iter(self.device_buffer)

    def __delitem__(self, key):
        self.device_keys.discard(key)
        del self.device_buffer[key]
