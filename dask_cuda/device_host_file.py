from zict import Buffer, File, Func
from zict.common import ZictBase
from distributed.protocol import deserialize_bytes, serialize_bytes
from distributed.worker import weight

from functools import partial
import os


def _is_device_object(obj):
    """
    Check if obj is a device object, by checking if it has a
    __cuda_array_interface__ attribute or is instance of cudf.DataFrame
    or cudf.Series.
    """
    try:
        import cudf
        _device_instances = [cudf.DataFrame, cudf.Series]
    except ImportError:
        _device_instances = []
    return (hasattr(obj, "__cuda_array_interface__") or
            any([isinstance(obj, inst) for inst in _device_instances]))


def _serialize_if_device(obj):
    """ Serialize an object if it's a device object """
    if _is_device_object(obj):
        return serialize_bytes(obj, on_error="raise")
    else:
        return obj


def _deserialize_if_device(obj):
    """ Deserialize an object if it's an instance of bytes """
    if isinstance(obj, bytes):
        return deserialize_bytes(obj)
    else:
        return obj


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
        disk once filled.
    local_dir: path
        Path where to store serialized objects on disk
    """

    def __init__(
        self, device_memory_limit=None, memory_limit=None, local_dir="dask-worker-space"
    ):
        path = os.path.join(local_dir, "storage")

        self.host_func = dict()
        self.disk_func = Func(
            partial(serialize_bytes, on_error="raise"), deserialize_bytes, File(path)
        )
        self.host_buffer = Buffer(
            self.host_func, self.disk_func, memory_limit, weight=weight
        )

        self.device_func = dict()
        self.device_host_func = Func(
            _serialize_if_device, _deserialize_if_device, self.host_buffer
        )
        self.device_buffer = Buffer(
            self.device_func, self.device_host_func, device_memory_limit, weight=weight
        )

        self.device = self.device_buffer.fast.d
        self.host = self.host_buffer.fast.d
        self.disk = self.host_buffer.slow.d

        # For Worker compatibility only, where `fast` is host memory buffer
        self.fast = self.host_buffer.fast

    def __setitem__(self, key, value):
        if _is_device_object(value):
            self.device_buffer[key] = value
        else:
            self.host_buffer[key] = value

    def __getitem__(self, key):
        if key in self.host_buffer:
            obj = self.host_buffer[key]
            del self.host_buffer[key]
            self.device_buffer[key] = _deserialize_if_device(obj)

        if key in self.device_buffer:
            return self.device_buffer[key]
        else:
            raise KeyError

    def __len__(self):
        return len(self.device_buffer)

    def __iter__(self):
        return iter(self.device_buffer)

    def __delitem__(self, i):
        del self.device_buffer[i]
