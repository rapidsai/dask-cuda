from zict import Buffer, File, Func
from zict.common import ZictBase
from distributed.protocol import deserialize_bytes, serialize_bytes
from distributed.worker import weight

from functools import partial
import os


def _is_device_object(obj):
    return hasattr(obj, '__cuda_array_interface__')


def _serialize_if_device(obj):
    if _is_device_object(obj):
        return serialize_bytes(obj, on_error='raise')
    else:
        return obj


def _deserialize_if_device(obj):
    if isinstance(obj, bytes):
        return deserialize_bytes(obj)
    else:
        return obj


class DeviceHostFile(ZictBase):
    def __init__(self, device_memory_limit=None, memory_limit=None,
                 local_dir='dask-worker-space', compress=False):
        path = os.path.join(local_dir, 'storage')

        self.host_func = dict()
        self.disk_func = Func(partial(serialize_bytes, on_error='raise'),
                              deserialize_bytes, File(path))
        self.host = Buffer(self.host_func, self.disk_func, memory_limit,
                           weight=weight)

        self.device_func = dict()
        self.device_host_func = Func(_serialize_if_device,
                                     _deserialize_if_device, self.host)
        self.device = Buffer(self.device_func, self.device_host_func,
                             device_memory_limit, weight=weight)

        self.fast = self.host.fast

    def __setitem__(self, key, value):
        if _is_device_object(value):
            self.device[key] = value
        else:
            self.host[key] = value

    def __getitem__(self, key):
        if key in self.host:
            obj = self.host[key]
            del self.host[key]
            self.device[key] = _deserialize_if_device(obj)

        if key in self.device:
            return self.device[key]
        else:
            raise KeyError

    def __len__(self):
        return self.device.__len__()

    def __iter__(self):
        return self.device.__iter__()

    def __delitem__(self, i):
        return self.device.__delitem__(i)
