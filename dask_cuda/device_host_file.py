from zict import Buffer, File, Func
from zict.common import ZictBase
from distributed.protocol import deserialize_bytes, serialize_bytelist
from distributed.worker import weight

try:
    from cytoolz import partial
except ImportError:
    from toolz import partial

import os


def _is_device_object(obj):
    return hasattr(obj, '__cuda_array_interface__')


class DeviceHostFile(ZictBase):
    def __init__(self, device_memory_limit=None, memory_limit=None,
                 local_dir='dask-worker-space', compress=False):
        path = os.path.join(local_dir, 'storage')

        self.device_func = dict()
        self.host_func = dict()
        self.disk_func = Func(partial(serialize_bytelist, on_error='raise'),
                              deserialize_bytes, File(path))

        self.host = Buffer(self.host_func, self.disk_func, memory_limit,
                           weight=weight)
        self.device = Buffer(self.device_func, self.host, device_memory_limit,
                             weight=weight)

        self.fast = self.host.fast

    def __setitem__(self, key, value):
        if _is_device_object(value):
            self.device[key] = value
        else:
            self.host[key] = value

    def __getitem__(self, key):
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
