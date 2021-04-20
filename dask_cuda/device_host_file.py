import functools
import logging
import os
import time

from zict import Buffer, File, Func
from zict.common import ZictBase

import dask
from distributed.protocol import (
    dask_deserialize,
    dask_serialize,
    deserialize,
    deserialize_bytes,
    serialize,
    serialize_bytelist,
)
from distributed.sizeof import safe_sizeof
from distributed.utils import nbytes

from . import proxy_object
from .is_device_object import is_device_object
from .utils import nvtx_annotate


class LoggedBuffer(Buffer):
    """Extends zict.Buffer with logging capabilities

    Two arguments `fast_name` and `slow_name` are passed to constructor that
    identify a user-friendly name for logging of where spilling is going from/to.
    For example, their names can be "Device" and "Host" to identify that spilling
    is happening from a CUDA device into system memory.
    """

    def __init__(self, *args, fast_name="Fast", slow_name="Slow", addr=None, **kwargs):
        self.addr = "Unknown Address" if addr is None else addr
        self.fast_name = fast_name
        self.slow_name = slow_name
        self.msg_template = (
            "Worker at <%s>: Spilled key %s with %s bytes from %s to %s in %s seconds"
        )

        # It is a bit hacky to forcefully capture the "distributed.worker" logger,
        # eventually it would be better to have a different logger. For now this
        # is ok, allowing users to read logs with client.get_worker_logs(), a
        # proper solution would require changes to Distributed.
        self.logger = logging.getLogger("distributed.worker")

        super().__init__(*args, **kwargs)

        self.total_time_fast_to_slow = 0.0
        self.total_time_slow_to_fast = 0.0

    def fast_to_slow(self, key, value):
        start = time.time()
        ret = super().fast_to_slow(key, value)
        total = time.time() - start
        self.total_time_fast_to_slow += total

        self.logger.info(
            self.msg_template
            % (
                self.addr,
                key,
                safe_sizeof(value),
                self.fast_name,
                self.slow_name,
                total,
            )
        )

        return ret

    def slow_to_fast(self, key):
        start = time.time()
        ret = super().slow_to_fast(key)
        total = time.time() - start
        self.total_time_slow_to_fast += total

        self.logger.info(
            self.msg_template
            % (self.addr, key, safe_sizeof(ret), self.slow_name, self.fast_name, total)
        )

        return ret

    def set_address(self, addr):
        self.addr = addr

    def get_total_spilling_time(self):
        return {
            (
                "Total spilling time from %s to %s" % (self.fast_name, self.slow_name)
            ): self.total_time_fast_to_slow,
            (
                "Total spilling time from %s to %s" % (self.slow_name, self.fast_name)
            ): self.total_time_slow_to_fast,
        }


class DeviceSerialized:
    """Store device object on the host

    This stores a device-side object as
    1.  A msgpack encodable header
    2.  A list of `bytes`-like objects (like NumPy arrays)
        that are in host memory
    """

    def __init__(self, header, frames):
        self.header = header
        self.frames = frames

    def __sizeof__(self):
        return sum(map(nbytes, self.frames))

    def __reduce_ex__(self, protocol):
        header, frames = device_serialize(self)
        frames = [f.obj for f in frames]
        return device_deserialize, (header, frames)


@dask_serialize.register(DeviceSerialized)
def device_serialize(obj):
    header = {"obj-header": obj.header}
    frames = obj.frames
    return header, frames


@dask_deserialize.register(DeviceSerialized)
def device_deserialize(header, frames):
    return DeviceSerialized(header["obj-header"], frames)


@nvtx_annotate("SPILL_D2H", color="red", domain="dask_cuda")
def device_to_host(obj: object) -> DeviceSerialized:
    header, frames = serialize(obj, serializers=("dask", "pickle"), on_error="raise")
    return DeviceSerialized(header, frames)


@nvtx_annotate("SPILL_H2D", color="green", domain="dask_cuda")
def host_to_device(s: DeviceSerialized) -> object:
    return deserialize(s.header, s.frames)


@nvtx_annotate("SPILL_D2H", color="red", domain="dask_cuda")
def pxy_obj_device_to_host(obj: object) -> proxy_object.ProxyObject:
    try:
        # Never re-serialize proxy objects.
        if obj._obj_pxy["serializers"] is None:
            return obj
    except (KeyError, AttributeError):
        pass

    # Notice, both the "dask" and the "pickle" serializer will
    # spill `obj` to main memory.
    return proxy_object.asproxy(obj, serializers=("dask", "pickle"))


@nvtx_annotate("SPILL_H2D", color="green", domain="dask_cuda")
def pxy_obj_host_to_device(s: proxy_object.ProxyObject) -> object:
    # Notice, we do _not_ deserialize at this point. The proxy
    # object automatically deserialize just-in-time.
    return s


class DeviceHostFile(ZictBase):
    """Manages serialization/deserialization of objects.

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
    log_spilling: bool
        If True, all spilling operations will be logged directly to
        distributed.worker with an INFO loglevel. This will eventually be
        replaced by a Dask configuration flag.
    """

    def __init__(
        self,
        device_memory_limit=None,
        memory_limit=None,
        local_directory=None,
        log_spilling=False,
    ):
        if local_directory is None:
            local_directory = dask.config.get("temporary-directory") or os.getcwd()

        if not os.path.exists(local_directory):
            os.makedirs(local_directory, exist_ok=True)
        local_directory = os.path.join(local_directory, "dask-worker-space")

        self.disk_func_path = os.path.join(local_directory, "storage")

        self.host_func = dict()
        self.disk_func = Func(
            functools.partial(serialize_bytelist, on_error="raise"),
            deserialize_bytes,
            File(self.disk_func_path),
        )

        host_buffer_kwargs = {}
        device_buffer_kwargs = {}
        buffer_class = Buffer
        if log_spilling is True:
            buffer_class = LoggedBuffer
            host_buffer_kwargs = {"fast_name": "Host", "slow_name": "Disk"}
            device_buffer_kwargs = {"fast_name": "Device", "slow_name": "Host"}

        if memory_limit == 0:
            self.host_buffer = self.host_func
        else:
            self.host_buffer = buffer_class(
                self.host_func,
                self.disk_func,
                memory_limit,
                weight=lambda k, v: safe_sizeof(v),
                **host_buffer_kwargs,
            )

        self.device_keys = set()
        self.device_func = dict()
        self.device_host_func = Func(device_to_host, host_to_device, self.host_buffer)
        self.device_buffer = Buffer(
            self.device_func,
            self.device_host_func,
            device_memory_limit,
            weight=lambda k, v: safe_sizeof(v),
            **device_buffer_kwargs,
        )

        self.device = self.device_buffer.fast.d
        self.host = self.host_buffer if memory_limit == 0 else self.host_buffer.fast.d
        self.disk = None if memory_limit == 0 else self.host_buffer.slow.d

        # For Worker compatibility only, where `fast` is host memory buffer
        self.fast = self.host_buffer if memory_limit == 0 else self.host_buffer.fast

    def __setitem__(self, key, value):
        if key in self.device_buffer:
            # Make sure we register the removal of an existing key
            del self[key]

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

    def set_address(self, addr):
        if isinstance(self.host_buffer, LoggedBuffer):
            self.host_buffer.set_address(addr)
        self.device_buffer.set_address(addr)

    def get_total_spilling_time(self):
        ret = {}
        if isinstance(self.device_buffer, LoggedBuffer):
            ret = {**ret, **self.device_buffer.get_total_spilling_time()}
        if isinstance(self.host_buffer, LoggedBuffer):
            ret = {**ret, **self.host_buffer.get_total_spilling_time()}
        return ret
