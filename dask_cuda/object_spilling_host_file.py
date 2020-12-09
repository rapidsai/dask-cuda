import threading
import time
import weakref
from typing import MutableMapping
from dask.sizeof import sizeof

from .proxify_device_object import proxify_device_object


class ObjectSpillingHostFile(MutableMapping):
    """Manages serialization/deserialization of objects.

    TODO: Three LRU cache levels are controlled, for device, host and disk.
    Each level takes care of serializing objects once its limit has been
    reached and pass it to the subsequent level. Similarly, each cache
    may deserialize the object, but storing it back in the appropriate
    cache, depending on the type of object being deserialized.

    Parameters
    ----------
    device_memory_limit: int
        Number of bytes of CUDA device memory for device LRU cache,
        spills to host cache once filled.
    TODO: memory_limit: int
        Number of bytes of host memory for host LRU cache, spills to
        disk once filled. Setting this to 0 means unlimited host memory,
        implies no spilling to disk.
    local_directory: path
        Path where to store serialized objects on disk
    """

    def __init__(
        self,
        device_memory_limit: int,
        **kwargs,
    ):
        self.device_memory_limit = device_memory_limit
        self.store = {}
        self.lock = threading.RLock()

        # self.proxied_id_to_proxy = Dict[int, proxy_object.ProxyObject] = {}
        # self.proxy_id_to_proxy: Dict[int, proxy_object.ProxyObject] = {}
        # self.proxies: Dict[int, proxy_object.ProxyObject] = {}

    def __contains__(self, key):
        return key in self.store

    def __len__(self):
        return len(self.store)

    def __iter__(self):
        return iter(self.store)

    def unspilled_proxies(self):
        found_proxies = []
        proxied_id_to_proxy = {}
        proxify_device_object(self.store, proxied_id_to_proxy, found_proxies)
        ret = list(proxied_id_to_proxy.values())
        assert len(ret) == len(set(id(p) for p in ret))  # No duplicates
        return ret

    def obj_mappings(self):
        # TODO: simplify and optimize
        proxied_id_to_proxy = {}
        buffer_to_proxies = {}

        for p in self.unspilled_proxies():
            proxied = p._obj_pxy["obj"]
            proxied_id = id(proxied)
            assert proxied_id not in proxied_id_to_proxy
            proxied_id_to_proxy[proxied_id] = p
            for buf in p._obj_pxy_get_device_memory_objects():
                l = buffer_to_proxies.get(buf, [])
                if id(p) not in set(id(i) for i in l):
                    l.append(p)
                buffer_to_proxies[buf] = l
        return proxied_id_to_proxy, buffer_to_proxies

    def __setitem__(self, key, value):
        with self.lock:
            found_proxies = []
            proxied_id_to_proxy, _ = self.obj_mappings()
            self.store[key] = proxify_device_object(
                value, proxied_id_to_proxy, found_proxies
            )
            last_access = time.time()
            self_weakref = weakref.ref(self)
            for p in found_proxies:
                p._obj_pxy["hostfile"] = self_weakref
                p._obj_pxy["last_access"] = last_access

            self.maybe_evict()

    def __getitem__(self, key):
        return self.store[key]

    def __delitem__(self, key):
        del self.store[key]

    def evict(self, proxy):
        proxy._obj_pxy_serialize(serializers=["dask", "pickle"])

    def buffer_info(self, ignores=()):
        buffers = []
        dev_mem_usage = 0
        for buf, proxies in self.obj_mappings()[1].items():
            last_access = max(p._obj_pxy.get("last_access", 0) for p in proxies)
            size = sizeof(buf)
            buffers.append((last_access, size, proxies))
            dev_mem_usage += size
        return buffers, dev_mem_usage

    def maybe_evict(self, extra_dev_mem=0, ignores=()):
        buffers, dev_mem_usage = self.buffer_info()
        dev_mem_usage += extra_dev_mem
        if dev_mem_usage > self.device_memory_limit:
            buffers.sort(key=lambda x: (x[0], -x[1]))
            for _, size, proxies in buffers:
                for p in proxies:
                    self.evict(p)
                dev_mem_usage -= size
                if dev_mem_usage <= self.device_memory_limit:
                    break
