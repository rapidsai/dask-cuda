import threading
import time
import weakref
from typing import DefaultDict, Dict, Hashable, List, MutableMapping, Tuple
from dask.sizeof import sizeof

from .proxify_device_object import proxify_device_object
from .proxy_object import ProxyObject


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

        self.key_to_proxies: DefaultDict[Hashable, List] = DefaultDict(list)
        self.proxied_id_to_proxy: Dict[int, ProxyObject] = {}

    def __contains__(self, key):
        return key in self.store

    def __len__(self):
        return len(self.store)

    def __iter__(self):
        with self.lock:
            return iter(self.store)

    def get_proxied_id_to_proxy(self) -> Dict[int, ProxyObject]:
        with self.lock:
            ret = {}
            for proxies in self.key_to_proxies.values():
                for proxy in proxies:
                    if not proxy._obj_pxy_serialized():
                        proxied_id = id(proxy._obj_pxy["obj"])
                        p = ret.get(proxied_id, None)
                        if p is None:
                            ret[proxied_id] = proxy
                        else:
                            assert id(p) == id(proxy)  # No duplicates
            return ret

    def get_dev_buffer_to_proxies(self) -> DefaultDict[Hashable, List[ProxyObject]]:
        # Notice, multiple proxy object can point to different non-overlapping
        # parts of the same device buffer.
        ret = DefaultDict(list)
        for proxy in self.get_proxied_id_to_proxy().values():
            for dev_buffer in proxy._obj_pxy_get_device_memory_objects():
                ret[dev_buffer].append(proxy)
        return ret

    def get_access_info(self) -> Tuple[int, List[Tuple[int, int, List[ProxyObject]]]]:
        total_dev_mem_usage = 0
        dev_buf_access = []
        for dev_buf, proxies in self.get_dev_buffer_to_proxies().items():
            last_access = max(p._obj_pxy.get("last_access", 0) for p in proxies)
            size = sizeof(dev_buf)
            dev_buf_access.append((last_access, size, proxies))
            total_dev_mem_usage += size
        return total_dev_mem_usage, dev_buf_access

    def __setitem__(self, key, value):
        with self.lock:
            found_proxies = []
            proxied_id_to_proxy = self.get_proxied_id_to_proxy()
            self.store[key] = proxify_device_object(
                value, proxied_id_to_proxy, found_proxies
            )
            last_access = time.time()
            self_weakref = weakref.ref(self)
            for p in found_proxies:
                p._obj_pxy["hostfile"] = self_weakref
                p._obj_pxy["last_access"] = last_access
                self.key_to_proxies[key].append(p)

            self.maybe_evict()

    def __getitem__(self, key):
        with self.lock:
            return self.store[key]

    def __delitem__(self, key):
        with self.lock:
            del self.store[key]
            self.key_to_proxies.pop(key, None)

    def evict(self, proxy):
        proxy._obj_pxy_serialize(serializers=("dask", "pickle"))

    def maybe_evict(self, extra_dev_mem=0):
        total_dev_mem_usage, dev_buf_access = self.get_access_info()
        total_dev_mem_usage += extra_dev_mem
        if total_dev_mem_usage > self.device_memory_limit:
            dev_buf_access.sort(key=lambda x: (x[0], -x[1]))
            for _, size, proxies in dev_buf_access:
                for p in proxies:
                    self.evict(p)
                total_dev_mem_usage -= size
                if total_dev_mem_usage <= self.device_memory_limit:
                    break
