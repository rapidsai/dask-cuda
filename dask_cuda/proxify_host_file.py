import threading
import time
import weakref
from collections import defaultdict
from typing import (
    DefaultDict,
    Dict,
    Hashable,
    Iterator,
    List,
    MutableMapping,
    Set,
    Tuple,
)

from dask.sizeof import sizeof

from .proxify_device_objects import proxify_device_objects
from .proxy_object import ProxyObject


class UnspilledProxies:
    """Class to track current unspilled proxies"""

    def __init__(self):
        self.dev_mem_usage = 0
        self.proxy_id_to_dev_mems: DefaultDict[int, Set[Hashable]] = defaultdict(set)
        self.dev_mem_to_proxy_ids: DefaultDict[Hashable, Set[int]] = defaultdict(set)

    def add(self, proxy: ProxyObject):
        proxy_id = id(proxy)
        if proxy_id not in self.proxy_id_to_dev_mems:
            for dev_mem in proxy._obj_pxy_get_device_memory_objects():
                self.proxy_id_to_dev_mems[proxy_id].add(dev_mem)
                ps = self.dev_mem_to_proxy_ids[dev_mem]
                if len(ps) == 0:
                    self.dev_mem_usage += sizeof(dev_mem)
                ps.add(proxy_id)

    def remove(self, proxy: ProxyObject):
        proxy_id = id(proxy)
        if proxy_id in self.proxy_id_to_dev_mems:
            for dev_mem in self.proxy_id_to_dev_mems.pop(proxy_id):
                self.dev_mem_to_proxy_ids[dev_mem].remove(proxy_id)
                if len(self.dev_mem_to_proxy_ids[dev_mem]) == 0:
                    del self.dev_mem_to_proxy_ids[dev_mem]
                    self.dev_mem_usage -= sizeof(dev_mem)

    def __iter__(self):
        return iter(self.proxy_id_to_dev_mems)


class ProxiesTally:
    """
    This class together with UnspilledProxies implements the tracking of current
    objects in device memory and the total memory usage. It turns out having to
    re-calculate device memory usage continuously is too expensive.

    We have to track four events:
    - When adding a new key to the host file
    - When removing a key from the host file
    - When a proxy in the host file is deserialized
    - When a proxy in the host file is serialized

    However, it gets a bit complicated because:
    - The value of a key in the host file can contain many proxy objects and a single
      proxy object can be referred from many keys
    - Multiple proxy objects can refer to the same underlying device memory object
    - Proxy objects are not hashable thus we have to use the `id()` as key in
      dictionaries

    ProxiesTally and UnspilledProxies implements this by carefully maintaining
    dictionaries that maps to/from keys, proxy objects, and device memory objects.
    """

    def __init__(self):
        self.lock = threading.RLock()
        self.proxy_id_to_proxy: Dict[int, ProxyObject] = {}
        self.key_to_proxy_ids: DefaultDict[Hashable, Set[int]] = defaultdict(set)
        self.proxy_id_to_keys: DefaultDict[int, Set[Hashable]] = defaultdict(set)
        self.unspilled_proxies = UnspilledProxies()

    def add_key(self, key, proxies: List[ProxyObject]):
        with self.lock:
            for proxy in proxies:
                proxy_id = id(proxy)
                self.proxy_id_to_proxy[proxy_id] = proxy
                self.key_to_proxy_ids[key].add(proxy_id)
                self.proxy_id_to_keys[proxy_id].add(key)
                if not proxy._obj_pxy_is_serialized():
                    self.unspilled_proxies.add(proxy)

    def del_key(self, key):
        with self.lock:
            for proxy_id in self.key_to_proxy_ids.pop(key, ()):
                self.proxy_id_to_keys[proxy_id].remove(key)
                if len(self.proxy_id_to_keys[proxy_id]) == 0:
                    del self.proxy_id_to_keys[proxy_id]
                    self.unspilled_proxies.remove(self.proxy_id_to_proxy.pop(proxy_id))

    def spill_proxy(self, proxy: ProxyObject):
        with self.lock:
            self.unspilled_proxies.remove(proxy)

    def unspill_proxy(self, proxy: ProxyObject):
        with self.lock:
            self.unspilled_proxies.add(proxy)

    def get_unspilled_proxies(self) -> Iterator[ProxyObject]:
        with self.lock:
            for proxy_id in self.unspilled_proxies:
                ret = self.proxy_id_to_proxy[proxy_id]
                assert not ret._obj_pxy_is_serialized()
                yield ret

    def get_proxied_id_to_proxy(self) -> Dict[int, ProxyObject]:
        return {id(p._obj_pxy["obj"]): p for p in self.get_unspilled_proxies()}

    def get_dev_mem_usage(self) -> int:
        return self.unspilled_proxies.dev_mem_usage


class ProxifyHostFile(MutableMapping):
    """Host file that proxify stored data

    This class is an alternative to the default disk-backed LRU dict used by
    workers in Distributed.

    It wraps all CUDA device objects in a ProxyObject instance and maintains
    `device_memory_limit` by spilling ProxyObject on-the-fly. This addresses
    some issues with the default DeviceHostFile host, which tracks device
    memory inaccurately see <https://github.com/rapidsai/dask-cuda/pull/451>

    Limitations
    -----------
    - For now, ProxifyHostFile doesn't support spilling to disk.
    - ProxyObject has some limitations and doesn't mimic the proxied object
      perfectly. See docs of ProxyObject for detail.
    - This is still experimental, expect bugs and API changes.

    Parameters
    ----------
    device_memory_limit: int
        Number of bytes of CUDA device memory used before spilling to host.
    """

    def __init__(self, device_memory_limit: int):
        self.device_memory_limit = device_memory_limit
        self.store = {}
        self.lock = threading.RLock()
        self.proxies_tally = ProxiesTally()

    def __contains__(self, key):
        return key in self.store

    def __len__(self):
        return len(self.store)

    def __iter__(self):
        with self.lock:
            return iter(self.store)

    def get_dev_buffer_to_proxies(self) -> DefaultDict[Hashable, List[ProxyObject]]:
        with self.lock:
            # Notice, multiple proxy object can point to different non-overlapping
            # parts of the same device buffer.
            ret = DefaultDict(list)
            for proxy in self.proxies_tally.get_unspilled_proxies():
                for dev_buffer in proxy._obj_pxy_get_device_memory_objects():
                    ret[dev_buffer].append(proxy)
            return ret

    def get_access_info(self) -> Tuple[int, List[Tuple[int, int, List[ProxyObject]]]]:
        with self.lock:
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
            proxied_id_to_proxy = self.proxies_tally.get_proxied_id_to_proxy()
            self.store[key] = proxify_device_objects(
                value, proxied_id_to_proxy, found_proxies
            )
            last_access = time.time()
            self_weakref = weakref.ref(self)
            for p in found_proxies:
                p._obj_pxy["hostfile"] = self_weakref
                p._obj_pxy["last_access"] = last_access
            self.proxies_tally.add_key(key, found_proxies)
            self.maybe_evict()

    def __getitem__(self, key):
        with self.lock:
            return self.store[key]

    def __delitem__(self, key):
        with self.lock:
            del self.store[key]
            self.proxies_tally.del_key(key)

    def evict(self, proxy):
        proxy._obj_pxy_serialize(serializers=("dask", "pickle"))

    def maybe_evict(self, extra_dev_mem=0):
        if (  # Shortcut when not evicting
            self.proxies_tally.get_dev_mem_usage() + extra_dev_mem
            <= self.device_memory_limit
        ):
            return

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
