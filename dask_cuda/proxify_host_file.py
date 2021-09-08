import abc
import logging
import threading
import time
import warnings
import weakref
from collections import defaultdict
from typing import (
    Any,
    DefaultDict,
    Dict,
    Hashable,
    Iterator,
    List,
    MutableMapping,
    Optional,
    Set,
    Tuple,
)
from weakref import ReferenceType

import dask
from dask.sizeof import sizeof

from .proxify_device_objects import proxify_device_objects, unproxify_device_objects
from .proxy_object import ProxyObject


class Proxies(abc.ABC):
    """Abstract base class to implement tracking of proxies

    This class is not threadsafe
    """

    def __init__(self):
        self._proxy_id_to_proxy: Dict[int, ReferenceType[ProxyObject]] = {}
        self._mem_usage = 0

    def __len__(self) -> int:
        return len(self._proxy_id_to_proxy)

    @abc.abstractmethod
    def mem_usage_add(self, proxy: ProxyObject) -> None:
        """Given a new proxy, update `self._mem_usage`"""

    @abc.abstractmethod
    def mem_usage_remove(self, proxy: ProxyObject) -> None:
        """Removal of proxy, update `self._mem_usage`"""

    def add(self, proxy: ProxyObject) -> None:
        """Add a proxy for tracking, calls `self.mem_usage_add`"""
        assert not self.contains_proxy_id(id(proxy))
        self._proxy_id_to_proxy[id(proxy)] = weakref.ref(proxy)
        self.mem_usage_add(proxy)

    def remove(self, proxy: ProxyObject) -> None:
        """Remove proxy from tracking, calls `self.mem_usage_remove`"""
        del self._proxy_id_to_proxy[id(proxy)]
        self.mem_usage_remove(proxy)
        if len(self._proxy_id_to_proxy) == 0:
            if self._mem_usage != 0:
                warnings.warn(
                    "ProxyManager is empty but the tally of "
                    f"{self} is {self._mem_usage} bytes. "
                    "Resetting the tally."
                )
                self._mem_usage = 0

    def __iter__(self) -> Iterator[ProxyObject]:
        for p in self._proxy_id_to_proxy.values():
            ret = p()
            if ret is not None:
                yield ret

    def contains_proxy_id(self, proxy_id: int) -> bool:
        return proxy_id in self._proxy_id_to_proxy

    def mem_usage(self) -> int:
        return self._mem_usage


class ProxiesOnHost(Proxies):
    """Implement tracking of proxies on the CPU

    This uses dask.sizeof to update memory usage.
    """

    def mem_usage_add(self, proxy: ProxyObject):
        self._mem_usage += sizeof(proxy)

    def mem_usage_remove(self, proxy: ProxyObject):
        self._mem_usage -= sizeof(proxy)


class ProxiesOnDevice(Proxies):
    """Implement tracking of proxies on the GPU

    This is a bit more complicated than ProxiesOnHost because we have to
    handle that multiple proxy objects can refer to the same underlying
    device memory object. Thus, we have to track aliasing and make sure
    we don't count down the memory usage prematurely.
    """

    def __init__(self):
        super().__init__()
        self.proxy_id_to_dev_mems: Dict[int, Set[Hashable]] = {}
        self.dev_mem_to_proxy_ids: DefaultDict[Hashable, Set[int]] = defaultdict(set)

    def mem_usage_add(self, proxy: ProxyObject):
        proxy_id = id(proxy)
        assert proxy_id not in self.proxy_id_to_dev_mems
        self.proxy_id_to_dev_mems[proxy_id] = set()
        for dev_mem in proxy._obj_pxy_get_device_memory_objects():
            self.proxy_id_to_dev_mems[proxy_id].add(dev_mem)
            ps = self.dev_mem_to_proxy_ids[dev_mem]
            if len(ps) == 0:
                self._mem_usage += sizeof(dev_mem)
            ps.add(proxy_id)

    def mem_usage_remove(self, proxy: ProxyObject):
        proxy_id = id(proxy)
        for dev_mem in self.proxy_id_to_dev_mems.pop(proxy_id):
            self.dev_mem_to_proxy_ids[dev_mem].remove(proxy_id)
            if len(self.dev_mem_to_proxy_ids[dev_mem]) == 0:
                del self.dev_mem_to_proxy_ids[dev_mem]
                self._mem_usage -= sizeof(dev_mem)


class ProxyManager:
    """
    This class together with Proxies, ProxiesOnHost, and ProxiesOnDevice
    implements the tracking of all known proxies and their total host/device
    memory usage. It turns out having to re-calculate memory usage continuously
    is too expensive.

    The idea is to have the ProxifyHostFile or the proxies themselves update
    their location (device or host). The manager then tallies the total memory usage.

    Notice, the manager only keeps weak references to the proxies.
    """

    def __init__(self, device_memory_limit: int):
        self.lock = threading.RLock()
        self._host = ProxiesOnHost()
        self._dev = ProxiesOnDevice()
        self._device_memory_limit = device_memory_limit

    def __repr__(self) -> str:
        return (
            f"<ProxyManager dev_limit={self._device_memory_limit}"
            f" host={self._host.mem_usage()}({len(self._host)})"
            f" dev={self._dev.mem_usage()}({len(self._dev)})>"
        )

    def __len__(self) -> int:
        return len(self._host) + len(self._dev)

    def pprint(self) -> str:
        ret = f"{self}:"
        if len(self) == 0:
            return ret + " Empty"
        ret += "\n"
        for proxy in self._host:
            ret += f"  host - {repr(proxy)}\n"
        for proxy in self._dev:
            ret += f"  dev  - {repr(proxy)}\n"
        return ret[:-1]  # Strip last newline

    def get_proxies_by_serializer(self, serializer: Optional[str]) -> Proxies:
        if serializer in ("dask", "pickle"):
            return self._host
        else:
            return self._dev

    def contains(self, proxy_id: int) -> bool:
        with self.lock:
            return self._host.contains_proxy_id(
                proxy_id
            ) or self._dev.contains_proxy_id(proxy_id)

    def add(self, proxy: ProxyObject) -> None:
        with self.lock:
            if not self.contains(id(proxy)):
                self.get_proxies_by_serializer(proxy._obj_pxy["serializer"]).add(proxy)

    def remove(self, proxy: ProxyObject) -> None:
        with self.lock:
            # Find where the proxy is located and remove it
            proxies: Optional[Proxies] = None
            if self._host.contains_proxy_id(id(proxy)):
                proxies = self._host
            if self._dev.contains_proxy_id(id(proxy)):
                assert proxies is None, "Proxy in multiple locations"
                proxies = self._dev
            assert proxies is not None, "Trying to remove unknown proxy"
            proxies.remove(proxy)

    def move(
        self,
        proxy: ProxyObject,
        from_serializer: Optional[str],
        to_serializer: Optional[str],
    ) -> None:
        with self.lock:
            src = self.get_proxies_by_serializer(from_serializer)
            dst = self.get_proxies_by_serializer(to_serializer)
            if src is not dst:
                src.remove(proxy)
                dst.add(proxy)

    def proxify(self, obj: object) -> object:
        with self.lock:
            found_proxies: List[ProxyObject] = []
            proxied_id_to_proxy: Dict[int, ProxyObject] = {}
            ret = proxify_device_objects(obj, proxied_id_to_proxy, found_proxies)
            last_access = time.monotonic()
            for p in found_proxies:
                p._obj_pxy["last_access"] = last_access
                if not self.contains(id(p)):
                    p._obj_pxy_register_manager(self)
                    self.add(p)
            self.maybe_evict()
            return ret

    def get_dev_buffer_to_proxies(self) -> DefaultDict[Hashable, List[ProxyObject]]:
        with self.lock:
            # Notice, multiple proxy object can point to different non-overlapping
            # parts of the same device buffer.
            ret = defaultdict(list)
            for proxy in self._dev:
                for dev_buffer in proxy._obj_pxy_get_device_memory_objects():
                    ret[dev_buffer].append(proxy)
            return ret

    def get_dev_access_info(
        self,
    ) -> Tuple[int, List[Tuple[int, int, List[ProxyObject]]]]:
        with self.lock:
            total_dev_mem_usage = 0
            dev_buf_access = []
            for dev_buf, proxies in self.get_dev_buffer_to_proxies().items():
                last_access = max(p._obj_pxy.get("last_access", 0) for p in proxies)
                size = sizeof(dev_buf)
                dev_buf_access.append((last_access, size, proxies))
                total_dev_mem_usage += size
            assert total_dev_mem_usage == self._dev.mem_usage()
            return total_dev_mem_usage, dev_buf_access

    def maybe_evict(self, extra_dev_mem=0) -> None:
        if (  # Shortcut when not evicting
            self._dev.mem_usage() + extra_dev_mem <= self._device_memory_limit
        ):
            return

        with self.lock:
            total_dev_mem_usage, dev_buf_access = self.get_dev_access_info()
            total_dev_mem_usage += extra_dev_mem
            if total_dev_mem_usage > self._device_memory_limit:
                dev_buf_access.sort(key=lambda x: (x[0], -x[1]))
                for _, size, proxies in dev_buf_access:
                    for p in proxies:
                        # Serialize to disk, which "dask" and "pickle" does
                        p._obj_pxy_serialize(serializers=("dask", "pickle"))
                    total_dev_mem_usage -= size
                    if total_dev_mem_usage <= self._device_memory_limit:
                        break


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
    compatibility_mode: bool or None
        Enables compatibility-mode, which means that items are un-proxified before
        retrieval. This makes it possible to get some of the JIT-unspill benefits
        without having to be ProxyObject compatible. In order to still allow specific
        ProxyObjects, set the `mark_as_explicit_proxies=True` when proxifying with
        `proxify_device_objects()`. If None, the "jit-unspill-compatibility-mode"
        config value are used, which defaults to False.
    """

    def __init__(self, device_memory_limit: int, compatibility_mode: bool = None):
        self.device_memory_limit = device_memory_limit
        self.store: Dict[Hashable, Any] = {}
        self.lock = threading.RLock()
        self.manager = ProxyManager(device_memory_limit)
        if compatibility_mode is None:
            self.compatibility_mode = dask.config.get(
                "jit-unspill-compatibility-mode", default=False
            )
        else:
            self.compatibility_mode = compatibility_mode

        # It is a bit hacky to forcefully capture the "distributed.worker" logger,
        # eventually it would be better to have a different logger. For now this
        # is ok, allowing users to read logs with client.get_worker_logs(), a
        # proper solution would require changes to Distributed.
        self.logger = logging.getLogger("distributed.worker")

    def __contains__(self, key):
        return key in self.store

    def __len__(self):
        return len(self.store)

    def __iter__(self):
        with self.lock:
            return iter(self.store)

    @property
    def fast(self):
        """Dask use this to trigger CPU-to-Disk spilling"""
        self.logger.warning(
            "JIT-Unspill doesn't support spilling to "
            "Disk, see <https://github.com/rapidsai/dask-cuda/issues/657>"
        )
        return None

    def __setitem__(self, key, value):
        with self.lock:
            if key in self.store:
                # Make sure we register the removal of an existing key
                del self[key]
            self.store[key] = self.manager.proxify(value)

    def __getitem__(self, key):
        with self.lock:
            ret = self.store[key]
        if self.compatibility_mode:
            ret = unproxify_device_objects(ret, skip_explicit_proxies=True)
            self.manager.maybe_evict()
        return ret

    def __delitem__(self, key):
        with self.lock:
            del self.store[key]
