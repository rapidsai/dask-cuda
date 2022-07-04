import abc
import gc
import io
import logging
import os
import os.path
import pathlib
import threading
import time
import traceback
import warnings
import weakref
from collections import defaultdict
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    Hashable,
    Iterable,
    List,
    MutableMapping,
    Optional,
    Set,
    Tuple,
    TypeVar,
)
from weakref import ReferenceType

import dask
from dask.sizeof import sizeof
from dask.utils import format_bytes
from distributed.protocol.compression import decompress, maybe_compress
from distributed.protocol.serialize import (
    merge_and_deserialize,
    register_serialization_family,
    serialize_and_split,
)

from . import proxify_device_objects as pdo
from .disk_io import SpillToDiskProperties, disk_read, disk_write
from .get_device_memory_objects import DeviceMemoryId, get_device_memory_ids
from .proxify_device_objects import proxify_device_objects, unproxify_device_objects
from .proxy_object import ProxyObject

T = TypeVar("T")


class Proxies(abc.ABC):
    """Abstract base class to implement tracking of proxies

    This class is not threadsafe
    """

    def __init__(self):
        self._proxy_id_to_proxy: Dict[int, ReferenceType[ProxyObject]] = {}
        self._mem_usage = 0
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._proxy_id_to_proxy)

    @abc.abstractmethod
    def mem_usage_add(self, proxy: ProxyObject) -> None:
        """Given a new proxy, update `self._mem_usage`"""

    @abc.abstractmethod
    def mem_usage_remove(self, proxy: ProxyObject) -> None:
        """Removal of proxy, update `self._mem_usage`"""

    @abc.abstractmethod
    def buffer_info(self) -> List[Tuple[float, int, List[ProxyObject]]]:
        """Return a list of buffer information

        The returned format is:
            `[(<access-time>, <size-of-buffer>, <list-of-proxies>), ...]
        """

    def add(self, proxy: ProxyObject) -> None:
        """Add a proxy for tracking, calls `self.mem_usage_add`"""
        assert not self.contains_proxy_id(id(proxy))
        with self._lock:
            self._proxy_id_to_proxy[id(proxy)] = weakref.ref(proxy)
        self.mem_usage_add(proxy)

    def remove(self, proxy: ProxyObject) -> None:
        """Remove proxy from tracking, calls `self.mem_usage_remove`"""
        with self._lock:
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

    def get_proxies(self) -> List[ProxyObject]:
        """Return a list of all proxies"""
        with self._lock:
            ret = []
            for p in self._proxy_id_to_proxy.values():
                proxy = p()
                if proxy is not None:
                    ret.append(proxy)
            return ret

    def get_proxies_by_ids(self, proxy_ids: Iterable[int]) -> List[ProxyObject]:
        """Return a list of proxies"""
        ret = []
        for proxy_id in proxy_ids:
            weakref_proxy = self._proxy_id_to_proxy.get(proxy_id)
            if weakref_proxy is not None:
                proxy = weakref_proxy()
                if proxy is not None:
                    ret.append(proxy)
        return ret

    def contains_proxy_id(self, proxy_id: int) -> bool:
        return proxy_id in self._proxy_id_to_proxy

    def mem_usage(self) -> int:
        return self._mem_usage


class ProxiesOnHost(Proxies):
    """Implement tracking of proxies on the CPU

    This uses dask.sizeof to update memory usage.
    """

    def mem_usage_add(self, proxy: ProxyObject) -> None:
        self._mem_usage += sizeof(proxy)

    def mem_usage_remove(self, proxy: ProxyObject) -> None:
        self._mem_usage -= sizeof(proxy)

    def buffer_info(self) -> List[Tuple[float, int, List[ProxyObject]]]:
        ret = []
        for p in self.get_proxies():
            size = sizeof(p)
            ret.append((p._pxy_get().last_access, size, [p]))
        return ret


class ProxiesOnDisk(ProxiesOnHost):
    """Implement tracking of proxies on the Disk"""


class ProxiesOnDevice(Proxies):
    """Implement tracking of proxies on the GPU

    This is a bit more complicated than ProxiesOnHost because we have to
    handle that multiple proxy objects can refer to the same underlying
    device memory object. Thus, we have to track aliasing and make sure
    we don't count down the memory usage prematurely.

    Notice, we only track direct aliasing thus multiple proxy objects can
    point to different non-overlapping parts of the same device buffer.
    In this case the tally of the total device memory usage is incorrect.
    """

    def __init__(self):
        super().__init__()
        self.proxy_id_to_dev_mems: Dict[int, Set[DeviceMemoryId]] = {}
        self.dev_mem_to_proxy_ids: DefaultDict[DeviceMemoryId, Set[int]] = defaultdict(
            set
        )

    def mem_usage_add(self, proxy: ProxyObject) -> None:
        proxy_id = id(proxy)
        assert proxy_id not in self.proxy_id_to_dev_mems
        self.proxy_id_to_dev_mems[proxy_id] = set()
        for dev_mem in get_device_memory_ids(proxy._pxy_get().obj):
            self.proxy_id_to_dev_mems[proxy_id].add(dev_mem)
            ps = self.dev_mem_to_proxy_ids[dev_mem]
            if len(ps) == 0:
                self._mem_usage += dev_mem.nbytes
            ps.add(proxy_id)

    def mem_usage_remove(self, proxy: ProxyObject) -> None:
        proxy_id = id(proxy)
        for dev_mem in self.proxy_id_to_dev_mems.pop(proxy_id):
            self.dev_mem_to_proxy_ids[dev_mem].remove(proxy_id)
            if len(self.dev_mem_to_proxy_ids[dev_mem]) == 0:
                del self.dev_mem_to_proxy_ids[dev_mem]
                self._mem_usage -= dev_mem.nbytes

    def buffer_info(self) -> List[Tuple[float, int, List[ProxyObject]]]:
        ret = []
        for dev_mem, proxy_ids in self.dev_mem_to_proxy_ids.items():
            proxies = self.get_proxies_by_ids(proxy_ids)
            last_access = max(p._pxy_get().last_access for p in proxies)
            ret.append((last_access, dev_mem.nbytes, proxies))
        return ret


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

    def __init__(self, device_memory_limit: int, memory_limit: int):
        self.lock = threading.RLock()
        self._disk = ProxiesOnDisk()
        self._host = ProxiesOnHost()
        self._dev = ProxiesOnDevice()
        self._device_memory_limit = device_memory_limit
        self._host_memory_limit = memory_limit

    def __repr__(self) -> str:
        with self.lock:
            return (
                f"<ProxyManager dev_limit={format_bytes(self._device_memory_limit)}"
                f" host_limit={format_bytes(self._host_memory_limit)}"
                f" disk={format_bytes(self._disk.mem_usage())}({len(self._disk)})"
                f" host={format_bytes(self._host.mem_usage())}({len(self._host)})"
                f" dev={format_bytes(self._dev.mem_usage())}({len(self._dev)})>"
            )

    def __len__(self) -> int:
        return len(self._disk) + len(self._host) + len(self._dev)

    def pprint(self) -> str:
        with self.lock:
            ret = f"{self}:"
            if len(self) == 0:
                return ret + " Empty"
            ret += "\n"
            for proxy in self._disk.get_proxies():
                ret += f"  disk - {repr(proxy)}\n"
            for proxy in self._host.get_proxies():
                ret += f"  host - {repr(proxy)}\n"
            for proxy in self._dev.get_proxies():
                ret += f"  dev  - {repr(proxy)}\n"
            return ret[:-1]  # Strip last newline

    def get_proxies_by_serializer(self, serializer: Optional[str]) -> Proxies:
        """Get Proxies collection by serializer"""
        if serializer == "disk":
            return self._disk
        elif serializer in ("dask", "pickle"):
            return self._host
        else:
            return self._dev

    def get_proxies_by_proxy_object(self, proxy: ProxyObject) -> Optional[Proxies]:
        """Get Proxies collection by proxy object"""
        proxy_id = id(proxy)
        if self._dev.contains_proxy_id(proxy_id):
            return self._dev
        if self._host.contains_proxy_id(proxy_id):
            return self._host
        if self._disk.contains_proxy_id(proxy_id):
            return self._disk
        return None

    def contains(self, proxy_id: int) -> bool:
        """Is the proxy in any of the Proxies collection?"""
        with self.lock:
            return (
                self._disk.contains_proxy_id(proxy_id)
                or self._host.contains_proxy_id(proxy_id)
                or self._dev.contains_proxy_id(proxy_id)
            )

    def add(self, proxy: ProxyObject, serializer: Optional[str]) -> None:
        """Add the proxy to the Proxies collection by that match the serializer"""
        with self.lock:
            old_proxies = self.get_proxies_by_proxy_object(proxy)
            new_proxies = self.get_proxies_by_serializer(serializer)
            if old_proxies is not new_proxies:
                if old_proxies is not None:
                    old_proxies.remove(proxy)
                new_proxies.add(proxy)

    def remove(self, proxy: ProxyObject) -> None:
        """Remove the proxy from the Proxies collection it is in"""
        with self.lock:
            # Find where the proxy is located (if found) and remove it
            proxies: Optional[Proxies] = None
            if self._disk.contains_proxy_id(id(proxy)):
                proxies = self._disk
            if self._host.contains_proxy_id(id(proxy)):
                assert proxies is None, "Proxy in multiple locations"
                proxies = self._host
            if self._dev.contains_proxy_id(id(proxy)):
                assert proxies is None, "Proxy in multiple locations"
                proxies = self._dev
            assert proxies is not None, "Trying to remove unknown proxy"
            proxies.remove(proxy)

    def validate(self):
        """Validate the state of the manager"""
        with self.lock:
            for serializer in ("disk", "dask", "cuda"):
                proxies = self.get_proxies_by_serializer(serializer)
                for p in proxies.get_proxies():
                    assert (
                        self.get_proxies_by_serializer(p._pxy_get().serializer)
                        is proxies
                    )
                with proxies._lock:
                    for i, p in proxies._proxy_id_to_proxy.items():
                        assert p() is not None
                        assert i == id(p())
                for p in proxies.get_proxies():
                    pxy = p._pxy_get()
                    if pxy.is_serialized():
                        header, _ = pxy.obj
                        assert header["serializer"] == pxy.serializer

    def proxify(self, obj: T) -> Tuple[T, bool]:
        """Proxify `obj` and add found proxies to the `Proxies` collections

        Returns the proxified object and a boolean, which is `True` when one or
        more incompatible-types were found.
        """
        incompatible_type_found = False
        with self.lock:
            found_proxies: List[ProxyObject] = []
            # In order detect already proxied object, proxify_device_objects()
            # needs a mapping from proxied objects to their proxy objects.
            proxied_id_to_proxy = {
                id(p._pxy_get().obj): p for p in self._dev.get_proxies()
            }
            ret = proxify_device_objects(obj, proxied_id_to_proxy, found_proxies)
            last_access = time.monotonic()
            for p in found_proxies:
                pxy = p._pxy_get()
                pxy.last_access = last_access
                if not self.contains(id(p)):
                    pxy.manager = self
                    self.add(proxy=p, serializer=pxy.serializer)
                if pdo.incompatible_types and isinstance(p, pdo.incompatible_types):
                    incompatible_type_found = True
        self.maybe_evict()
        return ret, incompatible_type_found

    def evict(
        self,
        nbytes: int,
        proxies_access: Callable[[], List[Tuple[float, int, List[ProxyObject]]]],
        serializer: Callable[[ProxyObject], None],
    ) -> int:
        """Evict buffers retrieved by calling `proxies_access`

        Calls `proxies_access` to retrieve a list of proxies and then spills
        enough proxies to free up at a minimum `nbytes` bytes. In order to
        spill a proxy, `serializer` is called.

        Parameters
        ----------
        nbytes: int
            Number of bytes to evict.
        proxies_access: callable
            Function that returns a list of proxies pack in a tuple like:
            `[(<access-time>, <size-of-buffer>, <list-of-proxies>), ...]
        serializer: callable
            Function that serialize the given proxy object.

        Return
        ------
        nbytes: int
            Number of bytes spilled.
        """
        freed_memory: int = 0
        proxies_to_serialize: List[ProxyObject] = []
        with self.lock:
            access = proxies_access()
            access.sort(key=lambda x: (x[0], -x[1]))
            for _, size, proxies in access:
                proxies_to_serialize.extend(proxies)
                freed_memory += size
                if freed_memory >= nbytes:
                    break

        serialized_proxies: Set[int] = set()
        for p in proxies_to_serialize:
            # Avoid trying to serialize the same proxy multiple times
            if id(p) not in serialized_proxies:
                serialized_proxies.add(id(p))
                serializer(p)
        return freed_memory

    def maybe_evict_from_device(self, extra_dev_mem=0) -> None:
        """Evict buffers until total memory usage is below device-memory-limit

        Adds `extra_dev_mem` to the current total memory usage when comparing
        against device-memory-limit.
        """
        mem_over_usage = (
            self._dev.mem_usage() + extra_dev_mem - self._device_memory_limit
        )
        if mem_over_usage > 0:
            self.evict(
                nbytes=mem_over_usage,
                proxies_access=self._dev.buffer_info,
                serializer=lambda p: p._pxy_serialize(serializers=("dask", "pickle")),
            )

    def maybe_evict_from_host(self, extra_host_mem=0) -> None:
        """Evict buffers until total memory usage is below host-memory-limit

        Adds `extra_host_mem` to the current total memory usage when comparing
        against device-memory-limit.
        """
        assert self._host_memory_limit is not None
        mem_over_usage = (
            self._host.mem_usage() + extra_host_mem - self._host_memory_limit
        )
        if mem_over_usage > 0:
            self.evict(
                nbytes=mem_over_usage,
                proxies_access=self._host.buffer_info,
                serializer=ProxifyHostFile.serialize_proxy_to_disk_inplace,
            )

    def maybe_evict(self, extra_dev_mem=0) -> None:
        self.maybe_evict_from_device(extra_dev_mem)
        if self._host_memory_limit:
            self.maybe_evict_from_host()


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
    memory_limit: int
        Number of bytes of host memory used before spilling to disk.
    local_directory: str or None, default None
        Path on local machine to store temporary files. Can be a string (like
        ``"path/to/files"``) or ``None`` to fall back on the value of
        ``dask.temporary-directory`` in the local Dask configuration, using the
        current working directory if this is not set.
        WARNING, this **cannot** change while running thus all serialization to
        disk are using the same directory.
    shared_filesystem: bool or None, default None
        Whether the `local_directory` above is shared between all workers or not.
        If ``None``, the "jit-unspill-shared-fs" config value are used, which
        defaults to False.
        Notice, a shared filesystem must support the `os.link()` operation.
    compatibility_mode: bool or None, default None
        Enables compatibility-mode, which means that items are un-proxified before
        retrieval. This makes it possible to get some of the JIT-unspill benefits
        without having to be ProxyObject compatible. In order to still allow specific
        ProxyObjects, set the `mark_as_explicit_proxies=True` when proxifying with
        `proxify_device_objects()`. If ``None``, the "jit-unspill-compatibility-mode"
        config value are used, which defaults to False.
    spill_on_demand: bool or None, default None
        Enables spilling when the RMM memory pool goes out of memory. If ``None``,
        the "spill-on-demand" config value are used, which defaults to True.
        Notice, enabling this does nothing when RMM isn't availabe or not used.
    gds_spilling: bool
        Enable GPUDirect Storage spilling. If ``None``, the "gds-spilling" config
        value are used, which defaults to ``False``.
    """

    # Notice, we define `_spill_to_disk` as a static variable because it is used by
    # the static register_disk_spilling() method.
    _spill_to_disk: Optional[SpillToDiskProperties] = None

    lock = threading.RLock()

    def __init__(
        self,
        *,
        device_memory_limit: int,
        memory_limit: int,
        local_directory: str = None,
        shared_filesystem: bool = None,
        compatibility_mode: bool = None,
        spill_on_demand: bool = None,
        gds_spilling: bool = None,
    ):
        # each value of self.store is a tuple containing the proxified
        # object, as well as a boolean indicating whether any
        # incompatible types were found when proxifying it
        self.store: Dict[Hashable, Tuple[Any, bool]] = {}
        self.manager = ProxyManager(device_memory_limit, memory_limit)

        # Create an instance of `SpillToDiskProperties` if it doesn't already exist
        path = pathlib.Path(
            os.path.join(
                local_directory
                or dask.config.get("temporary-directory")
                or os.getcwd(),
                "dask-worker-space",
                "jit-unspill-disk-storage",
            )
        ).resolve()
        if ProxifyHostFile._spill_to_disk is None:
            ProxifyHostFile._spill_to_disk = SpillToDiskProperties(
                path, shared_filesystem, gds_spilling
            )
        elif ProxifyHostFile._spill_to_disk.root_dir != path:
            raise ValueError("Cannot change the JIT-Unspilling disk path")

        self.register_disk_spilling()

        if compatibility_mode is None:
            self.compatibility_mode = dask.config.get(
                "jit-unspill-compatibility-mode", default=False
            )
        else:
            self.compatibility_mode = compatibility_mode
        if spill_on_demand is None:
            spill_on_demand = dask.config.get("spill-on-demand", default=True)
        # `None` in this context means: never initialize
        self.spill_on_demand_initialized = False if spill_on_demand else None

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

    def initialize_spill_on_demand_once(self):
        """Register callback function to handle RMM out-of-memory exceptions

        This function is idempotent and should be called at least once. Currently, we
        do this in __setitem__ instead of in __init__ because a Dask worker might re-
        initiate the RMM pool and its resource adaptors after creating ProxifyHostFile.
        """
        if self.spill_on_demand_initialized is False:
            self.spill_on_demand_initialized = True
            try:
                import rmm.mr

                assert hasattr(rmm.mr, "FailureCallbackResourceAdaptor")
            except (ImportError, AssertionError):
                pass
            else:

                def oom(nbytes: int) -> bool:
                    """Try to handle an out-of-memory error by spilling"""
                    memory_freed = self.manager.evict(
                        nbytes=nbytes,
                        proxies_access=self.manager._dev.buffer_info,
                        serializer=lambda p: p._pxy_serialize(
                            serializers=("dask", "pickle")
                        ),
                    )
                    gc.collect()
                    if memory_freed > 0:
                        return True  # Ask RMM to retry the allocation
                    else:
                        with io.StringIO() as f:
                            traceback.print_stack(file=f)
                            f.seek(0)
                            tb = f.read()
                        self.logger.warning(
                            "RMM allocation of %s failed, spill-on-demand couldn't "
                            "find any device memory to spill:\n%s\ntraceback:\n%s\n",
                            format_bytes(nbytes),
                            self.manager.pprint(),
                            tb,
                        )
                        # Since we didn't find anything to spill, we give up.
                        return False

                current_mr = rmm.mr.get_current_device_resource()
                mr = rmm.mr.FailureCallbackResourceAdaptor(current_mr, oom)
                rmm.mr.set_current_device_resource(mr)

    def evict(self) -> int:
        """Manually evict 1% of host limit.

        Dask uses this to trigger CPU-to-Disk spilling. We don't know how much
        we need to spill but Dask will call `evict()` repeatedly until enough
        is spilled. We ask for 1% each time.

        Return
        ------
        nbytes: int
            Number of bytes spilled or -1 if nothing to spill.
        """
        assert self.manager._host_memory_limit is not None
        ret = self.manager.evict(
            nbytes=int(self.manager._host_memory_limit * 0.01),
            proxies_access=self.manager._host.buffer_info,
            serializer=ProxifyHostFile.serialize_proxy_to_disk_inplace,
        )
        gc.collect()
        return ret if ret > 0 else -1

    @property
    def fast(self):
        """Alternative access to `.evict()` used by Dask

        Dask expects `.fast.evict()` to be availabe for manually triggering
        of CPU-to-Disk spilling.
        """
        if len(self.manager._host) == 0:
            return False  # We have nothing in host memory to spill

        class EvictDummy:
            @staticmethod
            def evict():
                ret = (
                    None,
                    None,
                    self.evict(),
                )
                gc.collect()
                return ret

        return EvictDummy()

    def __setitem__(self, key, value):
        with self.lock:
            self.initialize_spill_on_demand_once()
            if key in self.store:
                # Make sure we register the removal of an existing key
                del self[key]
            self.store[key] = self.manager.proxify(value)

    def __getitem__(self, key):
        with self.lock:
            ret, incompatible_type_found = self.store[key]
        if self.compatibility_mode:
            ret = unproxify_device_objects(ret, skip_explicit_proxies=True)
            self.manager.maybe_evict()
        elif incompatible_type_found:
            # Notice, we only call `unproxify_device_objects()` when `key`
            # contains incompatible types.
            ret = unproxify_device_objects(ret, only_incompatible_types=True)
            self.manager.maybe_evict()
        return ret

    def __delitem__(self, key):
        with self.lock:
            del self.store[key]

    @classmethod
    def register_disk_spilling(cls) -> None:
        """Register Dask serializers that writes to disk

        This is a static method because the registration of a Dask
        serializer/deserializer pair is a global operation thus we can
        only register one such pair. This means that all instances of
        the ``ProxifyHostFile`` end up using the same ``local_directory``.
        """
        assert cls._spill_to_disk is not None

        def disk_dumps(x):
            # When using GDS, we prepend "cuda" to serializers to keep the CUDA
            # objects on the GPU. Otherwise the "dask" or "pickle" serializer will
            # copy everything to host memory.
            serializers = ["dask", "pickle"]
            if cls._spill_to_disk.gds_enabled:
                serializers = ["cuda"] + serializers
            serialize_header, frames = serialize_and_split(
                x, serializers=serializers, on_error="raise"
            )
            if frames:
                compression, frames = zip(*map(maybe_compress, frames))
            else:
                compression = []
            serialize_header["compression"] = compression
            serialize_header["count"] = len(frames)
            return (
                {
                    "serializer": "disk",
                    "disk-io-header": disk_write(
                        path=cls._spill_to_disk.gen_file_path(),
                        frames=frames,
                        shared_filesystem=cls._spill_to_disk.shared_filesystem,
                        gds=cls._spill_to_disk.gds_enabled,
                    ),
                    "serialize-header": serialize_header,
                },
                [],
            )

        def disk_loads(header, frames):
            assert frames == []
            frames = disk_read(
                header["disk-io-header"], gds=cls._spill_to_disk.gds_enabled
            )
            if "compression" in header["serialize-header"]:
                frames = decompress(header["serialize-header"], frames)
            return merge_and_deserialize(header["serialize-header"], frames)

        register_serialization_family("disk", disk_dumps, disk_loads)

    @classmethod
    def serialize_proxy_to_disk_inplace(cls, proxy: ProxyObject) -> None:
        """Serialize `proxy` to disk.

        Avoid de-serializing if `proxy` is serialized using "dask" or
        "pickle". In this case the already serialized data is written
        directly to disk.

        Parameters
        ----------
        proxy : ProxyObject
            Proxy object to serialize using the "disk" serialize.
        """
        assert cls._spill_to_disk is not None
        pxy = proxy._pxy_get(copy=True)
        if pxy.is_serialized():
            header, frames = pxy.obj
            if header["serializer"] in ("dask", "pickle"):
                pxy.obj = (
                    {
                        "serializer": "disk",
                        "disk-io-header": disk_write(
                            path=cls._spill_to_disk.gen_file_path(),
                            frames=frames,
                            shared_filesystem=cls._spill_to_disk.shared_filesystem,
                        ),
                        "serialize-header": header,
                    },
                    [],
                )
                pxy.serializer = "disk"
                proxy._pxy_set(pxy)
                return
        proxy._pxy_serialize(serializers=("disk",), proxy_detail=pxy)
