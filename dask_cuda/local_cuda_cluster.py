import copy
import os

import dask
from dask.distributed import LocalCluster
from distributed.system import MEMORY_LIMIT
from distributed.utils import parse_bytes

from .device_host_file import DeviceHostFile
from .utils import (
    CPUAffinity,
    get_cpu_affinity,
    get_device_total_memory,
    get_n_gpus,
)


def _ucx_net_devices(dev, ucx_net_devices):
    net_dev = None
    if callable(ucx_net_devices):
        net_dev = ucx_net_devices(dev)
    elif isinstance(ucx_net_devices, str):
        if ucx_net_devices == "auto":
            # If TopologicalDistance from ucp is available, we set the UCX
            # net device to the closest network device explicitly.
            from ucp._libs.topological_distance import TopologicalDistance

            net_dev = ""
            td = TopologicalDistance()
            ibs = td.get_cuda_distances_from_device_index(dev, "openfabrics")
            if len(ibs) > 0:
                net_dev += ibs[0]["name"] + ":1,"
            ifnames = td.get_cuda_distances_from_device_index(dev, "network")
            if len(ifnames) > 0:
                net_dev += ifnames[0]["name"]
        else:
            net_dev = ucx_net_devices
    return net_dev


def cuda_visible_devices(i, visible=None):
    """ Cycling values for CUDA_VISIBLE_DEVICES environment variable

    Examples
    --------
    >>> cuda_visible_devices(0, range(4))
    '0,1,2,3'
    >>> cuda_visible_devices(3, range(8))
    '3,4,5,6,7,0,1,2'
    """
    if visible is None:
        try:
            visible = map(int, os.environ["CUDA_VISIBLE_DEVICES"].split(","))
        except KeyError:
            visible = range(get_n_gpus())
    visible = list(visible)

    L = visible[i:] + visible[:i]
    return ",".join(map(str, L))


class LocalCUDACluster(LocalCluster):
    """ A variant of LocalCluster that uses one GPU per process

    This assigns a different CUDA_VISIBLE_DEVICES environment variable to each
    worker process.

    Parameters
    ----------
    CUDA_VISIBLE_DEVICES: str
        String like ``"0,1,2,3"`` or ``[0, 1, 2, 3]`` to restrict activity to
        different GPUs

    See Also
    --------
    LocalCluster
    """

    def __init__(
        self,
        n_workers=None,
        threads_per_worker=1,
        processes=True,
        memory_limit=None,
        device_memory_limit=None,
        CUDA_VISIBLE_DEVICES=None,
        data=None,
        local_directory=None,
        ucx_net_devices=None,
        **kwargs,
    ):
        if CUDA_VISIBLE_DEVICES is None:
            CUDA_VISIBLE_DEVICES = cuda_visible_devices(0)
        if isinstance(CUDA_VISIBLE_DEVICES, str):
            CUDA_VISIBLE_DEVICES = CUDA_VISIBLE_DEVICES.split(",")
        CUDA_VISIBLE_DEVICES = list(map(int, CUDA_VISIBLE_DEVICES))
        if n_workers is None:
            n_workers = len(CUDA_VISIBLE_DEVICES)
        if memory_limit is None:
            memory_limit = MEMORY_LIMIT / n_workers
        self.host_memory_limit = memory_limit
        self.device_memory_limit = device_memory_limit

        if not processes:
            raise ValueError(
                "Processes are necessary in order to use multiple GPUs with Dask"
            )

        if self.device_memory_limit is None:
            self.device_memory_limit = get_device_total_memory(0)
        elif isinstance(self.device_memory_limit, str):
            self.device_memory_limit = parse_bytes(self.device_memory_limit)

        if data is None:
            data = (
                DeviceHostFile,
                {
                    "device_memory_limit": self.device_memory_limit,
                    "memory_limit": self.host_memory_limit,
                    "local_directory": local_directory
                    or dask.config.get("temporary-directory")
                    or os.getcwd(),
                },
            )

        if ucx_net_devices == "auto":
            try:
                from ucp._libs.topological_distance import TopologicalDistance  # noqa
            except ImportError:
                raise ValueError(
                    "ucx_net_devices set to 'auto' but UCX-Py is not "
                    "installed or it's compiled without hwloc support"
                )
        elif ucx_net_devices == "":
            raise ValueError("ucx_net_devices can not be an empty string")
        self.ucx_net_devices = ucx_net_devices

        super().__init__(
            n_workers=0,
            threads_per_worker=threads_per_worker,
            memory_limit=memory_limit,
            processes=True,
            data=data,
            local_directory=local_directory,
            **kwargs,
        )

        self.new_spec["options"]["preload"] = self.new_spec["options"].get(
            "preload", []
        ) + ["dask_cuda.initialize"]
        self.new_spec["options"]["preload_argv"] = self.new_spec["options"].get(
            "preload_argv", []
        ) + ["--create-cuda-context"]

        self.cuda_visible_devices = CUDA_VISIBLE_DEVICES
        self.scale(n_workers)
        self.sync(self._correct_state)

    def new_worker_spec(self):
        try:
            name = min(set(self.cuda_visible_devices) - set(self.worker_spec))
        except Exception:
            raise ValueError(
                "Can not scale beyond visible devices", self.cuda_visible_devices
            )

        spec = copy.deepcopy(self.new_spec)
        worker_count = self.cuda_visible_devices.index(name)
        visible_devices = cuda_visible_devices(worker_count, self.cuda_visible_devices)
        spec["options"].update(
            {
                "env": {
                    "CUDA_VISIBLE_DEVICES": visible_devices,
                },
                "plugins": {CPUAffinity(get_cpu_affinity(worker_count))},
            }
        )

        net_dev = _ucx_net_devices(visible_devices.split(",")[0], self.ucx_net_devices)
        if net_dev is not None:
            spec["options"]["env"]["UCX_NET_DEVICES"] = net_dev

        return {name: spec}
