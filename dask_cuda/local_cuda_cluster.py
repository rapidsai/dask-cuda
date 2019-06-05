import copy
import os

from dask.distributed import LocalCluster
from distributed.worker import TOTAL_MEMORY

from .utils import get_n_gpus


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
        CUDA_VISIBLE_DEVICES=None,
        **kwargs
    ):
        if n_workers is None:
            n_workers = get_n_gpus()
        if CUDA_VISIBLE_DEVICES is None:
            CUDA_VISIBLE_DEVICES = cuda_visible_devices(0)
        if isinstance(CUDA_VISIBLE_DEVICES, str):
            CUDA_VISIBLE_DEVICES = CUDA_VISIBLE_DEVICES.split(",")
        CUDA_VISIBLE_DEVICES = list(map(int, CUDA_VISIBLE_DEVICES))
        if memory_limit is None:
            memory_limit = TOTAL_MEMORY / n_workers
        if not processes:
            raise ValueError(
                "Processes are necessary in order to use multiple GPUs with Dask"
            )

        super().__init__(
            n_workers=0,
            threads_per_worker=threads_per_worker,
            memory_limit=memory_limit,
            processes=True,
            **kwargs
        )

        self.new_spec["options"]["preload"] = self.new_spec["options"].get(
            "preload", []
        ) + ["dask_cuda.initialize_context"]

        self.cuda_visible_devices = CUDA_VISIBLE_DEVICES
        self.scale(n_workers)

    def new_worker_spec(self):
        try:
            name = min(set(self.cuda_visible_devices) - set(self.worker_spec))
        except Exception:
            raise ValueError(
                "Can not scale beyond visible devices", self.cuda_visible_devices
            )

        spec = copy.deepcopy(self.new_spec)
        ii = self.cuda_visible_devices.index(name)
        spec["options"].update(
            {
                "env": {
                    "CUDA_VISIBLE_DEVICES": cuda_visible_devices(
                        ii, self.cuda_visible_devices
                    )
                }
            }
        )

        return name, spec
