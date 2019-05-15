import os
import warnings

from tornado import gen

from dask.distributed import LocalCluster
from distributed.diskutils import WorkSpace
from distributed.nanny import Nanny
from distributed.worker import Worker, TOTAL_MEMORY
from distributed.utils import parse_bytes, warn_on_duration

from .device_host_file import DeviceHostFile
from .utils import get_n_gpus, get_device_total_memory


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
            visible = list(map(int, os.environ["CUDA_VISIBLE_DEVICES"].split(",")))
        except KeyError:
            visible = list(range(get_n_gpus()))

    L = visible[i:] + visible[:i]
    return ",".join(map(str, L))


class LocalCUDACluster(LocalCluster):
    def __init__(
        self,
        n_workers=None,
        threads_per_worker=1,
        processes=True,
        memory_limit=None,
        device_memory_limit=None,
        **kwargs,
    ):
        if n_workers is None:
            n_workers = get_n_gpus()
        if not processes:
            raise NotImplementedError("Need processes to segment GPUs")
        if n_workers > get_n_gpus():
            raise ValueError("Can not specify more processes than GPUs")
        if memory_limit is None:
            memory_limit = TOTAL_MEMORY / n_workers
        self.host_memory_limit = memory_limit
        self.device_memory_limit = device_memory_limit

        LocalCluster.__init__(
            self,
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            memory_limit=memory_limit,
            **kwargs,
        )

    @gen.coroutine
    def _start(self, ip=None, n_workers=0):
        """
        Start all cluster services.
        """
        if self.status == "running":
            return
        if (ip is None) and (not self.scheduler_port) and (not self.processes):
            # Use inproc transport for optimization
            scheduler_address = "inproc://"
        elif ip is not None and ip.startswith("tls://"):
            scheduler_address = "%s:%d" % (ip, self.scheduler_port)
        else:
            if ip is None:
                ip = "127.0.0.1"
            scheduler_address = (ip, self.scheduler_port)
        self.scheduler.start(scheduler_address)

        yield [
            self._start_worker(
                **self.worker_kwargs,
                env={"CUDA_VISIBLE_DEVICES": cuda_visible_devices(i)},
            )
            for i in range(n_workers)
        ]

        self.status = "running"

        raise gen.Return(self)

    @gen.coroutine
    def _start_worker(self, death_timeout=60, **kwargs):
        if self.status and self.status.startswith("clos"):
            warnings.warn("Tried to start a worker while status=='%s'" % self.status)
            return

        if self.processes:
            W = Nanny
            kwargs["quiet"] = True
        else:
            W = Worker

        local_dir = kwargs.get("local_dir", "dask-worker-space")
        with warn_on_duration(
            "1s",
            "Creating scratch directories is taking a surprisingly long time. "
            "This is often due to running workers on a network file system. "
            "Consider specifying a local-directory to point workers to write "
            "scratch data to a local disk.",
        ):
            _workspace = WorkSpace(os.path.abspath(local_dir))
            _workdir = _workspace.new_work_dir(prefix="worker-")
            local_dir = _workdir.dir_path

        device_index = int(kwargs["env"]["CUDA_VISIBLE_DEVICES"].split(",")[0])
        if self.device_memory_limit is None:
            self.device_memory_limit = get_device_total_memory(device_index)
        elif isinstance(self.device_memory_limit, str):
            self.device_memory_limit = parse_bytes(self.device_memory_limit)
        data = DeviceHostFile(
            device_memory_limit=self.device_memory_limit,
            memory_limit=self.host_memory_limit,
            local_dir=local_dir,
        )

        w = yield W(
            self.scheduler.address,
            loop=self.loop,
            death_timeout=death_timeout,
            silence_logs=self.silence_logs,
            data=data,
            **kwargs,
        )

        self.workers.append(w)

        while w.status != "closed" and w.worker_address not in self.scheduler.workers:
            yield gen.sleep(0.01)

        if w.status == "closed" and self.scheduler.status == "running":
            self.workers.remove(w)
            raise gen.TimeoutError("Worker failed to start")

        raise gen.Return(w)
