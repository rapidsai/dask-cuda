from tornado import gen
from numba import cuda

import dask
from distributed import Worker
from distributed.worker import logger
from distributed.compatibility import unicode
from distributed.utils import format_bytes, ignoring, parse_bytes, PeriodicCallback

from .device_host_file import DeviceHostFile


def get_device_total_memory():
    """ Return total memory of CUDA device from current context """
    return cuda.current_context().get_memory_info()[1]  # (free, total)


def get_device_used_memory():
    """ Return used memory of CUDA device from current context """
    memory_info = cuda.current_context().get_memory_info()  # (free, total)
    return memory_info[1] - memory_info[0]


def parse_device_memory_limit(memory_limit, ncores):
    """ Parse device memory limit input """
    if memory_limit is None or memory_limit == 0 or memory_limit == "auto":
        memory_limit = int(get_device_total_memory())
    with ignoring(ValueError, TypeError):
        x = float(memory_limit)
        if isinstance(x, float) and x <= 1:
            return int(x * get_device_total_memory())

    if isinstance(memory_limit, (unicode, str)):
        return parse_bytes(memory_limit)
    else:
        return int(memory_limit)


class CUDAWorker(Worker):
    """ CUDA Worker node in a Dask distributed cluster

    Parameters
    ----------
    device_memory_limit: int, float, string
        Number of bytes of CUDA device memory that this worker should use.
        Set to zero for no limit or 'auto' for 100% of memory use.
        Use strings or numbers like 5GB or 5e9
    device_memory_target_fraction: float
        Fraction of CUDA device memory to try to stay beneath
    device_memory_spill_fraction: float
        Fraction of CUDA device memory at which we start spilling to disk
    device_memory_pause_fraction: float
        Fraction of CUDA device memory at which we stop running new tasks

    Note: CUDAWorker is a subclass fo distributed.Worker, only parameters
    specific for CUDAWorker are listed here. For a complete list of
    parameters, refer to that.
    """

    def __init__(self, *args, **kwargs):
        self.device_memory_limit = kwargs.pop(
            "device_memory_limit", get_device_total_memory()
        )

        if "device_memory_target_fraction" in kwargs:
            self.device_memory_target_fraction = kwargs.pop(
                "device_memory_target_fraction"
            )
        else:
            self.device_memory_target_fraction = dask.config.get(
                "distributed.worker.device-memory.target"
            )
        if "device_memory_spill_fraction" in kwargs:
            self.device_memory_spill_fraction = kwargs.pop(
                "device_memory_spill_fraction"
            )
        else:
            self.device_memory_spill_fraction = dask.config.get(
                "distributed.worker.device-memory.spill"
            )
        if "device_memory_pause_fraction" in kwargs:
            self.device_memory_pause_fraction = kwargs.pop(
                "device_memory_pause_fraction"
            )
        else:
            self.device_memory_pause_fraction = dask.config.get(
                "distributed.worker.device-memory.pause"
            )

        super().__init__(*args, **kwargs)

        self.device_memory_limit = parse_device_memory_limit(
            self.device_memory_limit, self.ncores
        )

        self.data = DeviceHostFile(
            device_memory_limit=self.device_memory_limit,
            memory_limit=self.memory_limit,
            local_dir=self.local_dir,
        )

        self._paused = False
        self._device_paused = False

        if self.device_memory_limit:
            self._device_memory_monitoring = False
            pc = PeriodicCallback(
                self.device_memory_monitor,
                self.memory_monitor_interval * 1000,
                io_loop=self.io_loop,
            )
            self.periodic_callbacks["device_memory"] = pc

    def _start(self, addr_on_port=0):
        super()._start(addr_on_port)
        if self.device_memory_limit:
            logger.info(
                "        Device Memory: %26s", format_bytes(self.device_memory_limit)
            )
            logger.info("-" * 49)

    def _check_for_pause(
        self,
        fraction,
        pause_fraction,
        used_memory,
        memory_limit,
        paused,
        free_func,
        worker_description,
    ):
        if pause_fraction and fraction > pause_fraction:
            # Try to free some memory while in paused state
            if free_func:
                free_func()
            if not self._paused:
                logger.warning(
                    "%s is at %d%% memory usage. Pausing worker.  "
                    "Process memory: %s -- Worker memory limit: %s",
                    worker_description,
                    int(fraction * 100),
                    format_bytes(used_memory),
                    format_bytes(memory_limit),
                )
                return True
        return False

    def _resume_message(self, fraction, used_memory, memory_limit, worker_description):
        logger.warning(
            "%s is at %d%% memory usage. Resuming worker. "
            "Process memory: %s -- Worker memory limit: %s",
            worker_description,
            int(fraction * 100),
            format_bytes(used_memory),
            format_bytes(memory_limit),
        )

    def _resume_worker(self):
        if self.paused and not (self._paused or self._device_paused):
            self.paused = False
            self.ensure_computing()

    @gen.coroutine
    def memory_monitor(self):
        """ Track this process's memory usage and act accordingly

        If we rise above (memory_spill_fraction * memory_limit) of
        memory use, start dumping data to disk. The default value for
        memory_spill_fraction is 0.7, defined via configuration
        'distributed.worker.memory.target'.

        If we rise above (memory_pause_fraction * memory_limit) of
        memory use , stop execution of new tasks. The default value
        for memory_pause_fraction is 0.8, defined via configuration
        'distributed.worker.memory.pause'.
        """
        if self._memory_monitoring:
            return
        self._memory_monitoring = True
        total = 0

        proc = self.monitor.proc
        memory = proc.memory_info().rss
        frac = memory / self.memory_limit

        # Pause worker threads if device memory use above
        # (self.memory_pause_fraction * 100)%
        old_pause_state = self._paused
        worker_description = "Worker"
        self._paused = self._check_for_pause(
            frac,
            self.memory_pause_fraction,
            memory,
            self.memory_limit,
            self._paused,
            self._throttled_gc.collect(),
            worker_description,
        )
        if old_pause_state and not self._paused:
            self._resume_message(frac, memory, self.memory_limit, worker_description)
        self._resume_worker()

        # Dump data to disk if memory use above
        # (self.memory_spill_fraction * 100)%
        if self.memory_spill_fraction and frac > self.memory_spill_fraction:
            target = self.memory_limit * self.memory_target_fraction
            count = 0
            need = memory - target
            while memory > target:
                if not self.data.host_buffer.fast:
                    logger.warning(
                        "Memory use is high but worker has no data "
                        "to store to disk.  Perhaps some other process "
                        "is leaking memory?  Process memory: %s -- "
                        "Worker memory limit: %s",
                        format_bytes(proc.memory_info().rss),
                        format_bytes(self.memory_limit),
                    )
                    break
                k, v, weight = self.data.host_buffer.fast.evict()
                del k, v
                total += weight
                count += 1
                yield gen.moment
                memory = proc.memory_info().rss
                if total > need and memory > target:
                    # Issue a GC to ensure that the evicted data is actually
                    # freed from memory and taken into account by the monitor
                    # before trying to evict even more data.
                    self._throttled_gc.collect()
                    memory = proc.memory_info().rss
            if count:
                logger.debug(
                    "Moved %d pieces of data and %s bytes to disk",
                    count,
                    format_bytes(total),
                )

        self._memory_monitoring = False
        raise gen.Return(total)

    @gen.coroutine
    def device_memory_monitor(self):
        """ Track this process's memory usage and act accordingly

        If we rise above (device_memory_spill_fraction * memory_limit) of
        device memory use, start dumping data to disk. The default value
        for device_memory_spill_fraction is 0.7, defined via configuration
        'distributed.worker.device-memory.target'.

        If we rise above (device_memory_pause_fraction * memory_limit) of
        device memory use, stop execution of new tasks. The default value
        for device_memory_pause_fraction is 0.8, defined via configuration
        'distributed.worker.device-memory.pause'.
        """
        if self._memory_monitoring:
            return
        self._device_memory_monitoring = True
        total = 0
        memory = get_device_used_memory()
        frac = memory / self.device_memory_limit

        # Pause worker threads if device memory use above
        # (self.device_memory_pause_fraction * 100)%
        old_pause_state = self._device_paused
        worker_description = "Worker's CUDA device"
        self._device_paused = self._check_for_pause(
            frac,
            self.device_memory_pause_fraction,
            memory,
            self.device_memory_limit,
            self._device_paused,
            None,
            worker_description,
        )
        if old_pause_state and not self._device_paused:
            self._resume_message(
                frac, memory, self.device_memory_limit, worker_description
            )
        self._resume_worker()

        # Dump device data to host if device memory use above
        # (self.device_memory_spill_fraction * 100)%
        if (
            self.device_memory_spill_fraction
            and frac > self.device_memory_spill_fraction
        ):
            target = self.device_memory_limit * self.device_memory_target_fraction
            count = 0
            while memory > target:
                if not self.data.device_buffer.fast:
                    logger.warning(
                        "CUDA device memory use is high but worker has "
                        "no data to store to host.  Perhaps some other "
                        "process is leaking memory?  Process memory: "
                        "%s -- Worker memory limit: %s",
                        format_bytes(get_device_used_memory()),
                        format_bytes(self.device_memory_limit),
                    )
                    break
                k, v, weight = self.data.device_buffer.fast.evict()
                del k, v
                total += weight
                count += 1
                yield gen.moment
                memory = get_device_used_memory()
            if count:
                logger.debug(
                    "Moved %d pieces of data and %s bytes to host memory",
                    count,
                    format_bytes(total),
                )

        self._device_memory_monitoring = False
        raise gen.Return(total)
