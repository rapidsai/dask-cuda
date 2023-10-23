from typing import Literal

import distributed
from distributed import Nanny, Worker


class MockWorker(Worker):
    """Mock Worker class preventing NVML from getting used by SystemMonitor.

    By preventing the Worker from initializing NVML in the SystemMonitor, we can
    mock test multiple devices in `CUDA_VISIBLE_DEVICES` behavior with single-GPU
    machines.
    """

    def __init__(self, *args, **kwargs):
        distributed.diagnostics.nvml.device_get_count = MockWorker.device_get_count
        self._device_get_count = distributed.diagnostics.nvml.device_get_count
        super().__init__(*args, **kwargs)

    def __del__(self):
        distributed.diagnostics.nvml.device_get_count = self._device_get_count

    @staticmethod
    def device_get_count():
        return 0


class IncreasedCloseTimeoutNanny(Nanny):
    """Increase `Nanny`'s close timeout.

    The internal close timeout mechanism of `Nanny` recomputes the time left to kill
    the `Worker` process based on elapsed time of the close task, which may leave
    very little time for the subprocess to shutdown cleanly, which may cause tests
    to fail when the system is under higher load. This class increases the default
    close timeout of 5.0 seconds that `Nanny` sets by default, which can be overriden
    via Distributed's public API.

    This class can be used with the `worker_class` argument of `LocalCluster` or
    `LocalCUDACluster` to provide a much higher default of 30.0 seconds.
    """

    async def close(  # type:ignore[override]
        self, timeout: float = 30.0, reason: str = "nanny-close"
    ) -> Literal["OK"]:
        return await super().close(timeout=timeout, reason=reason)
