from distributed import Nanny

from .cuda_worker import CUDAWorker


class CUDANanny(Nanny):
    def __init__(self, scheduler_ip=None, **kwargs):
        Nanny.__init__(self, worker_class=CUDAWorker,
                       scheduler_ip=scheduler_ip,
                       **kwargs)
