from distributed import Nanny

from .cuda_worker import CUDAWorker


class CUDANanny(Nanny):
    """ A process to manage CUDAWorker processes

    This is a subclass of Nanny, with the only difference
    being worker_class=CUDAWorker.
    """
    def __init__(self, scheduler_ip=None, **kwargs):
        Nanny.__init__(self, worker_class=CUDAWorker,
                       scheduler_ip=scheduler_ip,
                       **kwargs)
