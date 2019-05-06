from distributed import Nanny

from .cuda_worker import CUDAWorker


class CUDANanny(Nanny):
    """ A process to manage CUDAWorker processes

    This is a subclass of Nanny, with the only difference
    being worker_class=CUDAWorker.
    """

    def __init__(self, *args, worker_class=CUDAWorker, **kwargs):
        Nanny.__init__(self, *args, worker_class=worker_class, **kwargs)
