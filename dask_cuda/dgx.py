import os

from dask.distributed import Nanny, SpecCluster, Scheduler

from .local_cuda_cluster import cuda_visible_devices


class CPUAffinity:
    def __init__(self, cores):
        self.cores = cores

    def setup(self, worker=None):
        os.sched_setaffinity(0, self.cores)


affinity = {  # nvidia-smi topo -m
    0: list(range(0, 20)) + list(range(40, 60)),
    1: list(range(0, 20)) + list(range(40, 60)),
    2: list(range(0, 20)) + list(range(40, 60)),
    3: list(range(0, 20)) + list(range(40, 60)),
    4: list(range(20, 40)) + list(range(60, 79)),
    5: list(range(20, 40)) + list(range(60, 79)),
    6: list(range(20, 40)) + list(range(60, 79)),
    7: list(range(20, 40)) + list(range(60, 79)),
}


def DGX(interface="ib", **kwargs):
    gpus = list(
        map(int, os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(","))
    )

    spec = {
        i: {
            "cls": Nanny,
            "options": {
                "env": {
                    "CUDA_VISIBLE_DEVICES": cuda_visible_devices(ii, gpus),
                    # 'UCX_NET_DEVICES': 'mlx5_%d:1' % (i // 2)
                    "UCX_TLS": "rc,cuda_copy,cuda_ipc",
                },
                "interface": interface + str(i // 2),
                "protocol": "ucx",
                "ncores": 1,
                "data": dict,
                "preload": ["dask_cuda.initialize_context"],
                "dashboard_address": ":0",
                "plugins": [CPUAffinity(affinity[i])],
            },
        }
        for ii, i in enumerate(gpus)
    }

    scheduler = {
        "cls": Scheduler,
        "options": {
            "interface": interface + str(gpus[0] // 2),
            "protocol": "ucx",
            "dashboard_address": ":8787",
        },
    }

    return SpecCluster(workers=spec, scheduler=scheduler, **kwargs)
