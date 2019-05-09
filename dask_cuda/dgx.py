import os

from dask.distributed import Nanny, SpecCluster, Scheduler

# from distributed.bokeh.scheduler import BokehScheduler

from .local_cuda_cluster import cuda_visible_devices


def DGX(**kwargs):
    gpus = map(
        int, os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(",")
    )

    spec = {
        i: {
            "cls": Nanny,
            "options": {
                "env": {
                    "CUDA_VISIBLE_DEVICES": cuda_visible_devices(i, range(8)),
                    # 'UCX_NET_DEVICES': 'mlx5_%d:1' % (i // 2)
                },
                "interface": "ib%d" % (i // 2),
                "protocol": "ucx",
                "ncores": 1,
                "data": dict,
            },
        }
        for i in gpus
    }

    scheduler = {
        "cls": Scheduler,
        "options": {
            "interface": "ib0",
            "protocol": "ucx",
            # "services": {("bokeh", "10.33.227.165:8900"): BokehScheduler},
        },
    }

    return SpecCluster(workers=spec, scheduler=scheduler, **kwargs)
