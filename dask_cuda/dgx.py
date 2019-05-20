import os

from dask.distributed import Nanny, SpecCluster, Scheduler
from distributed.utils import get_ip_interface

# from distributed.bokeh.scheduler import BokehScheduler

from .local_cuda_cluster import cuda_visible_devices


def DGX(**kwargs):
    ethernet_host = get_ip_interface("enp1s0f0")
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
                    # "UCX_TLS": "rc,cuda_copy",
                },
                "interface": "ib%d" % (i // 2),
                "protocol": "ucx",
                "ncores": 1,
                "data": dict,
            },
        }
        for ii, i in enumerate(gpus)
    }

    scheduler = {
        "cls": Scheduler,
        "options": {
            "interface": "ib0",
            "protocol": "ucx",
            "dashboard_address": ethernet_host + ":8787",
        },
    }

    return SpecCluster(workers=spec, scheduler=scheduler, **kwargs)
