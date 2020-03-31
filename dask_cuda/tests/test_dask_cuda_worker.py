from __future__ import absolute_import, division, print_function

import os
from time import sleep

from distributed import Client
from distributed.metrics import time
from distributed.utils_test import loop  # noqa: F401
from distributed.utils_test import popen
from dask_cuda.utils import get_gpu_count

import pytest


def test_cuda_visible_devices(loop):  # noqa: F811
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,7,8"
    try:
        with popen(["dask-scheduler", "--port", "9359", "--no-dashboard"]):
            with popen(
                [
                    "dask-cuda-worker",
                    "127.0.0.1:9359",
                    "--host",
                    "127.0.0.1",
                    "--device-memory-limit",
                    "1 MB",
                    "--no-dashboard",
                ]
            ):
                with Client("127.0.0.1:9359", loop=loop) as client:
                    start = time()
                    while True:
                        if len(client.scheduler_info()["workers"]) == 4:
                            break
                        else:
                            assert time() - start < 10
                            sleep(0.1)

                    def get_visible_devices():
                        return os.environ["CUDA_VISIBLE_DEVICES"]

                    # verify 4 workers with the 4 expected CUDA_VISIBLE_DEVICES
                    result = client.run(get_visible_devices)
                    expected = {"2,3,7,8": 1, "3,7,8,2": 1, "7,8,2,3": 1, "8,2,3,7": 1}
                    for v in result.values():
                        del expected[v]
                    assert len(expected) == 0
    finally:
        del os.environ["CUDA_VISIBLE_DEVICES"]


def test_rmm_pool(loop):  # noqa: F811
    rmm = pytest.importorskip("rmm")
    with popen(["dask-scheduler", "--port", "9369", "--no-dashboard"]):
        with popen(
            [
                "dask-cuda-worker",
                "127.0.0.1:9369",
                "--host",
                "127.0.0.1",
                "--rmm-pool-size",
                "2 GB",
                "--no-dashboard",
            ]
        ):
            with Client("127.0.0.1:9369", loop=loop) as client:
                start = time()
                while True:
                    if len(client.scheduler_info()["workers"]) == get_gpu_count():
                        break
                    else:
                        assert time() - start < 10
                        sleep(0.1)

                memory_info = client.run(rmm.get_info)
                for v in memory_info.values():
                    assert v.total == 2000000000
