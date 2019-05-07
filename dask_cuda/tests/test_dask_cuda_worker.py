from __future__ import print_function, division, absolute_import

import pytest

pytest.importorskip("requests")

import sys, os
from time import sleep
from toolz import first

from distributed import Client
from distributed.metrics import time
from distributed.utils import sync, tmpfile
from distributed.utils_test import popen, slow, terminate_process, wait_for_port
from distributed.utils_test import loop  # noqa: F401

from dask_cuda.utils import get_n_gpus
from dask_cuda.local_cuda_cluster import cuda_visible_devices


def test_cuda_visible_devices_worker(loop):
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,7,8"
    try:
        with popen(["dask-scheduler", "--port", "9359", "--no-bokeh"]) as sched:
            with popen(
                [
                    "dask-cuda-worker",
                    "127.0.0.1:9359",
                    "--host",
                    "127.0.0.1",
                    "--no-bokeh",
                    "--worker-class",
                    "Worker",
                ]
            ) as worker:
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


def test_cuda_visible_devices_cudaworker_single(loop):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    try:
        with popen(["dask-scheduler", "--port", "9359", "--no-bokeh"]) as sched:
            with popen(
                [
                    "dask-cuda-worker",
                    "127.0.0.1:9359",
                    "--host",
                    "127.0.0.1",
                    "--no-bokeh",
                ]
            ) as worker:
                with Client("127.0.0.1:9359", loop=loop) as client:
                    start = time()
                    while True:
                        if len(client.scheduler_info()["workers"]) == 1:
                            break
                        else:
                            assert time() - start < 10
                            sleep(0.1)

                    def get_visible_devices():
                        return os.environ["CUDA_VISIBLE_DEVICES"]

                    # verify 4 workers with the 4 expected CUDA_VISIBLE_DEVICES
                    result = client.run(get_visible_devices)
                    expected = {"0": 1}
                    for v in result.values():
                        del expected[v]
                    assert len(expected) == 0
    finally:
        del os.environ["CUDA_VISIBLE_DEVICES"]


def test_cuda_visible_devices_cudaworker(loop):
    n_gpus = get_n_gpus()
    if n_gpus < 2:
        pytest.skip("More than 1 GPU required for test")
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices(0)
    try:
        with popen(["dask-scheduler", "--port", "9359", "--no-bokeh"]) as sched:
            with popen(
                [
                    "dask-cuda-worker",
                    "127.0.0.1:9359",
                    "--host",
                    "127.0.0.1",
                    "--no-bokeh",
                ]
            ) as worker:
                with Client("127.0.0.1:9359", loop=loop) as client:
                    start = time()
                    while True:
                        if len(client.scheduler_info()["workers"]) == n_gpus:
                            break
                        else:
                            assert time() - start < 10
                            sleep(0.1)

                    def get_visible_devices():
                        return os.environ["CUDA_VISIBLE_DEVICES"]

                    # verify n_gpus workers with CUDA_VISIBLE_DEVICES
                    # in proper order
                    result = client.run(get_visible_devices)
                    expected = {
                        cuda_visible_devices(i, [j for j in range(n_gpus)]): 1
                        for i in range(n_gpus)
                    }
                    for v in result.values():
                        del expected[v]
                    assert len(expected) == 0
    finally:
        del os.environ["CUDA_VISIBLE_DEVICES"]
