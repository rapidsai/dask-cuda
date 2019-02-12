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


def test_cuda_visible_devices(loop):
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,7,8"
    try:
        with popen(["dask-scheduler", "--port", "9359", "--no-bokeh"]) as sched:
            with popen(
                [
                    "dask-cuda-worker",
                    "127.0.0.1:9359",
                    "--host",
                    "127.0.0.1",
                    "--nprocs",
                    "4",
                    "--no-bokeh",
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
