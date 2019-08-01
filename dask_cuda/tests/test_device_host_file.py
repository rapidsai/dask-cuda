import numpy as np
import cupy
from dask_cuda.device_host_file import DeviceHostFile
from random import randint

import pytest
from cupy.testing import assert_array_equal


@pytest.mark.parametrize("num_host_arrays", [1, 10, 100])
@pytest.mark.parametrize("num_device_arrays", [1, 10, 100])
@pytest.mark.parametrize("array_size_range", [(1, 1000), (100, 100), (1000, 1000)])
def test_device_host_file_short(
    tmp_path, num_device_arrays, num_host_arrays, array_size_range
):
    tmpdir = tmp_path / "storage"
    tmpdir.mkdir()
    dhf = DeviceHostFile(
        device_memory_limit=1024 * 16, memory_limit=1024 * 16, local_directory=tmpdir
    )

    host = [
        ("x-%d" % i, np.random.random(randint(*array_size_range)))
        for i in range(num_host_arrays)
    ]
    device = [
        ("dx-%d" % i, cupy.random.random(randint(*array_size_range)))
        for i in range(num_device_arrays)
    ]

    import random

    full = host + device
    random.shuffle(full)

    for i in full:
        dhf[i[0]] = i[1]

    random.shuffle(full)

    for i in full:
        assert_array_equal(i[1], dhf[i[0]])
        del dhf[i[0]]

    assert set(dhf.device.keys()) == set()
    assert set(dhf.host.keys()) == set()
    assert set(dhf.disk.keys()) == set()


def test_device_host_file_step_by_step(tmp_path):
    tmpdir = tmp_path / "storage"
    tmpdir.mkdir()
    dhf = DeviceHostFile(
        device_memory_limit=1024 * 16, memory_limit=1024 * 16, local_directory=tmpdir
    )

    a = np.random.random(1000)
    b = cupy.random.random(1000)

    dhf["a1"] = a

    assert set(dhf.device.keys()) == set()
    assert set(dhf.host.keys()) == set(["a1"])
    assert set(dhf.disk.keys()) == set()

    dhf["b1"] = b

    assert set(dhf.device.keys()) == set(["b1"])
    assert set(dhf.host.keys()) == set(["a1"])
    assert set(dhf.disk.keys()) == set()

    dhf["b2"] = b
    assert set(dhf.device.keys()) == set(["b1", "b2"])
    assert set(dhf.host.keys()) == set(["a1"])
    assert set(dhf.disk.keys()) == set()

    dhf["b3"] = b
    assert set(dhf.device.keys()) == set(["b2", "b3"])
    assert set(dhf.host.keys()) == set(["a1", "b1"])
    assert set(dhf.disk.keys()) == set()

    dhf["a2"] = a
    assert set(dhf.device.keys()) == set(["b2", "b3"])
    assert set(dhf.host.keys()) == set(["a2", "b1"])
    assert set(dhf.disk.keys()) == set(["a1"])

    dhf["b4"] = b
    assert set(dhf.device.keys()) == set(["b3", "b4"])
    assert set(dhf.host.keys()) == set(["a2", "b2"])
    assert set(dhf.disk.keys()) == set(["a1", "b1"])

    dhf["b4"] = b
    assert set(dhf.device.keys()) == set(["b3", "b4"])
    assert set(dhf.host.keys()) == set(["a2", "b2"])
    assert set(dhf.disk.keys()) == set(["a1", "b1"])

    assert_array_equal(dhf["a1"], a)
    del dhf["a1"]
    assert_array_equal(dhf["a2"], a)
    del dhf["a2"]
    assert_array_equal(dhf["b1"], b)
    del dhf["b1"]
    assert_array_equal(dhf["b2"], b)
    del dhf["b2"]
    assert_array_equal(dhf["b3"], b)
    del dhf["b3"]
    assert_array_equal(dhf["b4"], b)
    del dhf["b4"]

    assert set(dhf.device.keys()) == set()
    assert set(dhf.host.keys()) == set()
    assert set(dhf.disk.keys()) == set()
