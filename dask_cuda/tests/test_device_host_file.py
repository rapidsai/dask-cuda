import numpy as np
import cupy
from dask_cuda.device_host_file import DeviceHostFile, host_to_device, device_to_host
from distributed.protocol import deserialize_bytes, serialize_bytelist
from random import randint
import dask.array as da

import pytest


@pytest.mark.parametrize("num_host_arrays", [1, 10, 100])
@pytest.mark.parametrize("num_device_arrays", [1, 10, 100])
@pytest.mark.parametrize("array_size_range", [(1, 1000), (100, 100), (1000, 1000)])
def test_device_host_file_short(
    tmp_path, num_device_arrays, num_host_arrays, array_size_range
):
    tmpdir = tmp_path / "storage"
    tmpdir.mkdir()
    dhf = DeviceHostFile(
        device_memory_limit=1024 * 16, memory_limit=1024 * 16, local_dir=tmpdir
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

    for k, v in full:
        dhf[k] = v

    random.shuffle(full)

    for k, original in full:
        acquired = dhf[k]
        da.assert_eq(original, acquired)
        del dhf[k]

    assert set(dhf.device.keys()) == set()
    assert set(dhf.host.keys()) == set()
    assert set(dhf.disk.keys()) == set()


def test_device_host_file_step_by_step(tmp_path):
    tmpdir = tmp_path / "storage"
    tmpdir.mkdir()
    dhf = DeviceHostFile(
        device_memory_limit=1024 * 16, memory_limit=1024 * 16, local_dir=tmpdir
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

    da.assert_eq(dhf["a1"], a)
    del dhf["a1"]
    da.assert_eq(dhf["a2"], a)
    del dhf["a2"]
    da.assert_eq(dhf["b1"], b)
    del dhf["b1"]
    da.assert_eq(dhf["b2"], b)
    del dhf["b2"]
    da.assert_eq(dhf["b3"], b)
    del dhf["b3"]
    da.assert_eq(dhf["b4"], b)
    del dhf["b4"]

    assert set(dhf.device.keys()) == set()
    assert set(dhf.host.keys()) == set()
    assert set(dhf.disk.keys()) == set()


def test_serialize_cupy():
    x = cupy.arange(10)

    obj = device_to_host(x)
    assert all(isinstance(part, (bytes, memoryview, np.ndarray)) for part in obj.parts)
    btslst = serialize_bytelist(obj)

    bts = deserialize_bytes(b"".join(btslst))
    y = host_to_device(bts)

    da.assert_eq(x, y)


def test_serialize_cudf():
    cudf = pytest.importorskip("cudf")
    dd = pytest.importorskip("dask.dataframe")

    df = cudf.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]})

    obj = device_to_host(df)
    assert all(isinstance(part, (bytes, memoryview, np.ndarray)) for part in obj.parts)
    btslst = serialize_bytelist(obj)

    bts = deserialize_bytes(b"".join(btslst))
    df2 = host_to_device(bts)

    dd.assert_eq(df, df2)
