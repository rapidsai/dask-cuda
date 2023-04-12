from random import randint

import numpy as np
import pytest

import dask.array
from distributed.protocol import (
    deserialize,
    deserialize_bytes,
    serialize,
    serialize_bytelist,
)

from dask_cuda.device_host_file import DeviceHostFile, device_to_host, host_to_device

cupy = pytest.importorskip("cupy")


def assert_eq(x, y):
    # Explicitly calling "cupy.asnumpy" to support `ProxyObject` because
    # "cupy" is hardcoded in `dask.array.normalize_to_array()`
    return dask.array.assert_eq(cupy.asnumpy(x), cupy.asnumpy(y))


@pytest.mark.parametrize("num_host_arrays", [1, 10, 100])
@pytest.mark.parametrize("num_device_arrays", [1, 10, 100])
@pytest.mark.parametrize("array_size_range", [(1, 1000), (100, 100), (1000, 1000)])
def test_device_host_file_short(
    tmp_path, num_device_arrays, num_host_arrays, array_size_range
):
    tmpdir = tmp_path / "storage"
    tmpdir.mkdir()
    dhf = DeviceHostFile(
        device_memory_limit=1024 * 16,
        memory_limit=1024 * 16,
        worker_local_directory=tmpdir,
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
        assert_eq(original, acquired)
        del dhf[k]

    assert set(dhf.device.keys()) == set()
    assert set(dhf.host.keys()) == set()
    assert set(dhf.disk.keys()) == set()
    assert set(dhf.others.keys()) == set()


def test_device_host_file_step_by_step(tmp_path):
    tmpdir = tmp_path / "storage"
    tmpdir.mkdir()
    dhf = DeviceHostFile(
        device_memory_limit=1024 * 16,
        memory_limit=1024 * 16,
        worker_local_directory=tmpdir,
    )

    a = np.random.random(1000)
    b = cupy.random.random(1000)

    dhf["a1"] = a
    assert set(dhf.device.keys()) == set()
    assert set(dhf.host.keys()) == set(["a1"])
    assert set(dhf.disk.keys()) == set()
    assert set(dhf.others.keys()) == set()

    dhf["b1"] = b
    assert set(dhf.device.keys()) == set(["b1"])
    assert set(dhf.host.keys()) == set(["a1"])
    assert set(dhf.disk.keys()) == set()
    assert set(dhf.others.keys()) == set()

    dhf["b2"] = b
    assert set(dhf.device.keys()) == set(["b1", "b2"])
    assert set(dhf.host.keys()) == set(["a1"])
    assert set(dhf.disk.keys()) == set()
    assert set(dhf.others.keys()) == set()

    dhf["b3"] = b
    assert set(dhf.device.keys()) == set(["b2", "b3"])
    assert set(dhf.host.keys()) == set(["a1", "b1"])
    assert set(dhf.disk.keys()) == set()
    assert set(dhf.others.keys()) == set()

    dhf["a2"] = a
    assert set(dhf.device.keys()) == set(["b2", "b3"])
    assert set(dhf.host.keys()) == set(["a2", "b1"])
    assert set(dhf.disk.keys()) == set(["a1"])
    assert set(dhf.others.keys()) == set()

    dhf["b4"] = b
    assert set(dhf.device.keys()) == set(["b3", "b4"])
    assert set(dhf.host.keys()) == set(["a2", "b2"])
    assert set(dhf.disk.keys()) == set(["a1", "b1"])
    assert set(dhf.others.keys()) == set()

    dhf["b4"] = b
    assert set(dhf.device.keys()) == set(["b3", "b4"])
    assert set(dhf.host.keys()) == set(["a2", "b2"])
    assert set(dhf.disk.keys()) == set(["a1", "b1"])
    assert set(dhf.others.keys()) == set()

    assert_eq(dhf["a1"], a)
    del dhf["a1"]
    assert_eq(dhf["a2"], a)
    del dhf["a2"]
    assert_eq(dhf["b1"], b)
    del dhf["b1"]
    assert_eq(dhf["b2"], b)
    del dhf["b2"]
    assert_eq(dhf["b3"], b)
    del dhf["b3"]
    assert_eq(dhf["b4"], b)
    del dhf["b4"]

    assert set(dhf.device.keys()) == set()
    assert set(dhf.host.keys()) == set()
    assert set(dhf.disk.keys()) == set()
    assert set(dhf.others.keys()) == set()

    dhf["x"] = b
    dhf["x"] = a
    assert set(dhf.device.keys()) == set()
    assert set(dhf.host.keys()) == set(["x"])
    assert set(dhf.others.keys()) == set()


@pytest.mark.parametrize("collection", [dict, list, tuple])
@pytest.mark.parametrize("length", [0, 1, 3, 6])
@pytest.mark.parametrize("value", [10, {"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]}])
def test_serialize_cupy_collection(collection, length, value):
    # Avoid running test for length 0 (no collection) multiple times
    if length == 0 and collection is not list:
        return

    if isinstance(value, dict):
        cudf = pytest.importorskip("cudf")
        dd = pytest.importorskip("dask.dataframe")
        x = cudf.DataFrame(value)
        assert_func = dd.assert_eq
    else:
        x = cupy.arange(10)
        assert_func = assert_eq

    if length == 0:
        obj = device_to_host(x)
    elif collection is dict:
        obj = device_to_host(dict(zip(range(length), (x,) * length)))
    else:
        obj = device_to_host(collection((x,) * length))

    if length > 0:
        assert all([h["serializer"] == "dask" for h in obj.header["sub-headers"]])
    else:
        assert obj.header["serializer"] == "dask"

    btslst = serialize_bytelist(obj)

    bts = deserialize_bytes(b"".join(btslst))
    res = host_to_device(bts)

    if length == 0:
        assert_func(res, x)
    else:
        assert isinstance(res, collection)
        values = res.values() if collection is dict else res
        [assert_func(v, x) for v in values]

    header, frames = serialize(obj, serializers=["pickle"], on_error="raise")

    assert len(frames) == (1 + len(obj.frames))

    obj2 = deserialize(header, frames)
    res = host_to_device(obj2)

    if length == 0:
        assert_func(res, x)
    else:
        assert isinstance(res, collection)
        values = res.values() if collection is dict else res
        [assert_func(v, x) for v in values]
