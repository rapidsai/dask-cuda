import operator
import os
import pickle
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas
import pytest
from packaging import version
from pandas.testing import assert_frame_equal, assert_series_equal

import dask
import dask.array
from dask.dataframe.core import has_parallel_type
from dask.sizeof import sizeof
from distributed import Client
from distributed.protocol.serialize import deserialize, serialize
from distributed.utils_test import gen_test

import dask_cuda
from dask_cuda import LocalCUDACluster, proxy_object
from dask_cuda.disk_io import SpillToDiskFile
from dask_cuda.proxify_device_objects import proxify_device_objects
from dask_cuda.proxify_host_file import ProxifyHostFile

# Make the "disk" serializer available and use a directory that are
# remove on exit.
if ProxifyHostFile._spill_to_disk is None:
    tmpdir = tempfile.TemporaryDirectory()
    ProxifyHostFile(
        worker_local_directory=tmpdir.name,
        device_memory_limit=1024,
        memory_limit=1024,
    )


@pytest.mark.parametrize("serializers", [None, ("dask", "pickle"), ("disk",)])
def test_proxy_object(serializers):
    """Check "transparency" of the proxy object"""

    org = bytearray(range(10))
    pxy = proxy_object.asproxy(org, serializers=serializers)

    assert len(org) == len(pxy)
    assert org[0] == pxy[0]
    assert 1 in pxy
    assert 10 not in pxy
    assert str(org) == str(pxy)
    assert "dask_cuda.proxy_object.ProxyObject at " in repr(pxy)
    assert "bytearray at " in repr(pxy)

    pxy._pxy_serialize(serializers=("dask", "pickle"))
    assert "dask_cuda.proxy_object.ProxyObject at " in repr(pxy)
    assert "bytearray (serialized='dask')" in repr(pxy)

    assert org == proxy_object.unproxy(pxy)
    assert org == proxy_object.unproxy(org)


class DummyObj:
    """Class that only "pickle" can serialize"""

    def __reduce__(self):
        return (DummyObj, ())


def test_proxy_object_serializer():
    """Check the serializers argument"""
    pxy = proxy_object.asproxy(DummyObj(), serializers=("dask", "pickle"))
    assert pxy._pxy_get().serializer == "pickle"
    assert "DummyObj (serialized='pickle')" in repr(pxy)

    with pytest.raises(ValueError) as excinfo:
        pxy = proxy_object.asproxy([42], serializers=("dask", "pickle"))
        assert "Cannot wrap a collection" in str(excinfo.value)


@pytest.mark.parametrize("serializers_first", [None, ("dask", "pickle"), ("disk",)])
@pytest.mark.parametrize("serializers_second", [None, ("dask", "pickle"), ("disk",)])
def test_double_proxy_object(serializers_first, serializers_second):
    """Check asproxy() when creating a proxy object of a proxy object"""
    serializer1 = serializers_first[0] if serializers_first else None
    serializer2 = serializers_second[0] if serializers_second else None
    org = bytearray(range(10))
    pxy1 = proxy_object.asproxy(org, serializers=serializers_first)
    assert pxy1._pxy_get().serializer == serializer1
    pxy2 = proxy_object.asproxy(pxy1, serializers=serializers_second)
    if serializers_second is None:
        # Check that `serializers=None` doesn't change the initial serializers
        assert pxy2._pxy_get().serializer == serializer1
    else:
        assert pxy2._pxy_get().serializer == serializer2
    assert pxy1 is pxy2


@pytest.mark.parametrize("serializers", [None, ("dask", "pickle"), ("disk",)])
@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_proxy_object_of_array(serializers, backend):
    """Check that a proxied array behaves as a regular (numpy or cupy) array"""

    np = pytest.importorskip(backend)

    # Make sure that equality works, which we use to test the other operators
    org = np.arange(10) + 1
    pxy = proxy_object.asproxy(org.copy(), serializers=serializers)
    assert all(org == pxy)
    assert all(org + 1 != pxy)

    # Check unary scalar operators
    for op in [int, float, complex, operator.index, oct, hex]:
        org = np.int64(42)
        pxy = proxy_object.asproxy(org.copy(), serializers=serializers)
        expect = op(org)
        got = op(pxy)
        assert type(expect) == type(got)
        assert expect == got

    # Check unary operators
    for op_str in ["neg", "pos", "abs", "inv"]:
        op = getattr(operator, op_str)
        org = np.arange(10) + 1
        pxy = proxy_object.asproxy(org.copy(), serializers=serializers)
        expect = op(org)
        got = op(pxy)
        assert type(expect) == type(got)
        assert all(expect == got)

    # Check binary operators that takes a scalar as second argument
    for op_str in ["rshift", "lshift", "pow"]:
        op = getattr(operator, op_str)
        org = np.arange(10) + 1
        pxy = proxy_object.asproxy(org.copy(), serializers=serializers)
        expect = op(org, 2)
        got = op(pxy, 2)
        assert type(expect) == type(got)
        assert all(expect == got)

    # Check binary operators
    for op_str in [
        "add",
        "eq",
        "floordiv",
        "ge",
        "gt",
        "le",
        "lshift",
        "lt",
        "mod",
        "mul",
        "ne",
        "or_",
        "sub",
        "truediv",
        "xor",
        "iadd",
        "ior",
        "iand",
        "ifloordiv",
        "ilshift",
        "irshift",
        "ipow",
        "imod",
        "imul",
        "isub",
        "ixor",
    ]:
        op = getattr(operator, op_str)
        org = np.arange(10) + 1
        pxy = proxy_object.asproxy(org.copy(), serializers=serializers)
        expect = op(org.copy(), org)
        got = op(org.copy(), pxy)
        assert isinstance(got, type(expect))
        assert all(expect == got)

        expect = op(org.copy(), org)
        got = op(pxy, org)
        assert isinstance(got, type(expect))
        assert all(expect == got)

        # Check proxy-proxy operations
        if "i" != op_str[0]:  # Skip in-place operators
            expect = op(org.copy(), org)
            got = op(pxy, proxy_object.asproxy(org.copy()))
            assert all(expect == got)

    # Check unary truth operators
    for op_str in ["not_", "truth"]:
        op = getattr(operator, op_str)
        org = np.arange(1) + 1
        pxy = proxy_object.asproxy(org.copy(), serializers=serializers)
        expect = op(org)
        got = op(pxy)
        assert type(expect) == type(got)
        assert expect == got

    # Check reflected methods
    for op_str in [
        "__radd__",
        "__rsub__",
        "__rmul__",
        "__rtruediv__",
        "__rfloordiv__",
        "__rmod__",
        "__rpow__",
        "__rlshift__",
        "__rrshift__",
        "__rxor__",
        "__ror__",
    ]:
        org = np.arange(10) + 1
        pxy = proxy_object.asproxy(org.copy(), serializers=serializers)
        expect = getattr(org, op_str)(org)
        got = getattr(org, op_str)(pxy)
        assert isinstance(got, type(expect))
        assert all(expect == got)


@pytest.mark.parametrize("serializers", [None, ["dask"], ["disk"]])
def test_proxy_object_of_cudf(serializers):
    """Check that a proxied cudf dataframe behaves as a regular dataframe"""
    cudf = pytest.importorskip("cudf")
    df = cudf.DataFrame({"a": range(10)})
    pxy = proxy_object.asproxy(df, serializers=serializers)
    assert_frame_equal(df.to_pandas(), pxy.to_pandas())


@pytest.mark.parametrize("proxy_serializers", [None, ["dask"], ["cuda"], ["disk"]])
@pytest.mark.parametrize("dask_serializers", [["dask"], ["cuda"]])
def test_serialize_of_proxied_cudf(proxy_serializers, dask_serializers):
    """Check that we can serialize a proxied cudf dataframe, which might
    be serialized already.
    """
    cudf = pytest.importorskip("cudf")
    df = cudf.DataFrame({"a": range(10)})
    pxy = proxy_object.asproxy(df, serializers=proxy_serializers)
    header, frames = serialize(pxy, serializers=dask_serializers, on_error="raise")
    pxy = deserialize(header, frames)
    assert_frame_equal(df.to_pandas(), pxy.to_pandas())


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_fixed_attribute_length(backend):
    """Test fixed attribute `x.__len__` access

    Notice, accessing fixed attributes shouldn't de-serialize the proxied object
    """
    np = pytest.importorskip(backend)

    # Access `len()`` of an array
    pxy = proxy_object.asproxy(np.arange(10), serializers=("dask",))
    assert len(pxy) == 10
    # Accessing the length shouldn't de-serialize the proxied object
    assert pxy._pxy_get().is_serialized()

    # Access `len()` of a scalar
    pxy = proxy_object.asproxy(np.array(10), serializers=("dask",))
    with pytest.raises(TypeError) as excinfo:
        len(pxy)
        assert "len() of unsized object" in str(excinfo.value)
        assert pxy._pxy_get().is_serialized()


def test_fixed_attribute_name():
    """Test fixed attribute `x.name` access

    Notice, accessing fixed attributes shouldn't de-serialize the proxied object
    """
    obj_without_name = SimpleNamespace()
    obj_with_name = SimpleNamespace(name="I have a name")

    # Access `name` of an array
    pxy = proxy_object.asproxy(obj_without_name, serializers=("pickle",))
    with pytest.raises(AttributeError) as excinfo:
        pxy.name
        assert "has no attribute 'name'" in str(excinfo.value)
        assert pxy._pxy_get().is_serialized()

    # Access `name` of a datatype
    pxy = proxy_object.asproxy(obj_with_name, serializers=("pickle",))
    assert pxy.name == "I have a name"
    assert pxy._pxy_get().is_serialized()


@pytest.mark.parametrize("jit_unspill", [True, False])
@gen_test(timeout=20)
async def test_spilling_local_cuda_cluster(jit_unspill):
    """Testing spilling of a proxied cudf dataframe in a local cuda cluster"""
    cudf = pytest.importorskip("cudf")
    dask_cudf = pytest.importorskip("dask_cudf")

    def task(x):
        assert isinstance(x, cudf.DataFrame)
        if jit_unspill:
            # Check that `x` is a proxy object and the proxied DataFrame is serialized
            assert "ProxyObject" in str(type(x))
            assert x._pxy_get().serializer == "dask"
        else:
            assert type(x) == cudf.DataFrame
        assert len(x) == 10  # Trigger deserialization
        return x

    # Notice, setting `device_memory_limit=1B` to trigger spilling
    async with LocalCUDACluster(
        n_workers=1,
        device_memory_limit="1B",
        jit_unspill=jit_unspill,
        asynchronous=True,
    ) as cluster:
        async with Client(cluster, asynchronous=True) as client:
            df = cudf.DataFrame({"a": range(10)})
            ddf = dask_cudf.from_cudf(df, npartitions=1)
            ddf = ddf.map_partitions(task, meta=df.head())
            got = await client.compute(ddf)
            if isinstance(got, pandas.Series):
                pytest.xfail(
                    "BUG fixed by <https://github.com/rapidsai/dask-cuda/pull/451>"
                )
            assert_frame_equal(got.to_pandas(), df.to_pandas())


@pytest.mark.parametrize("obj", [bytearray(10), bytearray(10**6)])
def test_serializing_to_disk(obj):
    """Check serializing to disk"""

    # Serialize from device to disk
    pxy = proxy_object.asproxy(obj)
    ProxifyHostFile.serialize_proxy_to_disk_inplace(pxy)
    assert pxy._pxy_get().serializer == "disk"
    assert obj == proxy_object.unproxy(pxy)

    # Serialize from host to disk
    pxy = proxy_object.asproxy(obj, serializers=("pickle",))
    ProxifyHostFile.serialize_proxy_to_disk_inplace(pxy)
    assert pxy._pxy_get().serializer == "disk"
    assert obj == proxy_object.unproxy(pxy)


@pytest.mark.parametrize("serializer", ["dask", "pickle", "disk"])
def test_multiple_deserializations(serializer):
    """Check for race conditions when accessing the ProxyDetail"""
    data1 = bytearray(10)
    proxy = proxy_object.asproxy(data1, serializers=(serializer,))
    pxy = proxy._pxy_get()
    data2 = proxy._pxy_deserialize()
    assert data1 == data2

    # Check that the spilled file still exist.
    if serializer == "disk":
        file_path = pxy.obj[0]["disk-io-header"]["path"]
        assert isinstance(file_path, SpillToDiskFile)
        assert file_path.exists()
        file_path = str(file_path)

    # Check that the spilled data within `pxy` is still available even
    # though `proxy` has been deserialized.
    data3 = pxy.deserialize()
    assert data1 == data3

    # Check that the spilled file has been removed now that all reference
    # to is has been deleted.
    if serializer == "disk":
        assert not os.path.exists(file_path)


@pytest.mark.parametrize("size", [10, 10**4])
@pytest.mark.parametrize(
    "serializers", [None, ["dask"], ["cuda", "dask"], ["pickle"], ["disk"]]
)
@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_serializing_array_to_disk(backend, serializers, size):
    """Check serializing arrays to disk"""

    np = pytest.importorskip(backend)
    obj = np.arange(size)

    # Serialize from host to disk
    pxy = proxy_object.asproxy(obj, serializers=serializers)
    ProxifyHostFile.serialize_proxy_to_disk_inplace(pxy)
    assert pxy._pxy_get().serializer == "disk"
    assert list(obj) == list(proxy_object.unproxy(pxy))


class _PxyObjTest(proxy_object.ProxyObject):
    """
    A class that:
        - defines `__dask_tokenize__` in order to avoid deserialization when
          calling `client.scatter()`
        - Asserts that no deserialization is performaned when communicating.
    """

    def __dask_tokenize__(self):
        return 42

    def _pxy_deserialize(self):
        if self._pxy_get().assert_on_deserializing:
            assert self._pxy_get().serializer is None
        return super()._pxy_deserialize()


@pytest.mark.parametrize("send_serializers", [None, ("dask", "pickle"), ("cuda",)])
@pytest.mark.parametrize("protocol", ["tcp", "ucx"])
@gen_test(timeout=120)
async def test_communicating_proxy_objects(protocol, send_serializers):
    """Testing serialization of cuDF dataframe when communicating"""
    cudf = pytest.importorskip("cudf")

    def task(x):
        # Check that the subclass survives the trip from client to worker
        assert isinstance(x, _PxyObjTest)
        serializers_used = x._pxy_get().serializer

        # Check that `x` is serialized with the expected serializers
        if protocol == "ucx":
            if send_serializers is None:
                assert serializers_used == "cuda"
            else:
                assert serializers_used == send_serializers[0]
        else:
            assert serializers_used == "dask"

    async with dask_cuda.LocalCUDACluster(
        n_workers=1,
        protocol=protocol,
        asynchronous=True,
    ) as cluster:
        async with Client(cluster, asynchronous=True) as client:
            df = cudf.DataFrame({"a": range(10)})
            df = proxy_object.asproxy(
                df, serializers=send_serializers, subclass=_PxyObjTest
            )

            # Notice, in one case we expect deserialization when communicating.
            # Since "tcp" cannot send device memory directly, it will be re-serialized
            # using the default dask serializers that spill the data to main memory.
            if protocol == "tcp" and send_serializers == ("cuda",):
                df._pxy_get().assert_on_deserializing = False
            else:
                df._pxy_get().assert_on_deserializing = True
            df = await client.scatter(df)
            await client.submit(task, df)


@pytest.mark.parametrize("protocol", ["tcp", "ucx"])
@pytest.mark.parametrize("shared_fs", [True, False])
@gen_test(timeout=20)
async def test_communicating_disk_objects(protocol, shared_fs):
    """Testing disk serialization of cuDF dataframe when communicating"""
    cudf = pytest.importorskip("cudf")
    ProxifyHostFile._spill_to_disk.shared_filesystem = shared_fs

    def task(x):
        # Check that the subclass survives the trip from client to worker
        assert isinstance(x, _PxyObjTest)
        serializer_used = x._pxy_get().serializer
        if shared_fs:
            assert serializer_used == "disk"
        else:
            assert serializer_used == "dask"

    async with dask_cuda.LocalCUDACluster(
        n_workers=1,
        protocol=protocol,
        asynchronous=True,
    ) as cluster:
        async with Client(cluster, asynchronous=True) as client:
            df = cudf.DataFrame({"a": range(10)})
            df = proxy_object.asproxy(df, serializers=("disk",), subclass=_PxyObjTest)
            df._pxy_get().assert_on_deserializing = False
            df = await client.scatter(df)
            await client.submit(task, df)


@pytest.mark.parametrize("array_module", ["numpy", "cupy"])
@pytest.mark.parametrize(
    "serializers", [None, ("dask", "pickle"), ("cuda", "dask", "pickle"), ("disk",)]
)
def test_pickle_proxy_object(array_module, serializers):
    """Check pickle of the proxy object"""
    array_module = pytest.importorskip(array_module)
    org = array_module.arange(10)
    pxy = proxy_object.asproxy(org, serializers=serializers)
    data = pickle.dumps(pxy)
    restored = pickle.loads(data)
    repr(restored)
    assert all(org == restored)


def test_pandas():
    """Check pandas operations on proxy objects"""
    pandas = pytest.importorskip("pandas")

    df1 = pandas.DataFrame({"a": range(10)})
    df2 = pandas.DataFrame({"a": range(10)})

    res = dask.dataframe.methods.concat([df1, df2])
    got = dask.dataframe.methods.concat([df1, df2])
    assert_frame_equal(res, got)

    got = dask.dataframe.methods.concat([proxy_object.asproxy(df1), df2])
    assert_frame_equal(res, got)

    got = dask.dataframe.methods.concat([df1, proxy_object.asproxy(df2)])
    assert_frame_equal(res, got)

    df1 = pandas.Series(range(10))
    df2 = pandas.Series(range(10))

    res = dask.dataframe.methods.concat([df1, df2])
    got = dask.dataframe.methods.concat([df1, df2])
    assert all(res == got)

    got = dask.dataframe.methods.concat([proxy_object.asproxy(df1), df2])
    assert all(res == got)

    got = dask.dataframe.methods.concat([df1, proxy_object.asproxy(df2)])
    assert all(res == got)


def test_from_cudf_of_proxy_object():
    """Check from_cudf() of a proxy object"""
    cudf = pytest.importorskip("cudf")
    dask_cudf = pytest.importorskip("dask_cudf")

    df = proxy_object.asproxy(cudf.DataFrame({"a": range(10)}))
    assert has_parallel_type(df)

    ddf = dask_cudf.from_cudf(df, npartitions=1)
    assert has_parallel_type(ddf)

    # Notice, the output is a dask-cudf dataframe and not a proxy object
    assert type(ddf) is dask_cudf.core.DataFrame


def test_proxy_object_parquet(tmp_path):
    """Check parquet read/write of a proxy object"""
    cudf = pytest.importorskip("cudf")
    tmp_path = tmp_path / "proxy_test.parquet"

    df = cudf.DataFrame({"a": range(10)})
    pxy = proxy_object.asproxy(df)
    pxy.to_parquet(str(tmp_path), engine="pyarrow")
    df2 = dask.dataframe.read_parquet(tmp_path)
    assert_frame_equal(df.to_pandas(), df2.compute())


def test_assignments():
    """Check assignment to a proxied dataframe"""
    cudf = pytest.importorskip("cudf")

    df = proxy_object.asproxy(cudf.DataFrame({"a": range(10)}))
    df.index = df["a"].copy(deep=False)


def test_concatenate3_of_proxied_cupy_arrays():
    """Check concatenate of cupy arrays"""
    from dask.array.core import concatenate3

    cupy = pytest.importorskip("cupy")
    org = cupy.arange(10)
    a = proxy_object.asproxy(org.copy())
    b = proxy_object.asproxy(org.copy())
    assert all(concatenate3([a, b]) == concatenate3([org.copy(), org.copy()]))


def test_tensordot_of_proxied_cupy_arrays():
    """Check tensordot of cupy arrays"""
    cupy = pytest.importorskip("cupy")

    org = cupy.arange(9).reshape((3, 3))
    a = proxy_object.asproxy(org.copy())
    b = proxy_object.asproxy(org.copy())
    res1 = dask.array.tensordot(a, b).flatten()
    res2 = dask.array.tensordot(org.copy(), org.copy()).flatten()
    assert all(res1 == res2)


def test_einsum_of_proxied_cupy_arrays():
    """Check tensordot of cupy arrays"""
    cupy = pytest.importorskip("cupy")

    org = cupy.arange(25).reshape(5, 5)
    res1 = dask.array.einsum("ii", org)
    a = proxy_object.asproxy(org.copy())
    res2 = dask.array.einsum("ii", a)
    assert all(res1.flatten() == res2.flatten())


@pytest.mark.parametrize(
    "np_func", [np.less, np.less_equal, np.greater, np.greater_equal, np.equal]
)
def test_array_ufucn_proxified_object(np_func):
    cudf = pytest.importorskip("cudf")

    np_array = np.array(100)
    ser = cudf.Series([1, 2, 3])
    proxy_obj = proxify_device_objects(ser)
    expected = np_func(ser, np_array)
    actual = np_func(proxy_obj, np_array)

    assert_series_equal(expected.to_pandas(), actual.to_pandas())


def test_cudf_copy():
    cudf = pytest.importorskip("cudf")
    df = cudf.DataFrame({"A": range(10)})
    df = proxify_device_objects(df)
    cpy = df.copy()
    assert_frame_equal(cpy.to_pandas(), df.to_pandas())


def test_cudf_fillna():
    cudf = pytest.importorskip("cudf")
    df = cudf.DataFrame({"A": range(10)})
    df = proxify_device_objects(df)
    df = df.fillna(0)


def test_sizeof_cupy():
    cupy = pytest.importorskip("cupy")
    cupy.cuda.set_allocator(None)
    a = cupy.arange(1e7)
    a_size = sizeof(a)
    pxy = proxy_object.asproxy(a)
    assert a_size == pytest.approx(sizeof(pxy))
    pxy._pxy_serialize(serializers=("dask",))
    assert a_size == pytest.approx(sizeof(pxy))
    assert pxy._pxy_get().is_serialized()
    pxy._pxy_cache = {}
    assert a_size == pytest.approx(sizeof(pxy))
    assert pxy._pxy_get().is_serialized()


def test_sizeof_cudf():
    cudf = pytest.importorskip("cudf")
    a = cudf.datasets.timeseries().reset_index()
    a_size = sizeof(a)
    pxy = proxy_object.asproxy(a)
    assert a_size == pytest.approx(sizeof(pxy))
    pxy._pxy_serialize(serializers=("dask",))
    assert a_size == pytest.approx(sizeof(pxy))
    assert pxy._pxy_get().is_serialized()
    # By clearing the cache, `sizeof(pxy)` now measure the serialized data
    # thus we have to increase the tolerance.
    pxy._pxy_cache = {}
    assert a_size == pytest.approx(sizeof(pxy), rel=1e-2)
    assert pxy._pxy_get().is_serialized()


def test_cupy_broadcast_to():
    cupy = pytest.importorskip("cupy")
    a = cupy.arange(10)
    a_b = np.broadcast_to(a, (10, 10))
    p_b = np.broadcast_to(proxy_object.asproxy(a), (10, 10))

    assert a_b.shape == p_b.shape
    assert (a_b == p_b).all()


def test_cupy_matmul():
    cupy = pytest.importorskip("cupy")
    if version.parse(cupy.__version__) >= version.parse("11.0"):
        pytest.xfail("See: https://github.com/rapidsai/dask-cuda/issues/995")
    a, b = cupy.arange(10), cupy.arange(10)
    c = a @ b
    assert c == proxy_object.asproxy(a) @ b
    assert c == a @ proxy_object.asproxy(b)
    assert c == proxy_object.asproxy(a) @ proxy_object.asproxy(b)


def test_cupy_imatmul():
    cupy = pytest.importorskip("cupy")
    if version.parse(cupy.__version__) >= version.parse("11.0"):
        pytest.xfail("See: https://github.com/rapidsai/dask-cuda/issues/995")
    a = cupy.arange(9).reshape(3, 3)
    c = a.copy()
    c @= a

    a1 = a.copy()
    a1 @= proxy_object.asproxy(a)
    assert (a1 == c).all()

    a2 = proxy_object.asproxy(a.copy())
    a2 @= a
    assert (a2 == c).all()
