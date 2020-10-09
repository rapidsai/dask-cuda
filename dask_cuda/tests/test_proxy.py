import operator

import pytest
from pandas.testing import assert_frame_equal

from distributed import Client
from distributed.protocol.serialize import deserialize, serialize

import dask_cudf

import dask_cuda
from dask_cuda import proxy_object


@pytest.mark.parametrize("serializers", [None, ["dask", "pickle"]])
def test_proxy_object(serializers):
    """Check "transparency" of the proxy object"""

    org = list(range(10))
    pxy = proxy_object.asproxy(org, serializers=serializers)

    assert len(org) == len(pxy)
    assert org[0] == pxy[0]
    assert 1 in pxy
    assert -1 not in pxy


@pytest.mark.parametrize("serializers_first", [None, ["dask", "pickle"]])
@pytest.mark.parametrize("serializers_second", [None, ["dask", "pickle"]])
def test_double_proxy_object(serializers_first, serializers_second):
    """Check asproxy() when creating a proxy object of a proxy object"""
    org = list(range(10))
    pxy1 = proxy_object.asproxy(org, serializers=serializers_first)
    assert pxy1._obj_pxy["serializers"] == serializers_first
    pxy2 = proxy_object.asproxy(pxy1, serializers=serializers_second)
    if serializers_second is None:
        # Check that `serializers=None` doesn't change the initial serializers
        assert pxy2._obj_pxy["serializers"] == serializers_first
    else:
        assert pxy2._obj_pxy["serializers"] == serializers_second
    assert pxy1 is pxy2


@pytest.mark.parametrize("serializers", [None, ["dask", "pickle"]])
def test_proxy_object_of_numpy(serializers):
    """Check that a proxied numpy array behaves as a regular dataframe"""

    np = pytest.importorskip("numpy")

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
        "ifloordiv",
        "ilshift",
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

    # Check unary truth operators
    for op_str in ["not_", "truth"]:
        op = getattr(operator, op_str)
        org = np.arange(1) + 1
        pxy = proxy_object.asproxy(org.copy(), serializers=serializers)
        expect = op(org)
        got = op(pxy)
        assert type(expect) == type(got)
        assert expect == got


@pytest.mark.parametrize("serializers", [None, ["dask"]])
def test_proxy_object_of_cudf(serializers):
    """Check that a proxied cudf dataframe behaves as a regular dataframe"""
    cudf = pytest.importorskip("cudf")
    df = cudf.DataFrame({"a": range(10)})
    pxy = proxy_object.asproxy(df, serializers=serializers)
    assert_frame_equal(df.to_pandas(), pxy.to_pandas())


@pytest.mark.parametrize("proxy_serializers", [None, ["dask"], ["cuda"]])
@pytest.mark.parametrize("dask_serializers", [["dask"], ["cuda"]])
def test_serialize_of_proxied_cudf(proxy_serializers, dask_serializers):
    """Check that we can serialize a proxied cudf dataframe, which might
    be serialized already.
    """
    cudf = pytest.importorskip("cudf")

    df = cudf.DataFrame({"a": range(10)})
    pxy = proxy_object.asproxy(df, serializers=proxy_serializers)
    header, frames = serialize(pxy, serializers=dask_serializers)
    pxy = deserialize(header, frames)
    assert_frame_equal(df.to_pandas(), pxy.to_pandas())


@pytest.mark.parametrize("jit_unspill", [True, False])
def test_spilling_local_cuda_cluster(jit_unspill):
    """Testing spelling of a proxied cudf dataframe in a local cuda cluster"""
    cudf = pytest.importorskip("cudf")

    def task(x):
        assert isinstance(x, cudf.DataFrame)
        if jit_unspill:
            # Check that `x` is a proxy object and the proxied DataFrame is serialized
            assert type(x) == proxy_object.ObjectProxy
            assert x._obj_pxy_get_meta()["serializers"] == ["dask", "pickle"]
        else:
            assert type(x) == cudf.DataFrame
        assert len(x) == 10  # Trigger deserialization
        return x

    # Notice, setting `device_memory_limit=1` to trigger spilling
    with dask_cuda.LocalCUDACluster(
        n_workers=1, device_memory_limit=1, jit_unspill=jit_unspill
    ) as cluster:
        with Client(cluster):
            df = cudf.DataFrame({"a": range(10)})
            ddf = dask_cudf.from_cudf(df, npartitions=1)
            ddf = ddf.map_partitions(task, meta=df.head())
            got = ddf.compute()
            assert_frame_equal(got.to_pandas(), df.to_pandas())


class _PxyObjTest(proxy_object.ObjectProxy):
    """
    A class that:
        - defines `__dask_tokenize__` in order to avoid deserialization when
          calling `client.scatter()`
        - Asserts that no deserialization is performaned when communicating.
    """

    def __dask_tokenize__(self):
        return 42

    def _obj_pxy_deserialize(self):
        if self.assert_on_deserializing:
            assert self._obj_pxy["serializers"] is None
        return super()._obj_pxy_deserialize()


@pytest.mark.parametrize("send_serializers", [None, ["dask", "pickle"], ["cuda"]])
@pytest.mark.parametrize("protocol", ["tcp", "ucx"])
def test_communicating_proxy_objects(protocol, send_serializers):
    """Testing serialization of cuDF dataframe when communicating"""
    cudf = pytest.importorskip("cudf")

    def task(x):
        # Check that the subclass survives the trip from client to worker
        assert isinstance(x, _PxyObjTest)
        serializers_used = list(x._obj_pxy_get_meta()["serializers"])

        # Check that `x` is serialized with the expected serializers
        if protocol == "ucx":
            if send_serializers is None:
                assert serializers_used == ["cuda", "dask", "pickle"]
            else:
                assert serializers_used == send_serializers
        else:
            assert serializers_used == ["dask", "pickle"]

    with dask_cuda.LocalCUDACluster(
        n_workers=1, protocol=protocol, enable_tcp_over_ucx=protocol == "ucx"
    ) as cluster:
        with Client(cluster) as client:
            df = cudf.DataFrame({"a": range(10)})
            df = proxy_object.asproxy(
                df, serializers=send_serializers, subclass=_PxyObjTest
            )

            # Notice, in one case we expect deserialization when communicating.
            # Since "tcp" cannot send device memory directly, it will be re-serialized
            # using the default dask serializers that spill the data to main memory.
            if protocol == "tcp" and send_serializers == ["cuda"]:
                df.assert_on_deserializing = False
            else:
                df.assert_on_deserializing = True
            df = client.scatter(df)
            client.submit(task, df).result()
