import pytest
from pandas.testing import assert_frame_equal

from distributed import Client
from distributed.protocol.serialize import deserialize, serialize

import dask_cudf

import dask_cuda
from dask_cuda import proxy_object


@pytest.mark.parametrize("serialize_obj", [True, False])
def test_proxy_object(serialize_obj):
    """Check "transparency" of the proxy object"""

    org = list(range(10))
    pxy = proxy_object.asproxy(org, serialize_obj=serialize_obj)

    assert len(org) == len(pxy)
    assert org[0] == pxy[0]
    assert 1 in pxy
    assert -1 not in pxy

    # TODO: check operators (when implemented)


@pytest.mark.parametrize("serialize_obj", [True, False])
def test_proxy_object_of_cudf(serialize_obj):
    """Check that a proxied cudf dataframe is behaviors as a regular dataframe"""
    cudf = pytest.importorskip("cudf")
    df = cudf.DataFrame({"a": range(10)})
    pxy = proxy_object.asproxy(df, serialize_obj=serialize_obj)
    assert_frame_equal(df.to_pandas(), pxy.to_pandas())


@pytest.mark.parametrize("serialize_obj", [True, False])
@pytest.mark.parametrize(
    "serializers", [["dask", "pickle"], ["cuda", "dask", "pickle"]]
)
def test_serialize_of_proxied_cudf(serialize_obj, serializers):
    """Check that we can serialize a proxied cudf dataframe, which might
    be serialized already.
    """
    cudf = pytest.importorskip("cudf")

    if "cuda" in serializers:
        pytest.skip("cuda serializer support not implemented")

    df = cudf.DataFrame({"a": range(10)})
    pxy = proxy_object.asproxy(df, serialize_obj=serialize_obj)
    header, frames = serialize(pxy, serializers=serializers)
    pxy = deserialize(header, frames)
    assert_frame_equal(df.to_pandas(), pxy.to_pandas())


@pytest.mark.parametrize("spill_proxy", [True, False])
def test_spilling_local_cuda_cluster(spill_proxy):
    """Testing spelling of a proxied cudf dataframe in a local cuda cluster"""
    cudf = pytest.importorskip("cudf")

    def task(x):
        assert isinstance(x, cudf.DataFrame)
        assert len(x) == 10  # Trigger deserialization
        return x

    # Notice, setting `device_memory_limit=1` to trigger spilling
    with dask_cuda.LocalCUDACluster(
        n_workers=1, device_memory_limit=1, spill_proxy=spill_proxy
    ) as cluster:
        with Client(cluster):
            df = cudf.DataFrame({"a": range(10)})
            ddf = dask_cudf.from_cudf(df, npartitions=1)
            ddf = ddf.map_partitions(task, meta=df.head())
            got = ddf.compute()
            assert_frame_equal(got.to_pandas(), df.to_pandas())
