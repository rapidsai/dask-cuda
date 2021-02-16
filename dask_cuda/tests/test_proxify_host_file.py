import pytest
from pandas.testing import assert_frame_equal

from dask.dataframe.shuffle import shuffle_group
from distributed import Client

import dask_cuda
import dask_cuda.proxify_device_objects
import dask_cuda.proxy_object
from dask_cuda.get_device_memory_objects import get_device_memory_objects
from dask_cuda.proxify_host_file import ProxifyHostFile

cupy = pytest.importorskip("cupy")
cupy.cuda.set_allocator(None)
itemsize = cupy.arange(1).nbytes


def test_one_item_limit():
    dhf = ProxifyHostFile(device_memory_limit=itemsize)
    dhf["k1"] = cupy.arange(1) + 1
    dhf["k2"] = cupy.arange(1) + 2

    # Check k1 is spilled because of the newer k2
    k1 = dhf["k1"]
    k2 = dhf["k2"]
    assert k1._obj_pxy_is_serialized()
    assert not k2._obj_pxy_is_serialized()

    # Accessing k1 spills k2 and unspill k1
    k1_val = k1[0]
    assert k1_val == 1
    assert k2._obj_pxy_is_serialized()

    # Duplicate arrays changes nothing
    dhf["k3"] = [k1, k2]
    assert not k1._obj_pxy_is_serialized()
    assert k2._obj_pxy_is_serialized()

    # Adding a new array spills k1 and k2
    dhf["k4"] = cupy.arange(1) + 4
    assert k1._obj_pxy_is_serialized()
    assert k2._obj_pxy_is_serialized()
    assert not dhf["k4"]._obj_pxy_is_serialized()

    # Accessing k2 spills k1 and k4
    k2[0]
    assert k1._obj_pxy_is_serialized()
    assert dhf["k4"]._obj_pxy_is_serialized()
    assert not k2._obj_pxy_is_serialized()

    # Deleting k2 does not change anything since k3 still holds a
    # reference to the underlying proxy object
    assert dhf.proxies_tally.get_dev_mem_usage() == itemsize
    p1 = list(dhf.proxies_tally.get_unspilled_proxies())
    assert len(p1) == 1
    del dhf["k2"]
    assert dhf.proxies_tally.get_dev_mem_usage() == itemsize
    p2 = list(dhf.proxies_tally.get_unspilled_proxies())
    assert len(p2) == 1
    assert p1[0] is p2[0]

    # Overwriting "k3" with a non-cuda object, should be noticed
    dhf["k3"] = "non-cuda-object"
    assert dhf.proxies_tally.get_dev_mem_usage() == 0


@pytest.mark.parametrize("jit_unspill", [True, False])
def test_local_cuda_cluster(jit_unspill):
    """Testing spilling of a proxied cudf dataframe in a local cuda cluster"""
    cudf = pytest.importorskip("cudf")
    dask_cudf = pytest.importorskip("dask_cudf")

    def task(x):
        assert isinstance(x, cudf.DataFrame)
        if jit_unspill:
            # Check that `x` is a proxy object and the proxied DataFrame is serialized
            assert "FrameProxyObject" in str(type(x))
            assert x._obj_pxy["serializers"] == ("dask", "pickle")
        else:
            assert type(x) == cudf.DataFrame
        assert len(x) == 10  # Trigger deserialization
        return x

    # Notice, setting `device_memory_limit=1B` to trigger spilling
    with dask_cuda.LocalCUDACluster(
        n_workers=1, device_memory_limit="1B", jit_unspill=jit_unspill
    ) as cluster:
        with Client(cluster):
            df = cudf.DataFrame({"a": range(10)})
            ddf = dask_cudf.from_cudf(df, npartitions=1)
            ddf = ddf.map_partitions(task, meta=df.head())
            got = ddf.compute()
            assert_frame_equal(got.to_pandas(), df.to_pandas())


def test_dataframes_share_dev_mem():
    cudf = pytest.importorskip("cudf")

    df = cudf.DataFrame({"a": range(10)})
    grouped = shuffle_group(df, "a", 0, 2, 2, False, 2)
    view1 = grouped[0]
    view2 = grouped[1]
    # Even though the two dataframe doesn't point to the same cudf.Buffer object
    assert view1["a"].data is not view2["a"].data
    # They still share the same underlying device memory
    assert view1["a"].data._owner._owner is view2["a"].data._owner._owner

    dhf = ProxifyHostFile(device_memory_limit=160)
    dhf["v1"] = view1
    dhf["v2"] = view2
    v1 = dhf["v1"]
    v2 = dhf["v2"]
    # The device_memory_limit is not exceeded since both dataframes share device memory
    assert not v1._obj_pxy_is_serialized()
    assert not v2._obj_pxy_is_serialized()
    # Now the device_memory_limit is exceeded, which should evict both dataframes
    dhf["k1"] = cupy.arange(1)
    assert v1._obj_pxy_is_serialized()
    assert v2._obj_pxy_is_serialized()


def test_cudf_get_device_memory_objects():
    cudf = pytest.importorskip("cudf")
    objects = [
        cudf.DataFrame({"a": range(10), "b": range(10)}, index=reversed(range(10))),
        cudf.MultiIndex(
            levels=[[1, 2], ["blue", "red"]], codes=[[0, 0, 1, 1], [1, 0, 1, 0]]
        ),
    ]
    res = get_device_memory_objects(objects)
    assert len(res) == 4, "We expect four buffer objects"
