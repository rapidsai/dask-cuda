import pytest

from pandas.testing import assert_frame_equal
from dask.dataframe.shuffle import shuffle_group
from distributed import Client
import dask_cuda
import dask_cuda.proxy_object
import dask_cuda.proxify_device_object
from dask_cuda.get_device_memory_objects import get_device_memory_objects
from dask_cuda.object_spilling_host_file import ObjectSpillingHostFile

cupy = pytest.importorskip("cupy")
cupy.cuda.set_allocator(None)
itemsize = cupy.arange(1).nbytes


def test_one_item_limit():
    dhf = ObjectSpillingHostFile(device_memory_limit=itemsize)
    dhf["k1"] = cupy.arange(1) + 1
    dhf["k2"] = cupy.arange(1) + 2

    # Check k1 is spilled because of the newer k2
    k1 = dhf["k1"]
    assert k1._obj_pxy_serialized()
    assert not dhf["k2"]._obj_pxy_serialized()

    # Accessing k1 spills k2 and unspill k1
    k1_val = k1[0]
    assert k1_val == 1
    k2 = dhf["k2"]
    assert k2._obj_pxy_serialized()

    # Duplicate arrays changes nothing
    dhf["k3"] = [k1, k2]
    assert not k1._obj_pxy_serialized()
    assert k2._obj_pxy_serialized()

    # Adding a new array spills k1 and k2
    dhf["k4"] = cupy.arange(1) + 4
    assert k1._obj_pxy_serialized()
    assert k2._obj_pxy_serialized()
    assert not dhf["k4"]._obj_pxy_serialized()

    # Deleting k2 does change anything since k3 still holds a
    # reference to the underlying proxy object
    dhf["k2"][0]
    assert dhf["k1"]._obj_pxy_serialized()
    assert not dhf["k2"]._obj_pxy_serialized()
    assert dhf["k4"]._obj_pxy_serialized()
    del dhf["k2"]
    assert not dhf["k3"][1]._obj_pxy_serialized()


@pytest.mark.parametrize("object_spilling", [True, False])
def test_local_cuda_cluster(object_spilling):
    """Testing spilling of a proxied cudf dataframe in a local cuda cluster"""
    cudf = pytest.importorskip("cudf")
    dask_cudf = pytest.importorskip("dask_cudf")

    def task(x):
        assert isinstance(x, cudf.DataFrame)
        if object_spilling:
            # Check that `x` is a proxy object and the proxied DataFrame is serialized
            assert type(x) is dask_cuda.proxify_device_object.FrameProxyObject
            assert x._obj_pxy["serializers"] == ("dask", "pickle")
        else:
            assert type(x) == cudf.DataFrame
        assert len(x) == 10  # Trigger deserialization
        return x

    # Notice, setting `device_memory_limit=1B` to trigger spilling
    with dask_cuda.LocalCUDACluster(
        n_workers=1, device_memory_limit="1B", object_spilling=object_spilling
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

    dhf = ObjectSpillingHostFile(device_memory_limit=160)
    dhf["v1"] = view1
    dhf["v2"] = view2
    v1 = dhf["v1"]
    v2 = dhf["v2"]
    # The device_memory_limit is not exceeded since both dataframes share device memory
    assert not v1._obj_pxy_serialized()
    assert not v2._obj_pxy_serialized()
    # Now the device_memory_limit is exceeded, which should evict both dataframes
    dhf["k1"] = cupy.arange(1)
    assert v1._obj_pxy_serialized()
    assert v2._obj_pxy_serialized()


def test_cudf_get_device_memory_objects():
    cudf = pytest.importorskip("cudf")
    objects = [
        cudf.DataFrame({"a": range(10), "b": range(10)}),
        cudf.MultiIndex(
            levels=[[1, 2], ["blue", "red"]], codes=[[0, 0, 1, 1], [1, 0, 1, 0]]
        ),
    ]
    res = get_device_memory_objects(objects)
    assert len(res) == 4, "We expect four buffer objects"