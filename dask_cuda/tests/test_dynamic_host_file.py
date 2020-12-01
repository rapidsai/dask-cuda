import pytest

from pandas.testing import assert_frame_equal
from distributed import Client
import dask_cuda
from dask_cuda.dynamic_host_file import DynamicHostFile

cupy = pytest.importorskip("cupy")
itemsize = cupy.arange(1).nbytes


def test_one_item_limit():
    dhf = DynamicHostFile(device_memory_limit=itemsize)
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


@pytest.mark.parametrize("dynamic_spill", [True, False])
def test_local_cuda_cluster(dynamic_spill):
    """Testing spilling of a proxied cudf dataframe in a local cuda cluster"""
    cudf = pytest.importorskip("cudf")
    dask_cudf = pytest.importorskip("dask_cudf")

    def task(x):
        assert isinstance(x, cudf.DataFrame)
        if dynamic_spill:
            # Check that `x` is a proxy object and the proxied DataFrame is serialized
            assert type(x) is dask_cuda.proxy_object.ProxyObject
            assert x._obj_pxy_get_meta()["serializers"] == ["dask", "pickle"]
        else:
            assert type(x) == cudf.DataFrame
        assert len(x) == 10  # Trigger deserialization
        return x

    # Notice, setting `device_memory_limit=1B` to trigger spilling
    with dask_cuda.LocalCUDACluster(
        n_workers=1, device_memory_limit="1B", dynamic_spill=dynamic_spill
    ) as cluster:
        with Client(cluster):
            df = cudf.DataFrame({"a": range(10)})
            ddf = dask_cudf.from_cudf(df, npartitions=1)
            ddf = ddf.map_partitions(task, meta=df.head())
            got = ddf.compute()
            assert_frame_equal(got.to_pandas(), df.to_pandas())
