from typing import Iterable

import numpy as np
import pandas
import pytest
from pandas.testing import assert_frame_equal

import dask
import dask.dataframe
from dask.dataframe.shuffle import shuffle_group
from dask.sizeof import sizeof
from distributed import Client
from distributed.client import wait
from distributed.worker import get_worker

import dask_cuda
import dask_cuda.proxify_device_objects
from dask_cuda.get_device_memory_objects import get_device_memory_objects
from dask_cuda.proxify_host_file import ProxifyHostFile
from dask_cuda.proxy_object import ProxyObject, asproxy

cupy = pytest.importorskip("cupy")
cupy.cuda.set_allocator(None)
one_item_array = lambda: cupy.arange(1)
one_item_nbytes = one_item_array().nbytes

# While testing we want to proxify `cupy.ndarray` even though
# it is on the ignore_type list by default.
dask_cuda.proxify_device_objects.dispatch.dispatch(cupy.ndarray)
dask_cuda.proxify_device_objects.ignore_types = ()


def is_proxies_equal(p1: Iterable[ProxyObject], p2: Iterable[ProxyObject]):
    """Check that two collections of proxies contains the same proxies (unordered)

    In order to avoid deserializing proxy objects when comparing them,
    this funcntion compares object IDs.
    """

    ids1 = sorted([id(p) for p in p1])
    ids2 = sorted([id(p) for p in p2])
    return ids1 == ids2


def test_one_dev_item_limit():
    dhf = ProxifyHostFile(device_memory_limit=one_item_nbytes, host_memory_limit=1000)

    a1 = one_item_array() + 42
    a2 = one_item_array()
    dhf["k1"] = a1
    dhf["k2"] = a2
    dhf.manager.validate()

    # Check k1 is spilled because of the newer k2
    k1 = dhf["k1"]
    k2 = dhf["k2"]
    assert k1._obj_pxy_is_serialized()
    assert not k2._obj_pxy_is_serialized()
    dhf.manager.validate()
    assert is_proxies_equal(dhf.manager._host, [k1])
    assert is_proxies_equal(dhf.manager._dev, [k2])

    # Accessing k1 spills k2 and unspill k1
    k1_val = k1[0]
    assert k1_val == 42
    assert k2._obj_pxy_is_serialized()
    dhf.manager.validate()
    assert is_proxies_equal(dhf.manager._host, [k2])
    assert is_proxies_equal(dhf.manager._dev, [k1])

    # Duplicate arrays changes nothing
    dhf["k3"] = [k1, k2]
    assert not k1._obj_pxy_is_serialized()
    assert k2._obj_pxy_is_serialized()
    dhf.manager.validate()
    assert is_proxies_equal(dhf.manager._host, [k2])
    assert is_proxies_equal(dhf.manager._dev, [k1])

    # Adding a new array spills k1 and k2
    dhf["k4"] = one_item_array()
    k4 = dhf["k4"]
    assert k1._obj_pxy_is_serialized()
    assert k2._obj_pxy_is_serialized()
    assert not dhf["k4"]._obj_pxy_is_serialized()
    dhf.manager.validate()
    assert is_proxies_equal(dhf.manager._host, [k1, k2])
    assert is_proxies_equal(dhf.manager._dev, [k4])

    # Accessing k2 spills k1 and k4
    k2[0]
    assert k1._obj_pxy_is_serialized()
    assert dhf["k4"]._obj_pxy_is_serialized()
    assert not k2._obj_pxy_is_serialized()
    dhf.manager.validate()
    assert is_proxies_equal(dhf.manager._host, [k1, k4])
    assert is_proxies_equal(dhf.manager._dev, [k2])

    # Deleting k2 does not change anything since k3 still holds a
    # reference to the underlying proxy object
    assert dhf.manager.get_dev_access_info()[0] == one_item_nbytes
    dhf.manager.validate()
    assert is_proxies_equal(dhf.manager._host, [k1, k4])
    assert is_proxies_equal(dhf.manager._dev, [k2])
    del dhf["k2"]
    dhf.manager.validate()
    assert is_proxies_equal(dhf.manager._host, [k1, k4])
    assert is_proxies_equal(dhf.manager._dev, [k2])

    # Overwriting "k3" with a non-cuda object and deleting `k2`
    # should empty the device
    dhf["k3"] = "non-cuda-object"
    del k2
    dhf.manager.validate()
    assert is_proxies_equal(dhf.manager._host, [k1, k4])
    assert is_proxies_equal(dhf.manager._dev, [])


def test_one_item_host_limit():
    host_memory_limit = sizeof(
        asproxy(one_item_array(), serializers=("dask", "pickle"))
    )
    dhf = ProxifyHostFile(
        device_memory_limit=one_item_nbytes, host_memory_limit=host_memory_limit
    )

    a1 = one_item_array() + 1
    a2 = one_item_array() + 2
    dhf["k1"] = a1
    dhf["k2"] = a2
    dhf.manager.validate()

    # Check k1 is spilled because of the newer k2
    k1 = dhf["k1"]
    k2 = dhf["k2"]
    assert k1._obj_pxy_is_serialized()
    assert not k2._obj_pxy_is_serialized()
    dhf.manager.validate()
    assert is_proxies_equal(dhf.manager._disk, [])
    assert is_proxies_equal(dhf.manager._host, [k1])
    assert is_proxies_equal(dhf.manager._dev, [k2])

    # Check k1 is spilled to disk and k2 is spilled to host
    dhf["k3"] = one_item_array() + 3
    k3 = dhf["k3"]
    dhf.manager.validate()
    assert is_proxies_equal(dhf.manager._disk, [k1])
    assert is_proxies_equal(dhf.manager._host, [k2])
    assert is_proxies_equal(dhf.manager._dev, [k3])

    dhf.manager.validate()

    # Accessing k2 spills k3 and unspill k2
    k2_val = k2[0]
    assert k2_val == 2
    dhf.manager.validate()
    assert is_proxies_equal(dhf.manager._disk, [k1])
    assert is_proxies_equal(dhf.manager._host, [k3])
    assert is_proxies_equal(dhf.manager._dev, [k2])

    # Adding a new array spill k3 to disk and k2 to host
    dhf["k4"] = one_item_array() + 4
    k4 = dhf["k4"]
    dhf.manager.validate()
    assert is_proxies_equal(dhf.manager._disk, [k1, k3])
    assert is_proxies_equal(dhf.manager._host, [k2])
    assert is_proxies_equal(dhf.manager._dev, [k4])

    # Accessing k1 unspills k1 directly to device and spills k4 to host
    k1_val = k1[0]
    assert k1_val == 1
    dhf.manager.validate()
    assert is_proxies_equal(dhf.manager._disk, [k2, k3])
    assert is_proxies_equal(dhf.manager._host, [k4])
    assert is_proxies_equal(dhf.manager._dev, [k1])


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
            assert x._obj_pxy["serializer"] == "dask"
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

    dhf = ProxifyHostFile(device_memory_limit=160, host_memory_limit=1000)
    dhf["v1"] = view1
    dhf["v2"] = view2
    v1 = dhf["v1"]
    v2 = dhf["v2"]
    # The device_memory_limit is not exceeded since both dataframes share device memory
    assert not v1._obj_pxy_is_serialized()
    assert not v2._obj_pxy_is_serialized()
    # Now the device_memory_limit is exceeded, which should evict both dataframes
    dhf["k1"] = one_item_array()
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


def test_externals():
    """Test adding objects directly to the manager

    Add an object directly to the manager makes it count against the
    device_memory_limit but isn't part of the store.

    Normally, we use __setitem__ to store objects in the hostfile and make it
    count against the device_memory_limit with the inherent consequence that
    the objects are not freeable before subsequential calls to __delitem__.
    This is a problem for long running tasks that want objects to count against
    the device_memory_limit while freeing them ASAP without explicit calls to
    __delitem__.
    """
    dhf = ProxifyHostFile(device_memory_limit=one_item_nbytes, host_memory_limit=1000)
    dhf["k1"] = one_item_array()
    k1 = dhf["k1"]
    k2 = dhf.manager.proxify(one_item_array())
    # `k2` isn't part of the store but still triggers spilling of `k1`
    assert len(dhf) == 1
    assert k1._obj_pxy_is_serialized()
    assert not k2._obj_pxy_is_serialized()
    assert is_proxies_equal(dhf.manager._host, [k1])
    assert is_proxies_equal(dhf.manager._dev, [k2])
    assert dhf.manager._dev._mem_usage == one_item_nbytes

    k1[0]  # Trigger spilling of `k2`
    assert not k1._obj_pxy_is_serialized()
    assert k2._obj_pxy_is_serialized()
    assert is_proxies_equal(dhf.manager._host, [k2])
    assert is_proxies_equal(dhf.manager._dev, [k1])
    assert dhf.manager._dev._mem_usage == one_item_nbytes

    k2[0]  # Trigger spilling of `k1`
    assert k1._obj_pxy_is_serialized()
    assert not k2._obj_pxy_is_serialized()
    assert is_proxies_equal(dhf.manager._host, [k1])
    assert is_proxies_equal(dhf.manager._dev, [k2])
    assert dhf.manager._dev._mem_usage == one_item_nbytes

    # Removing `k2` also removes it from the tally
    del k2
    assert is_proxies_equal(dhf.manager._host, [k1])
    assert is_proxies_equal(dhf.manager._dev, [])
    assert dhf.manager._dev._mem_usage == 0


def test_proxify_device_objects_of_cupy_array():
    """Check that a proxied array behaves as a regular cupy array

    Notice, in this test we add `cupy.ndarray` to the ignore_types temporarily.
    """
    cupy = pytest.importorskip("cupy")
    dask_cuda.proxify_device_objects.ignore_types = (cupy.ndarray,)
    try:
        # Make sure that equality works, which we use to test the other operators
        org = cupy.arange(9).reshape((3, 3)) + 1
        pxy = dask_cuda.proxify_device_objects.proxify_device_objects(
            org.copy(), {}, []
        )
        assert (org == pxy).all()
        assert (org + 1 != pxy).all()

        for op in [cupy.dot]:
            res = op(org, org)
            assert (op(pxy, pxy) == res).all()
            assert (op(org, pxy) == res).all()
            assert (op(pxy, org) == res).all()
    finally:
        dask_cuda.proxify_device_objects.ignore_types = ()


@pytest.mark.parametrize("npartitions", [1, 2, 3])
@pytest.mark.parametrize("compatibility_mode", [True, False])
def test_compatibility_mode_dataframe_shuffle(compatibility_mode, npartitions):
    cudf = pytest.importorskip("cudf")

    def is_proxy_object(x):
        return "ProxyObject" in str(type(x))

    with dask.config.set(jit_unspill_compatibility_mode=compatibility_mode):
        with dask_cuda.LocalCUDACluster(n_workers=1, jit_unspill=True) as cluster:
            with Client(cluster):
                ddf = dask.dataframe.from_pandas(
                    cudf.DataFrame({"key": np.arange(10)}), npartitions=npartitions
                )
                res = ddf.shuffle(on="key", shuffle="tasks").persist()

                # With compatibility mode on, we shouldn't encounter any proxy objects
                if compatibility_mode:
                    assert "ProxyObject" not in str(type(res.compute()))
                res = res.map_partitions(is_proxy_object).compute()
                res = res.to_list()

                if compatibility_mode:
                    assert not any(res)  # No proxy objects
                else:
                    assert all(res)  # Only proxy objects


def test_spill_to_disk():
    """
    Test Dask triggering CPU-to-Disk spilling,
    which we do not support at the moment
    """

    with dask.config.set({"distributed.worker.memory.terminate": 0}):
        with dask_cuda.LocalCUDACluster(
            n_workers=1, memory_limit=100, jit_unspill=True
        ) as cluster:
            with Client(cluster) as client:
                ddf = dask.dataframe.from_pandas(
                    pandas.DataFrame({"key": np.arange(1000)}), npartitions=1
                )
                ddf = ddf.persist()
                wait(ddf)

                def f():
                    """Trigger a memory_monitor() and reset memory_limit"""
                    w = get_worker()

                    async def y():
                        await w.memory_monitor()
                        w.memory_limit = 10 ** 6

                    w.loop.add_callback(y)

                wait(client.submit(f))
                assert "JIT-Unspill doesn't support spilling to Disk" in str(
                    client.get_worker_logs()
                )
