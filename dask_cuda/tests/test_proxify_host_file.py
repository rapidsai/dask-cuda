from typing import Iterable
from unittest.mock import patch

import numpy as np
import pytest
from pandas.testing import assert_frame_equal

import dask
import dask.dataframe
from dask.dataframe.shuffle import shuffle_group
from dask.sizeof import sizeof
from dask.utils import format_bytes
from distributed import Client
from distributed.utils_test import gen_test

import dask_cuda
import dask_cuda.proxify_device_objects
from dask_cuda.get_device_memory_objects import get_device_memory_ids
from dask_cuda.proxify_host_file import ProxifyHostFile
from dask_cuda.proxy_object import ProxyObject, asproxy, unproxy
from dask_cuda.utils import get_device_total_memory

cupy = pytest.importorskip("cupy")
cupy.cuda.set_allocator(None)
one_item_array = lambda: cupy.arange(1)
one_item_nbytes = one_item_array().nbytes

# While testing we don't want to unproxify `cupy.ndarray` even though
# it is on the incompatible_types list by default.
dask_cuda.proxify_device_objects.dispatch.dispatch(cupy.ndarray)
dask_cuda.proxify_device_objects.incompatible_types = ()  # type: ignore


@pytest.fixture(scope="module")
def root_dir(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("jit-unspill")
    # Make the "disk" serializer available and use a tmp directory
    if ProxifyHostFile._spill_to_disk is None:
        ProxifyHostFile(
            worker_local_directory=tmpdir.name,
            device_memory_limit=1024,
            memory_limit=1024,
        )
    assert ProxifyHostFile._spill_to_disk is not None

    # In order to use the same tmp dir, we use `root_dir` for all
    # ProxifyHostFile creations. Notice, we use `..` to remove the
    # `jit-unspill-disk-storage` part added by the
    # ProxifyHostFile implicitly.
    return str(ProxifyHostFile._spill_to_disk.root_dir / "..")


def is_proxies_equal(p1: Iterable[ProxyObject], p2: Iterable[ProxyObject]):
    """Check that two collections of proxies contains the same proxies (unordered)

    In order to avoid deserializing proxy objects when comparing them,
    this function compares object IDs.
    """

    ids1 = sorted([id(p) for p in p1])
    ids2 = sorted([id(p) for p in p2])
    return ids1 == ids2


def test_one_dev_item_limit(root_dir):
    dhf = ProxifyHostFile(
        worker_local_directory=root_dir,
        device_memory_limit=one_item_nbytes,
        memory_limit=1000,
    )

    a1 = one_item_array() + 42
    a2 = one_item_array()
    dhf["k1"] = a1
    dhf["k2"] = a2
    dhf.manager.validate()

    # Check k1 is spilled because of the newer k2
    k1 = dhf["k1"]
    k2 = dhf["k2"]
    assert k1._pxy_get().is_serialized()
    assert not k2._pxy_get().is_serialized()
    dhf.manager.validate()
    assert is_proxies_equal(dhf.manager._host.get_proxies(), [k1])
    assert is_proxies_equal(dhf.manager._dev.get_proxies(), [k2])

    # Accessing k1 spills k2 and unspill k1
    k1_val = k1[0]
    assert k1_val == 42
    assert k2._pxy_get().is_serialized()
    dhf.manager.validate()
    assert is_proxies_equal(dhf.manager._host.get_proxies(), [k2])
    assert is_proxies_equal(dhf.manager._dev.get_proxies(), [k1])

    # Duplicate arrays changes nothing
    dhf["k3"] = [k1, k2]
    assert not k1._pxy_get().is_serialized()
    assert k2._pxy_get().is_serialized()
    dhf.manager.validate()
    assert is_proxies_equal(dhf.manager._host.get_proxies(), [k2])
    assert is_proxies_equal(dhf.manager._dev.get_proxies(), [k1])

    # Adding a new array spills k1 and k2
    dhf["k4"] = one_item_array()
    k4 = dhf["k4"]
    assert k1._pxy_get().is_serialized()
    assert k2._pxy_get().is_serialized()
    assert not dhf["k4"]._pxy_get().is_serialized()
    dhf.manager.validate()
    assert is_proxies_equal(dhf.manager._host.get_proxies(), [k1, k2])
    assert is_proxies_equal(dhf.manager._dev.get_proxies(), [k4])

    # Accessing k2 spills k1 and k4
    k2[0]
    assert k1._pxy_get().is_serialized()
    assert dhf["k4"]._pxy_get().is_serialized()
    assert not k2._pxy_get().is_serialized()
    dhf.manager.validate()
    assert is_proxies_equal(dhf.manager._host.get_proxies(), [k1, k4])
    assert is_proxies_equal(dhf.manager._dev.get_proxies(), [k2])

    # Deleting k2 does not change anything since k3 still holds a
    # reference to the underlying proxy object
    assert dhf.manager._dev.mem_usage() == one_item_nbytes
    dhf.manager.validate()
    assert is_proxies_equal(dhf.manager._host.get_proxies(), [k1, k4])
    assert is_proxies_equal(dhf.manager._dev.get_proxies(), [k2])
    del dhf["k2"]
    dhf.manager.validate()
    assert is_proxies_equal(dhf.manager._host.get_proxies(), [k1, k4])
    assert is_proxies_equal(dhf.manager._dev.get_proxies(), [k2])

    # Overwriting k3 with a non-cuda object and deleting k2
    # should empty the device
    dhf["k3"] = "non-cuda-object"
    del k2
    dhf.manager.validate()
    assert is_proxies_equal(dhf.manager._host.get_proxies(), [k1, k4])
    assert is_proxies_equal(dhf.manager._dev.get_proxies(), [])

    # Adding the underlying proxied of k1 doesn't change anything.
    # The host file detects that k1_ary is already proxied by the
    # existing proxy object k1.
    k1_ary = unproxy(k1)
    dhf["k5"] = k1_ary
    dhf.manager.validate()
    assert is_proxies_equal(dhf.manager._host.get_proxies(), [k4])
    assert is_proxies_equal(dhf.manager._dev.get_proxies(), [k1])

    # Clean up
    del k1, k4
    dhf.clear()
    assert len(dhf.manager) == 0


def test_one_item_host_limit(capsys, root_dir):
    memory_limit = sizeof(asproxy(one_item_array(), serializers=("dask", "pickle")))
    dhf = ProxifyHostFile(
        worker_local_directory=root_dir,
        device_memory_limit=one_item_nbytes,
        memory_limit=memory_limit,
    )

    a1 = one_item_array() + 1
    a2 = one_item_array() + 2
    dhf["k1"] = a1
    dhf["k2"] = a2
    dhf.manager.validate()

    # Check k1 is spilled because of the newer k2
    k1 = dhf["k1"]
    k2 = dhf["k2"]
    assert k1._pxy_get().is_serialized()
    assert not k2._pxy_get().is_serialized()
    dhf.manager.validate()
    assert is_proxies_equal(dhf.manager._disk.get_proxies(), [])
    assert is_proxies_equal(dhf.manager._host.get_proxies(), [k1])
    assert is_proxies_equal(dhf.manager._dev.get_proxies(), [k2])

    # Check k1 is spilled to disk and k2 is spilled to host
    dhf["k3"] = one_item_array() + 3
    k3 = dhf["k3"]
    dhf.manager.validate()
    assert is_proxies_equal(dhf.manager._disk.get_proxies(), [k1])
    assert is_proxies_equal(dhf.manager._host.get_proxies(), [k2])
    assert is_proxies_equal(dhf.manager._dev.get_proxies(), [k3])
    dhf.manager.validate()

    # Accessing k2 spills k3 and unspill k2
    k2_val = k2[0]
    assert k2_val == 2
    dhf.manager.validate()
    assert is_proxies_equal(dhf.manager._disk.get_proxies(), [k1])
    assert is_proxies_equal(dhf.manager._host.get_proxies(), [k3])
    assert is_proxies_equal(dhf.manager._dev.get_proxies(), [k2])

    # Adding a new array spill k3 to disk and k2 to host
    dhf["k4"] = one_item_array() + 4
    k4 = dhf["k4"]
    dhf.manager.validate()
    assert is_proxies_equal(dhf.manager._disk.get_proxies(), [k1, k3])
    assert is_proxies_equal(dhf.manager._host.get_proxies(), [k2])
    assert is_proxies_equal(dhf.manager._dev.get_proxies(), [k4])

    # Accessing k1 unspills k1 directly to device and spills k4 to host
    k1_val = k1[0]
    assert k1_val == 1
    dhf.manager.validate()
    assert is_proxies_equal(dhf.manager._disk.get_proxies(), [k2, k3])
    assert is_proxies_equal(dhf.manager._host.get_proxies(), [k4])
    assert is_proxies_equal(dhf.manager._dev.get_proxies(), [k1])

    # Clean up
    del k1, k2, k3, k4
    dhf.clear()
    assert len(dhf.manager) == 0


def test_spill_on_demand(root_dir):
    """
    Test spilling on demand by disabling the device_memory_limit
    and allocating two large buffers that will otherwise fail because
    of spilling on demand.
    """
    rmm = pytest.importorskip("rmm")
    if not hasattr(rmm.mr, "FailureCallbackResourceAdaptor"):
        pytest.skip("RMM doesn't implement FailureCallbackResourceAdaptor")

    total_mem = get_device_total_memory()
    dhf = ProxifyHostFile(
        worker_local_directory=root_dir,
        device_memory_limit=2 * total_mem,
        memory_limit=2 * total_mem,
        spill_on_demand=True,
    )
    for i in range(2):
        dhf[i] = rmm.DeviceBuffer(size=total_mem // 2 + 1)


@pytest.mark.parametrize("jit_unspill", [True, False])
@gen_test(timeout=20)
async def test_local_cuda_cluster(jit_unspill):
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
    async with dask_cuda.LocalCUDACluster(
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
            assert_frame_equal(got.to_pandas(), df.to_pandas())


def test_dataframes_share_dev_mem(root_dir):
    cudf = pytest.importorskip("cudf")

    df = cudf.DataFrame({"a": range(10)})
    grouped = shuffle_group(df, "a", 0, 2, 2, False, 2)
    view1 = grouped[0]
    view2 = grouped[1]
    # Even though the two dataframe doesn't point to the same cudf.Buffer object
    assert view1["a"].data is not view2["a"].data
    # They still share the same underlying device memory
    view1["a"].data.get_ptr(mode="read") == view2["a"].data.get_ptr(mode="read")

    dhf = ProxifyHostFile(
        worker_local_directory=root_dir, device_memory_limit=160, memory_limit=1000
    )
    dhf["v1"] = view1
    dhf["v2"] = view2
    v1 = dhf["v1"]
    v2 = dhf["v2"]
    # The device_memory_limit is not exceeded since both dataframes share device memory
    assert not v1._pxy_get().is_serialized()
    assert not v2._pxy_get().is_serialized()
    # Now the device_memory_limit is exceeded, which should evict both dataframes
    dhf["k1"] = one_item_array()
    assert v1._pxy_get().is_serialized()
    assert v2._pxy_get().is_serialized()


def test_cudf_get_device_memory_objects():
    cudf = pytest.importorskip("cudf")
    objects = [
        cudf.DataFrame({"a": range(10), "b": range(10)}, index=reversed(range(10))),
        cudf.MultiIndex(
            levels=[[1, 2], ["blue", "red"]], codes=[[0, 0, 1, 1], [1, 0, 1, 0]]
        ),
    ]
    res = get_device_memory_ids(objects)
    assert len(res) == 4, "We expect four buffer objects"


def test_externals(root_dir):
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
    dhf = ProxifyHostFile(
        worker_local_directory=root_dir,
        device_memory_limit=one_item_nbytes,
        memory_limit=1000,
    )
    dhf["k1"] = one_item_array()
    k1 = dhf["k1"]
    k2, incompatible_type_found = dhf.manager.proxify(one_item_array())
    assert not incompatible_type_found
    # `k2` isn't part of the store but still triggers spilling of `k1`
    assert len(dhf) == 1
    assert k1._pxy_get().is_serialized()
    assert not k2._pxy_get().is_serialized()
    assert is_proxies_equal(dhf.manager._host.get_proxies(), [k1])
    assert is_proxies_equal(dhf.manager._dev.get_proxies(), [k2])
    assert dhf.manager._dev._mem_usage == one_item_nbytes

    k1[0]  # Trigger spilling of `k2`
    assert not k1._pxy_get().is_serialized()
    assert k2._pxy_get().is_serialized()
    assert is_proxies_equal(dhf.manager._host.get_proxies(), [k2])
    assert is_proxies_equal(dhf.manager._dev.get_proxies(), [k1])
    assert dhf.manager._dev._mem_usage == one_item_nbytes

    k2[0]  # Trigger spilling of `k1`
    assert k1._pxy_get().is_serialized()
    assert not k2._pxy_get().is_serialized()
    assert is_proxies_equal(dhf.manager._host.get_proxies(), [k1])
    assert is_proxies_equal(dhf.manager._dev.get_proxies(), [k2])
    assert dhf.manager._dev._mem_usage == one_item_nbytes

    # Removing `k2` also removes it from the tally
    del k2
    assert is_proxies_equal(dhf.manager._host.get_proxies(), [k1])
    assert is_proxies_equal(dhf.manager._dev.get_proxies(), [])
    assert dhf.manager._dev._mem_usage == 0


@patch("dask_cuda.proxify_device_objects.incompatible_types", (cupy.ndarray,))
def test_incompatible_types(root_dir):
    """Check that ProxifyHostFile unproxifies `cupy.ndarray` on retrieval

    Notice, in this test we add `cupy.ndarray` to the incompatible_types temporarily.
    """
    cupy = pytest.importorskip("cupy")
    cudf = pytest.importorskip("cudf")
    dhf = ProxifyHostFile(
        worker_local_directory=root_dir, device_memory_limit=100, memory_limit=100
    )

    # We expect `dhf` to unproxify `a1` (but not `a2`) on retrieval
    a1, a2 = (cupy.arange(9), cudf.Series([1, 2, 3]))
    dhf["a"] = (a1, a2)
    b1, b2 = dhf["a"]
    assert a1 is b1
    assert isinstance(b2, ProxyObject)
    assert a2 is unproxy(b2)


@pytest.mark.parametrize("npartitions", [1, 2, 3])
@pytest.mark.parametrize("compatibility_mode", [True, False])
@gen_test(timeout=30)
async def test_compatibility_mode_dataframe_shuffle(compatibility_mode, npartitions):
    cudf = pytest.importorskip("cudf")

    def is_proxy_object(x):
        return "ProxyObject" in str(type(x))

    with dask.config.set(jit_unspill_compatibility_mode=compatibility_mode):
        async with dask_cuda.LocalCUDACluster(
            n_workers=1, jit_unspill=True, asynchronous=True
        ) as cluster:
            async with Client(cluster, asynchronous=True) as client:
                ddf = dask.dataframe.from_pandas(
                    cudf.DataFrame({"key": np.arange(10)}), npartitions=npartitions
                )
                res = ddf.shuffle(on="key", shuffle="tasks").persist()

                # With compatibility mode on, we shouldn't encounter any proxy objects
                if compatibility_mode:
                    assert "ProxyObject" not in str(type(await client.compute(res)))
                res = await client.compute(res.map_partitions(is_proxy_object))
                res = res.to_list()

                if compatibility_mode:
                    assert not any(res)  # No proxy objects
                else:
                    assert all(res)  # Only proxy objects


@gen_test(timeout=60)
async def test_worker_force_spill_to_disk():
    """Test Dask triggering CPU-to-Disk spilling"""
    cudf = pytest.importorskip("cudf")

    with dask.config.set({"distributed.worker.memory.terminate": False}):
        async with dask_cuda.LocalCUDACluster(
            n_workers=1, device_memory_limit="1MB", jit_unspill=True, asynchronous=True
        ) as cluster:
            async with Client(cluster, asynchronous=True) as client:
                # Create a df that are spilled to host memory immediately
                df = cudf.DataFrame({"key": np.arange(10**8)})
                ddf = dask.dataframe.from_pandas(df, npartitions=1).persist()
                await ddf

                async def f(dask_worker):
                    """Trigger a memory_monitor() and reset memory_limit"""
                    w = dask_worker
                    # Set a host memory limit that triggers spilling to disk
                    w.memory_manager.memory_pause_fraction = False
                    memory = w.monitor.proc.memory_info().rss
                    w.memory_manager.memory_limit = memory - 10**8
                    w.memory_manager.memory_target_fraction = 1
                    print(w.memory_manager.data)
                    await w.memory_manager.memory_monitor(w)
                    # Check that host memory are freed
                    assert w.monitor.proc.memory_info().rss < memory - 10**7
                    w.memory_manager.memory_limit = memory * 10  # Un-limit

                client.run(f)
                log = str(await client.get_worker_logs())
                # Check that the worker doesn't complain about unmanaged memory
                assert "Unmanaged memory use is high" not in log


def test_on_demand_debug_info():
    """Test worker logging when on-demand-spilling fails"""
    rmm = pytest.importorskip("rmm")
    if not hasattr(rmm.mr, "FailureCallbackResourceAdaptor"):
        pytest.skip("RMM doesn't implement FailureCallbackResourceAdaptor")

    rmm_pool_size = 2**20

    def task():
        (
            rmm.DeviceBuffer(size=rmm_pool_size // 2),
            rmm.DeviceBuffer(size=rmm_pool_size // 2),
            rmm.DeviceBuffer(size=rmm_pool_size),  # Trigger OOM
        )

    with dask_cuda.LocalCUDACluster(
        n_workers=1,
        jit_unspill=True,
        rmm_pool_size=rmm_pool_size,
        rmm_maximum_pool_size=rmm_pool_size,
        rmm_track_allocations=True,
    ) as cluster:
        with Client(cluster) as client:
            # Warmup, which trigger the initialization of spill on demand
            client.submit(range, 10).result()

            # Submit too large RMM buffer
            with pytest.raises(MemoryError, match="Maximum pool size exceeded"):
                client.submit(task).result()

            log = str(client.get_worker_logs())
            size = format_bytes(rmm_pool_size)
            assert f"WARNING - RMM allocation of {size} failed" in log
            assert f"RMM allocs: {size}" in log
            assert "traceback:" in log
