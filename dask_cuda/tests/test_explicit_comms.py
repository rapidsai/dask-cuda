# Copyright (c) 2021-2025 NVIDIA CORPORATION.

import asyncio
import multiprocessing as mp
import os
import signal
import time
from functools import partial
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import dask
from dask import dataframe as dd
from dask.dataframe.shuffle import partitioning_index
from dask.dataframe.utils import assert_eq
from distributed import Client
from distributed.deploy.local import LocalCluster

import dask_cuda
from dask_cuda.explicit_comms import comms
from dask_cuda.explicit_comms.dataframe.shuffle import (
    _contains_shuffle_expr,
    shuffle as explicit_comms_shuffle,
)
from dask_cuda.utils_test import IncreasedCloseTimeoutNanny

mp = mp.get_context("spawn")  # type: ignore
ucp = pytest.importorskip("ucp")


# Notice, all of the following tests is executed in a new process such
# that UCX options of the different tests doesn't conflict.


async def my_rank(state, arg):
    return state["rank"] + arg


def _test_local_cluster(protocol):
    with LocalCluster(
        protocol=protocol,
        dashboard_address=None,
        n_workers=4,
        threads_per_worker=1,
        worker_class=IncreasedCloseTimeoutNanny,
        processes=True,
    ) as cluster:
        with Client(cluster) as client:
            c = comms.CommsContext(client)
            assert sum(c.run(my_rank, 0)) == sum(range(4))


@pytest.mark.parametrize("protocol", ["tcp", "ucx", "ucxx"])
def test_local_cluster(protocol):
    p = mp.Process(target=_test_local_cluster, args=(protocol,))
    p.start()
    p.join()
    assert not p.exitcode


def _test_dataframe_merge_empty_partitions(nrows, npartitions):
    with LocalCluster(
        protocol="tcp",
        dashboard_address=None,
        n_workers=npartitions,
        threads_per_worker=1,
        worker_class=IncreasedCloseTimeoutNanny,
        processes=True,
    ) as cluster:
        with Client(cluster):
            df1 = pd.DataFrame({"key": np.arange(nrows), "payload1": np.arange(nrows)})
            key = np.arange(nrows)
            np.random.shuffle(key)
            df2 = pd.DataFrame({"key": key, "payload2": np.arange(nrows)})
            expected = df1.merge(df2).set_index("key")
            ddf1 = dd.from_pandas(df1, npartitions=npartitions)
            ddf2 = dd.from_pandas(df2, npartitions=npartitions)

            for batchsize in (-1, 1, 2):
                with dask.config.set(
                    explicit_comms=True, explicit_comms_batchsize=batchsize
                ):
                    ddf3 = ddf1.merge(ddf2, on=["key"]).set_index("key")
                    got = ddf3.compute()
                    pd.testing.assert_frame_equal(got, expected)


def test_dataframe_merge_empty_partitions():
    # Notice, we use more partitions than rows
    p = mp.Process(target=_test_dataframe_merge_empty_partitions, args=(2, 4))
    p.start()
    p.join()
    assert not p.exitcode


def check_partitions(df, npartitions):
    """Check that all values in `df` hashes to the same"""
    dtypes = {}
    for col, dtype in df.dtypes.items():
        if pd.api.types.is_numeric_dtype(dtype):
            dtypes[col] = np.float64
    if not dtypes:
        dtypes = None

    hashes = partitioning_index(df, npartitions, cast_dtype=dtypes)
    if len(hashes) > 0:
        return len(hashes.unique()) == 1
    else:
        return True


def _test_dataframe_shuffle(backend, protocol, n_workers, _partitions):
    if backend == "cudf":
        cudf = pytest.importorskip("cudf")

    with LocalCluster(
        protocol=protocol,
        dashboard_address=None,
        n_workers=n_workers,
        threads_per_worker=1,
        worker_class=IncreasedCloseTimeoutNanny,
        processes=True,
    ) as cluster:
        with Client(cluster):
            comms.default_comms()
            np.random.seed(42)
            df = pd.DataFrame({"key": np.random.randint(0, high=100, size=100)})
            if backend == "cudf":
                df = cudf.DataFrame.from_pandas(df)

            if _partitions:
                df["_partitions"] = 0

            for input_nparts in range(1, 5):
                for output_nparts in range(1, 5):
                    ddf1 = dd.from_pandas(df.copy(), npartitions=input_nparts)
                    # To reduce test runtime, we change the batchsizes here instead
                    # of using a test parameter.
                    for batchsize in (-1, 1, 2):
                        with dask.config.set(explicit_comms_batchsize=batchsize):
                            ddf = explicit_comms_shuffle(
                                ddf1,
                                ["_partitions"] if _partitions else ["key"],
                                npartitions=output_nparts,
                                batchsize=batchsize,
                            ).persist()

                            assert ddf.npartitions == output_nparts

                            if _partitions:
                                # If "_partitions" is the hash key, we expect all but
                                # the first partition to be empty
                                assert_eq(ddf.partitions[0].compute(), df)
                                assert all(
                                    len(ddf.partitions[i].compute()) == 0
                                    for i in range(1, ddf.npartitions)
                                )
                            else:
                                # Check that each partition hashes to the same value
                                result = ddf.map_partitions(
                                    check_partitions, output_nparts
                                ).compute()
                                assert all(result.to_list())

                                # Check the values (ignoring the row order)
                                expected = df.sort_values("key")
                                got = ddf.compute().sort_values("key")
                                assert_eq(got, expected)

                                # Check that partitioning is consistent with "tasks"
                                ddf_tasks = ddf1.shuffle(
                                    ["key"],
                                    npartitions=output_nparts,
                                    shuffle_method="tasks",
                                )
                                for i in range(output_nparts):
                                    expected_partition = ddf_tasks.partitions[
                                        i
                                    ].compute()["key"]
                                    actual_partition = ddf.partitions[i].compute()[
                                        "key"
                                    ]
                                    if backend == "cudf":
                                        expected_partition = (
                                            expected_partition.values_host
                                        )
                                        actual_partition = actual_partition.values_host
                                    else:
                                        expected_partition = expected_partition.values
                                        actual_partition = actual_partition.values
                                    assert all(
                                        np.sort(expected_partition)
                                        == np.sort(actual_partition)
                                    )


@pytest.mark.parametrize("nworkers", [1, 2, 3])
@pytest.mark.parametrize("backend", ["pandas", "cudf"])
@pytest.mark.parametrize("protocol", ["tcp", "ucx", "ucxx"])
@pytest.mark.parametrize("_partitions", [True, False])
def test_dataframe_shuffle(backend, protocol, nworkers, _partitions):
    if backend == "cudf":
        pytest.importorskip("cudf")

    p = mp.Process(
        target=_test_dataframe_shuffle, args=(backend, protocol, nworkers, _partitions)
    )
    p.start()
    p.join()
    assert not p.exitcode


@pytest.mark.parametrize("in_cluster", [True, False])
def _test_dask_use_explicit_comms(in_cluster):
    def check_shuffle():
        """Check if shuffle use explicit-comms by search for keys named
        'explicit-comms-shuffle'
        """
        name = "explicit-comms-shuffle"
        ddf = dd.from_pandas(pd.DataFrame({"key": np.arange(10)}), npartitions=2)
        with dask.config.set(explicit_comms=False):
            res = ddf.shuffle(on="key", npartitions=4)
            assert all(name not in str(key) for key in res.dask)
        with dask.config.set(explicit_comms=True):
            res = ddf.shuffle(on="key", npartitions=4)
            if in_cluster:
                assert any(name in str(key) for key in res.dask)
            else:  # If not in cluster, we cannot use explicit comms
                assert all(name not in str(key) for key in res.dask)

        if in_cluster:
            # We check environment variables by setting an illegal batchsize
            with patch.dict(
                os.environ,
                {"DASK_EXPLICIT_COMMS": "1", "DASK_EXPLICIT_COMMS_BATCHSIZE": "-2"},
            ):
                dask.config.refresh()  # Trigger re-read of the environment variables
                with pytest.raises(ValueError, match="explicit-comms-batchsize"):
                    ddf.shuffle(on="key", npartitions=4).dask

    if in_cluster:
        with LocalCluster(
            protocol="tcp",
            dashboard_address=None,
            n_workers=2,
            threads_per_worker=1,
            worker_class=IncreasedCloseTimeoutNanny,
            processes=True,
        ) as cluster:
            with Client(cluster):
                check_shuffle()
    else:
        check_shuffle()


@pytest.mark.parametrize("in_cluster", [True, False])
def test_dask_use_explicit_comms(in_cluster):
    def _timeout(process, function, timeout):
        if process.is_alive():
            function()
        timeout = time.time() + timeout
        while process.is_alive() and time.time() < timeout:
            time.sleep(0.1)

    p = mp.Process(target=_test_dask_use_explicit_comms, args=(in_cluster,))
    p.start()

    # Timeout before killing process
    _timeout(p, lambda: None, 60.0)

    # Send SIGINT (i.e., KeyboardInterrupt) hoping we get a stack trace.
    _timeout(p, partial(p._popen._send_signal, signal.SIGINT), 3.0)

    # SIGINT didn't work, kill process.
    _timeout(p, p.kill, 3.0)

    assert not p.is_alive()
    assert p.exitcode == 0


def _test_dataframe_shuffle_merge(backend, protocol, n_workers):
    if backend == "cudf":
        cudf = pytest.importorskip("cudf")

    with LocalCluster(
        protocol=protocol,
        dashboard_address=None,
        n_workers=n_workers,
        threads_per_worker=1,
        worker_class=IncreasedCloseTimeoutNanny,
        processes=True,
    ) as cluster:
        with Client(cluster):
            nrows = n_workers * 10

            # Let's make some dataframes that we can join on the "key" column
            df1 = pd.DataFrame({"key": np.arange(nrows), "payload1": np.arange(nrows)})
            key = np.arange(nrows)
            np.random.shuffle(key)
            df2 = pd.DataFrame(
                {"key": key[nrows // 3 :], "payload2": np.arange(nrows)[nrows // 3 :]}
            )
            expected = df1.merge(df2, on="key").set_index("key")

            if backend == "cudf":
                df1 = cudf.DataFrame.from_pandas(df1)
                df2 = cudf.DataFrame.from_pandas(df2)

            ddf1 = dd.from_pandas(df1, npartitions=n_workers + 1)
            ddf2 = dd.from_pandas(
                df2, npartitions=n_workers - 1 if n_workers > 1 else 1
            )
            with dask.config.set(explicit_comms=True):
                got = ddf1.merge(ddf2, on="key").set_index("key").compute()
            assert_eq(got, expected)


@pytest.mark.parametrize("nworkers", [1, 2, 4])
@pytest.mark.parametrize("backend", ["pandas", "cudf"])
@pytest.mark.parametrize("protocol", ["tcp", "ucx", "ucxx"])
def test_dataframe_shuffle_merge(backend, protocol, nworkers):
    if backend == "cudf":
        pytest.importorskip("cudf")
    p = mp.Process(
        target=_test_dataframe_shuffle_merge, args=(backend, protocol, nworkers)
    )
    p.start()
    p.join()
    assert not p.exitcode


def _test_jit_unspill(protocol):
    import cudf

    with dask_cuda.LocalCUDACluster(
        protocol=protocol,
        dashboard_address=None,
        n_workers=1,
        threads_per_worker=1,
        jit_unspill=True,
        device_memory_limit="1B",
    ) as cluster:
        with Client(cluster):
            np.random.seed(42)
            df = cudf.DataFrame.from_pandas(
                pd.DataFrame({"key": np.random.random(100)})
            )
            ddf = dd.from_pandas(df.copy(), npartitions=4)
            ddf = explicit_comms_shuffle(ddf, ["key"])

            # Check the values of `ddf` (ignoring the row order)
            expected = df.sort_values("key")
            got = ddf.compute().sort_values("key")
            assert_eq(got, expected)


@pytest.mark.parametrize("protocol", ["tcp", "ucx", "ucxx"])
def test_jit_unspill(protocol):
    pytest.importorskip("cudf")

    p = mp.Process(target=_test_jit_unspill, args=(protocol,))
    p.start()
    p.join()
    assert not p.exitcode


def _test_lock_workers(scheduler_address, ranks):
    async def f(info):
        worker = info["worker"]
        if hasattr(worker, "running"):
            assert not worker.running
        worker.running = True
        await asyncio.sleep(0.5)
        assert worker.running
        worker.running = False

    with Client(scheduler_address) as client:
        c = comms.CommsContext(client)
        c.run(f, workers=[c.worker_addresses[r] for r in ranks], lock_workers=True)


def test_lock_workers():
    """
    Testing `run(...,lock_workers=True)` by spawning 30 runs with overlapping
    and non-overlapping worker sets.
    """
    try:
        from distributed import MultiLock  # noqa F401
    except ImportError as e:
        pytest.skip(str(e))

    with LocalCluster(
        protocol="tcp",
        dashboard_address=None,
        n_workers=4,
        threads_per_worker=5,
        worker_class=IncreasedCloseTimeoutNanny,
        processes=True,
    ) as cluster:
        ps = []
        for _ in range(5):
            for ranks in [[0, 1], [1, 3], [2, 3]]:
                ps.append(
                    mp.Process(
                        target=_test_lock_workers,
                        args=(cluster.scheduler_address, ranks),
                    )
                )
                ps[-1].start()

        for p in ps:
            p.join()

        assert all(p.exitcode == 0 for p in ps)


def test_create_destroy_create():
    # https://github.com/rapidsai/dask-cuda/issues/1450
    assert len(comms._comms_cache) == 0
    with LocalCluster(n_workers=1) as cluster:
        with Client(cluster) as client:
            context = comms.default_comms()
            scheduler_addresses_old = list(client.scheduler_info()["workers"].keys())
            comms_addresses_old = list(comms.default_comms().worker_addresses)
            assert comms.default_comms() is context
            assert len(comms._comms_cache) == 1

            # Add a worker, which should have a new comms object
            cluster.scale(2)
            client.wait_for_workers(2, timeout=5)
            context2 = comms.default_comms()
            assert context is not context2
            assert len(comms._comms_cache) == 2

    del context
    del context2
    assert len(comms._comms_cache) == 0
    assert scheduler_addresses_old == comms_addresses_old

    # A new cluster should have a new comms object. Previously, this failed
    # because we referenced the old cluster's addresses.
    with LocalCluster(n_workers=1) as cluster:
        with Client(cluster) as client:
            scheduler_addresses_new = list(client.scheduler_info()["workers"].keys())
            comms_addresses_new = list(comms.default_comms().worker_addresses)

    assert scheduler_addresses_new == comms_addresses_new


def test_scaled_cluster_gets_new_comms_context():
    # Ensure that if we create a CommsContext, scale the cluster,
    # and create a new CommsContext, then the new CommsContext
    # should include the new worker.
    # https://github.com/rapidsai/dask-cuda/issues/1450

    name = "explicit-comms-shuffle"
    ddf = dd.from_pandas(pd.DataFrame({"key": np.arange(10)}), npartitions=2)

    with LocalCluster(n_workers=2) as cluster:
        with Client(cluster) as client:
            context_1 = comms.default_comms()

            def check(dask_worker, session_id: int):
                has_state = hasattr(dask_worker, "_explicit_comm_state")
                has_state_for_session = (
                    has_state and session_id in dask_worker._explicit_comm_state
                )
                if has_state_for_session:
                    n_workers = dask_worker._explicit_comm_state[session_id]["nworkers"]
                else:
                    n_workers = None
                return {
                    "has_state": has_state,
                    "has_state_for_session": has_state_for_session,
                    "n_workers": n_workers,
                }

            result_1 = client.run(check, session_id=context_1.sessionId)
            expected_values = {
                "has_state": True,
                "has_state_for_session": True,
                "n_workers": 2,
            }
            expected_1 = {
                k: expected_values for k in client.scheduler_info()["workers"]
            }
            assert result_1 == expected_1

            # Run a shuffle with the initial setup as a sanity test
            with dask.config.set(explicit_comms=True):
                shuffled = ddf.shuffle(on="key", npartitions=4)
                assert any(name in str(key) for key in shuffled.dask)
                result = shuffled.compute()

            with dask.config.set(explicit_comms=False):
                shuffled = ddf.shuffle(on="key", npartitions=4)
                expected = shuffled.compute()

            assert_eq(result, expected)

            # --- Scale the cluster ---
            cluster.scale(3)
            client.wait_for_workers(3, timeout=5)

            context_2 = comms.default_comms()
            result_2 = client.run(check, session_id=context_2.sessionId)
            expected_values = {
                "has_state": True,
                "has_state_for_session": True,
                "n_workers": 3,
            }
            expected_2 = {
                k: expected_values for k in client.scheduler_info()["workers"]
            }
            assert result_2 == expected_2

            # Run a shuffle with the new setup
            with dask.config.set(explicit_comms=True):
                shuffled = ddf.shuffle(on="key", npartitions=4)
                assert any(name in str(key) for key in shuffled.dask)
                result = shuffled.compute()

            with dask.config.set(explicit_comms=False):
                shuffled = ddf.shuffle(on="key", npartitions=4)
                expected = shuffled.compute()

            assert_eq(result, expected)


def test_contains_shuffle_expr():
    df = dd.from_pandas(pd.DataFrame({"key": np.arange(10)}), npartitions=2)
    assert not _contains_shuffle_expr(df)

    with dask.config.set(explicit_comms=True):
        shuffled = df.shuffle(on="key")

        assert _contains_shuffle_expr(shuffled)
        assert not _contains_shuffle_expr(df)

        # this requires an active client.
        with LocalCluster(n_workers=1) as cluster:
            with Client(cluster):
                explict_shuffled = explicit_comms_shuffle(df, ["key"])
                assert not _contains_shuffle_expr(explict_shuffled)
