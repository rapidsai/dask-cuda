import multiprocessing as mp

import numpy as np
import pandas as pd
import pytest

import dask
from dask import dataframe as dd
from dask.dataframe.shuffle import partitioning_index
from distributed import Client
from distributed.deploy.local import LocalCluster

import cudf
from cudf.tests.utils import assert_eq

from dask_cuda.explicit_comms import (
    CommsContext,
    dataframe_merge,
    dataframe_shuffle,
)

mp = mp.get_context("spawn")
ucp = pytest.importorskip("ucp")

# Notice, all of the following tests is executed in a new process such
# that UCX options of the different tests doesn't conflict.


async def my_rank(state):
    return state["rank"]


def _test_local_cluster(protocol):
    with LocalCluster(
        protocol=protocol,
        dashboard_address=None,
        n_workers=4,
        threads_per_worker=1,
        processes=True,
    ) as cluster:
        with Client(cluster) as client:
            comms = CommsContext(client)
            assert sum(comms.run(my_rank)) == sum(range(4))


@pytest.mark.parametrize("protocol", ["tcp", "ucx"])
def test_local_cluster(protocol):
    p = mp.Process(target=_test_local_cluster, args=(protocol,))
    p.start()
    p.join()
    assert not p.exitcode


def _test_dataframe_merge(backend, protocol, n_workers):
    dask.config.update(
        dask.config.global_config,
        {
            "ucx": {
                "TLS": "tcp,sockcm,cuda_copy",
            },
        },
        priority="new",
    )

    with LocalCluster(
        protocol=protocol,
        dashboard_address=None,
        n_workers=n_workers,
        threads_per_worker=1,
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
            expected = df1.merge(df2).set_index("key")

            if backend == "cudf":
                df1 = cudf.DataFrame.from_pandas(df1)
                df2 = cudf.DataFrame.from_pandas(df2)

            ddf1 = dd.from_pandas(df1, npartitions=n_workers + 1)
            ddf2 = dd.from_pandas(
                df2, npartitions=n_workers - 1 if n_workers > 1 else 1
            )
            ddf3 = dataframe_merge(ddf1, ddf2, on="key").set_index("key")
            got = ddf3.compute()

            if backend == "cudf":
                assert_eq(got, expected)

            else:
                pd.testing.assert_frame_equal(got, expected)


@pytest.mark.parametrize("nworkers", [1, 2, 4])
@pytest.mark.parametrize("backend", ["pandas", "cudf"])
@pytest.mark.parametrize("protocol", ["tcp", "ucx"])
def test_dataframe_merge(backend, protocol, nworkers):
    p = mp.Process(target=_test_dataframe_merge, args=(backend, protocol, nworkers))
    p.start()
    p.join()
    assert not p.exitcode


def _test_dataframe_merge_empty_partitions(nrows, npartitions):
    with LocalCluster(
        protocol="tcp",
        dashboard_address=None,
        n_workers=npartitions,
        threads_per_worker=1,
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
            ddf3 = dataframe_merge(ddf1, ddf2, on="key").set_index("key")
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
    hashes = partitioning_index(df, npartitions)
    if len(hashes) > 0:
        return len(hashes.unique()) == 1
    else:
        return True


def _test_dataframe_shuffle(backend, protocol, n_workers):
    dask.config.update(
        dask.config.global_config,
        {
            "ucx": {
                "TLS": "tcp,sockcm,cuda_copy",
            },
        },
        priority="new",
    )

    with LocalCluster(
        protocol=protocol,
        dashboard_address=None,
        n_workers=n_workers,
        threads_per_worker=1,
        processes=True,
    ) as cluster:
        with Client(cluster):
            nrows_per_worker = 5
            np.random.seed(42)
            df = pd.DataFrame({"key": np.random.random(n_workers * nrows_per_worker)})
            if backend == "cudf":
                df = cudf.DataFrame.from_pandas(df)
            ddf = dd.from_pandas(df, npartitions=n_workers)

            ddf = dataframe_shuffle(ddf, "key")

            # Check that each partition of `ddf` hashes to the same value
            result = ddf.map_partitions(check_partitions, n_workers).compute()
            assert all(result.to_list())


@pytest.mark.parametrize("nworkers", [1, 2, 4])
@pytest.mark.parametrize("backend", ["pandas", "cudf"])
@pytest.mark.parametrize("protocol", ["tcp", "ucx"])
def test_dataframe_shuffle(backend, protocol, nworkers):
    p = mp.Process(target=_test_dataframe_shuffle, args=(backend, protocol, nworkers))
    p.start()
    p.join()
    assert not p.exitcode
