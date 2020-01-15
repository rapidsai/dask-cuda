import multiprocessing as mp

from distributed import Client
from distributed.deploy.local import LocalCluster
from dask_cuda.explicit_comms import CommsContext, cudf_merge

import pandas as pd
import dask.dataframe as dd

import numpy as np
import pytest
import cudf
import cupy

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


def test_local_cluster():
    for protocol in ["tcp", "ucx"]:
        p = mp.Process(target=_test_local_cluster, args=(protocol,))
        p.start()
        p.join()
        assert not p.exitcode


def _test_cudf_merge(protocol):
    n_workers = 2
    with LocalCluster(
        protocol=protocol,
        dashboard_address=None,
        n_workers=n_workers,
        threads_per_worker=1,
        processes=True,
    ) as cluster:
        with Client(cluster) as client:
            comms = CommsContext(client)
            nrows = n_workers * 10
            df1 = pd.DataFrame({"key": np.arange(nrows), "payload1": np.arange(nrows)})
            df2 = pd.DataFrame({"key": np.arange(nrows), "payload2": np.arange(nrows)})

            ddf1 = dd.from_pandas(
                cudf.DataFrame.from_pandas(df1), npartitions=n_workers
            )
            ddf2 = dd.from_pandas(
                cudf.DataFrame.from_pandas(df2), npartitions=n_workers
            )
            ddf3 = cudf_merge(ddf1, ddf2).set_index("key")

            got = ddf3.compute().to_pandas()
            got.index.names = ["key"]  # TODO: this shouldn't be needed
            expected = df1.merge(df2).set_index("key")

            pd.testing.assert_frame_equal(got, expected)


def test_cudf_merge():
    for protocol in ["tcp", "ucx"]:
        p = mp.Process(target=_test_cudf_merge, args=(protocol,))
        p.start()
        p.join()
        assert not p.exitcode
