import multiprocessing as mp

from distributed import Client
from distributed.deploy.local import LocalCluster
from dask_cuda.explicit_comms import CommsContext

import numpy
import pytest

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
