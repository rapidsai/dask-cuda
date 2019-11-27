import os
import multiprocessing as mp
import numpy
import pytest
import dask.array as da
import dask
from distributed.deploy.local import LocalCluster
from distributed import Client

mp = mp.get_context("spawn")
ucp = pytest.importorskip("ucp")

# Notice, all of the following tests is executed in a new process such
# that UCX options of the different tests doesn't conflict.
# Furthermore, all tests do some computation to trigger initialization
# of UCX before retrieving the current config.


def _test_global_option(seg_size):
    """Test setting UCX options through dask's global config"""
    dask.config.set({"ucx": {"SEG_SIZE": seg_size, "TLS": "tcp,sockcm,cuda_copy"}})

    with LocalCluster(
        protocol="ucx",
        dashboard_address=None,
        n_workers=1,
        threads_per_worker=1,
        processes=True,
    ) as cluster:
        with Client(cluster):
            res = da.from_array(numpy.arange(10000), chunks=(1000,))
            res = res.sum().compute()
            assert res == 49995000
            conf = ucp.get_config()
            assert conf["SEG_SIZE"] == seg_size


def test_global_option():
    for seg_size in ["2K", "1M", "2M"]:
        p = mp.Process(target=_test_global_option, args=(seg_size,))
        p.start()
        p.join()
        assert not p.exitcode
