import pytest

import psutil
import os

from distributed.utils_test import gen_test
from dask.distributed import Client

from dask_cuda import DGX


@pytest.mark.skipif(
    "ib0" not in psutil.net_if_addrs(), reason="Infiniband interface ib0 not found"
)
@gen_test(timeout=20)
async def test_dgx():
    async with DGX(enable_infiniband=False, enable_nvlink=False, asynchronous=True) as cluster:
        async with Client(cluster, asynchronous=True):
            pass


@pytest.mark.skipif(
    "ib0" not in psutil.net_if_addrs(), reason="Infiniband interface ib0 not found"
)
@gen_test(timeout=20)
async def test_dgx_ucx():
    ucp = pytest.importorskip("ucp")
    ucx_env = {
        "UCX_TLS": "rc,tcp,sockcm,cuda_copy,cuda_ipc",
        "UCX_SOCKADDR_TLS_PRIORITY": "sockcm",
    }
    os.environ.update(ucx_env)

    async with DGX(protocol="ucx", asynchronous=True) as cluster:
        async with Client(cluster, asynchronous=True):
            pass
