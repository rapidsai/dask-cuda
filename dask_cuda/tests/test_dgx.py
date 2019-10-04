import pytest

import psutil
import os

from distributed.utils_test import gen_test
from dask.distributed import Client

from dask_cuda import DGX
from dask_cuda.utils import get_ucx_env


@pytest.mark.skipif(
    "ib0" not in psutil.net_if_addrs(), reason="Infiniband interface ib0 not found"
)
@gen_test(timeout=20)
async def test_dgx():
    async with DGX(
        enable_infiniband=False, enable_nvlink=False, asynchronous=True
    ) as cluster:
        async with Client(cluster, asynchronous=True):
            pass


@pytest.mark.skipif(
    "ib0" not in psutil.net_if_addrs(), reason="Infiniband interface ib0 not found"
)
@gen_test(timeout=20)
async def test_dgx_tcp_over_ucx():
    ucp = pytest.importorskip("ucp")
    with pytest.warns(UserWarning):
        ucx_env = get_ucx_env(enable_infiniband=True, enable_nvlink=True)
        os.environ.update(ucx_env)

        async with DGX(
            protocol="ucx",
            enable_tcp_over_ucx=True,
            asynchronous=True,
        ) as cluster:
            async with Client(cluster, asynchronous=True):
                pass


@pytest.mark.skipif(
    "ib0" not in psutil.net_if_addrs(), reason="Infiniband interface ib0 not found"
)
@gen_test(timeout=20)
async def test_dgx_ucx():
    ucp = pytest.importorskip("ucp")
    with pytest.warns(UserWarning):
        ucx_env = get_ucx_env(enable_infiniband=True, enable_nvlink=True)
        os.environ.update(ucx_env)

        async with DGX(
            protocol="ucx",
            enable_infiniband=True,
            enable_nvlink=True,
            asynchronous=True,
        ) as cluster:
            async with Client(cluster, asynchronous=True):
                pass
