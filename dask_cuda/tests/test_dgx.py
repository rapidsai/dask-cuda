import pytest

import psutil
import os

import dask.array as da

from distributed.utils_test import gen_test
from dask.distributed import Client

from dask_cuda import DGX
from dask_cuda.initialize import initialize
from dask_cuda.utils import get_ucx_env

cupy = pytest.importorskip("cupy")


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
    ucx_env = get_ucx_env(enable_tcp=True)
    os.environ.update(ucx_env)

    ucp = pytest.importorskip("ucp")

    async with DGX(
        protocol="ucx", enable_tcp_over_ucx=True, asynchronous=True
    ) as cluster:
        async with Client(cluster, asynchronous=True):
            pass


@pytest.mark.skipif(
    "ib0" not in psutil.net_if_addrs(), reason="Infiniband interface ib0 not found"
)
@pytest.mark.parametrize(
    "params",
    [
        {"enable_tcp": True, "enable_infiniband": False, "enable_nvlink": False},
        {"enable_tcp": True, "enable_infiniband": True, "enable_nvlink": True},
    ],
)
@pytest.mark.asyncio
async def test_dgx_ucx_infiniband_nvlink(params):
    ucp = pytest.importorskip("ucp")

    enable_tcp = params["enable_tcp"]
    enable_infiniband = params["enable_infiniband"]
    enable_nvlink = params["enable_nvlink"]

    initialize(
        create_cuda_context=True,
        enable_tcp_over_ucx=enable_tcp,
        enable_infiniband=enable_infiniband,
        enable_nvlink=enable_nvlink,
    )

    async with DGX(
        interface="enp1s0f0",
        protocol="ucx",
        enable_tcp_over_ucx=enable_tcp,
        enable_infiniband=enable_infiniband,
        enable_nvlink=enable_nvlink,
        asynchronous=True,
    ) as cluster:
        async with Client(cluster, asynchronous=True) as client:
            rs = da.random.RandomState(RandomState=cupy.random.RandomState)
            a = rs.normal(10, 1, (int(1e4), int(1e4)), chunks=(int(2.5e3), int(2.5e3)))
            x = a + a.T

            res = await client.compute(x)
