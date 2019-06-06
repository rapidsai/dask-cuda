import psutil
import pytest
from distributed.utils_test import gen_test

from dask.distributed import Client
from dask_cuda import DGX


@pytest.mark.skipif(
    'ib0' not in psutil.net_if_addrs(),
    reason='Infiniband interface ib0 not found'
)
@gen_test(timeout=20)
async def test_dgx():
    async with DGX(asynchronous=True) as cluster:
        async with Client(cluster, asynchronous=True):
            pass
