import pytest

import dask.array as da
from distributed import Client

from dask_cuda import LocalCUDACluster

cupy = pytest.importorskip("cupy")


@pytest.mark.parametrize("protocol", ["ucx", "ucxx", "tcp"])
def test_ucx_from_array(protocol):
    if protocol == "ucx":
        pytest.importorskip("ucp")
    elif protocol == "ucxx":
        pytest.importorskip("ucxx")

    N = 10_000
    with LocalCUDACluster(protocol=protocol) as cluster:
        with Client(cluster):
            val = da.from_array(cupy.arange(N), chunks=(N // 10,)).sum().compute()
            assert val == (N * (N - 1)) // 2
