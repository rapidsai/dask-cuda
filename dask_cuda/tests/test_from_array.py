# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import pytest

import dask.array as da
from distributed import Client

from dask_cuda import LocalCUDACluster

cupy = pytest.importorskip("cupy")


@pytest.mark.parametrize("protocol", ["ucx", "tcp"])
def test_ucx_from_array(protocol):
    N = 10_000
    with LocalCUDACluster(protocol=protocol) as cluster:
        with Client(cluster):
            val = da.from_array(cupy.arange(N), chunks=(N // 10,)).sum().compute()
            assert val == (N * (N - 1)) // 2
