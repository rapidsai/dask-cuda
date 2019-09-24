import pytest

import os

from dask_cuda.utils import get_cpu_affinity_list, get_n_gpus, unpack_bitmask


def test_get_n_gpus():
    assert isinstance(get_n_gpus(), int)

    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
        assert get_n_gpus() == 3
    finally:
        del os.environ["CUDA_VISIBLE_DEVICES"]


@pytest.mark.parametrize(
    "params",
    [
        {
            "input": [1152920405096267775, 0],
            "output": [i for i in range(20)] + [i + 40 for i in range(20)],
        },
        {
            "input": [17293823668613283840, 65535],
            "output": [i + 20 for i in range(20)] + [i + 60 for i in range(20)],
        },
        {"input": [18446744073709551615, 0], "output": [i for i in range(64)]},
        {"input": [0, 18446744073709551615], "output": [i + 64 for i in range(64)]},
    ],
)
def test_unpack_bitmask(params):
    assert unpack_bitmask(params["input"]) == params["output"]


def test_cpu_affinity():
    for i in range(get_n_gpus()):
        affinity = get_cpu_affinity_list(i)
        os.sched_setaffinity(0, affinity)
        assert list(os.sched_getaffinity(0)) == affinity
