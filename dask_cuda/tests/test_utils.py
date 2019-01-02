import os

from dask_cuda.utils import get_n_gpus

def test_get_n_gpus():
    assert isinstance(get_n_gpus(), int)

    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
        assert get_n_gpus() == 3
    finally:
        del os.environ['CUDA_VISIBLE_DEVICES']
