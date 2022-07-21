import os

import pytest


@pytest.fixture(autouse=True)
def revert_cuda_visible_devices(request):
    def _revert():
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]

    request.addfinalizer(_revert)
