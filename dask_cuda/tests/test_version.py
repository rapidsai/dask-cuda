# Copyright (c) 2024, NVIDIA CORPORATION.

import dask_cuda


def test_version_constants_are_populated():
    # __git_commit__ will only be non-empty in a built distribution
    assert isinstance(dask_cuda.__git_commit__, str)

    # __version__ should always be non-empty
    assert isinstance(dask_cuda.__version__, str)
    assert len(dask_cuda.__version__) > 0
