# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

from dask_cuda.device_host_file import DeviceHostFile


@pytest.mark.parametrize("device_memory_limit", [None, 1])
def test_device_host_file_mapping_views_include_host_only_keys(
    tmp_path, device_memory_limit
):
    dhf = DeviceHostFile(
        device_memory_limit=device_memory_limit,
        memory_limit=None,
        worker_local_directory=tmp_path,
    )
    values = {
        "numpy": np.arange(3),
        "pandas": pd.DataFrame({"x": [1, 2, 3]}),
    }

    for key, value in values.items():
        dhf[key] = value

    assert len(dhf) == len(values)
    assert set(dhf) == set(values)
    assert list(dhf).count("numpy") == 1
    assert list(dhf).count("pandas") == 1
    np.testing.assert_array_equal(dhf["numpy"], values["numpy"])
    pd.testing.assert_frame_equal(dhf["pandas"], values["pandas"])
