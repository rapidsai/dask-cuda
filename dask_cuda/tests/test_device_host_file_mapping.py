# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

from dask_cuda.device_host_file import DeviceHostFile
from dask_cuda.is_spillable_object import is_spillable_object


class HostValue:
    pass


class SpillableValue:
    pass


class DeviceValue:
    __cuda_array_interface__ = {
        "shape": (),
        "typestr": "<f8",
        "data": (0, False),
        "version": 3,
    }


@is_spillable_object.register(SpillableValue)
def is_spillable_test_value(_):
    return True


def _make_value(kind):
    if kind == "host":
        return HostValue()
    if kind == "others":
        return SpillableValue()
    if kind == "device":
        return DeviceValue()
    raise ValueError(kind)


def _assert_key_location(dhf, key, location):
    assert (key in dhf.others) is (location == "others")
    assert (key in dhf.device_buffer) is (location == "device")
    assert (key in dhf.host_buffer) is (location == "host")
    assert (key in dhf.device_keys) is (location == "device")


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


@pytest.mark.parametrize(
    "first_kind, second_kind",
    [
        ("others", "host"),
        ("others", "device"),
        ("host", "others"),
        ("device", "others"),
        ("device", "host"),
        ("host", "device"),
    ],
)
def test_device_host_file_overwrites_discard_previous_storage(
    tmp_path, first_kind, second_kind
):
    dhf = DeviceHostFile(
        device_memory_limit=None,
        memory_limit=None,
        worker_local_directory=tmp_path,
    )
    key = "x"
    first = _make_value(first_kind)
    second = _make_value(second_kind)

    dhf[key] = first
    assert dhf[key] is first
    _assert_key_location(dhf, key, first_kind)

    dhf[key] = second

    assert dhf[key] is second
    assert len(dhf) == 1
    assert list(dhf) == [key]
    _assert_key_location(dhf, key, second_kind)


@pytest.mark.parametrize("kind", ["host", "device", "others"])
def test_device_host_file_delete_discards_storage(tmp_path, kind):
    dhf = DeviceHostFile(
        device_memory_limit=None,
        memory_limit=None,
        worker_local_directory=tmp_path,
    )
    key = "x"

    dhf[key] = _make_value(kind)
    del dhf[key]

    assert len(dhf) == 0
    assert list(dhf) == []
    _assert_key_location(dhf, key, None)
    with pytest.raises(KeyError):
        dhf[key]
