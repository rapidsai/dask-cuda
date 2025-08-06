# SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import patch

import pynvml
import pytest
from numba import cuda

from dask.config import canonical_name

from dask_cuda.utils import (
    cuda_visible_devices,
    get_cpu_affinity,
    get_device_total_memory,
    get_gpu_count,
    get_n_gpus,
    get_preload_options,
    get_ucx_config,
    has_device_memory_resource,
    nvml_device_index,
    parse_cuda_visible_device,
    parse_device_memory_limit,
    unpack_bitmask,
)
from dask_cuda.utils_test import get_ucx_implementation


@patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1,2"})
def test_get_n_gpus():
    assert isinstance(get_n_gpus(), int)

    assert get_n_gpus() == 3


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


def test_unpack_bitmask_single_value():
    with pytest.raises(TypeError):
        unpack_bitmask(1)


def test_cpu_affinity():
    for i in range(get_n_gpus()):
        affinity = get_cpu_affinity(i)
        os.sched_setaffinity(0, affinity)
        assert os.sched_getaffinity(0) == set(affinity)


def test_cpu_affinity_and_cuda_visible_devices():
    affinity = dict()
    for i in range(get_n_gpus()):
        # The negative here would be `device = 0` as required for CUDA runtime
        # calls.
        device = nvml_device_index(0, cuda_visible_devices(i))
        affinity[device] = get_cpu_affinity(device)

    for i in range(get_n_gpus()):
        assert get_cpu_affinity(i) == affinity[i]


def test_get_device_total_memory():
    for i in range(get_n_gpus()):
        with cuda.gpus[i]:
            total_mem = get_device_total_memory(i)
            if has_device_memory_resource():
                assert type(total_mem) is int
                assert total_mem > 0
            else:
                assert total_mem is None


@pytest.mark.parametrize(
    "protocol",
    ["ucx", "ucx-old"],
)
def test_get_preload_options_default(protocol):
    get_ucx_implementation(protocol)

    opts = get_preload_options(
        protocol=protocol,
        create_cuda_context=True,
    )

    assert "preload" in opts
    assert opts["preload"] == ["dask_cuda.initialize"]
    assert "preload_argv" in opts
    assert opts["preload_argv"] == ["--create-cuda-context"]


@pytest.mark.parametrize(
    "protocol",
    ["ucx", "ucx-old"],
)
@pytest.mark.parametrize("enable_tcp", [True, False])
@pytest.mark.parametrize("enable_infiniband", [True, False])
@pytest.mark.parametrize("enable_nvlink", [True, False])
def test_get_preload_options(protocol, enable_tcp, enable_infiniband, enable_nvlink):
    get_ucx_implementation(protocol)

    opts = get_preload_options(
        protocol=protocol,
        create_cuda_context=True,
        enable_tcp_over_ucx=enable_tcp,
        enable_infiniband=enable_infiniband,
        enable_nvlink=enable_nvlink,
    )

    assert "preload" in opts
    assert opts["preload"] == ["dask_cuda.initialize"]
    assert "preload_argv" in opts
    assert "--create-cuda-context" in opts["preload_argv"]

    if enable_tcp:
        assert "--enable-tcp-over-ucx" in opts["preload_argv"]
    if enable_infiniband:
        assert "--enable-infiniband" in opts["preload_argv"]
    if enable_nvlink:
        assert "--enable-nvlink" in opts["preload_argv"]


@pytest.mark.parametrize(
    "protocol",
    ["ucx", "ucx-old"],
)
@pytest.mark.parametrize("enable_tcp_over_ucx", [True, False, None])
@pytest.mark.parametrize("enable_nvlink", [True, False, None])
@pytest.mark.parametrize("enable_infiniband", [True, False, None])
def test_get_ucx_config(
    protocol, enable_tcp_over_ucx, enable_infiniband, enable_nvlink
):
    get_ucx_implementation(protocol)

    kwargs = {
        "enable_tcp_over_ucx": enable_tcp_over_ucx,
        "enable_infiniband": enable_infiniband,
        "enable_nvlink": enable_nvlink,
    }
    ucx_config = get_ucx_config(**kwargs)

    assert ucx_config[canonical_name("create_cuda_context", ucx_config)] is True

    if enable_tcp_over_ucx is not None:
        assert ucx_config[canonical_name("tcp", ucx_config)] is enable_tcp_over_ucx
    else:
        if (
            enable_infiniband is not True
            and enable_nvlink is not True
            and not (enable_infiniband is None and enable_nvlink is None)
        ):
            assert ucx_config[canonical_name("tcp", ucx_config)] is True
        else:
            assert ucx_config[canonical_name("tcp", ucx_config)] is None

    if enable_infiniband is not None:
        assert ucx_config[canonical_name("infiniband", ucx_config)] is enable_infiniband
    else:
        if (
            enable_tcp_over_ucx is not True
            and enable_nvlink is not True
            and not (enable_tcp_over_ucx is None and enable_nvlink is None)
        ):
            assert ucx_config[canonical_name("infiniband", ucx_config)] is True
        else:
            assert ucx_config[canonical_name("infiniband", ucx_config)] is None

    if enable_nvlink is not None:
        assert ucx_config[canonical_name("nvlink", ucx_config)] is enable_nvlink
    else:
        if (
            enable_tcp_over_ucx is not True
            and enable_infiniband is not True
            and not (enable_tcp_over_ucx is None and enable_infiniband is None)
        ):
            assert ucx_config[canonical_name("nvlink", ucx_config)] is True
        else:
            assert ucx_config[canonical_name("nvlink", ucx_config)] is None

    if any(
        opt is not None
        for opt in [enable_tcp_over_ucx, enable_infiniband, enable_nvlink]
    ) and not all(
        opt is False for opt in [enable_tcp_over_ucx, enable_infiniband, enable_nvlink]
    ):
        assert ucx_config[canonical_name("cuda-copy", ucx_config)] is True
    else:
        assert ucx_config[canonical_name("cuda-copy", ucx_config)] is None


def test_parse_visible_devices():
    pynvml.nvmlInit()
    indices = []
    uuids = []
    for index in range(get_gpu_count()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        try:
            uuid = pynvml.nvmlDeviceGetUUID(handle).decode("utf-8")
        except AttributeError:
            uuid = pynvml.nvmlDeviceGetUUID(handle)

        assert parse_cuda_visible_device(index) == index
        assert parse_cuda_visible_device(uuid) == uuid

        indices.append(str(index))
        uuids.append(uuid)

    index_devices = ",".join(indices)
    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": index_devices}):
        for index in range(get_gpu_count()):
            visible = cuda_visible_devices(index)
            assert visible.split(",")[0] == str(index)

    uuid_devices = ",".join(uuids)
    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": uuid_devices}):
        for index in range(get_gpu_count()):
            visible = cuda_visible_devices(index)
            assert visible.split(",")[0] == str(uuids[index])

    with pytest.raises(ValueError):
        parse_cuda_visible_device("Foo")

    with pytest.raises(TypeError):
        parse_cuda_visible_device(None)
        parse_cuda_visible_device([])


def test_parse_device_bytes():
    total = get_device_total_memory(0)

    assert parse_device_memory_limit(None) is None
    assert parse_device_memory_limit(0) is None
    assert parse_device_memory_limit("0") is None
    assert parse_device_memory_limit("0.0") is None
    assert parse_device_memory_limit("0 GiB") is None

    assert parse_device_memory_limit(1) == 1
    assert parse_device_memory_limit("1") == 1

    assert parse_device_memory_limit(1000000000) == 1000000000
    assert parse_device_memory_limit("1GB") == 1000000000

    if has_device_memory_resource(0):
        assert parse_device_memory_limit(1.0) == total
        assert parse_device_memory_limit("1.0") == total

        assert parse_device_memory_limit(0.8) == int(total * 0.8)
        assert parse_device_memory_limit(0.8, alignment_size=256) == int(
            total * 0.8 // 256 * 256
        )

        assert parse_device_memory_limit("default") == parse_device_memory_limit(0.8)
    else:
        assert parse_device_memory_limit("default") is None

        with pytest.raises(ValueError):
            assert parse_device_memory_limit(1.0) == total
        with pytest.raises(ValueError):
            assert parse_device_memory_limit("1.0") == total
        with pytest.raises(ValueError):
            assert parse_device_memory_limit(0.8) == int(total * 0.8)
        with pytest.raises(ValueError):
            assert parse_device_memory_limit(0.8, alignment_size=256) == int(
                total * 0.8 // 256 * 256
            )


def test_parse_device_memory_limit():
    total = get_device_total_memory(0)

    assert parse_device_memory_limit(None) is None
    assert parse_device_memory_limit(0) is None
    assert parse_device_memory_limit("0") is None
    assert parse_device_memory_limit(0.0) is None
    assert parse_device_memory_limit("0 GiB") is None

    assert parse_device_memory_limit(1) == 1
    assert parse_device_memory_limit("1") == 1

    assert parse_device_memory_limit("auto") == total

    assert parse_device_memory_limit(1000000000) == 1000000000
    assert parse_device_memory_limit("1GB") == 1000000000

    if has_device_memory_resource(0):
        assert parse_device_memory_limit(1.0) == total
        assert parse_device_memory_limit("1.0") == total

        assert parse_device_memory_limit(0.8) == int(total * 0.8)
        assert parse_device_memory_limit(0.8, alignment_size=256) == int(
            total * 0.8 // 256 * 256
        )
        assert parse_device_memory_limit("default") == parse_device_memory_limit(0.8)
    else:
        assert parse_device_memory_limit("default") is None

        with pytest.raises(ValueError):
            assert parse_device_memory_limit(1.0) == total
        with pytest.raises(ValueError):
            assert parse_device_memory_limit("1.0") == total
        with pytest.raises(ValueError):
            assert parse_device_memory_limit(0.8) == int(total * 0.8)
        with pytest.raises(ValueError):
            assert parse_device_memory_limit(0.8, alignment_size=256) == int(
                total * 0.8 // 256 * 256
            )


def test_has_device_memory_resoure():
    has_memory_resource = has_device_memory_resource()
    total = get_device_total_memory(0)

    if has_memory_resource:
        # Tested only in devices with a memory resource
        assert total == parse_device_memory_limit("auto")
    else:
        # Tested only in devices without a memory resource
        assert total is None


def test_parse_visible_mig_devices():
    pynvml.nvmlInit()
    for index in range(get_gpu_count()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        try:
            mode = pynvml.nvmlDeviceGetMigMode(handle)[0]
        except pynvml.NVMLError:
            # if not a MIG device, i.e. a normal GPU, skip
            continue
        if mode:
            # Just checks to see if there are any MIG enabled GPUS.
            # If there is one, check if the number of mig instances
            # in that GPU is <= to count, where count gives us the
            # maximum number of MIG devices/instances that can exist
            # under a given parent NVML device.
            count = pynvml.nvmlDeviceGetMaxMigDeviceCount(handle)
            miguuids = []
            for i in range(count):
                try:
                    mighandle = pynvml.nvmlDeviceGetMigDeviceHandleByIndex(
                        device=handle, index=i
                    )
                    miguuids.append(mighandle)
                except pynvml.NVMLError:
                    pass
            assert len(miguuids) <= count
