# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import sys
import types

import pytest

import dask_cuda.cuda_worker as cuda_worker_module
from dask_cuda.cuda_worker import CUDAWorker


@pytest.fixture
def captured(monkeypatch):
    """Stub out everything that needs a GPU or a running process."""
    calls = []

    def fake_worker_plugins(**kwargs):
        calls.append(kwargs)
        return ()

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(cuda_worker_module, "worker_plugins", fake_worker_plugins)
    monkeypatch.setattr(cuda_worker_module, "Nanny", lambda *a, **kw: object())
    monkeypatch.setattr(
        cuda_worker_module, "worker_data_function", lambda **kw: lambda i: {}
    )
    monkeypatch.setattr(cuda_worker_module, "nvml_device_index", lambda i, v: 0)
    monkeypatch.setattr(cuda_worker_module, "initialize", lambda **kw: None)
    monkeypatch.setitem(sys.modules, "rmm", types.ModuleType("rmm"))
    return calls


def _make(**kwargs):
    return CUDAWorker(scheduler="tcp://127.0.0.1:8786", **kwargs)


def test_external_lib_list_string_is_split(captured):
    _make(rmm_allocator_external_lib_list="torch, cupy")
    assert captured[0]["rmm_allocator_external_lib_list"] == ["torch", "cupy"]


def test_external_lib_list_list_is_kept(captured):
    _make(rmm_allocator_external_lib_list=["torch", "cupy"])
    assert captured[0]["rmm_allocator_external_lib_list"] == ["torch", "cupy"]


def test_external_lib_list_invalid_type(captured):
    with pytest.raises(ValueError, match="rmm_allocator_external_lib_list"):
        _make(rmm_allocator_external_lib_list=("torch",))
