# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import functools
import importlib.metadata

import packaging.version


@functools.lru_cache(maxsize=None)
def get_dask_version() -> packaging.version.Version:
    return packaging.version.parse(importlib.metadata.version("dask"))


@functools.lru_cache(maxsize=None)
def DASK_2025_4_0():
    # dask 2025.4.0 isn't currently released, so we're relying
    # on strictly greater than here.
    return get_dask_version() > packaging.version.parse("2025.3.0")


@functools.lru_cache(maxsize=None)
def get_cuda_core_version() -> packaging.version.Version:
    return packaging.version.parse(importlib.metadata.version("cuda-core"))


@functools.lru_cache(maxsize=None)
def CUDA_CORE_0_5_0():
    return get_cuda_core_version() > packaging.version.parse("0.5.0")
