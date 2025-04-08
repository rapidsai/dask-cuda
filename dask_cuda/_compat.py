# Copyright (c) 2025 NVIDIA CORPORATION.

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
