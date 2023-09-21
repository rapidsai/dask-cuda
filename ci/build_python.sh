#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION.

set -euo pipefail

source rapids-env-update

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-logger "Begin py build"

rapids-conda-retry mambabuild \
  conda/recipes/dask-cuda

rapids-upload-conda-to-s3 python
