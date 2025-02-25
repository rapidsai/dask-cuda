#!/bin/bash
# Copyright (c) 2022-2025, NVIDIA CORPORATION.

set -euo pipefail

rapids-configure-conda-channels

source rapids-date-string

rapids-print-env

rapids-generate-version > ./VERSION

rapids-logger "Begin py build"
conda config --set path_conflict prevent

RAPIDS_PACKAGE_VERSION=$(head -1 ./VERSION) rapids-conda-retry build \
  conda/recipes/dask-cuda

rapids-upload-conda-to-s3 python
