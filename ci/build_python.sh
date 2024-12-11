#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION.

set -euo pipefail

rapids-configure-conda-channels

source rapids-date-string

rapids-print-env

rapids-generate-version > ./VERSION

rapids-logger "Begin py build"
conda config --set path_conflict prevent

RAPIDS_PACKAGE_VERSION=$(head -1 ./VERSION) rapids-conda-retry mambabuild \
  conda/recipes/dask-cuda

rapids-upload-conda-to-s3 python
