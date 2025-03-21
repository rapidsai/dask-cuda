#!/bin/bash
# Copyright (c) 2022-2025, NVIDIA CORPORATION.

set -euo pipefail

rapids-configure-conda-channels

# Setting channel priority per-repo until all RAPIDS can build using strict channel priority
# This will be replaced when we port this recipe to `rattler-build`
conda config --set channel_priority strict
# `rapids-configure-conda-channels` should only insert `rapidsai` channel into release builds
conda config --remove channels rapidsai

source rapids-date-string

rapids-print-env

rapids-generate-version > ./VERSION

rapids-logger "Begin py build"
conda config --set path_conflict prevent

RAPIDS_PACKAGE_VERSION=$(head -1 ./VERSION) rapids-conda-retry build \
  conda/recipes/dask-cuda

rapids-upload-conda-to-s3 python
