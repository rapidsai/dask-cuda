#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

source rapids-date-string

rapids-generate-version > ./VERSION

rapids-pip-retry wheel . -w dist -v --no-deps --disable-pip-version-check
./ci/validate_wheel.sh dist

RAPIDS_PY_WHEEL_NAME="dask-cuda" RAPIDS_PY_WHEEL_PURE="1" rapids-upload-wheels-to-s3 dist
