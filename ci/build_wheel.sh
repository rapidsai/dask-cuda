#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.

set -euo pipefail

wheel_dir=${RAPIDS_WHEEL_BLD_OUTPUT_DIR}

source rapids-date-string

rapids-generate-version > ./VERSION

rapids-pip-retry wheel . -w "${wheel_dir}" -v --no-deps --disable-pip-version-check
./ci/validate_wheel.sh "${wheel_dir}"

RAPIDS_PY_WHEEL_NAME="dask-cuda" RAPIDS_PY_WHEEL_PURE="1" rapids-upload-wheels-to-s3 python "${wheel_dir}"
