#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

source rapids-configure-sccache
source rapids-date-string

rapids-generate-version > ./VERSION

python -m pip wheel . -w dist -vvv --no-deps --disable-pip-version-check

RAPIDS_PY_WHEEL_NAME="dask-cuda" rapids-upload-wheels-to-s3 dist
