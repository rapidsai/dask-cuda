#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.

set -euo pipefail

source rapids-date-string
source rapids-init-pip

rapids-generate-version > ./VERSION

rapids-pip-retry wheel . -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" -v --no-deps --disable-pip-version-check
./ci/validate_wheel.sh "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"
