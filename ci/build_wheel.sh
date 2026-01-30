#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-date-string
RAPIDS_INIT_PIP_REMOVE_NVIDIA_INDEX="true"
export RAPIDS_INIT_PIP_REMOVE_NVIDIA_INDEX
source rapids-init-pip

rapids-generate-version > ./VERSION

RAPIDS_PIP_WHEEL_ARGS=(
  -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"
  -v
  --no-deps
  --disable-pip-version-check
)

# unset PIP_CONSTRAINT (set by rapids-init-pip)... it doesn't affect builds as of pip 25.3, and
# results in an error from 'pip wheel' when set and --build-constraint is also passed
unset PIP_CONSTRAINT
rapids-pip-retry wheel \
    "${RAPIDS_PIP_WHEEL_ARGS[@]}" \
    .

./ci/validate_wheel.sh "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"

RAPIDS_PACKAGE_NAME="$(rapids-package-name wheel_python dask-cuda --pure)"
export RAPIDS_PACKAGE_NAME
