#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

source rapids-configure-sccache
source rapids-date-string

package_name=dask-cuda
version=$(rapids-generate-version)

sed -i "s/^version = .*/version = \"${version}\"/g" "pyproject.toml"

python -m pip wheel . -w dist -vvv --no-deps --disable-pip-version-check

RAPIDS_PY_WHEEL_NAME="${package_name}" rapids-upload-wheels-to-s3 dist
