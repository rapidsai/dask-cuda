#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

source rapids-configure-sccache
source rapids-date-string

package_name=dask-cuda
version=$(rapids-generate-version)
commit=$(git rev-parse HEAD)

echo "${version}" | tr -d '"' > VERSION
sed -i "/^__git_commit__/ s/= .*/= \"${commit}\"/g" "dask_cuda/_version.py"

python -m pip wheel . -w dist -vvv --no-deps --disable-pip-version-check

RAPIDS_PY_WHEEL_NAME="${package_name}" rapids-upload-wheels-to-s3 dist
