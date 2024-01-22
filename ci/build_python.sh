#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION.

set -euo pipefail

rapids-configure-conda-channels

source rapids-configure-sccache

source rapids-date-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

package_name="dask_cuda"

version=$(rapids-generate-version)
commit=$(git rev-parse HEAD)

echo "${version}" | tr -d '"' > VERSION
sed -i "/^__git_commit__/ s/= .*/= \"${commit}\"/g" "${package_name}/_version.py"

rapids-logger "Begin py build"

RAPIDS_PACKAGE_VERSION=${version} rapids-conda-retry mambabuild \
  conda/recipes/dask-cuda

rapids-upload-conda-to-s3 python
