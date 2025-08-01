#!/bin/bash
# Copyright (c) 2022-2025, NVIDIA CORPORATION.

set -euo pipefail

source rapids-date-string

rapids-print-env

rapids-generate-version > ./VERSION
RAPIDS_PACKAGE_VERSION=$(head -1 ./VERSION)
export RAPIDS_PACKAGE_VERSION

# populates `RATTLER_CHANNELS` array and `RATTLER_ARGS` array
source rapids-rattler-channel-string

rapids-logger "Building dask-cuda"

RBB_CHANNEL="$(rapids-get-pr-conda-artifact rapidsai/rapids-build-backend 73 python)"

rattler-build build --recipe conda/recipes/dask-cuda \
                    "${RATTLER_ARGS[@]}" \
                    "${RATTLER_CHANNELS[@]}" \
                    --channel "$RBB_CHANNEL"

# remove build_cache directory to avoid uploading the entire source tree
# tracked in https://github.com/prefix-dev/rattler-build/issues/1424
rm -rf "$RAPIDS_CONDA_BLD_OUTPUT_DIR"/build_cache
