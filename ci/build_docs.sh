#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

rapids-logger "Downloading artifacts from previous jobs"
PYTHON_CHANNEL=$(rapids-download-from-github "$(rapids-package-name "conda_python" dask-cuda --pure)")

rapids-logger "Create test conda environment"
. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

rapids-dependency-file-generator \
    --output conda \
    --file-key docs \
    --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" \
    --prepend-channel "${PYTHON_CHANNEL}" \
    | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n docs
conda activate docs

rapids-print-env

RAPIDS_DOCS_DIR="$(mktemp -d)"
export RAPIDS_DOCS_DIR

rapids-logger "Build Python docs"
pushd docs
sphinx-build -b dirhtml ./source _html
mkdir -p "${RAPIDS_DOCS_DIR}/dask-cuda/"html
mv _html/* "${RAPIDS_DOCS_DIR}/dask-cuda/html"
popd

RAPIDS_VERSION_NUMBER="$(rapids-version-major-minor)" rapids-upload-docs
