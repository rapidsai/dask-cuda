#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.

set -euo pipefail

rapids-logger "Downloading artifacts from previous jobs"
# rapids-download-conda-from-s3 doesn't have handling for noarch packages
# these are uncommon enough for RAPIDS that we handle this one as a special case
PYTHON_CHANNEL="/tmp/python_channel"
rapids-download-from-s3 "dask-cuda_conda_python_noarch.tar.gz" "${PYTHON_CHANNEL}"

rapids-logger "Create test conda environment"
. /opt/conda/etc/profile.d/conda.sh

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
