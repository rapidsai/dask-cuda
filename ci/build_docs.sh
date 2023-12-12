#!/bin/bash

set -euo pipefail

rapids-logger "Create test conda environment"
. /opt/conda/etc/profile.d/conda.sh

rapids-dependency-file-generator \
    --output conda \
    --file_key docs \
    --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

rapids-mamba-retry env create --force -f env.yaml -n docs
conda activate docs

rapids-print-env

rapids-logger "Downloading artifacts from previous jobs"

PYTHON_CHANNEL=$(rapids-download-conda-from-s3 python)

rapids-mamba-retry install \
    --channel "${PYTHON_CHANNEL}" \
    dask-cuda

export RAPIDS_VERSION_NUMBER="24.02"
export RAPIDS_DOCS_DIR="$(mktemp -d)"

rapids-logger "Build Python docs"
pushd docs
sphinx-build -b dirhtml ./source _html
sphinx-build -b text ./source _text
mkdir -p "${RAPIDS_DOCS_DIR}/dask-cuda/"{html,txt}
mv _html/* "${RAPIDS_DOCS_DIR}/dask-cuda/html"
mv _text/* "${RAPIDS_DOCS_DIR}/dask-cuda/txt"
popd

rapids-upload-docs
