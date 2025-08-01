#!/bin/bash
# Copyright (c) 2020-2025, NVIDIA CORPORATION.

set -euo pipefail

rapids-logger "Create checks conda environment"
. /opt/conda/etc/profile.d/conda.sh

rapids-dependency-file-generator \
  --output conda \
  --file-key checks \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

mkdir /tmp/rbb
pushd /tmp/rbb
wget -O ../rbb.zip https://github.com/rapidsai/rapids-build-backend/actions/runs/16683467298/artifacts/3670230978
unzip ../rbb.zip
popd

rapids-mamba-retry env create --yes -f env.yaml -n checks
conda activate checks

# Run pre-commit checks
pre-commit run --hook-stage manual --all-files --show-diff-on-failure
