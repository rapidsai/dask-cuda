#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

set -eou pipefail

RAPIDS_PY_WHEEL_NAME="dask-cuda" rapids-download-wheels-from-s3 python ./dist

# echo to expand wildcard before adding `[extra]` requires for pip
python -m pip install $(echo ./dist/dask_cuda*.whl)[test]

python -m pytest -n 8 ./python/dask_cuda/tests
