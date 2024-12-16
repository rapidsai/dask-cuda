#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -eou pipefail

RAPIDS_PY_VERSION="312" RAPIDS_PY_WHEEL_NAME="dask-cuda" rapids-download-wheels-from-s3 ./dist

# echo to expand wildcard before adding `[extra]` requires for pip
python -m pip install $(echo ./dist/dask_cuda*.whl)[test]

python -m pytest -n 8 ./python/dask_cuda/tests
