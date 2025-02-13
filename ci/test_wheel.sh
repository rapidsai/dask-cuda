#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

set -eou pipefail

RAPIDS_PY_WHEEL_NAME="dask-cuda" RAPIDS_PY_WHEEL_PURE="1" rapids-download-wheels-from-s3 python ./dist

# Install cuda-suffixed dependencies b/c while `dask-cuda` has no cuda suffix, the test dependencies do
rapids-dependency-file-generator \
    --output requirements \
    --file-key "test_python" \
    --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION};cuda_suffixed=true" \
| tee /tmp/requirements-test.txt

rapids-logger "Installing test dependencies"
# echo to expand wildcard
python -m pip install -v --prefer-binary -r /tmp/requirements-test.txt "$(echo ./dist/dask_cuda*.whl)"

rapids-logger "pytest dask-cuda"
./ci/run_pytest.sh \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-dask-cuda.xml"

rapids-logger "Run local benchmark"
./ci/run_benchmarks.sh
