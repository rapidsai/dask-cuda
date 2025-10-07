#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

set -eou pipefail

source rapids-init-pip

RAPIDS_CUDA_MAJOR="${RAPIDS_CUDA_VERSION%%.*}"
export RAPIDS_CUDA_MAJOR

DASK_CUDA_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="dask-cuda" RAPIDS_PY_WHEEL_PURE="1" rapids-download-wheels-from-github python)

# Install cuda-suffixed dependencies b/c while `dask-cuda` has no cuda suffix, the test dependencies do
rapids-dependency-file-generator \
    --output requirements \
    --file-key "test_python" \
    --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION};cuda_suffixed=true" \
| tee /tmp/requirements-test.txt

rapids-logger "Installing test dependencies"
# echo to expand wildcard
rapids-pip-retry install \
  -v \
  --prefer-binary \
  -r /tmp/requirements-test.txt \
  "$(echo "${DASK_CUDA_WHEELHOUSE}"/dask_cuda*.whl)[cu${RAPIDS_CUDA_MAJOR}]"

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}
mkdir -p "${RAPIDS_TESTS_DIR}"

EXITCODE=0
# shellcheck disable=SC2317
set_exit_code() {
    EXITCODE=$?
    rapids-logger "Test failed with error ${EXITCODE}"
}
trap set_exit_code ERR
set +e

rapids-logger "pytest dask-cuda"
./ci/run_pytest.sh \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-dask-cuda.xml"

rapids-logger "Run local benchmark"
./ci/run_benchmarks.sh

rapids-logger "Test script exiting with latest error code: $EXITCODE"
exit ${EXITCODE}
