#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -eou pipefail

RAPIDS_INIT_PIP_REMOVE_NVIDIA_INDEX="true"
export RAPIDS_INIT_PIP_REMOVE_NVIDIA_INDEX
source rapids-init-pip

RAPIDS_CUDA_MAJOR="${RAPIDS_CUDA_VERSION%%.*}"
export RAPIDS_CUDA_MAJOR

DASK_CUDA_WHEELHOUSE=$(rapids-download-from-github "$(rapids-package-name "wheel_python" "dask-cuda" --pure)")

rapids-logger "Installing test dependencies"
# echo to expand wildcard
rapids-pip-retry install \
  -v \
  --prefer-binary \
  "$(echo "${DASK_CUDA_WHEELHOUSE}"/dask_cuda*.whl)[cu${RAPIDS_CUDA_MAJOR},test,test-cu${RAPIDS_CUDA_MAJOR}]"

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}
mkdir -p "${RAPIDS_TESTS_DIR}"

EXITCODE=0
# shellcheck disable=SC2329
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
