#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -eou pipefail

source rapids-init-pip

RAPIDS_CUDA_MAJOR="${RAPIDS_CUDA_VERSION%%.*}"
export RAPIDS_CUDA_MAJOR

DASK_CUDA_WHEELHOUSE=$(rapids-download-from-github "$(rapids-package-name "wheel_python" "dask-cuda" --pure)")

# generate constraints (possibly pinning to oldest supported versions of dependencies)
rapids-generate-pip-constraints py_test "${PIP_CONSTRAINT}"

rapids-logger "Installing test dependencies"

# notes:
#
#   * echo to expand wildcard before adding `[test]` requires for pip
#   * just providing --constraint="${PIP_CONSTRAINT}" to be explicit, and because
#     that environment variable is ignored if any other --constraint are passed via the CLI
#
rapids-pip-retry install \
  -v \
  --prefer-binary \
  --constraint "${PIP_CONSTRAINT}" \
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
