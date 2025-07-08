#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

set -eou pipefail

source rapids-init-pip

DASK_CUDA_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="dask-cuda" RAPIDS_PY_WHEEL_PURE="1" rapids-download-wheels-from-github python)

# Install cuda-suffixed dependencies b/c while `dask-cuda` has no cuda suffix, the test dependencies do
rapids-dependency-file-generator \
    --output requirements \
    --file-key "test_python" \
    --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION};cuda_suffixed=true" \
| tee /tmp/requirements-test.txt

rapids-logger "Installing test dependencies"
# echo to expand wildcard
rapids-pip-retry install -v --prefer-binary -r /tmp/requirements-test.txt "$(echo "${DASK_CUDA_WHEELHOUSE}"/dask_cuda*.whl)"

rapids-logger "pytest dask-cuda"
./ci/run_pytest.sh \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-dask-cuda.xml" \
  -k "not ucxx"

rapids-logger "Run local benchmark"
./ci/run_benchmarks.sh

# Run rapids-dask-dependency tests without `distributed-ucxx`, ensuring the protocol
# selection mechanism works also on "legacy" environments where only `ucx-py` is
# installed.
# TODO: remove as part of https://github.com/rapidsai/dask-cuda/issues/1517
distributed_ucxx_package_name="$(pip list | grep distributed-ucxx | awk '{print $1}')"
pip uninstall "${distributed_ucxx_package_name}"
./ci/run_pytest.sh \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-dask-cuda-rdd-protocol-selection.xml" \
  -k "test_rdd_protocol"
