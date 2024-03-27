#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Generate Python testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file_key test_python \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

rapids-logger "Downloading artifacts from previous jobs"
PYTHON_CHANNEL=$(rapids-download-conda-from-s3 python)

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}
RAPIDS_COVERAGE_DIR=${RAPIDS_COVERAGE_DIR:-"${PWD}/coverage-results"}
mkdir -p "${RAPIDS_TESTS_DIR}" "${RAPIDS_COVERAGE_DIR}"

rapids-print-env

rapids-mamba-retry install \
  --channel "${PYTHON_CHANNEL}" \
  dask-cuda

rapids-logger "Check GPU usage"
nvidia-smi

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "pytest dask-cuda"
pushd dask_cuda
DASK_CUDA_TEST_SINGLE_GPU=1 \
DASK_CUDA_WAIT_WORKERS_MIN_TIMEOUT=20 \
UCXPY_IFNAME=eth0 \
UCX_WARN_UNUSED_ENV_VARS=n \
UCX_MEMTYPE_CACHE=n \
timeout 60m pytest \
  -vv \
  --durations=0 \
  --capture=no \
  --cache-clear \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-dask-cuda.xml" \
  --cov-config=../pyproject.toml \
  --cov=dask_cuda \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/dask-cuda-coverage.xml" \
  --cov-report=term \
  tests -k "not ucxx"
popd

rapids-logger "Run local benchmark"
python dask_cuda/benchmarks/local_cudf_shuffle.py \
  --partition-size="1 KiB" \
  -d 0 \
  --runs 1 \
  --backend dask

python dask_cuda/benchmarks/local_cudf_shuffle.py \
  --partition-size="1 KiB" \
  -d 0 \
  --runs 1 \
  --backend explicit-comms

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
