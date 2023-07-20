#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Generate Python testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file_key test_python \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

rapids-mamba-retry env create --force -f env.yaml -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

# Possible cuda versions 11, 12
CONDA_CUDA_VERSION=$(echo ${RAPIDS_CUDA_VERSION} | cut -d. -f1)
rapids-logger "CONDA_VUDA_VERSION=${CONDA_CUDA_VERSION}"

# Don't change commit unless communicated.
COMMIT="c67e306"

LIBCUDF_CHANNEL_20=$(rapids-get-artifact ci/cudf/pull-request/13599/${COMMIT}/cudf_conda_cpp_cuda${CONDA_CUDA_VERSION}_$(arch).tar.gz)

CUDF_CHANNEL_20=$(rapids-get-artifact ci/cudf/pull-request/13599/${COMMIT}/cudf_conda_python_cuda${CONDA_CUDA_VERSION}_${RAPIDS_PY_VERSION//.}_$(arch).tar.gz)

rapids-logger $LIBCUDF_CHANNEL_20
rapids-logger $CUDF_CHANNEL_20

# Force remove packages
rapids-mamba-retry remove --force cudf libcudf dask-cudf pandas python-tzdata

# Install the removed packages from the custom artifact channels.
rapids-mamba-retry install \
  --channel "${CUDF_CHANNEL_20}" \
  --channel "${LIBCUDF_CHANNEL_20}" \
  --channel dask/label/dev \
  --channel conda-forge \
  cudf libcudf dask-cudf pandas==2.0.2 python-tzdata

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
timeout 40m pytest \
  -vv \
  --capture=no \
  --cache-clear \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-dask-cuda.xml" \
  --cov-config=../pyproject.toml \
  --cov=dask_cuda \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/dask-cuda-coverage.xml" \
  --cov-report=term \
  tests
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
