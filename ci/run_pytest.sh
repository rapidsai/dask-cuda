#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

set -euo pipefail

# Support invoking run_pytest.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../dask_cuda

rapids-logger "pytest dask-cuda"
pushd dask_cuda
DASK_CUDA_TEST_SINGLE_GPU=1 \
  DASK_CUDA_WAIT_WORKERS_MIN_TIMEOUT=20 \
  UCXPY_IFNAME=eth0 \
  UCX_WARN_UNUSED_ENV_VARS=n \
  UCX_MEMTYPE_CACHE=n \
  timeout 90m pytest \
  -vv \
  --capture=no \
  --cache-clear \
  "$@" \
  -k "not ucxx" \
  tests
popd
