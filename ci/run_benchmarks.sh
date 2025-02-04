#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

set -euo pipefail

# Support invoking run_benchmarks.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../dask_cuda

rapids-logger "Run local benchmark"
python benchmarks/local_cudf_shuffle.py \
  --partition-size="1 KiB" \
  -d 0 \
  --runs 1 \
  --backend dask

python benchmarks/local_cudf_shuffle.py \
  --partition-size="1 KiB" \
  -d 0 \
  --runs 1 \
  --backend explicit-comms

python benchmarks/local_cudf_shuffle.py \
  --disable-rmm \
  --partition-size="1 KiB" \
  -d 0 \
  --runs 1 \
  --backend explicit-comms

python benchmarks/local_cudf_shuffle.py \
  --disable-rmm-pool \
  --partition-size="1 KiB" \
  -d 0 \
  --runs 1 \
  --backend explicit-comms

python benchmarks/local_cudf_shuffle.py \
  --rmm-pool-size 2GiB \
  --partition-size="1 KiB" \
  -d 0 \
  --runs 1 \
  --backend explicit-comms

python benchmarks/local_cudf_shuffle.py \
  --rmm-pool-size 2GiB \
  --rmm-maximum-pool-size 4GiB \
  --partition-size="1 KiB" \
  -d 0 \
  --runs 1 \
  --backend explicit-comms

python benchmarks/local_cudf_shuffle.py \
  --rmm-pool-size 2GiB \
  --rmm-maximum-pool-size 4GiB \
  --enable-rmm-async \
  --partition-size="1 KiB" \
  -d 0 \
  --runs 1 \
  --backend explicit-comms

python benchmarks/local_cudf_shuffle.py \
  --rmm-pool-size 2GiB \
  --rmm-maximum-pool-size 4GiB \
  --enable-rmm-managed \
  --partition-size="1 KiB" \
  -d 0 \
  --runs 1 \
  --backend explicit-comms
