#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Support invoking run_benchmarks.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../dask_cuda

python benchmarks/local_cudf_shuffle.py \
  --partition-size="1 KiB" \
  -d 0 \
  --runs 1 \
  --backend dask \
  "$@"
