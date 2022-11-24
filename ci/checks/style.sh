#!/bin/bash
# Copyright (c) 2020, NVIDIA CORPORATION.
################################################################################
# dask-cuda Style Tester
################################################################################

# Ignore errors and set path
set +e
PATH=/opt/conda/bin:$PATH

# Activate common conda env
. /opt/conda/etc/profile.d/conda.sh
conda activate rapids

# Run pre-commit checks
pre-commit run --hook-stage manual --all-files
