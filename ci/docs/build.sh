#!/bin/bash
# Copyright (c) 2021, NVIDIA CORPORATION.
######################################
# Dask-CUDA Docs build script for CI #
######################################

if [ -z "$PROJECT_WORKSPACE" ]; then
    echo ">>>> ERROR: Could not detect PROJECT_WORKSPACE in environment"
    echo ">>>> WARNING: This script contains git commands meant for automated building, do not run locally"
    exit 1
fi

export DOCS_WORKSPACE=$WORKSPACE/docs
export PATH=/conda/bin:/usr/local/cuda/bin:$PATH
export HOME=$WORKSPACE
export PROJECT_WORKSPACE=/rapids/dask-cuda
export PROJECTS=(dask-cuda)

gpuci_logger "Check environment..."
env

gpuci_logger "Check GPU usage..."
nvidia-smi

gpuci_logger "Activate conda env..."
. /opt/conda/etc/profile.d/conda.sh
conda activate rapids

gpuci_logger "Check versions..."
python --version
$CC --version
$CXX --version
conda info
conda config --show-sources
conda list --show-channel-urls

# Dask-CUDA Sphinx build
gpuci_logger "Build Dask-CUDA docs..."
cd $PROJECT_WORKSPACE/docs
make html

# commit to website
cd $DOCS_WORKSPACE

if [ ! -d "api/dask-cuda/$BRANCH_VERSION" ]; then
    mkdir -p api/dask-cuda/$BRANCH_VERSION
fi
rm -rf $DOCS_WORKSPACE/api/dask-cuda/$BRANCH_VERSION/*

mv $PROJECT_WORKSPACE/docs/build/html/* $DOCS_WORKSPACE/api/dask-cuda/$BRANCH_VERSION
