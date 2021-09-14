#!/usr/bin/env bash
# Copyright (c) 2019, NVIDIA CORPORATION.
################################################################################
# dask-cuda cpu build
################################################################################
set -e

# Set path and build parallel level
export PATH=/opt/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=${PARALLEL_LEVEL:-4}

# Set home to the job's workspace
export HOME="$WORKSPACE"

# Determine CUDA release version
export CUDA_REL=${CUDA_VERSION%.*}

# Setup 'gpuci_conda_retry' for build retries (results in 2 total attempts)
export GPUCI_CONDA_RETRY_MAX=1
export GPUCI_CONDA_RETRY_SLEEP=30

# Switch to project root; also root of repo checkout
cd "$WORKSPACE"

# While conda provides these during conda-build, they are also necessary during
# the setup.py build for PyPI
export GIT_DESCRIBE_TAG=`git describe --abbrev=0 --tags`
export GIT_DESCRIBE_NUMBER=`git rev-list ${GIT_DESCRIBE_TAG}..HEAD --count`

# If nightly build, append current YYMMDD to version
if [[ "$BUILD_MODE" = "branch" && "$SOURCE_BRANCH" = branch-* ]] ; then
  export VERSION_SUFFIX=`date +%y%m%d`
fi

################################################################################
# SETUP - Check environment
################################################################################

gpuci_logger "Check environment variables"
env

gpuci_logger "Activate conda env"
. /opt/conda/etc/profile.d/conda.sh
conda activate rapids

# Remove rapidsai-nightly channel if we are building main branch
if [ "$SOURCE_BRANCH" = "main" ]; then
  conda config --system --remove channels rapidsai-nightly
fi

gpuci_logger "Check compiler versions"
python --version
$CC --version
$CXX --version

gpuci_logger "Check conda environment"
conda info
conda config --show-sources
conda list --show-channel-urls

# FIX Added to deal with Anancoda SSL verification issues during conda builds
conda config --set ssl_verify False

pip install git+https://github.com/dask/dask.git@main
pip install git+https://github.com/dask/distributed.git@main

################################################################################
# BUILD - Package builds
################################################################################

gpuci_logger "Build conda pkg for libcudf"
gpuci_conda_retry build conda/recipes/dask-cuda --python=${PYTHON_VER}

rm -rf dist/
python setup.py sdist bdist_wheel

################################################################################
# UPLOAD - Packages
################################################################################

gpuci_logger "Upload conda pkg..."
source ci/cpu/upload.sh

gpuci_logger "Upload pypi pkg..."
source ci/cpu/upload-pypi.sh
