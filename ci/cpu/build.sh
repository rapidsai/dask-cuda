#!/usr/bin/env bash
# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

# Whether to keep `dask/label/dev` channel in the env. If INSTALL_DASK_MAIN=0,
# `dask/label/dev` channel is removed.
export INSTALL_DASK_MAIN=0

# Dask version to install when `INSTALL_DASK_MAIN=0`
export DASK_STABLE_VERSION="2023.1.1"

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

# Remove `rapidsai-nightly` & `dask/label/dev` channel if we are building main branch
if [ "$SOURCE_BRANCH" = "main" ]; then
  conda config --system --remove channels rapidsai-nightly
  conda config --system --remove channels dask/label/dev
elif [[ "${INSTALL_DASK_MAIN}" == 0 ]]; then
# Remove `dask/label/dev` channel if INSTALL_DASK_MAIN=0
  conda config --system --remove channels dask/label/dev
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

# Install latest nightly version for dask and distributed if needed
if [[ "${INSTALL_DASK_MAIN}" == 1 ]]; then
  gpuci_logger "Installing dask and distributed from dask nightly channel"
  gpuci_mamba_retry install -c dask/label/dev \
    "dask/label/dev::dask" \
    "dask/label/dev::distributed"
else
  gpuci_logger "gpuci_mamba_retry install conda-forge::dask==${DASK_STABLE_VERSION} conda-forge::distributed==${DASK_STABLE_VERSION} conda-forge::dask-core==${DASK_STABLE_VERSION} --force-reinstall"
  gpuci_mamba_retry install conda-forge::dask==${DASK_STABLE_VERSION} conda-forge::distributed==${DASK_STABLE_VERSION} conda-forge::dask-core==${DASK_STABLE_VERSION} --force-reinstall
fi


################################################################################
# BUILD - Package builds
################################################################################

# FIXME: Move boa install to gpuci/rapidsai
gpuci_mamba_retry install -c conda-forge boa

gpuci_logger "Build conda pkg for dask-cuda"
gpuci_conda_retry mambabuild conda/recipes/dask-cuda --python=${PYTHON}

rm -rf dist/
python setup.py sdist bdist_wheel

################################################################################
# UPLOAD - Packages
################################################################################

gpuci_logger "Upload conda pkg..."
source ci/cpu/upload.sh

gpuci_logger "Upload pypi pkg..."
source ci/cpu/upload-pypi.sh
