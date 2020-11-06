#!/usr/bin/env bash
# Copyright (c) 2020, NVIDIA CORPORATION.
################################################################################
# dask-cuda cpu build
################################################################################
set -e

# Logger function for build status output
function gpuci_logger() {
  echo -e "\n>>>> $@\n"
}

# Set path and build parallel level
export PATH=/opt/conda/bin:/usr/local/cuda/bin:$PATH

# Set home to the job's workspace
export HOME=$WORKSPACE

# Switch to project root; also root of repo checkout
cd $WORKSPACE

# Get latest tag and number of commits since tag
export GIT_DESCRIBE_TAG=`git describe --abbrev=0 --tags`
export GIT_DESCRIBE_NUMBER=`git rev-list ${GIT_DESCRIBE_TAG}..HEAD --count`

# Setup 'gpuci_conda_retry' for build retries (results in 2 total attempts)
export GPUCI_CONDA_RETRY_MAX=1
export GPUCI_CONDA_RETRY_SLEEP=30

################################################################################
# SETUP - Check environment
################################################################################

gpuci_logger "Get env"
env

gpuci_logger "Activate conda env"
. /opt/conda/etc/profile.d/conda.sh
conda activate rapids

gpuci_logger "Check versions"
python --version
$CC --version
$CXX --version
conda info
conda config --show-sources
conda list --show-channel-urls

# FIX Added to deal with Anancoda SSL verification issues during conda builds
conda config --set ssl_verify False

pip install git+https://github.com/dask/distributed.git@master

# If nightly build, append current YYMMDD to version
if [[ "$BUILD_MODE" = "branch" && "$SOURCE_BRANCH" = branch-* ]] ; then
  export VERSION_SUFFIX=`date +%y%m%d`
fi

################################################################################
# BUILD - Package builds
################################################################################

gpuci_conda_retry build conda/recipes/dask-cuda --python=${PYTHON}

rm -rf dist/
python setup.py sdist bdist_wheel

################################################################################
# UPLOAD - Packages
################################################################################

gpuci_logger "Upload conda pkg"
source ci/cpu/upload-anaconda.sh

gpuci_logger "Upload pypi pkg"
source ci/cpu/upload-pypi.sh