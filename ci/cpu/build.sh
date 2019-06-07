#!/usr/bin/env bash
# Copyright (c) 2019, NVIDIA CORPORATION.
################################################################################
# dask-cuda cpu build
################################################################################
set -e

# Logger function for build status output
function logger() {
  echo -e "\n>>>> $@\n"
}

# Set path and build parallel level
export PATH=/conda/bin:/usr/local/cuda/bin:$PATH

# Set home to the job's workspace
export HOME=$WORKSPACE

# Switch to project root; also root of repo checkout
cd $WORKSPACE

# Get latest tag and number of commits since tag
export GIT_DESCRIBE_TAG=`git describe --abbrev=0 --tags`
export GIT_DESCRIBE_NUMBER=`git rev-list ${GIT_DESCRIBE_TAG}..HEAD --count`

################################################################################
# SETUP - Check environment
################################################################################

logger "Get env..."
env

logger "Activate conda env..."
source activate gdf

logger "Check versions..."
python --version
gcc --version
g++ --version
conda list

# FIX Added to deal with Anancoda SSL verification issues during conda builds
conda config --set ssl_verify False

pip install git+https://github.com/dask/distributed.git@master

################################################################################
# BUILD - Conda package build
################################################################################

conda build conda/recipes/dask-cuda --python=${PYTHON}

################################################################################
# UPLOAD - Conda package
################################################################################

logger "Upload conda pkg..."
source ci/cpu/upload-anaconda.sh
