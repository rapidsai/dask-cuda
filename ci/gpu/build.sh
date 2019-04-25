#!/usr/bin/env bash
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

# Enable NumPy's __array_function__ protocol (needed for NumPy 1.16.x,
# will possibly be enabled by default starting on 1.17)
export NUMPY_EXPERIMENTAL_ARRAY_FUNCTION=1

################################################################################
# SETUP - Install additional packages
################################################################################

# Use dask master until there's a release with
# https://github.com/dask/dask/pull/4715
pip install --upgrade git+https://github.com/dask/dask

# Use dask-distributed master until there's a release with
# https://github.com/dask/distributed/pull/2625
pip install --upgrade git+https://github.com/dask/distributed

# Install CuPy for tests
pip install cupy-cuda100==6.0.0rc1

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

################################################################################
# TEST - Run tests
################################################################################

pip install -e .
pip install pytest
pytest --cache-clear --junitxml=${WORKSPACE}/junit-libgdf.xml -v
