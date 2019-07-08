#!/usr/bin/env bash
set -e
NUMARGS=$#
ARGS=$*

# Logger function for build status output
function logger() {
  echo -e "\n>>>> $@\n"
}

# Arg parsing function
function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

# Set path and build parallel level
export PATH=/conda/bin:/usr/local/cuda/bin:$PATH
export CUDA_REL=${CUDA//./}

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

# FIX Added to deal with Anancoda SSL verification issues during conda builds
conda config --set ssl_verify False

conda install \
    'cudf=0.8' \
    'dask-cudf=0.8'

pip install git+https://github.com/dask/distributed.git@master

################################################################################
# SETUP - Install additional packages
################################################################################

if hasArg --skip-tests; then
    logger "Skipping Tests..."
else
    # Install CuPy for tests
    pip install cupy-cuda${CUDA_REL}==6.0.0

    ################################################################################
    # TEST - Run tests
    ################################################################################

    pip install -e .
    pip install pytest pytest-asyncio
    py.test --cache-clear --junitxml=${WORKSPACE}/junit-dask-cuda.xml -v --cov-config=.coveragerc --cov=dask_cuda --cov-report=xml:${WORKSPACE}/dask-cuda-coverage.xml --cov-report term


    conda install codecov
    codecov -t $CODECOV_TOKEN
fi