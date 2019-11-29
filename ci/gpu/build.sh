#!/bin/bash
# Copyright (c) 2018, NVIDIA CORPORATION.
##############################################
# dask-cuda GPU build and test script for CI #
##############################################
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
export PARALLEL_LEVEL=4
export CUDA_REL=${CUDA_VERSION%.*}
export CUDA_REL2=${CUDA//./}

# Set home to the job's workspace
export HOME=$WORKSPACE

# Parse git describe
cd $WORKSPACE
export GIT_DESCRIBE_TAG=`git describe --tags`
export MINOR_VERSION=`echo $GIT_DESCRIBE_TAG | grep -o -E '([0-9]+\.[0-9]+)'`
export UCX_PATH=$CONDA_PREFIX

# Enable NumPy's __array_function__ protocol (needed for NumPy 1.16.x,
# will possibly be enabled by default starting on 1.17)
export NUMPY_EXPERIMENTAL_ARRAY_FUNCTION=1

################################################################################
# SETUP - Check environment
################################################################################

logger "Check environment..."
env

logger "Check GPU usage..."
nvidia-smi

logger "Activate conda env..."
source activate gdf
conda list

# Fixing Numpy version to avoid RuntimeWarning: numpy.ufunc size changed, may
# indicate binary incompatibility. Expected 192 from C header, got 216 from PyObject
conda install "cudatoolkit=$CUDA_REL" \
              "cupy>=6.5.0" "numpy=1.16.4" \
              "cudf=${MINOR_VERSION}" "dask-cudf=${MINOR_VERSION}" \
              "dask>=2.3.0" "distributed>=2.3.2"

# needed for asynccontextmanager in py36
conda install -c conda-forge "async_generator" "automake" "libtool" \
                              "cmake" "automake" "autoconf" "cython" \
                              "pytest" "pkg-config" "pytest-asyncio"
conda list

# Install the master version of dask and distributed
logger "pip install git+https://github.com/dask/distributed.git --upgrade"
pip install "git+https://github.com/dask/distributed.git" --upgrade
logger "pip install git+https://github.com/dask/dask.git --upgrade"
pip install "git+https://github.com/dask/dask.git" --upgrade

logger "Check versions..."
python --version
$CC --version
$CXX --version
conda list

################################################################################
# BUILD - Build ucx
################################################################################

logger "Build ucx"
git clone https://github.com/openucx/ucx
cd ucx
git checkout v1.7.x
ls
./autogen.sh
mkdir build
cd build
../configure --prefix=$CONDA_PREFIX --enable-debug --with-cuda=$CUDA_HOME --enable-mt CPPFLAGS="-I//$CUDA_HOME/include"
make -j install
cd $WORKSPACE


################################################################################
# Installing ucx-py
################################################################################

logger "pip install git+https://github.com/rapidsai/ucx-py.git --upgrade"
pip install "git+https://github.com/rapidsai/ucx-py.git" --upgrade


################################################################################
# BUILD - Build dask-cuda
################################################################################

logger "Build dask-cuda..."
cd $WORKSPACE
python -m pip install -e .


################################################################################
# TEST - Run py.tests for ucx-py
################################################################################

if hasArg --skip-tests; then
    logger "Skipping Tests..."
else
    logger "Python py.test for dask-cuda..."
    cd $WORKSPACE
    ls dask_cuda/tests/
    UCXPY_IFNAME=eth0 UCX_WARN_UNUSED_ENV_VARS=n UCX_MEMTYPE_CACHE=n py.test -vs --cache-clear --junitxml=${WORKSPACE}/junit-dask-cuda.xml --cov-config=.coveragerc --cov=dask_cuda --cov-report=xml:${WORKSPACE}/dask-cuda-coverage.xml --cov-report term dask_cuda/tests/
fi
