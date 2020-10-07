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

# Install dask and distributed from master branch. Usually needed during
# development time and disabled before a new dask-cuda release.
export INSTALL_DASK_MASTER=1

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
              "cudf=${MINOR_VERSION}" "dask-cudf=${MINOR_VERSION}" \
              "ucx-py=$MINOR_VERSION.*" "ucx-proc=*=gpu" \
              "rapids-build-env=$MINOR_VERSION.*"

# https://docs.rapids.ai/maintainers/depmgmt/ 
# conda remove -f rapids-build-env
# conda install "your-pkg=1.0.0"


conda list

# Install the master version of dask and distributed
if [[ "${INSTALL_DASK_MASTER}" == 1 ]]; then
    logger "pip install git+https://github.com/dask/distributed.git --upgrade"
    pip install "git+https://github.com/dask/distributed.git" --upgrade
    logger "pip install git+https://github.com/dask/dask.git --upgrade"
    pip install "git+https://github.com/dask/dask.git" --upgrade
fi

logger "Check versions..."
python --version
$CC --version
$CXX --version
conda list

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
    UCXPY_IFNAME=eth0 UCX_WARN_UNUSED_ENV_VARS=n UCX_MEMTYPE_CACHE=n py.test -vs --cache-clear --basetemp=${WORKSPACE}/dask-cuda-tmp --junitxml=${WORKSPACE}/junit-dask-cuda.xml --cov-config=.coveragerc --cov=dask_cuda --cov-report=xml:${WORKSPACE}/dask-cuda-coverage.xml --cov-report term dask_cuda/tests/

    logger "Running dask.distributed GPU tests"
    # Test downstream packages, which requires Python v3.7
    if [ $(python -c "import sys; print(sys.version_info[1])") -ge "7" ]; then
        logger "TEST OF DASK/UCX..."
        py.test --cache-clear -vs `python -c "import distributed.protocol.tests.test_cupy as m;print(m.__file__)"`
        py.test --cache-clear -vs `python -c "import distributed.protocol.tests.test_numba as m;print(m.__file__)"`
        py.test --cache-clear -vs `python -c "import distributed.protocol.tests.test_rmm as m;print(m.__file__)"`
        py.test --cache-clear -vs `python -c "import distributed.protocol.tests.test_collection_cuda as m;print(m.__file__)"`
        py.test --cache-clear -vs `python -c "import distributed.tests.test_nanny as m;print(m.__file__)"`
        py.test --cache-clear -vs `python -c "import distributed.diagnostics.tests.test_nvml as m;print(m.__file__)"`
    fi
fi
