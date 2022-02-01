#!/bin/bash
# Copyright (c) 2020, NVIDIA CORPORATION.
##############################################
# dask-cuda GPU build and test script for CI #
##############################################
set -e
NUMARGS=$#
ARGS=$*

# Arg parsing function
function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

# Set path and build parallel level
export PATH=/opt/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=${PARALLEL_LEVEL:-4}
export CUDA_REL=${CUDA_VERSION%.*}
export CUDA_REL2=${CUDA//./}

# Set home to the job's workspace
export HOME="$WORKSPACE"

# Parse git describe
cd "$WORKSPACE"
export GIT_DESCRIBE_TAG=`git describe --tags`
export MINOR_VERSION=`echo $GIT_DESCRIBE_TAG | grep -o -E '([0-9]+\.[0-9]+)'`
export UCX_PATH=$CONDA_PREFIX
export UCXPY_VERSION=0.26.*
unset GIT_DESCRIBE_TAG

# Enable NumPy's __array_function__ protocol (needed for NumPy 1.16.x,
# will possibly be enabled by default starting on 1.17)
export NUMPY_EXPERIMENTAL_ARRAY_FUNCTION=1

# Install dask and distributed from main branch. Usually needed during
# development time and disabled before a new dask-cuda release.
export INSTALL_DASK_MAIN=1

################################################################################
# SETUP - Check environment
################################################################################

gpuci_logger "Check environment"
env

gpuci_logger "Check GPU usage"
nvidia-smi

gpuci_logger "Activate conda env"
. /opt/conda/etc/profile.d/conda.sh
conda activate rapids

conda info
conda config --show-sources
conda list --show-channel-urls

# Installing cucim in order to test GDS spilling
# Pin pytest-asyncio because latest versions modify the default asyncio
# `event_loop_policy`. See https://github.com/dask/distributed/pull/4212 .
gpuci_mamba_retry install "cudf=${MINOR_VERSION}" \
              "dask-cudf=${MINOR_VERSION}" \
              "ucx-py=${UCXPY_VERSION}" \
              "ucx-proc=*=gpu" \
              "cucim" \
              "pytest-asyncio=<0.14.0"

# Install latest nightly version for dask and distributed if needed
if [[ "${INSTALL_DASK_MAIN}" == 1 ]]; then
  gpuci_logger "Installing dask and distributed from dask nightly channel"
  gpuci_mamba_retry install -c dask/label/dev \
    "dask/label/dev::dask" \
    "dask/label/dev::distributed"
fi

gpuci_logger "Check versions"
python --version
$CC --version
$CXX --version

conda info
conda config --show-sources
conda list --show-channel-urls

################################################################################
# BUILD - Build dask-cuda
################################################################################

# TODO: Move boa install to gpuci/rapidsai
gpuci_mamba_retry install boa

gpuci_logger "Build and install dask-cuda"
cd "${WORKSPACE}"
CONDA_BLD_DIR="${WORKSPACE}/.conda-bld"
gpuci_conda_retry mambabuild --croot "${CONDA_BLD_DIR}" conda/recipes/dask-cuda --python="${PYTHON}"
gpuci_mamba_retry install -c "${CONDA_BLD_DIR}" dask-cuda

################################################################################
# TEST - Run pytests for ucx-py
################################################################################

if hasArg --skip-tests; then
    gpuci_logger "Skipping Tests"
else
    gpuci_logger "Python pytest for dask-cuda"
    cd "$WORKSPACE"
    ls dask_cuda/tests/
    DASK_CUDA_TEST_SINGLE_GPU=1 UCXPY_IFNAME=eth0 UCX_WARN_UNUSED_ENV_VARS=n UCX_MEMTYPE_CACHE=n pytest -vs --cache-clear --basetemp="$WORKSPACE/dask-cuda-tmp" --junitxml="$WORKSPACE/junit-dask-cuda.xml" --cov-config=.coveragerc --cov=dask_cuda --cov-report=xml:"$WORKSPACE/dask-cuda-coverage.xml" --cov-report term dask_cuda/tests/

    logger "Run local benchmark..."
    python dask_cuda/benchmarks/local_cudf_shuffle.py --partition-size="1 KiB" -d 0  --runs 1 --backend dask
    python dask_cuda/benchmarks/local_cudf_shuffle.py --partition-size="1 KiB" -d 0  --runs 1 --backend explicit-comms
fi

if [ -n "${CODECOV_TOKEN}" ]; then
    codecov -t $CODECOV_TOKEN
fi
