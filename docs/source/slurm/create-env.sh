#!/bin/bash

set -x

#module load cuda/11.0.3
module load cuda/10.2.89.0
export PATH=/gpfs/fs1/bzaitlen/miniconda3/bin:$PATH
source /gpfs/fs1/bzaitlen/miniconda3/bin/activate
ENV=`date +"%Y%m%d-nightly-0.17"`

mamba create -n $ENV -c rapidsai-nightly -c nvidia -c conda-forge \
    automake make libtool pkg-config cudatoolkit=10.2 \
    libhwloc psutil python=3.8 setuptools cython \
    cudf=0.17 dask-cudf ipython ipdb pygithub --yes --quiet


conda activate $ENV
git clone http://github.com/quasiben/dask-cuda /tmp/dask-cuda
cd /tmp/dask-cuda
git checkout more-rmm-options
python -m pip install .
cd -
git clone https://github.com/openucx/ucx /tmp/ucx
cd /tmp/ucx
git checkout v1.8.x
git clean -fdx
# apply UCX IB registration cache patches, improves overall
# CUDA IB performance when using a memory pool
curl -LO https://raw.githubusercontent.com/rapidsai/ucx-split-feedstock/master/recipe/add-page-alignment.patch
curl -LO https://raw.githubusercontent.com/rapidsai/ucx-split-feedstock/master/recipe/ib_registration_cache.patch
git apply ib_registration_cache.patch && git apply add-page-alignment.patch
./autogen.sh
mkdir -p build
cd build
ls $CUDA_HOME
../contrib/configure-release \
    --prefix="${CONDA_PREFIX}" \
    --enable-cma \
    --enable-mt \
    --enable-numa \
    --with-gnu-ld \
    --with-cm \
    --with-rdmacm \
    --with-verbs \
    --with-rc \
    --with-ud \
    --with-dc \
    --with-dm \
    --with-cuda="${CUDA_HOME}"
make -j install
cd -
git clone https://github.com/rapidsai/ucx-py.git /tmp/ucx-py
cd /tmp/ucx-py
python -m pip install .

