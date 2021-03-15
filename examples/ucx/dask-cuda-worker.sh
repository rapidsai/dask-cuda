#!/bin/bash

CLUSTER_TYPE=$1

# set up environment variables
DASK_UCX__CUDA_COPY=True
DASK_UCX__TCP=True
DASK_RMM__POOL_SIZE=1GB
WORKER_FLAGS=(
    "--enable-tcp-over-ucx"
    "--rmm-pool-size=${DASK_RMM__POOL_SIZE}"
    )

if [ $CLUSTER_TYPE == "NVLINK" ]; then
    DASK_UCX__NVLINK=True
    INTERFACE=enp1s0f0
    WORKER_FLAGS+=("--enable-nvlink")
elif [ $CLUSTER_TYPE == "IB" ]; then
    DASK_UCX__INFINIBAND=True
    DASK_UCX__RDMACM=True
    DASK_UCX__NET_DEVICES=mlx5_0:1
    INTERFACE=ib0
    WORKER_FLAGS+=(
        "--enable-infiniband"
        "--enable-rdmacm"
        "--net-devices=auto"
        )
elif [ $CLUSTER_TYPE == "NVLINK_IB" ]; then
    DASK_UCX__NVLINK=True
    DASK_UCX__INFINIBAND=True
    DASK_UCX__RDMACM=True
    DASK_UCX__NET_DEVICES=mlx5_0:1
    INTERFACE=ib0
    WORKER_FLAGS+=(
        "--enable-nvlink"
        "--enable-infiniband"
        "--enable-rdmacm"
        "--net-devices=auto"
        )
fi

# initialize scheduler
dask-scheduler --protocol ucx --interface $INTERFACE &

# initialize workers
dask-cuda-worker ucx://10.33.225.162:8786 ${WORKER_FLAGS[*]}
