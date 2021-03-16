#!/bin/bash

usage() {
    echo "usage: $0 [-a <scheduler_address>] [-i <interface>] [-r <rmm_pool_size>] [-t <transports>]" >&2
    exit 1
    } 

# parse arguments
address=localhost
rmm_pool_size=1GB

while getopts ":a:i:r:t:" flag; do
    case "${flag}" in
        a) address=${OPTARG};;
        i) interface=${OPTARG};;
        r) rmm_pool_size=${OPTARG};;
        t) transport=${OPTARG};;
        *) usage;;
    esac
done

if [ -z ${interface+x} ] && ! [ -z ${transport+x} ]; then
    echo "$0: interface must be specified with -i if using NVLink or InfiniBand"
    exit 1
fi

# set up environment variables/flags
DASK_UCX__CUDA_COPY=True
DASK_UCX__TCP=True
DASK_RMM__POOL_SIZE=$rmm_pool_size

scheduler_flags="--protocol ucx"
worker_flags="--enable-tcp-over-ucx --rmm-pool-size ${rmm_pool_size}"

if ! [ -z ${interface+x} ]; then
    scheduler_flags+=" --interface ${interface}"
fi
if [[ $transport == *"nvlink"* ]]; then
    DASK_UCX__NVLINK=True

    worker_flags+=" --enable-nvlink"
fi
if [[ $transport == *"ib"* ]]; then
    DASK_UCX__INFINIBAND=True
    DASK_UCX__RDMACM=True
    DASK_UCX__NET_DEVICES=mlx5_0:1

    worker_flags+=" --enable-infiniband --enable-rdmacm --net-devices=auto"
fi

# initialize scheduler
dask-scheduler $scheduler_flags &

# initialize workers
dask-cuda-worker ucx://${address}:8786 $worker_flags
