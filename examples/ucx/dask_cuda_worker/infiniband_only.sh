#!/bin/bash

# Dask configuration
DASK_UCX__CUDA_COPY=True
DASK_UCX__TCP=True
DASK_UCX__INFINIBAND=True
DASK_UCX__RDMACM=True
DASK_UCX__NET_DEVICES=mlx5_0:1
DASK_RMM__POOL_SIZE=1GB

# start up scheduler
dask-scheduler --protocol ucx --interface ib0

# start up workers
dask-cuda-worker ucx://<scheduler_address>:8786 \
    --enable-tcp-over-ucx \
    --enable-infiniband \
    --enable-rdmacm \
    --net-devices="auto" \
    --rmm-pool-size="1GB"

# this cluster still needs to be connected to a client