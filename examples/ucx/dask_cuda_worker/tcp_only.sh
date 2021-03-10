#!/bin/bash

# Dask configuration
DASK_UCX__CUDA_COPY=True
DASK_UCX__TCP=True
DASK_UCX__NET_DEVICES=enp1s0f0
DASK_RMM__POOL_SIZE=1GB

# start up scheduler
dask-scheduler --protocol ucx --interface eth0

# start up workers
dask-cuda-worker ucx://<scheduler_address>:8786 \
    --enable-tcp-over-ucx \
    --rmm-pool-size="1GB"

# this cluster still needs to be connected to a client