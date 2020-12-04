Worker
======

Dask-CUDA workers extend the standard Dask worker in two ways:

1) Advanced networking configuration
2) GPU Memory Pool configuration

These configurations can be defined in the single cluster use case with ``LocalCUDACluster`` or passed to workers on the cli with ``dask-cuda-worker``

Single Cluster configuration
----------------------------
Dask-CUDA can be configured for single machine clusters with multiple GPUs such as as DGX1 or DGX2.  Below is an example of configuring a single machine Dask cluster on a DGX2 with an RMM pool and NVLink enabled

.. code-block:: python

    from dask.distributed import Client
    from dask_cuda import LocalCUDACluster
    from dask_cuda.initialize import initialize

    # Configurations
    protocol = "ucx"
    interface = "enp6s0"    # DGX-2
    enable_tcp_over_ucx = True
    enable_nvlink = True
    enable_infiniband = False

    initialize(
        create_cuda_context=True,
        enable_tcp_over_ucx=enable_tcp_over_ucx,
        enable_infiniband=enable_infiniband,
        enable_nvlink=enable_nvlink,
    )

    cluster = LocalCUDACluster(local_directory="/tmp/USERNAME",
                            protocol=protocol,
                            interface=interface,
                            enable_tcp_over_ucx=enable_tcp_over_ucx,
                            enable_infiniband=enable_infiniband,
                            enable_nvlink=enable_nvlink,
                            rmm_pool_size="25GB",
                        )
    client = Client(cluster)


Command Line Tool
-----------------

New configuration options::

    --interface TEXT                The external interface used to connect to
                                    the scheduler, usually an ethernet interface
                                    is used for connection, and not an
                                    InfiniBand interface (if one is available).

    --device-memory-limit TEXT      Bytes of memory per CUDA device that the
                                    worker can use. This can be an integer
                                    (bytes), float (fraction of total device
                                    memory), string (like 5GB or 5000M), 'auto',
                                    or zero for no memory management (i.e.,
                                    allow full device memory usage).

    --rmm-pool-size TEXT            If specified, initialize each worker with an
                                    RMM pool of the given size, otherwise no RMM
                                    pool is created. This can be an integer
                                    (bytes) or string (like 5GB or 5000M).
                                    NOTE: This size is a per worker (i.e., per
                                    GPU) configuration, and not cluster-wide!

    --enable-tcp-over-ucx / --disable-tcp-over-ucx
                                    Enable TCP communication over UCX
    --enable-infiniband / --disable-infiniband
                                    Enable InfiniBand communication
    --enable-nvlink / --disable-nvlink
                                    Enable NVLink communication
    --net-devices TEXT              When None (default), 'UCX_NET_DEVICES' will
                                    be left to its default. Otherwise, it must
                                    be a non-empty string with the interface
                                    name. Normally used only with --enable-
                                    infiniband to specify the interface to be
                                    used by the worker, such as 'mlx5_0:1' or
                                    'ib0'.
