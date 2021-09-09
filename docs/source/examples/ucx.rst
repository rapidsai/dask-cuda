Enabling UCX communication
==========================

A CUDA cluster using UCX communication can be started automatically with LocalCUDACluster or manually with the ``dask-cuda-worker`` CLI tool.
In either case, a ``dask.distributed.Client`` must be made for the worker cluster using the same Dask UCX configuration; see `UCX Integration -- Configuration <../ucx.html#configuration>`_ for details on all available options.

LocalCUDACluster
----------------

When using LocalCUDACluster with UCX communication, all required UCX configuration is handled through arguments supplied at construction; see `API -- Cluster <../api.html#cluster>`_ for a complete list of these arguments.
To connect a client to a cluster with all supported transports and an RMM pool:

.. code-block:: python

    from dask.distributed import Client
    from dask_cuda import LocalCUDACluster

    cluster = LocalCUDACluster(
        protocol="ucx",
        interface="ib0",
        enable_tcp_over_ucx=True,
        enable_nvlink=True,
        enable_infiniband=True,
        enable_rdmacm=True,
        rmm_pool_size="1GB"
    )
    client = Client(cluster)

.. note::
    For UCX 1.9 (deprecated) and older, it's necessary to pass ``ucx_net_devices="auto"`` to ``LocalCUDACluster``. UCX 1.11 and above is capable of selecting InfiniBand devices automatically.

dask-cuda-worker
----------------

When using ``dask-cuda-worker`` with communication, the scheduler, workers, and client must all be started manually, each using the same UCX configuration.

Scheduler
^^^^^^^^^

UCX configuration options will need to be specified for ``dask-scheduler`` as environment variables; see `Dask Configuration -- Environment Variables <https://docs.dask.org/en/latest/configuration.html#environment-variables>`_ for more details on the mapping between environment variables and options.

To start a Dask scheduler using UCX with all supported transports and an gigabyte RMM pool:

.. code-block:: bash

    $ DASK_DISTRIBUTED__COMM__UCX__CUDA_COPY=True \
    > DASK_DISTRIBUTED__COMM__UCX__TCP=True \
    > DASK_DISTRIBUTED__COMM__UCX__NVLINK=True \
    > DASK_DISTRIBUTED__COMM__UCX__INFINIBAND=True \
    > DASK_DISTRIBUTED__COMM__UCX__RDMACM=True \
    > DASK_DISTRIBUTED__RMM__POOL_SIZE=1GB \
    > dask-scheduler --protocol ucx --interface ib0

We communicate to the scheduler that we will be using UCX with the ``--protocol`` option, and that we will be using InfiniBand with the ``--interface`` option.

.. note::
    For UCX 1.9 (deprecated) and older it's also necessary to set ``DASK_DISTRIBUTED__COMM__UCX__NET_DEVICES=mlx5_0:1``, where ``"mlx5_0:1"`` is our UCX net device; because the scheduler does not rely upon Dask-CUDA, it cannot automatically detect InfiniBand interfaces, so we must specify one explicitly. UCX 1.11 and above is capable of selecting InfiniBand devices automatically.

Workers
^^^^^^^

All UCX configuration options have analogous options in ``dask-cuda-worker``; see `API -- Worker <../api.html#worker>`_ for a complete list of these options.
To start a cluster with all supported transports and an RMM pool:

.. code-block:: bash

    $ dask-cuda-worker ucx://<scheduler_address>:8786 \
    > --enable-tcp-over-ucx \
    > --enable-nvlink \
    > --enable-infiniband \
    > --enable-rdmacm \
    > --rmm-pool-size="1GB"

.. note::
    For UCX 1.9 (deprecated) and older it's also necessary to set ``--net-devices="auto"``. UCX 1.11 and above is capable of selecting InfiniBand devices automatically.

Client
^^^^^^

A client can be configured to use UCX by using ``dask_cuda.initialize``, a utility which takes the same UCX configuring arguments as LocalCUDACluster and adds them to the current Dask configuration used when creating it; see `API -- Client initialization <../api.html#client-initialization>`_ for a complete list of arguments.
To connect a client to the cluster we have made:

.. code-block:: python

    from dask.distributed import Client
    from dask_cuda.initialize import initialize

    initialize(
        enable_tcp_over_ucx=True,
        enable_nvlink=True,
        enable_infiniband=True,
        enable_rdmacm=True,
    )
    client = Client("ucx://<scheduler_address>:8786")

.. note::
    For UCX 1.9 (deprecated) and older it's also necessary to set ``net_devices="mlx5_0:1"``, where ``"mlx5_0:1"`` is our UCX net device; because the client does not rely upon Dask-CUDA, it cannot automatically detect InfiniBand interfaces, so we must specify one explicitly. UCX 1.11 and above is capable of selecting InfiniBand devices automatically.
