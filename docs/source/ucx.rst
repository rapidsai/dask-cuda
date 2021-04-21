UCX Integration
===============

Communication can be a major bottleneck in distributed systems.
Dask-CUDA addresses this by supporting integration with `UCX <https://www.openucx.org/>`_, an optimized communication framework that provides high-performance networking and supports a variety of transport methods, including `NVLink <https://www.nvidia.com/en-us/data-center/nvlink/>`_ and `Infiniband <https://www.mellanox.com/pdf/whitepapers/IB_Intro_WP_190.pdf>`_ for systems with specialized hardware, and TCP for systems without it.
This integration is enabled through `UCX-Py <https://ucx-py.readthedocs.io/>`_, an interface that provides Python bindings for UCX.


Requirements
------------

Hardware
^^^^^^^^

To use UCX with NVLink or InfiniBand, relevant GPUs must be connected with NVLink bridges or NVIDIA Mellanox InfiniBand Adapters, respectively.
NVIDIA provides comparison charts for both `NVLink bridges <https://www.nvidia.com/en-us/design-visualization/nvlink-bridges/>`_ and `InfiniBand adapters <https://www.nvidia.com/en-us/networking/infiniband-adapters/>`_.

Software
^^^^^^^^

UCX integration requires an environment with both UCX and UCX-Py installed; see `UCX-Py Installation <https://ucx-py.readthedocs.io/en/latest/install.html>`_ for detailed instructions on this process.

When using UCX, each NVLink and InfiniBand memory buffer must create a mapping between each unique pair of processes they are transferred across; this can be quite costly, potentially in the range of hundreds of milliseconds per mapping.
For this reason, it is strongly recommended to use `RAPIDS Memory Manager (RMM) <https://github.com/rapidsai/rmm>`_ to allocate a memory pool that is only prone to a single mapping operation, which all subsequent transfers may rely upon.
A memory pool also prevents the Dask scheduler from deserializing CUDA data, which will cause a crash.

Configuration
^^^^^^^^^^^^^

In addition to installations of UCX and UCX-Py on your system, several options must be specified within your Dask configuration to enable the integration.
Typically, these will affect ``UCX_TLS`` and ``UCX_SOCKADDR_TLS_PRIORITY``, environment variables used by UCX to decide what transport methods to use and which to prioritize, respectively.
However, some will affect related libraries, such as RMM:

- ``ucx.cuda_copy: true`` -- **required.**

  Adds ``cuda_copy`` to ``UCX_TLS``, enabling CUDA transfers over UCX.

- ``ucx.tcp: true`` -- **required.**

  Adds ``tcp`` to ``UCX_TLS``, enabling TCP transfers over UCX; this is required for very small transfers which are inefficient for NVLink and InfiniBand.

- ``ucx.nvlink: true`` -- **required for NVLink.**

  Adds ``cuda_ipc`` to ``UCX_TLS``, enabling NVLink transfers over UCX; affects intra-node communication only.

- ``ucx.infiniband: true`` -- **required for InfiniBand.**

  Adds ``rc`` to ``UCX_TLS``, enabling InfiniBand transfers over UCX.

- ``ucx.rdmacm: true`` -- **recommended for InfiniBand.**

  Replaces ``sockcm`` with ``rdmacm`` in ``UCX_TLS`` and ``UCX_SOCKADDR_TLS_PRIORITY``, enabling remote direct memory access (RDMA) for InfiniBand transfers.
  This is recommended by UCX for use with InfiniBand, and will not work if InfiniBand is disabled.

- ``ucx.net-devices: <str>`` -- **recommended.**

  Explicitly sets ``UCX_NET_DEVICES`` instead of defaulting to ``"all"``, which can result in suboptimal performance.
  If using InfiniBand, set to ``"auto"`` to automatically detect the InfiniBand interface closest to each GPU.
  If InfiniBand is disabled, set to a UCX-compatible ethernet interface, e.g. ``"enp1s0f0"`` on a DGX-1.
  All available UCX-compatible interfaces can be listed by running ``ucx_info -d``.

  .. warning::
      Setting ``ucx.net-devices: "auto"`` assumes that all InfiniBand interfaces on the system are connected and properly configured; undefined behavior may occur otherwise.
  

- ``rmm.pool-size: <str|int>`` -- **recommended.**

  Allocates an RMM pool of the specified size for the process; size can be provided with an integer number of bytes or in human readable format, e.g. ``"4GB"``.
  It is recommended to set the pool size to at least the minimum amount of memory used by the process; if possible, one can map all GPU memory to a single pool, to be utilized for the lifetime of the process.

.. note::
    These options can be used with mainline Dask/Distributed.
    However, some features are exclusive to Dask-CUDA, such as the automatic detection of InfiniBand interfaces. 
    See :doc:`Specializations for GPU Usage <specializations>` for more details on the benefits of using Dask-CUDA.


Usage
-----

Dask-CUDA workers using UCX communication can be started manually with the ``dask-cuda-worker`` CLI tool or automatically with ``LocalCUDACluster``.
In either case, a ``dask.distributed.Client`` must be made for the worker cluster using the same UCX configuration.

dask-cuda-worker
^^^^^^^^^^^^^^^^

A Dask cluster with UCX support can be started using the ``dask-cuda-worker`` CLI tool with a Dask scheduler which has been configured for UCX.
This must be used for cases where a multi-node cluster is needed, as ``LocalCUDACluster`` will only start single-node clusters.

Scheduler
"""""""""

UCX configuration options will need to be specified for ``dask-scheduler`` as environment variables; see `Dask Configuration <https://docs.dask.org/en/latest/configuration.html#environment-variables>`_ for more details on the mapping between environment variables and options.

To start a Dask scheduler using UCX with all supported transports and a 1 gigabyte RMM pool:

.. code-block:: bash

    DASK_UCX__CUDA_COPY=True \
    DASK_UCX__TCP=True \
    DASK_UCX__NVLINK=True \
    DASK_UCX__INFINIBAND=True \
    DASK_UCX__RDMACM=True \
    DASK_UCX__NET_DEVICES=mlx5_0:1 \
    DASK_RMM__POOL_SIZE=1GB \
    dask-scheduler --protocol ucx --interface ib0

Note the specification of ``mlx5_0:1`` as our UCX net device; because the scheduler does not rely upon Dask-CUDA, it cannot automatically detect InfiniBand interfaces, so we must specify one explicitly.
We communicate to the scheduler that we will be using UCX with the ``--protocol`` option, and that we will be using InfiniBand with the ``--interface`` option.

To start the same Dask scheduler as above but only using NVLink:

.. code-block:: bash

    DASK_UCX__CUDA_COPY=True \
    DASK_UCX__TCP=True \
    DASK_UCX__NVLINK=True \
    DASK_RMM__POOL_SIZE=1GB \
    dask-scheduler --protocol ucx --interface eth0

Note that we no longer specify a net device, as this generally can be skipped when using a non-InfiniBand interface.

Workers
"""""""

All the relevant Dask configuration options for UCX have analogous parameters in ``dask-cuda-worker``; see :doc:`Worker <worker>` for a complete list of these options.

To start workers with all supported transports and a 1 gigabyte RMM pool:

.. code-block:: bash

    dask-cuda-worker ucx://<scheduler_address>:8786 \
    --enable-tcp-over-ucx \
    --enable-nvlink \
    --enable-infiniband \
    --enable-rdmacm \
    --net-devices="auto" \
    --rmm-pool-size="1GB"

Client
""""""

The UCX configurations used by the scheduler and client must be the same.
This can be ensured by using ``dask_cuda.initialize``, a utility which takes the same UCX configuring arguments as ``LocalCUDACluster`` and adds them to the current Dask configuration used when creating the client; see the :doc:`API reference <api>` for a complete list of arguments.

To connect a client to a cluster with all supported transports:

.. code-block:: python

    from dask.distributed import Client
    from dask_cuda.initialize import initialize

    initialize(
        enable_tcp_over_ucx=True,
        enable_nvlink=True,
        enable_infiniband=True,
        enable_rdmacm=True,
        net_devices="mlx5_0:1",
    )
    client = Client("ucx://<scheduler_address>:8786")

Note the specification of ``"mlx5_0:1"`` as our net device; because the scheduler and client do not rely upon Dask-CUDA, they cannot automatically detect InfiniBand interfaces, so we must specify one explicitly.

LocalCUDACluster
^^^^^^^^^^^^^^^^

All options available to ``dask-cuda-worker`` are also available as arguments for ``LocalCUDACluster``; see the :doc:`API reference <api>` for a complete list of arguments.
When creating a ``LocalCUDACluster``, ``dask_cuda.initialize`` is run automatically to ensure the Dask configuration is consistent with the cluster, so that a client can be connected to the cluster with no additional setup.

To start a cluster and client with all supported transports and a 1 gigabyte RMM pool:

.. code-block:: python

    from dask.distributed import Client
    from dask_cuda import LocalCUDACluster

    cluster = LocalCUDACluster(
        protocol="ucx",
        interface="ib0", # passed to the scheduler
        enable_tcp_over_ucx=True,
        enable_nvlink=True,
        enable_infiniband=True,
        enable_rdmacm=True,
        ucx_net_devices="auto",
        rmm_pool_size="1GB"
    )
    client = Client(cluster)