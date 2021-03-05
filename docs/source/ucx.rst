UCX Integration
===============

Communication can be a major bottleneck in distributed systems.
Dask-CUDA addresses this by supporting integration with `UCX <https://www.openucx.org/>`_, an optimized communication framework that provides high-performance networking and supports a variety of transport methods, including:

- NVLink for *something ubiquitous to NVLink systems*
- InfiniBand for *something ubiquitous to InfiniBand systems*
- TCP for systems that do not have specialized hardware

This integration is enabled through `UCX-Py <https://ucx-py.readthedocs.io/>`_, an interface that provides Python bindings for UCX.


Requirements
------------

Hardware
^^^^^^^^

*Go into more detail on what hardware is required for NVLink, InfiniBand support*

Software
^^^^^^^^

*Does UCX integration require anything else other than UCX-Py?*

When using UCX, each NVLink and InfiniBand memory buffer must create a mapping between each unique pair of processes they are transferred across; this can be quite costly, taking up to 100 ms per mapping.
For this reason, it is strongly recommended to use `RAPIDS Memory Manager (RMM) <https://github.com/rapidsai/rmm>`_ to allocate a memory pool that is only prone to a single mapping operation, which all subsequent transfers may rely upon.

Configuration
^^^^^^^^^^^^^

In addition to installations of UCX and UCX-Py on your system, several options must be specified within your Dask configuration to enable the integration.
Typically, these will affect ``UCX_TLS`` and ``UCX_SOCKADDR_TLS_PRIORITY``, environment variables used by UCX to decide what transport methods to use and which to prioritize, respectively.
However, some will affect related libraries, such as RMM:

- ``ucx.cuda_copy: true`` -- **required.**
    
    Adds ``cuda_copy`` to ``UCX_TLS``, enabling *all* transfers over UCX.
    *Is this accurate to say?*

- ``ucx.tcp: true`` -- **required.**

    Adds ``tcp`` to ``UCX_TLS``, enabling TCP transfers over UCX if NVLink or InfiniBand are unavailable or disabled.
    *Why is this required if they are available and enabled?*

- ``ucx.nvlink: true`` -- required for NVLink.

    Adds ``cuda_ipc`` to ``UCX_TLS``, enabling NVLink transfers over UCX; affects intra-node communication only.

- ``ucx.infiniband: true`` -- required for InfiniBand.

    Adds ``rc`` to ``UCX_TLS``, enabling InfiniBand transfers over UCX; affects inter-node communication only.


- ``ucx.rdmacm: true`` -- recommended for InfiniBand.

    Replaces ``sockcm`` with ``rdmacm`` in ``UCX_TLS`` and ``UCX_SOCKADDR_TLS_PRIORITY``, *enabling remote direct memory access (RDMA) for connection management.*
    This is recommended by UCX for use with InfiniBand, and will have no effect if InfiniBand tranfers are disabled.

- ``ucx.net-devices: <str>`` -- recommended.

    Explicitly sets ``UCX_NET_DEVICES`` instead of defaulting to ``"all"``, which can result in suboptimal performance.
    If using InfiniBand, set to the desired IB device, e.g. ``"mlx5_0:1"``.
    If InfiniBand is disabled, set to the ethernet device, e.g. ``"enp1s0f0"`` on a DGX-1.
    All available UCX-compatible devices can be listed by running ``ucx_info -d`` or ``ifconfig`` *(are all the options under ifconfig compatible)*.

- ``rmm.pool-size: <str | int>`` -- recommended.

    Allocates an RMM pool of the specified size for the process; size can be provided with an integer number of bytes or in human readable format, e.g. ``"4GB"``.
    In addition to reducing the cost of mapping incurred by memory transfers, a pool can prevent the Dask scheduler from deserializing CUDA data and causing a crash.
    It is recommended to set the pool size to at least the minimum amount of memory used by the process; if possible, one can map all GPU memory to a single pool, to be utilized for the lifetime of the process.


.. note::
    These options can also be used with mainline Dask/Distributed.
    However, this may disable a variety of features, such as *put some features exclusive to Dask-CUDA here*. 
    See :doc:`Specializations for GPU Usage <specializations>` for more details on the benefits of using Dask-CUDA.


Important notes
---------------

- Automatic detection of InfiniBand interfaces: it's especially important to note the usage of ``--net-devices="auto"`` in ``dask-cuda-worker``, which will automatically determine the InfiniBand interface that's closest to each GPU. For safety, this option can only be used if ``--enable-infiniband`` is specified. Be warned that this mode assumes all InfiniBand interfaces on the system are connected and properly configured, undefined behavior may occur otherwise.


Launching Scheduler, Workers and Clients Separately
---------------------------------------------------

The first way for starting a Dask cluster with UCX support is to start each process separately. The processes are ``dask-scheduler``, ``dask-cuda-worker`` and the client process utilizing ``distributed.Client`` that will connect to the cluster. Details follow for each of the processes.

dask-scheduler
^^^^^^^^^^^^^^

The ``dask-scheduler`` has no parameters for UCX configuration -- different from what we will see for ``dask-cuda-worker`` on the next section -- for that reason we rely on Dask environment variables. Here's how to start the scheduler with all transports that are currently supported by Dask-CUDA:

.. code-block:: bash

    DASK_RMM__POOL_SIZE=1GB DASK_UCX__CUDA_COPY=True DASK_UCX__TCP=True DASK_UCX__NVLINK=True DASK_UCX__INFINIBAND=True DASK_UCX__RDMACM=True DASK_UCX__NET_DEVICES=mlx5_0:1 dask-scheduler --protocol ucx --interface ib0

Note above how we use ``DASK_UCX__NET_DEVICES=mlx5_0:1`` (the Mellanox name for ``ib0``) and the same interface with ``--interface ib0``. If the system doesn't have an InfiniBand interface available, you would normally use the main network interface, such as ``eth0``, as seen below:

.. code-block:: bash

    DASK_RMM__POOL_SIZE=1GB DASK_UCX__CUDA_COPY=True DASK_UCX__TCP=True DASK_UCX__NVLINK=True dask-scheduler --protocol ucx --interface eth0

Setting ``DASK_UCX__NET_DEVICES`` when using an interface that isn't an InfiniBand can generally be skipped.


dask-cuda-worker
^^^^^^^^^^^^^^^^

All ``DASK_*`` configurations described above have analogous parameters in ``dask-cuda-worker`` which are preferred over the regular configurations used for ``dask-scheduler`` due to some specializations, such as ``--net-devices="auto"`` which will correctly assign the topologically closest IB interface to the GPU of each worker, something that's not possible with ``DASK_UCX__NET_DEVICES``.

- ``--disable-tcp-over-ucx`` (default) is analogous to ``DASK_UCX__TCP=False``, ``--enable-tcp-over-ucx`` is equivalent to ``DASK_UCX__TCP=True``;
- ``--disable-nvlink`` (default) is analogous to ``DASK_UCX__NVLINK=False``, ``--enable-nvlink`` is equivalent to ``DASK_UCX__NVLINK=True``;
- ``--disable-infiniband`` (default) is analogous to ``DASK_UCX__INFINIBAND=False``, ``--enable-infiniband`` is equivalent to ``DASK_UCX__INFINIBAND=True``;
- ``--net-devices`` (default ``None``, implies ``UCX_NET_DEVICES=all``) equivalent to ``DASK_UCX__NET_DEVICES``;
- ``--rmm-pool-size`` equivalent to ``DASK_RMM__POOL_SIZE``.

Here's how to start workers with all transports that are currently relevant for us:

.. code-block:: bash

    dask-cuda-worker ucx://SCHEDULER_IB0_IP:8786 --enable-tcp-over-ucx --enable-nvlink --enable-infiniband -- enable-rdmacm --net-devices="auto" --rmm-pool-size="30GB"


client
^^^^^^

The same configurations used for the scheduler should be used by the client. One possible exception is ``DASK_RMM__POOL_SIZE``, at this time it's unclear whether this is necessary or not, but using that should not cause any issues nevertheless.

One can use ``os.environ`` inside the client script, it's important to set them at the very top before importing anything other than ``os``. See example below:

.. code-block:: python

    import os

    os.environ["DASK_RMM__POOL_SIZE"] = "1GB"
    os.environ["DASK_UCX__CUDA_COPY"] = "True"  # os.environ needs using strings, not Python True/False
    os.environ["DASK_UCX__TCP"] = "True"
    os.environ["DASK_UCX__NVLINK"] = "True"
    os.environ["DASK_UCX__INFINIBAND"] = "True"
    os.environ["DASK_UCX__NET_DEVICES"] = "mlx5_0:1"

    from distributed import Client

    client = Client("ucx://SCHEDULER_IB0_IP:8786")  # SCHEDULER_IB0_IP must be the IP of ib0 on the node where scheduler runs

    # Client code goes here


Starting a local cluster (single-node only)
-------------------------------------------

All options discussed previously are also available in ``LocalCUDACluster``. It is shown below how to start a local cluster with all UCX capabilities enabled:

.. code-block:: python

    import os

    # The options here are to be used by the client only,
    # inherent options for the Dask scheduler and workers
    # have to be passed to LocalCUDACluster
    os.environ["DASK_RMM__POOL_SIZE"] = "1GB"
    os.environ["DASK_UCX__CUDA_COPY"] = "True"  # os.environ needs using strings, not Python True/False
    os.environ["DASK_UCX__TCP"] = "True"
    os.environ["DASK_UCX__NVLINK"] = "True"
    os.environ["DASK_UCX__INFINIBAND"] = "True"
    os.environ["DASK_UCX__NET_DEVICES"] = "mlx5_0:1"

    from dask.distributed import Client
    from dask_cuda import LocalCUDACluster
    from dask_cuda.initialize import initialize

    cluster = LocalCUDACluster(
        protocol = "ucx"
        interface = "ib0"  # Interface -- used for the scheduler
        enable_tcp_over_ucx = True
        enable_nvlink = True
        enable_infiniband = True
        ucx_net_devices="auto"
        rmm_pool_size="24GB"
    )
    client = Client(cluster)

    # Client code goes here
