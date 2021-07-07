UCX Integration
===============

Communication can be a major bottleneck in distributed systems.
Dask-CUDA addresses this by supporting integration with `UCX <https://www.openucx.org/>`_, an optimized communication framework that provides high-performance networking and supports a variety of transport methods, including `NVLink <https://www.nvidia.com/en-us/data-center/nvlink/>`_ and `InfiniBand <https://www.mellanox.com/pdf/whitepapers/IB_Intro_WP_190.pdf>`_ for systems with specialized hardware, and TCP for systems without it.
This integration is enabled through `UCX-Py <https://ucx-py.readthedocs.io/>`_, an interface that provides Python bindings for UCX.

Hardware requirements
---------------------

To use UCX with NVLink or InfiniBand, relevant GPUs must be connected with NVLink bridges or NVIDIA Mellanox InfiniBand Adapters, respectively.
NVIDIA provides comparison charts for both `NVLink bridges <https://www.nvidia.com/en-us/design-visualization/nvlink-bridges/>`_ and `InfiniBand adapters <https://www.nvidia.com/en-us/networking/infiniband-adapters/>`_.

Software requirements
---------------------

UCX integration requires an environment with both UCX and UCX-Py installed; see `UCX-Py Installation <https://ucx-py.readthedocs.io/en/latest/install.html>`_ for detailed instructions on this process.

When using UCX, each NVLink and InfiniBand memory buffer must create a mapping between each unique pair of processes they are transferred across; this can be quite costly, potentially in the range of hundreds of milliseconds per mapping.
For this reason, it is strongly recommended to use `RAPIDS Memory Manager (RMM) <https://github.com/rapidsai/rmm>`_ to allocate a memory pool that is only prone to a single mapping operation, which all subsequent transfers may rely upon.
A memory pool also prevents the Dask scheduler from deserializing CUDA data, which will cause a crash.

Configuration
-------------

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

  For optimal performance with UCX 1.11 and above, it is recommended to also set the environment variables ``UCX_MAX_RNDV_RAILS=1`` and ``UCX_MEMTYPE_REG_WHOLE_ALLOC_TYPES=cuda``, see documentation `here <https://ucx-py.readthedocs.io/en/latest/configuration.html#ucx-max-rndv-rails>`_ and `here <https://ucx-py.readthedocs.io/en/latest/configuration.html#ucx-memtype-reg-whole-alloc-types>`_ for more details on those variables.

- ``ucx.rdmacm: true`` -- **recommended for InfiniBand.**

  Replaces ``sockcm`` with ``rdmacm`` in ``UCX_SOCKADDR_TLS_PRIORITY``, enabling remote direct memory access (RDMA) for InfiniBand transfers.
  This is recommended by UCX for use with InfiniBand, and will not work if InfiniBand is disabled.

- ``ucx.net-devices: <str>`` -- **recommended for UCX 1.9 and older.**

  Explicitly sets ``UCX_NET_DEVICES`` instead of defaulting to ``"all"``, which can result in suboptimal performance.
  If using InfiniBand, set to ``"auto"`` to automatically detect the InfiniBand interface closest to each GPU.
  If InfiniBand is disabled, set to a UCX-compatible ethernet interface, e.g. ``"enp1s0f0"`` on a DGX-1.
  All available UCX-compatible interfaces can be listed by running ``ucx_info -d``.

  UCX 1.11 and above is capable of identifying closest interfaces without setting ``"auto"``, it is recommended not to set ``ucx.net-devices``, but some recommendations for optimal performance apply, see the documentation on ``ucx.infiniband`` above fore details.

  .. warning::
      Setting ``ucx.net-devices: "auto"`` assumes that all InfiniBand interfaces on the system are connected and properly configured; undefined behavior may occur otherwise.


- ``rmm.pool-size: <str|int>`` -- **recommended.**

  Allocates an RMM pool of the specified size for the process; size can be provided with an integer number of bytes or in human readable format, e.g. ``"4GB"``.
  It is recommended to set the pool size to at least the minimum amount of memory used by the process; if possible, one can map all GPU memory to a single pool, to be utilized for the lifetime of the process.

.. note::
    These options can be used with mainline Dask.distributed.
    However, some features are exclusive to Dask-CUDA, such as the automatic detection of InfiniBand interfaces. 
    See `Dask-CUDA -- Motivation <index.html#motivation>`_ for more details on the benefits of using Dask-CUDA.

Usage
-----

See `Enabling UCX communication <examples/ucx.html>`_ for examples of UCX usage with different supported transports.
