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

.. warning::
    Dask-CUDA must create worker CUDA contexts during cluster initialization, and properly ordering that task is critical for correct UCX configuration.
    If a CUDA context already exists for this process at the time of cluster initialization, unexpected behavior can occur.
    To avoid this, it is advised to initialize any UCX-enabled clusters before doing operations that would result in a CUDA context being created.
    Depending on the library, even an import can force CUDA context creation.

    For some RAPIDS libraries (e.g. cuDF), setting ``RAPIDS_NO_INITIALIZE=1`` at runtime will delay or disable their CUDA context creation, allowing for improved compatibility with UCX-enabled clusters and preventing runtime warnings.


Configuration
-------------

Automatic
~~~~~~~~~

Beginning with Dask-CUDA 22.02 and assuming UCX >= 1.11.1, specifying UCX transports is now optional.

A local cluster can now be started with ``LocalCUDACluster(protocol="ucx")``, implying automatic UCX transport selection (``UCX_TLS=all``). Starting a cluster separately -- scheduler, workers and client as different processes -- is also possible, as long as Dask scheduler is created with ``dask scheduler --protocol="ucx"`` and connecting a ``dask cuda worker`` to the scheduler will imply automatic UCX transport selection, but that requires the Dask scheduler and client to be started with ``DASK_DISTRIBUTED__COMM__UCX__CREATE_CUDA_CONTEXT=True``. See `Enabling UCX communication <../examples/ucx/>`_ for more details examples of UCX usage with automatic configuration.

Configuring transports manually is still possible, please refer to the subsection below.

Manual
~~~~~~

In addition to installations of UCX and UCX-Py on your system, for manual configuration several options must be specified within your Dask configuration to enable the integration.
Typically, these will affect ``UCX_TLS`` and ``UCX_SOCKADDR_TLS_PRIORITY``, environment variables used by UCX to decide what transport methods to use and which to prioritize, respectively.
However, some will affect related libraries, such as RMM:

- ``distributed.comm.ucx.cuda_copy: true`` -- **required.**

  Adds ``cuda_copy`` to ``UCX_TLS``, enabling CUDA transfers over UCX.

- ``distributed.comm.ucx.tcp: true`` -- **required.**

  Adds ``tcp`` to ``UCX_TLS``, enabling TCP transfers over UCX; this is required for very small transfers which are inefficient for NVLink and InfiniBand.

- ``distributed.comm.ucx.nvlink: true`` -- **required for NVLink.**

  Adds ``cuda_ipc`` to ``UCX_TLS``, enabling NVLink transfers over UCX; affects intra-node communication only.

- ``distributed.comm.ucx.infiniband: true`` -- **required for InfiniBand.**

  Adds ``rc`` to ``UCX_TLS``, enabling InfiniBand transfers over UCX.

  For optimal performance with UCX 1.11 and above, it is recommended to also set the environment variables ``UCX_MAX_RNDV_RAILS=1`` and ``UCX_MEMTYPE_REG_WHOLE_ALLOC_TYPES=cuda``, see documentation `here <https://ucx-py.readthedocs.io/en/latest/configuration.html#ucx-max-rndv-rails>`_ and `here <https://ucx-py.readthedocs.io/en/latest/configuration.html#ucx-memtype-reg-whole-alloc-types>`_ for more details on those variables.

- ``distributed.comm.ucx.rdmacm: true`` -- **recommended for InfiniBand.**

  Replaces ``sockcm`` with ``rdmacm`` in ``UCX_SOCKADDR_TLS_PRIORITY``, enabling remote direct memory access (RDMA) for InfiniBand transfers.
  This is recommended by UCX for use with InfiniBand, and will not work if InfiniBand is disabled.

- ``distributed.rmm.pool-size: <str|int>`` -- **recommended.**

  Allocates an RMM pool of the specified size for the process; size can be provided with an integer number of bytes or in human readable format, e.g. ``"4GB"``.
  It is recommended to set the pool size to at least the minimum amount of memory used by the process; if possible, one can map all GPU memory to a single pool, to be utilized for the lifetime of the process.

.. note::
    These options can be used with mainline Dask.distributed.
    However, some features are exclusive to Dask-CUDA, such as the automatic detection of InfiniBand interfaces.
    See `Dask-CUDA -- Motivation <../#motivation>`_ for more details on the benefits of using Dask-CUDA.

Usage
-----

See `Enabling UCX communication <../examples/ucx/>`_ for examples of UCX usage with different supported transports.

Running in a fork-starved environment
-------------------------------------

Many high-performance networking stacks do not support the user
application calling ``fork()`` after the network substrate is
initialized. Symptoms include jobs randomly hanging, or crashing,
especially when using a large number of workers. To mitigate against
this when using Dask-CUDA's UCX integration, processes launched via
multiprocessing should use the start processes using the
`"forkserver"
<https://docs.python.org/dev/library/multiprocessing.html#contexts-and-start-methods>`_
method. When launching workers using `dask cuda worker <../quickstart/#dask-cuda-worker>`_, this can be
achieved by passing ``--multiprocessing-method forkserver`` as an
argument. In user code, the method can be controlled with the
``distributed.worker.multiprocessing-method`` configuration key in
``dask``. One must take care to, in addition, manually ensure that the
forkserver is running before launching any jobs. A run script should
therefore do something like the following:

.. code-block::

   import dask

   if __name__ == "__main__":
       import multiprocessing.forkserver as f
       f.ensure_running()
       with dask.config.set(
           {"distributed.worker.multiprocessing-method": "forkserver"}
       ):
           run_analysis(...)


.. note::

   In addition to this, at present one must also set
   ``PTXCOMPILER_CHECK_NUMBA_CODEGEN_PATCH_NEEDED=0`` in the
   environment to avoid a subprocess call from `ptxcompiler
   <https://github.com/rapidsai/ptxcompiler>`_

.. note::

   To confirm that no bad fork calls are occurring, start jobs with
   ``UCX_IB_FORK_INIT=n``. UCX will produce a warning ``UCX  WARN  IB:
   ibv_fork_init() was disabled or failed, yet a fork() has been
   issued.`` if the application calls ``fork()``.
