Enabling UCX communication
==========================

A CUDA cluster using UCX communication can be started automatically with LocalCUDACluster or manually with the ``dask cuda worker`` CLI tool.
In either case, a ``dask.distributed.Client`` must be made for the worker cluster using the same Dask UCX configuration; see `UCX Integration -- Configuration <../../ucx/#configuration>`_ for details on all available options.

LocalCUDACluster with Automatic Configuration
---------------------------------------------

Automatic configuration was introduced in Dask-CUDA 22.02 and requires UCX >= 1.11.1. This allows the user to specify only the UCX protocol and let UCX decide which transports to use.

To connect a client to a cluster with automatically-configured UCX and an RMM pool:

.. code-block:: python

    from dask.distributed import Client
    from dask_cuda import LocalCUDACluster

    cluster = LocalCUDACluster(
        protocol="ucx",
        interface="ib0",
        rmm_pool_size="1GB"
    )
    client = Client(cluster)

.. note::
    The ``interface="ib0"`` is intentionally specified above to ensure RDMACM is used in systems that support InfiniBand. On systems that don't support InfiniBand or where RDMACM isn't required, the ``interface`` argument may be omitted or specified to listen on a different interface.

LocalCUDACluster with Manual Configuration
------------------------------------------

When using LocalCUDACluster with UCX communication and manual configuration, all required UCX configuration is handled through arguments supplied at construction; see `API -- Cluster <../../api/#cluster>`_ for a complete list of these arguments.
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

``dask cuda worker`` with Automatic Configuration
-------------------------------------------------

When using ``dask cuda worker`` with UCX communication and automatic configuration, the scheduler, workers, and client must all be started manually, but without specifying any UCX transports explicitly. This is only supported in Dask-CUDA 22.02 and newer and requires UCX >= 1.11.1.

Scheduler
^^^^^^^^^

For automatic UCX configuration, we must ensure a CUDA context is created on the scheduler before UCX is initialized. This can be satisfied by specifying the ``DASK_DISTRIBUTED__COMM__UCX__CREATE_CUDA_CONTEXT=True`` environment variable when creating the scheduler.

To start a Dask scheduler using UCX with automatic configuration and one GB of RMM pool:

.. code-block:: bash

    $ DASK_DISTRIBUTED__COMM__UCX__CREATE_CUDA_CONTEXT=True \
    > DASK_DISTRIBUTED__RMM__POOL_SIZE=1GB \
    > dask scheduler --protocol ucx --interface ib0

.. note::
    The ``interface="ib0"`` is intentionally specified above to ensure RDMACM is used in systems that support InfiniBand. On systems that don't support InfiniBand or where RDMACM isn't required, the ``interface`` argument may be omitted or specified to listen on a different interface.

    We specify ``UCX_MEMTYPE_REG_WHOLE_ALLOC_TYPES=cuda`` above for optimal performance with InfiniBand, see details `here <https://ucx-py.readthedocs.io/en/latest/configuration.html#ucx-memtype-reg-whole-alloc-types>`__. If not using InfiniBand, that option may be omitted. In UCX 1.12 and newer, that option is default and may be omitted as well even when using InfiniBand.

Workers
^^^^^^^

To start workers with automatic UCX configuration and an RMM pool of 14GB per GPU:

.. code-block:: bash

    $ UCX_MEMTYPE_REG_WHOLE_ALLOC_TYPES=cuda
    > dask cuda worker ucx://<scheduler_address>:8786 \
    > --rmm-pool-size="14GB" \
    > --interface="ib0"

.. note::
    Analogous to the scheduler setup, the ``interface="ib0"`` is intentionally specified above to ensure RDMACM is used in systems that support InfiniBand. On systems that don't support InfiniBand or where RDMACM isn't required, the ``interface`` argument may be omitted or specified to listen on a different interface.

    We specify ``UCX_MEMTYPE_REG_WHOLE_ALLOC_TYPES=cuda`` above for optimal performance with InfiniBand, see details `here <https://ucx-py.readthedocs.io/en/latest/configuration.html#ucx-memtype-reg-whole-alloc-types>`__. If not using InfiniBand, that option may be omitted. In UCX 1.12 and newer, that option is default and may be omitted as well even when using InfiniBand.

Client
^^^^^^

To connect a client to the cluster with automatic UCX configuration we started:

.. code-block:: python

    import os

    os.environ["UCX_MEMTYPE_REG_WHOLE_ALLOC_TYPES"] = "cuda"

    import dask
    from dask.distributed import Client

    with dask.config.set({"distributed.comm.ucx.create_cuda_context": True}):
        client = Client("ucx://<scheduler_address>:8786")

Alternatively, the ``with dask.config.set`` statement from the example above may be omitted and the ``DASK_DISTRIBUTED__COMM__UCX__CREATE_CUDA_CONTEXT=True`` environment variable specified instead:

.. code-block:: python

    import os

    os.environ["UCX_MEMTYPE_REG_WHOLE_ALLOC_TYPES"] = "cuda"
    os.environ["DASK_DISTRIBUTED__COMM__UCX__CREATE_CUDA_CONTEXT"] = "True"

    from dask.distributed import Client

    client = Client("ucx://<scheduler_address>:8786")

.. note::
    We specify ``UCX_MEMTYPE_REG_WHOLE_ALLOC_TYPES=cuda`` above for optimal performance with InfiniBand, see details `here <https://ucx-py.readthedocs.io/en/latest/configuration.html#ucx-memtype-reg-whole-alloc-types>`_. If not using InfiniBand, that option may be omitted. In UCX 1.12 and newer, that option is default and may be omitted as well even when using InfiniBand.

``dask cuda worker`` with Manual Configuration
----------------------------------------------

When using ``dask cuda worker`` with UCX communication and manual configuration, the scheduler, workers, and client must all be started manually, each using the same UCX configuration.

Scheduler
^^^^^^^^^

UCX configuration options will need to be specified for ``dask scheduler`` as environment variables; see `Dask Configuration -- Environment Variables <https://docs.dask.org/en/latest/configuration.html#environment-variables>`_ for more details on the mapping between environment variables and options.

To start a Dask scheduler using UCX with all supported transports and an gigabyte RMM pool:

.. code-block:: bash

    $ DASK_DISTRIBUTED__COMM__UCX__CUDA_COPY=True \
    > DASK_DISTRIBUTED__COMM__UCX__TCP=True \
    > DASK_DISTRIBUTED__COMM__UCX__NVLINK=True \
    > DASK_DISTRIBUTED__COMM__UCX__INFINIBAND=True \
    > DASK_DISTRIBUTED__COMM__UCX__RDMACM=True \
    > DASK_DISTRIBUTED__RMM__POOL_SIZE=1GB \
    > dask scheduler --protocol ucx --interface ib0

We communicate to the scheduler that we will be using UCX with the ``--protocol`` option, and that we will be using InfiniBand with the ``--interface`` option.

Workers
^^^^^^^

All UCX configuration options have analogous options in ``dask cuda worker``; see `API -- Worker <../../api/#worker>`_ for a complete list of these options.
To start a cluster with all supported transports and an RMM pool:

.. code-block:: bash

    $ dask cuda worker ucx://<scheduler_address>:8786 \
    > --enable-tcp-over-ucx \
    > --enable-nvlink \
    > --enable-infiniband \
    > --enable-rdmacm \
    > --rmm-pool-size="1GB"

Client
^^^^^^

A client can be configured to use UCX by using ``dask_cuda.initialize``, a utility which takes the same UCX configuring arguments as LocalCUDACluster and adds them to the current Dask configuration used when creating it; see `API -- Client initialization <../../api/#client-initialization>`_ for a complete list of arguments.
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
