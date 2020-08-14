UCX Integration
===============

Communication can be a major bottleneck in distributed systems, and that's no different in Dask and Dask-CUDA. To address that, Dask-CUDA provides integration with `UCX-Py <https://ucx-py.readthedocs.io/>`_, a Python interface for the `UCX <https://www.openucx.org/>`_ communication framework.  UCX is a low-level library that provides high-performance networking and supports several transports, including NVLink and InfiniBand for systems that have such specialized hardware, as well as TCP for those that do not.



Relevant Settings
-----------------

In this list below we shall see all the Dask configurations relevant for using with Dask-CUDA currently. Note that these options can also be used with mainline Dask/Distributed if you can't use Dask-CUDA for some reason. Note that many improvments for running GPU code is readily available in Dask-CUDA -- but not in mainline Dask/Distributed -- thus, we recommend using Dask-CUDA when possible.
In the list below we describe the relevant Dask configurations for using UCX with Dask-CUDA.

.. note::
    These options can also be used with mainline Dask/Distributed outside of Dask-CUDA, however, we recommend using Dask-CUDA when possible. See the :doc:`Specializations for GPU Usage <specializations>` page for details on the benefits of using Dask-CUDA.


- ``DASK_UCX__CUDA_COPY=True`` (default: ``False``): *Always required for UCX*, adds ``cuda_copy`` to ``UCX_TLS`` -- required for all CUDA transfers in UCX, both intra- and inter-node;
- ``DASK_UCX__TCP=True`` (default: ``False``): *Always required for UCX*, adds ``tcp`` to ``UCX_TLS`` -- required for all TCP transfers (e.g., where NVLink or IB is not available/disabled) in UCX, both intra- and inter-node;
- ``DASK_UCX__NVLINK=True`` (default: ``False``): Adds ``cuda_ipc`` to ``UCX_TLS`` -- required for all NVLink transfers in UCX, only affects intra-node.
- ``DASK_UCX__INFINIBAND=True`` (defalt: ``False``): Adds ``rc`` to ``UCX_TLS`` -- required for all InfiniBand transfers in UCX, only affects inter-node.
- ``DASK_UCX__RDMACM=True`` (default: ``False``): Replaces ``sockcm`` by ``rdmacm`` in ``UCX_TLS`` and ``UCX_SOCKADDR_TLS_PRIORITY``. ``rdmacm`` is the recommended method by UCX to use with IB and currently won't work if ``DASK_UCX__INFINIBAND=False``.
- ``DASK_UCX__NET_DEVICES=mlx5_0:1`` (default: ``None``, causes UCX to decide what device to use, possibly being suboptimal, implies ``UCX_NET_DEVICES=all``): this is very important when ``DASK_UCX__INFINIBAND=True`` to ensure the scheduler is connected over the InfiniBand interface. When ``DASK_UCX__INFINIBAND=False`` it's recommended to use the ethernet device instead, e.g., ``DASK_UCX__NET_DEVICES=enp1s0f0`` on a DGX-1.
- ``DASK_RMM__POOL_SIZE=1GB``: allocates an RMM pool for the process. In some circumstances, the Dask scheduler will deserialize CUDA data and cause a crash if there's no pool.


Important notes
---------------

* CUDA Memory Pool: With UCX, all NVLink and InfiniBand memory buffers have to be mapped from one process to another upon the first request for transfer of that buffer between a single pair of processes. This can be quite costly, consuming up to 100ms only for mapping, plus the transfer time itself. For this reason it is strongly recommended to use a `RAPIDS Memory Manager (RMM) <https://github.com/rapidsai/rmm>`_ memory pool in such cases, incurring in a single mapping of the pool and all subsequent transfers will not be required to repeat that process. It is recommened to also keep the memory pool size to at least the minimum amount of memory used by the application, if possible one can map all GPU memory to a single pool and utilize that pool for the entire lifetime of the application.

* Automatic detection of InfiniBand interfaces: it's especially important to note the usage of ``--net-devices="auto"`` in ``dask-cuda-worker``, which will automatically determine the InfiniBand interface that's closest to each GPU. For safety, this option can only be used if ``--enable-infiniband`` is specified. Be warned that this mode assumes all InfiniBand interfaces on the system are connected and properly configured, undefined behavior may occur otherwise.


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
