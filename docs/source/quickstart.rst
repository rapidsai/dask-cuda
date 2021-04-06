Quickstart
==========

A Dask-CUDA cluster can be created using either LocalCUDACluster or ``dask-cuda-worker`` from the command line.

LocalCUDACluster
----------------

To create a Dask-CUDA cluster using all available GPUs and connect a Dask.distributed `Client <https://distributed.dask.org/en/latest/client.html>`_ to it:

.. code-block:: python

    from dask_cuda import LocalCUDACluster
    from dask.distributed import Client

    cluster = LocalCUDACluster()
    client = Client(cluster)

To create a cluster using only 2 GPUs:

.. code-block:: python

    cluster = LocalCUDACluster(CUDA_VISIBLE_DEVICES="0,1")  # can also use n_workers=2

dask-cuda-worker
----------------

To create an cluster from the command line, Dask-CUDA workers must be connected to a scheduler started with ``dask-scheduler``::

    $ dask-scheduler
    distributed.scheduler - INFO -   Scheduler at:  tcp://127.0.0.1:8786

    $ dask-cuda-worker 127.0.0.1:8786

Unlike LocalCUDACluster, the number of GPUs cannot be provided as an option, and must be specified as an environment variable::

    $ CUDA_VISIBLE_DEVICES=0,1 dask-cuda-worker 127.0.0.1:8786

To connect a client to this cluster:

.. code-block:: python

    from dask.distributed import Client

    client = Client("127.0.0.1:8786")

Other stuff
-----------

*Ideally some basic examples of defining GPU-centric operations here*
