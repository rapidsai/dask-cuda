Quickstart
==========

A Dask-CUDA cluster can be created using either LocalCUDACluster or ``dask-cuda-worker`` from the command line.

LocalCUDACluster
----------------

To create a standard Dask-CUDA cluster and connect a Dask.distributed `Client <https://distributed.dask.org/en/latest/client.html>`_ to it:

.. code-block:: python

    from dask_cuda import LocalCUDACluster
    from dask.distributed import Client

    cluster = LocalCUDACluster()
    client = Client(cluster)

dask-cuda-worker
----------------

To create an equivalent cluster from the command line, Dask-CUDA workers must be connected to a scheduler started with ``dask-scheduler``::

    $ dask-scheduler
    distributed.scheduler - INFO -   Scheduler at:  tcp://127.0.0.1:8786

    $ dask-cuda-worker 127.0.0.1:8786

To connect a client to this cluster:

.. code-block:: python

    from dask.distributed import Client

    client = Client("127.0.0.1:8786")

Other stuff
-----------

*Ideally some basic examples of defining GPU-centric operations here*
