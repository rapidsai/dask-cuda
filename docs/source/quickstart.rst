Quickstart
==========

A Dask-CUDA cluster can be created using either LocalCUDACluster or ``dask cuda worker`` from the command line.

LocalCUDACluster
----------------

To create a Dask-CUDA cluster using all available GPUs and connect a Dask.distributed `Client <https://distributed.dask.org/en/latest/client.html>`_ to it:

.. code-block:: python

    from dask_cuda import LocalCUDACluster
    from dask.distributed import Client

    cluster = LocalCUDACluster()
    client = Client(cluster)

.. tip::

   Be sure to include an ``if __name__ == "__main__":`` block when using :py:class:`dask_cuda.LocalCUDACluster` in a standalone Python script. See `standalone Python scripts <https://docs.dask.org/en/stable/scheduling.html#standalone-python-scripts>`_ for more details.

``dask cuda worker``
--------------------

To create an equivalent cluster from the command line, Dask-CUDA workers must be connected to a scheduler started with ``dask scheduler``:

.. code-block:: bash

    $ dask scheduler
    distributed.scheduler - INFO -   Scheduler at:  tcp://127.0.0.1:8786

    $ dask cuda worker 127.0.0.1:8786

To connect a client to this cluster:

.. code-block:: python

    from dask.distributed import Client

    client = Client("127.0.0.1:8786")
