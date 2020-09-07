Quickstart
==========


Setup
-----


::

    conda create -n dask-cuda -c rapidsai -c nvidia -c conda-forge \
      cudatoolkit=<CUDA version> cudf dask-cuda distributed python=3.7

Creating a Dask-CUDA Cluster
----------------------------

Notebook
~~~~~~~~

.. code-block:: python

    from dask_cuda import LocalCUDACluster
    from dask.distributed import Client

    # Create a Dask Cluster with one worker per GPU
    cluster = LocalCUDACluster()
    client = Client(cluster)