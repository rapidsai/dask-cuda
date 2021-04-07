Spilling from device
====================

By default, Dask-CUDA enables spilling from GPU to host memory when a GPU reaches a memory utilization of 80%.
This can be changed to suit the needs of a workload, or disabled altogether, by explicitly setting ``device_memory_limit``.
This parameter accepts an integer or string memory size, or a float representing a percentage of the GPU's total memory:

.. code-block:: python

    from dask_cuda import LocalCUDACluster

    cluster = LocalCUDACluster(device_memory_limit=50000)  # spilling after 50000 bytes
    cluster = LocalCUDACluster(device_memory_limit="5GB")  # spilling after 5 GB
    cluster = LocalCUDACluster(device_memory_limit=0.3)    # spilling after 30% memory utilization

Memory spilling can be disabled by setting ``device_memory_limit`` to 0:

.. code-block:: python

    cluster = LocalCUDACluster(device_memory_limit=0)  # spilling disabled

The same applies for ``dask-cuda-worker``, and spilling can be controlled by setting ``--device-memory-limit``::

    $ dask-scheduler
    distributed.scheduler - INFO -   Scheduler at:  tcp://127.0.0.1:8786

    $ dask-cuda-worker --device-memory-limit 50000
    $ dask-cuda-worker --device-memory-limit 5GB
    $ dask-cuda-worker --device-memory-limit 0.3
    $ dask-cuda-worker --device-memory-limit 0