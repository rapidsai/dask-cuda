Specializations for GPU Usage
=============================

It is known that main line Dask and Distributed packages can be used to leverage GPU computing, utilizing libraries such as cuDF, CuPy and Numba. So why use Dask-CUDA instead? This section aims to answer this question.

Automatic Instantiation of One-Worker-Per-GPU
---------------------------------------------

Using the ``dask-cuda-worker`` or ``LocalCUDACluster`` will automatically launch one worker for each GPU available on the node from where it was executed, avoiding the need for users to select GPUs in their application and thus reducing code complexity.

Controlling Number of Workers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Users can control the number of workers by explicitly defining ``CUDA_VISIBLE_DEVICES`` in either the ``dask-cuda-worker`` CLI or using ``LocalCUDACluster``.  For example, the following will launch 3 workers on devices, 1, 2, and 3:

.. code-block:: bash

    LocalCUDACluster(CUDA_VISIBLE_DEVICES='0,1,2')
    or
    CUDA_VISIBLE_DEVICES=0,1,2 dask-cuda-workewr

Users can also use UUID of the device as an inputs to ``CUDA_VISIBLE_DEVICES``.  UUIDs should begin with prefix 'GPU-' or 'MIG-GPU': `GPU-9baca7f5-0f2f-01ac-6b05-8da14d6e9005`, for example.


Spilling From Device
--------------------

For applications that do not fit in GPU memory, Dask-CUDA supports spilling from device memory to host memory when the GPU can't fit more data. The spilling mechanism is automatically triggered once the user-defined limit is reached, such limit can be set via the ``--device-memory-limit`` and ``device_memory_limit`` arguments for ``dask-cuda-worker`` and ``LocalCUDACluster``, respectively.

Previously, spilling was disabled by default, but since Dask-CUDA 0.17 the default has been changed to ``0.8`` -- spilling will begin when Dask-CUDA device memory utilization reaches 80% of the device's total memory.  Behavior can configured with ``--device-memory-limit`` flag.  Users can disable spilling by setting ``--device-memory-limit=0`` or ``device_memory_limit=0``.

CPU Affinity
------------

To improve performance, setting CPU affinity for each GPU is automatically done, preventing memory transfers from taking sub-optimal paths.

Automatic Selection of InfiniBand Device
----------------------------------------

When InfiniBand is activated, Dask-CUDA is also capable of selecting the topologically closest InfiniBand device to each GPU, thus ensuring optimal path and improving performance even further by using GPU Remote Direct Memory Access (RDMA) when available. See the :doc:`UCX <ucx>` page for more details.
