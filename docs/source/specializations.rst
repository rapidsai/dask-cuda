Specializations for GPU Usage
=============================

It is known that main line Dask and Distributed packages can be used to leverage GPU computing, utilizing libraries such as cuDF, CuPy and Numba. So why use Dask-CUDA instead? This section aims to answer this question.

Automatic Instantiation of One-Worker-Per-GPU
---------------------------------------------

Using the ``dask-cuda-worker`` or ``LocalCUDACluster`` will automatically launch one worker for each GPU available on the node where that is running, avoiding the need for users to select GPUs in their application and thus reducing code complexity.

Spilling From Device
--------------------

For applications that do not fit in GPU memory, Dask-CUDA supports spilling from device memory to host memory when the GPU can't fit more data. The spilling mechanism is automatically triggered once the user-defined limit is reached, such limit can be set via the ``--device-memory-limit`` and ``device_memory_limit`` arguments for ``dask-cuda-worker`` and ``LocalCUDACluster``, respectively.

CPU Affinity
------------

To improve performance, setting CPU affinity for each GPU is automatically done, preventing memory transfers from taking sub-optimal paths.

Automatic Selection of InfiniBand Device
----------------------------------------

When UCX is activated, Dask-CUDA is also capable of selecting the topologically closest InfiniBand device to each GPU, thus ensuring optimal path and improving performance even further by using Remote Direct Memory Access (RDMA) when available.
