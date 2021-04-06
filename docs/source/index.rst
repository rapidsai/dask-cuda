Dask-CUDA
=========

Dask-CUDA is a library for distributed computing in Python using GPUs.
It extends `Dask.distributed <https://distributed.dask.org/en/latest/>`_'s single-machine `LocalCluster <https://docs.dask.org/en/latest/setup/single-distributed.html#localcluster>`_ and `Worker <https://distributed.dask.org/en/latest/worker.html>`_ for use in GPU workloads.

Motivation
----------

While Distributed can be used to leverage GPU workloads through libraries such as `cuDF <https://docs.rapids.ai/api/cudf/stable/>`_, `CuPy <https://cupy.dev/>`_, and `Numba <https://numba.pydata.org/>`_, Dask-CUDA offers several unique features unavailable to Distributed:

- **Automatic instantiation of per-GPU workers** -- Using Dask-CUDA's LocalCUDACluster or ``dask-cuda-worker`` CLI will automatically launch one worker for each GPU available on the executing node, avoiding the need to explicitly select GPUs.
- **Automatic setting of CPU affinity**  -- The setting of CPU affinity for each GPU is done automatically, preventing memory transfers from taking suboptimal paths. 
- **Automatic selection of InfiniBand devices** -- When UCX communication is enabled over InfiniBand, Dask-CUDA automatically selects the optimal InfiniBand device for each GPU (see :doc:`UCX <ucx>` for instructions on enabling UCX communication).
- **Memory spilling from GPU** -- For memory-intensive workloads, Dask-CUDA supports spilling from device to host memory when a GPU reaches the default or user-specified memory utilization limit.

Contents
--------

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   install
   quickstart
   api

.. toctree::
   :maxdepth: 1
   :caption: Additional Features

   ucx
