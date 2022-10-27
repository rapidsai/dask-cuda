.. _controlling-number-of-workers:

Controlling number of workers
=============================

Users can restrict activity to specific GPUs by explicitly setting ``CUDA_VISIBLE_DEVICES``; for a LocalCUDACluster, this can provided as a keyword argument.
For example, to restrict activity to the first two indexed GPUs:

.. code-block:: python

    from dask_cuda import LocalCUDACluster

    cluster = LocalCUDACluster(CUDA_VISIBLE_DEVICES="0,1")

LocalCUDACluster can also take an ``n_workers`` argument, which will restrict activity to the first N GPUs listed in ``CUDA_VISIBLE_DEVICES``.
This argument can be used on its own or in conjunction with ``CUDA_VISIBLE_DEVICES``:

.. code-block:: python

    cluster = LocalCUDACluster(n_workers=2)                                # will use GPUs 0,1
    cluster = LocalCUDACluster(CUDA_VISIBLE_DEVICES="3,4,5", n_workers=2)  # will use GPUs 3,4

When using ``dask-cuda-worker``, ``CUDA_VISIBLE_DEVICES`` must be provided as an environment variable:

.. code-block:: bash

    $ dask-scheduler
    distributed.scheduler - INFO -   Scheduler at:  tcp://127.0.0.1:8786

    $ CUDA_VISIBLE_DEVICES=0,1 dask-cuda-worker 127.0.0.1:8786

GPUs can also be selected by their UUIDs, which can be acquired using `NVIDIA System Management Interface <https://developer.nvidia.com/nvidia-system-management-interface>`_:

.. code-block:: bash

    $ nvidia-smi -L
    GPU 0: Tesla V100-SXM2-32GB (UUID: GPU-dae76d0e-3414-958a-8f3e-fc6682b36f31)
    GPU 1: Tesla V100-SXM2-32GB (UUID: GPU-60f2c95a-c564-a078-2a14-b4ff488806ca)

These UUIDs can then be passed to ``CUDA_VISIBLE_DEVICES`` in place of a GPU index:

.. code-block:: python

    cluster = LocalCUDACluster(CUDA_VISIBLE_DEVICES="GPU-dae76d0e-3414-958a-8f3e-fc6682b36f31")

.. code-block:: bash

    $ CUDA_VISIBLE_DEVICES="GPU-dae76d0e-3414-958a-8f3e-fc6682b36f31" \
    > dask-cuda-worker 127.0.0.1:8786
