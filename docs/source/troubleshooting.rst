Troubleshooting
===============

This is a list of common issues encountered with Dask-CUDA and various systems.

Wrong Device Indexing
---------------------

It's common to rely on the device indexing presented by ``nvidia-smi`` when creating workers, and that is the default
in Dask-CUDA.  In most cases, ``nvidia-smi`` provides a one-to-one mapping with ``CUDA_VISIBLE_DEVICES``, but in some
systems the ordering may not match. While, ``CUDA_VISIBLE_DEVICES`` indexes GPUs by their PCI Bus ID, ``nvidia-smi``
orders by fastest GPUs.  Issues are commonly seen in
`DGX Station A100 <https://www.nvidia.com/en-us/data-center/dgx-station-a100/>`_ that contains 4 A100 GPUs, plus a
Display GPU, but the Display GPU may not be the last GPU according to the PCI Bus ID. To correct that and ensure the
mapping according to the PCI Bus ID, it's necessary to set the ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` environment variable
when starting the Python process:

.. code-block:: bash

    $ CUDA_DEVICE_ORDER=PCI_BUS_ID python
    $ CUDA_DEVICE_ORDER=PCI_BUS_ID ipython
    $ CUDA_DEVICE_ORDER=PCI_BUS_ID jupyter lab
    $ CUDA_DEVICE_ORDER=PCI_BUS_ID dask-cuda-worker ...


For the DGX Station A100, the display GPU is commonly the fourth in the PCI Bus ID ordering, thus one needs to use GPUs
0, 1, 2 and 4 for Dask-CUDA:

.. code-block:: python

    >>> from dask_cuda import LocalCUDACluster
    >>> cluster = LocalCUDACluster(CUDA_VISIBLE_DEVICES=[0, 1, 2, 4])
