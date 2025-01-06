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

Setting CPU Affinity Failure
----------------------------

Setting the proper CPU affinity for a Dask-CUDA worker is important to ensure optimal performance, particularly when
memory transfers to/from system memory is necessary. In Dask-CUDA this is an automatic feature that attempts to
determine the appropriate CPU affinity for each worker according to the GPU that worker is targeting.

There are situations where setting the CPU affinity may fail, the more common case involves workload managers and job
schedulers used by large compute clusters, such as Slurm.

Within a node with multiple physical CPU (i.e., multiple CPU sockets) and multiple GPUs, in such systems it is common
for GPUs to be directly connected to a specific physical CPU to balance resources. Consider for example a node with 4
GPUs and 40 CPU cores, where the CPU cores are split between two physical CPUs, in this case GPUs 0 and 1 may be
connected to CPUs 0-19 and GPUs 2 and 3 may be connected to CPUs 20-39. In a setup like this, if the node is entirely
assigned to the Dask-CUDA job, most likely setting CPU affinity will succeed, however, it is still possible that the
job assigns the wrong CPUs 20-39 to GPUs 0 and 1, or CPUs 0-19 to GPUs 2 and 3, in this case setting the CPU affinity
will be impossible, since the correct CPU/GPU resources are not available to the job. When this happens, the best
Dask-CUDA can do is raise a warning that redirects you to this sections and not set any CPU affinity, letting the
operating system handle all transfers as it sees fit, even if they may follow a suboptimal path.

If after following the instructions contained in this section, including consulting your cluster's manual and
administrators, please [file an issue under the Dask-CUDA repository](https://github.com/rapidsai/dask-cuda/issues),
including the output for all commands below, they must be executed from the allocated cluster job:

- ``conda list``, if environment was installed with conda or uses a RAPIDS provided Docker image;
- ``pip list``, if environment was installed with pip;
- ``nvidia-smi``;
- ``nvidia-smi topo -m``;
- ``python print_affinity.py``, the code for ``print_affinity.py`` immediately follows.

.. code-block:: python

    # print_affinity.py
    import math
    from multiprocessing import cpu_count

    import pynvml

    pynvml.nvmlInit()
    for i in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        cpu_affinity = pynvml.nvmlDeviceGetCpuAffinity(handle, math.ceil(cpu_count() / 64))
        print(f"GPU {i}: list(cpu_affinity)")

Slurm
~~~~~

The more commonly observed cases of this issue have been reported on Slurm clusters. Common ways to resolve this
normally involve providing a specific subset of CPUs to the job with one of the following arguments:

- `--cpus-per-task=N`: the number of CPUs the job will have allocated, you may need to ask for all CPUs to ensure
  the GPUs have all CPUs relevant to them available;
- `--exclusive`: to ensure exclusive allocation of CPUs to the job.

Unfortunately, providing exact solutions for all existing clust configurations is not possible, therefore make
make sure to consult your cluster's manual and administrator for detailed information and further troubleshooting.
