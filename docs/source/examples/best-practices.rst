Best Practices
==============


Multi-GPUs
~~~~~~~~~~

When using multiple GPUs, if possible, it's best to co-locate as many GPUs on the same physical node.  This could be a
DGX, a cloud instance with `multi-gpu options <https://rapids.ai/cloud>`_ , a high-density GPU HPC instance, etc.  This is done for
two reasons:

- 1. Moving data between GPUs is costly and performance decreases when computation stops due to communication overheads, Host-to-Device/Device-to-Host transfers, etc
- 2. Mutli-GPU instances often come with accelerated networking like `NVLink <https://www.nvidia.com/en-us/data-center/nvlink/>`_.  These accelerated
networking paths usually have much higher throughput/bandwidth compared with traditional networking *and* don't force and H-to-D/D-to-H transfers.  See `
Accelerated Networking`_ for more discussion

.. code-block:: python

    from dask_cuda import LocalCUDACluster

    cluster = LocalCUDACluster(CUDA_VISIBLE_DEVICES="0,1",
                               protocol="ucx")

GPU Memory Management
~~~~~~~~~~~~~~~~~~~~~

When using Dask-CUDA, especially with RAPIDS, it's best to use `RMM <https://docs.rapids.ai/api/rmm/stable/>`_ (RAPIDS Memory Manager)
to pre-allocate memory on the GPU.  Allocating memory, while fast, takes a small amount of times, however, one can easily make
hundreds of thousand or even millions of allocations in trivial workflows.  With RMM, allocations are sub-sampled from a larger
pool and greatly reduces the allocation time and thereby increases performance:

.. code-block:: python

    from dask_cuda import LocalCUDACluster

    cluster = LocalCUDACluster(CUDA_VISIBLE_DEVICES="0,1",
                               protocol="ucx",
                               rmm_pool_size="30GB")


We also recommend allocating most, though not all, of the GPU memory space.  We do this so that temporary allocations, like those made deep inside CUDA libraries (libcudf, cupy, xgboost, etc) have room and the job is not halted due to an Out-Of-Memory (``OOM``) error.


Accelerated Networking
~~~~~~~~~~~~~~~~~~~~~~

As discussed in `Multi-GPUs`_, accelerated networking has better bandwidth/throughput compared with traditional networking hardware and does
not force any costly H-to-D/D-to-H transfers.  Dask-CUDA can leverage accelerated networking hardware with `UCX-Py <https://ucx-py.readthedocs.io/en/latest/>`_.

As an example, let's compare a merge benchmark when using 2 GPUs connected with NVLink.  First we'll run with standard TCP comms:

::

    python local_cudf_merge.py -d 0,1 -p tcp -c 50_000_000 --rmm-pool-size 30GB


In the above, we used 2 GPUs (2 dask-cuda-workers), pre-allocated 30GB of GPU ram (to make gpu memory allocations faster), and used TCP comms
when Dask needed to move data back-and-forth between workers results in an average wall clock time of: ``19.72 s +/- 694.36 ms``::

    ================================================================================
    Wall clock                | Throughput
    --------------------------------------------------------------------------------
    20.09 s                   | 151.93 MiB/s
    20.33 s                   | 150.10 MiB/s
    18.75 s                   | 162.75 MiB/s
    ================================================================================
    Throughput                | 154.73 MiB/s +/- 3.14 MiB/s
    Bandwidth                 | 139.22 MiB/s +/- 2.98 MiB/s
    Wall clock                | 19.72 s +/- 694.36 ms
    ================================================================================
    (w1,w2)                   | 25% 50% 75% (total nbytes)
    --------------------------------------------------------------------------------
    (0,1)                     | 138.48 MiB/s 150.16 MiB/s 157.36 MiB/s (8.66 GiB)
    (1,0)                     | 107.01 MiB/s 162.38 MiB/s 188.59 MiB/s (8.66 GiB)
    ================================================================================
    Worker index              | Worker address
    --------------------------------------------------------------------------------
    0                         | tcp://127.0.0.1:44055
    1                         | tcp://127.0.0.1:41095
    ================================================================================


To compare, we'll now change the ``procotol`` from ``tcp`` to ``ucx``:

    python local_cudf_merge.py -d 0,1 -p ucx -c 50_000_000 --rmm-pool-size 28GB



With UCX and NVLink, we greatly reduced the wall clock time to: ``347.43 ms +/- 5.41 ms``.::

    ================================================================================
    Wall clock                | Throughput
    --------------------------------------------------------------------------------
    354.87 ms                 | 8.40 GiB/s
    345.24 ms                 | 8.63 GiB/s
    342.18 ms                 | 8.71 GiB/s
    ================================================================================
    Throughput                | 8.58 GiB/s +/- 78.96 MiB/s
    Bandwidth                 | 6.98 GiB/s +/- 46.05 MiB/s
    Wall clock                | 347.43 ms +/- 5.41 ms
    ================================================================================
    (w1,w2)                   | 25% 50% 75% (total nbytes)
    --------------------------------------------------------------------------------
    (0,1)                     | 17.38 GiB/s 17.94 GiB/s 18.88 GiB/s (8.66 GiB)
    (1,0)                     | 16.55 GiB/s 17.80 GiB/s 18.87 GiB/s (8.66 GiB)
    ================================================================================
    Worker index              | Worker address
    --------------------------------------------------------------------------------
    0                         | ucx://127.0.0.1:35954
    1                         | ucx://127.0.0.1:53584
    ================================================================================

