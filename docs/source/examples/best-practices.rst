Best Practices
==============


Multi-GPU Machines
~~~~~~~~~~~~~~~~~~

When choosing between two multi-GPU setups, it is best to pick the one where most GPUs are co-located with one-another.  This could be a
`DGX <https://www.nvidia.com/en-us/data-center/dgx-systems/>`_, a cloud instance with `multi-gpu options <https://rapids.ai/cloud>`_ , a high-density GPU HPC instance, etc.  This is done for two reasons:

- Moving data between GPUs is costly and performance decreases when computation stops due to communication overheads, Host-to-Device/Device-to-Host transfers, etc
- Multi-GPU instances often come with accelerated networking like `NVLink <https://www.nvidia.com/en-us/data-center/nvlink/>`_.  These accelerated networking paths usually have much higher throughput/bandwidth compared with traditional networking *and* don't force and Host-to-Device/Device-to-Host transfers.  See `Accelerated Networking`_ for more discussion.

.. code-block:: python

    from dask_cuda import LocalCUDACluster

    cluster = LocalCUDACluster(n_workers=2)                                # will use GPUs 0,1
    cluster = LocalCUDACluster(CUDA_VISIBLE_DEVICES="3,4")                 # will use GPUs 3,4

For more discussion on controlling number of workers/using multiple GPUs see :ref:`controlling-number-of-workers` .

GPU Memory Management
~~~~~~~~~~~~~~~~~~~~~

When using Dask-CUDA, especially with RAPIDS, it's best to use an |rmm-pool|__ to pre-allocate memory on the GPU.  Allocating memory, while fast, takes a small amount of time, however, one can easily make
hundreds of thousand or even millions of allocations in trivial workflows causing significant performance degradations.  With an RMM pool, allocations are sub-sampled from a larger pool and this greatly reduces the allocation time and thereby increases performance:


  .. |rmm-pool| replace:: :abbr:`RMM (RAPIDS Memory Manager)` pool
  __ https://docs.rapids.ai/api/rmm/stable/


.. code-block:: python

    from dask_cuda import LocalCUDACluster

    cluster = LocalCUDACluster(CUDA_VISIBLE_DEVICES="0,1",
                               protocol="ucx",
                               rmm_pool_size="30GB")


We also recommend allocating most, though not all, of the GPU memory space. We do this because the `CUDA Context <https://stackoverflow.com/questions/43244645/what-is-a-cuda-context#:~:text=The%20context%20holds%20all%20the,memory%20for%20zero%20copy%2C%20etc.>`_ takes a non-zero amount (typically 200-500 MBs) of GPU RAM on the device.

Additionally, when using `Accelerated Networking`_ , we only need to register a single IPC handle for the whole pool (which is expensive, but only done once) since from the IPC point of viewer there's only a single allocation. As opposed to just using RMM without a pool where each new allocation must be registered with IPC.

Spilling from Device
~~~~~~~~~~~~~~~~~~~~

Dask-CUDA offers several different ways to enable automatic spilling from device memory.
The best method often depends on the specific workflow. For classic ETL workloads using
`Dask cuDF <https://docs.rapids.ai/api/dask-cudf/stable/>`_, native cuDF spilling is usually
the best place to start. See :ref:`Dask-CUDA's spilling documentation <spilling-from-device>`
for more details.

Accelerated Networking
~~~~~~~~~~~~~~~~~~~~~~

As discussed in `Multi-GPU Machines`_, accelerated networking has better bandwidth/throughput compared with traditional networking hardware and does
not force any costly Host-to-Device/Device-to-Host transfers.  Dask-CUDA can leverage accelerated networking hardware with `UCXX <https://docs.rapids.ai/api/ucxx/nightly/>`_.

As an example, let's compare a merge benchmark when using 2 GPUs connected with NVLink.  First we'll run with standard TCP comms:

::

    python local_cudf_merge.py -d 0,1 -p tcp -c 50_000_000 --rmm-pool-size 30GB


In the above, we used 2 GPUs (2 dask-cuda-workers), pre-allocated 30GB of GPU RAM (to make gpu memory allocations faster), and used TCP comms
when Dask needed to move data back-and-forth between workers. This setup results in an average wall clock time of: ``19.72 s +/- 694.36 ms``::

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

    python local_cudf_merge.py -d 0,1 -p ucx -c 50_000_000 --rmm-pool-size 30GB



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
