.. _spilling-from-device:

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

The same applies for ``dask cuda worker``, and spilling can be controlled by setting ``--device-memory-limit``:

.. code-block::

    $ dask scheduler
    distributed.scheduler - INFO -   Scheduler at:  tcp://127.0.0.1:8786

    $ dask cuda worker --device-memory-limit 50000
    $ dask cuda worker --device-memory-limit 5GB
    $ dask cuda worker --device-memory-limit 0.3
    $ dask cuda worker --device-memory-limit 0


.. _cudf-spilling:

cuDF Spilling
-------------

When executing an ETL workflow with `Dask cuDF <https://docs.rapids.ai/api/dask-cudf/stable/>`_
(i.e. Dask DataFrame), it is usually best to leverage `native spilling support in cuDF
<https://docs.rapids.ai/api/cudf/stable/developer_guide/library_design/#spilling-to-host-memory>`_.

Native cuDF spilling has an important advantage over Dask-CUDA's default GPU-to-host
spilling: the latter tracks task outputs as whole units, so intermediate data created
inside a task generally cannot be spilled until the task finishes. With cuDF spilling,
individual device buffers can be spilled or unspilled while the task is still running.

When deploying a ``LocalCUDACluster``, cuDF spilling can be enabled with the ``enable_cudf_spill`` argument:

.. code-block::

    >>> from distributed import Client​
    >>> from dask_cuda import LocalCUDACluster​

    >>> cluster = LocalCUDACluster(n_workers=10, enable_cudf_spill=True)​
    >>> client = Client(cluster)​

The same applies for ``dask cuda worker``:

.. code-block::

    $ dask scheduler
    distributed.scheduler - INFO -   Scheduler at:  tcp://127.0.0.1:8786

    $ dask cuda worker --enable-cudf-spill


Statistics
~~~~~~~~~~

When cuDF spilling is enabled, it is also possible to have cuDF collect basic
spill statistics. Collecting this information can be a useful way to understand
the performance of memory-intensive workflows using cuDF.

When deploying a ``LocalCUDACluster``, cuDF spilling can be enabled with the
``cudf_spill_stats`` argument:

.. code-block::

    >>> cluster = LocalCUDACluster(n_workers=10, enable_cudf_spill=True, cudf_spill_stats=1)​

The same applies for ``dask cuda worker``:

.. code-block::

    $ dask cuda worker --enable-cudf-spill --cudf-spill-stats 1

To have each dask-cuda worker print spill statistics within the workflow, do something like:

.. code-block::

    def spill_info():
        from cudf.core.buffer.spill_manager import get_global_manager
        print(get_global_manager().statistics)
    client.submit(spill_info)

See the `cuDF spilling documentation
<https://docs.rapids.ai/api/cudf/stable/developer_guide/library_design/#statistics>`_
for more information on the available spill-statistics options.

Limitations
~~~~~~~~~~~

Although cuDF spilling is the best option for most ETL workflows using Dask cuDF,
it will be much less effective if that workflow converts between ``cudf.DataFrame``
and other data formats (e.g. ``cupy.ndarray``). Once the underlying device buffers
are "exposed" to external memory references, they become "unspillable" by cuDF.
In cases like this (e.g., Dask-CUDA combined with XGBoost), you may need to tune
``device_memory_limit``, use smaller partitions, or restructure the workflow so that
data stays in cuDF longer.
