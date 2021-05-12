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

The same applies for ``dask-cuda-worker``, and spilling can be controlled by setting ``--device-memory-limit``:

.. code-block::

    $ dask-scheduler
    distributed.scheduler - INFO -   Scheduler at:  tcp://127.0.0.1:8786

    $ dask-cuda-worker --device-memory-limit 50000
    $ dask-cuda-worker --device-memory-limit 5GB
    $ dask-cuda-worker --device-memory-limit 0.3
    $ dask-cuda-worker --device-memory-limit 0


JIT-Unspill
-----------
The regular spilling in Dask and Dask-CUDA has some significate issues. Instead of tracking individual objects, it tracks task outputs.
This means that a task returning a collection of CUDA objects will either spill all of the CUDA objects or none of them.
Other issues includes *object duplication*, *wrong spilling order*, and *non-tracking of sharing device buffers*
(see: https://github.com/dask/distributed/issues/4568#issuecomment-805049321)


In order to address all of these issues, Dask-CUDA introduces JIT-Unspilling, which can be enabled
by using the ``jit_unspill`` argument:

.. code-block::

    >>> import dask​
    >>> from distributed import Client​
    >>> from dask_cuda import LocalCUDACluster​

    >>> cluster = LocalCUDACluster(n_workers=10, device_memory_limit="1GB", jit_unspill=True)​
    >>> client = Client(cluster)​

    >>> with dask.config.set(jit_unspill=True):​
    ...   cluster = LocalCUDACluster(n_workers=10, device_memory_limit="1GB")​
    ...   client = Client(cluster)


.. code-block::

    $ dask-scheduler
    distributed.scheduler - INFO -   Scheduler at:  tcp://127.0.0.1:8786

    $ dask-cuda-worker --enable-jit-unspill​
    $ DASK_JIT_UNSPILL=True dask-cuda-worker​


JIT-Unspill wraps all CUDA objects, such as ``cudf.Dataframe``, in a ``ProxyObject``:

.. code-block::

    class ProxyObject:
        """Object wrapper/proxy for serializable objects

        This is used by ProxifyHostFile to delay deserialization of returned objects.

        Objects proxied by an instance of this class will be JIT-deserialized when
        accessed. The instance behaves as the proxied object and can be accessed/used
        just like the proxied object.

        ProxyObject has some limitations and doesn't mimic the proxied object perfectly.
        Thus, if encountering problems remember that it is always possible to use unproxy()
        to access the proxied object directly or disable JIT deserialization completely
        with `jit_unspill=False`.

        Type checking using instance() works as expected but direct type checking
        doesn't:
        >>> import numpy as np
        >>> from dask_cuda.proxy_object import asproxy
        >>> x = np.arange(3)
        >>> isinstance(asproxy(x), type(x))
        True
        >>>  type(asproxy(x)) is type(x)
        False
