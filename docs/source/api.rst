API
===

Cluster
-------
.. currentmodule:: dask_cuda
.. autoclass:: LocalCUDACluster
   :members:

CLI
---

Worker
~~~~~~
.. click:: dask_cuda.cli:worker
   :prog: dask cuda
   :nested: none

Cluster configuration
~~~~~~~~~~~~~~~~~~~~~
.. click:: dask_cuda.cli:config
   :prog: dask cuda
   :nested: none

Client initialization
---------------------
.. currentmodule:: dask_cuda.initialize
.. autofunction:: initialize


Explicit-comms
--------------
.. deprecated:: 26.4.0
   The explicit comms feature is deprecated and will be removed in a future version.

.. currentmodule:: dask_cuda.explicit_comms.comms
.. autoclass:: CommsContext
   :members:

.. currentmodule:: dask_cuda.explicit_comms.dataframe.shuffle
.. autofunction:: shuffle
