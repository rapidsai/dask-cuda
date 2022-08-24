API
===

Cluster
-------
.. currentmodule:: dask_cuda
.. autoclass:: LocalCUDACluster
   :members:

Worker
------
.. click:: dask_cuda.cli.dask_cuda_worker:cuda
   :prog: dask cuda
   :nested: none

Client initialization
---------------------
.. currentmodule:: dask_cuda.initialize
.. autofunction:: initialize


Explicit-comms
--------------
.. currentmodule:: dask_cuda.explicit_comms.comms
.. autoclass:: CommsContext
   :members:

