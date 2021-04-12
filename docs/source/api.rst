API
===

Cluster
-------
.. currentmodule:: dask_cuda
.. autoclass:: LocalCUDACluster
   :members:

Worker
------
.. click:: dask_cuda.cli.dask_cuda_worker:main
   :prog: dask-cuda-worker
   :nested: none

Client initialization
---------------------
.. currentmodule:: dask_cuda.initialize
.. autofunction:: initialize
