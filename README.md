Dask CUDA
=========

Various utilities to improve interoperation between Dask and CUDA-enabled
systems.

This repository is designed to be a catch-all for Dask and CUDA utilities.
It is experimental and should not be relied upon.

Currently Includes
------------------

-   `LocalCUDACluster`: a subclass of `dask.distributed.LocalCluster` that
    eases deployment on single-node multi-GPU systems.


Example
-------

```python
from dask_cuda import LocalCUDACluster
from dask.distributed import Client

cluster = LocalCUDACluster()
client = Client(cluster)
```
