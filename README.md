Dask CUDA
=========

Various utilities to improve deployment and management of Dask workers on
CUDA-enabled systems.

This library is experimental, and its API is subject to change at any time
without notice.

Example
-------

```python
from dask_cuda import LocalCUDACluster
from dask.distributed import Client

cluster = LocalCUDACluster()
client = Client(cluster)
```
