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

Documentation is available [here](https://docs.rapids.ai/api/dask-cuda/nightly/).

What this is not
----------------

This library does not automatically convert your Dask code to run on GPUs.

It only helps with deployment and management of Dask workers in multi-GPU
systems.  Parallelizing GPU libraries like [RAPIDS](https://rapids.ai) and
[CuPy](https://cupy.chainer.org) with Dask is an ongoing effort.  You may wish
to read about this effort at [blog.dask.org](https://blog.dask.org) for more
information.  Additional information about Dask-CUDA can also be found in the
[docs](https://docs.rapids.ai/api/dask-cuda/nightly/).
