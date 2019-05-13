Dask CUDA
=========

Various utilities to improve deployment and management of Dask workers on
CUDA-enabled systems.

This library is experimental, and its API is subject to change at any time
without notice.


What this is not
----------------

This library does not automatically convert your Dask code to run on GPUs.

It only helps with deployment and management of Dask workers in multi-GPU
systems.  Parallelizing GPU libraries like [RAPIDS](https://rapids.ai) and
[CuPy](https://cupy.chainer.org) with Dask is an ongoing effort.  You may wish
to read about this effort at [blog.dask.org](https://blog.dask.org) for more
information..


Installation
------------

Create a new Conda environment or activate an existing one:

    conda create -n dask-cuda
    conda activate dask-cuda

Install dependencies:
    
    conda install -c conda-forge "numpy>=1.16" "dask>=1.2.1"

    # For CUDA Toolkit version 9.2 
    conda install "cudatoolkit=9.2"
    pip install "cupy-cuda92>=6.0.0rc1"  # The packge in conda is too old

    # For CUDA Toolkit version 10.0 
    conda install "cudatoolkit=10.0"
    pip install "cupy-cuda100>=6.0.0rc1"  # The packge in conda is too old

Required environment variables:

    # Enable the __array_function__ protocol in NumPy
    export NUMPY_EXPERIMENTAL_ARRAY_FUNCTION=1

    # Since cupy was installed using pip, we need to make 
    # sure that cupy can find the CUDA shared library.
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

Install the `dask-cuda` pip package:

    pip install dask-cuda

Or install directly from the `dask-cuda` repo:

    git clone https://github.com/rapidsai/dask-cuda
    cd dask-cuda
    pip install .


Testing
-------

Install `pytest`:

    conda install pytest

If not already downloaded, get the source code that includes all tests:

    git clone https://github.com/rapidsai/dask-cuda
    cd dask-cuda

Run all tests:

    py.test dask_cuda


Example
-------

```python
from dask.distributed import Client, wait
from dask_cuda import LocalCUDACluster
import dask.array as da
import cupy
import numpy

if __name__ == '__main__':
    cluster = LocalCUDACluster()
    client = Client(cluster)

    # Tell Dask to use the random from cupy
    rs = da.random.RandomState(RandomState=cupy.random.RandomState)

    x = rs.random(1000, chunks=(100,))
    res = x.sum().compute()
    print("Sum of %d random numbers: %f" % (len(x), res))

    x = rs.random((10**5, 10*3), chunks=(10**4, 10**3))

    # Notice, because of the new __array_function__ protocol NumPy will 
    # delegate the SVD calculation to dask and cupy
    u, s, v = numpy.linalg.svd(x)

    print("Sum of `u` in a random SVD: %d" % u.sum().compute())
```

