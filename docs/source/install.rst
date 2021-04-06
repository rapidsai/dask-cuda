Installation
============

To use Dask-CUDA on your system, you will need the following:

- At least one CUDA-compatible GPU
- NVIDIA CUDA Toolkit
- *Anything else?*

Once these requirements are satisfied, Dask-CUDA can be install using ``conda``, ``pip``, or from source.

Conda
-----

To install the latest version of Dask-CUDA::

    conda install -c rapidsai -c nvidia -c conda-forge dask-cuda

Pip
---

::

    python -m pip install dask-cuda

Source
------

To install Dask-CUDA from source, the source code repository must be cloned from GitHub::

    git clone https://github.com/rapidsai/dask-cuda.git
    cd dask-cuda
    python -m pip install .

Optional libraries
------------------

Dask-CUDA is a part of the `RAPIDS <https://rapids.ai/>`_ suite of open-source software libraries for GPU-accelerated data science, and works well in conjunction with them.
See `RAPIDS -- Getting Started <https://rapids.ai/start.html>`_ for instructions on how to install these libraries.
