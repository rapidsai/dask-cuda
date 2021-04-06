Installation
============

To use Dask-CUDA on your system, you will need the following:

- At least one CUDA-compatible GPU
- NVIDIA CUDA Toolkit
- *Anything else?*

Once these requirements are satisfied, Dask-CUDA can be install using ``conda``, ``pip``, or from source.

Conda
-----

To install the latest version of Dask-CUDA on a system using CUDA Toolkit 11.0::

    conda install -c rapidsai -c nvidia -c conda-forge dask-cuda cudatoolkit=11.0

Pip
---

To do the same with ``pip``::

    python -m pip install dask-cuda

*Is there any way to specify toolkit version here? Or is pip install not recommended*

Source
------

To install Dask-CUDA from source, the source code repository must be cloned from GitHub::

    git clone https://github.com/rapidsai/dask-cuda.git
    cd dask-cuda
    python -m pip install .
