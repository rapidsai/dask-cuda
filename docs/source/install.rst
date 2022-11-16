Installation
============

Dask-CUDA can be installed using ``conda``, ``pip``, or from source.

Conda
-----

To use Dask-CUDA on your system, you will need:

- NVIDIA drivers for your GPU; see `NVIDIA Driver Installation Quickstart Guide <https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html>`_ for installation instructions
- A version of NVIDIA CUDA Toolkit compatible with the installed driver version; see Table 1 of `CUDA Compatibility -- Binary Compatibility <https://docs.nvidia.com/deploy/cuda-compatibility/index.html#binary-compatibility>`_ for an overview of CUDA Toolkit driver requirements

Once the proper CUDA Toolkit version has been determined, it can be installed using along with Dask-CUDA using ``conda``.
To install the latest version of Dask-CUDA along with CUDA Toolkit 11.5:

.. code-block:: bash

    conda install -c rapidsai -c conda-forge -c nvidia dask-cuda cudatoolkit=11.5

Pip
---

When working outside of a Conda environment, CUDA Toolkit can be downloaded and installed from `NVIDIA's website <https://developer.nvidia.com/cuda-toolkit>`_; this package also contains the required NVIDIA drivers.
To install the latest version of Dask-CUDA:

.. code-block:: bash

    python -m pip install dask-cuda

Source
------

To install Dask-CUDA from source, the source code repository must be cloned from GitHub:

.. code-block:: bash

    git clone https://github.com/rapidsai/dask-cuda.git
    cd dask-cuda
    python -m pip install .

Other RAPIDS libraries
----------------------

Dask-CUDA is a part of the `RAPIDS <https://rapids.ai/>`_ suite of open-source software libraries for GPU-accelerated data science, and works well in conjunction with them.
See `RAPIDS -- Getting Started <https://rapids.ai/start.html>`_ for instructions on how to install these libraries.
Keep in mind that these libraries will require:

- At least one CUDA-compliant GPU
- A system installation of `CUDA <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html>`_
