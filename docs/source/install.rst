Installation
============

To use Dask-CUDA on your system, you will need the `NVIDIA CUDA Toolkit <https://developer.nvidia.com/cuda-toolkit>`_.
Once this requirement is satisfied, Dask-CUDA can be installed using ``conda``, ``pip``, or from source.

Conda
-----

To install the latest version of Dask-CUDA on a system with CUDA Toolkit 11.0:

.. code-block:: bash

    conda install -c rapidsai -c nvidia -c conda-forge dask-cuda cudatoolkit=11.0

Pip
---

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