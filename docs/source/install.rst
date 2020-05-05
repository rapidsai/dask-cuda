Install
=======

Prerequisites
-------------

UCX depends on the following system libraries being present: ``libibcm``,
``libibverbs``, ``librdmacm``, and ``libnuma`` (``numactl`` on Enterprise
Linux).  Please install these with your Linux system's package manager. When
building from source you will also need the ``*-dev`` (``*-devel`` on
Enterprise Linux) packages as well.

Conda
-----

Some preliminary Conda packages can be installed as so. Replace
``<CUDA version>`` with either ``9.2``, ``10.0``, or ``10.1``. These are
available both on ``rapidsai`` and ``rapidsai-nightly``.

With GPU support:

::

    conda create -n ucx -c conda-forge -c rapidsai \
      cudatoolkit=<CUDA version> ucx-proc=*=gpu ucx ucx-py python=3.7

Without GPU support:

::

    conda create -n ucx -c conda-forge -c rapidsai \
      ucx-proc=*=cpu ucx ucx-py python=3.7

Note: These use UCX's ``v1.7.x`` branch.

Source
------

The following instructions assume you'll be using ucx-py on a CUDA enabled system and is in a `Conda environment <https://docs.conda.io/projects/conda/en/latest/>`_.


Build Dependencies
~~~~~~~~~~~~~~~~~~

::

    conda create -n ucx -c conda-forge \
        automake make libtool pkg-config \
        libhwloc \
        python=3.7 setuptools cython>=0.29.14,<3.0.0a0

Test Dependencies
~~~~~~~~~~~~~~~~~

::

    conda install -n ucx -c rapidsai -c nvidia -c conda-forge \
        pytest pytest-asyncio \
        cupy numba>=0.46 rmm \
        distributed

UCX
~~~

::

    conda activate ucx
    git clone https://github.com/openucx/ucx
    cd ucx
    git checkout v1.7.x
    ./autogen.sh
    mkdir build
    cd build
    # Performance build
    ../contrib/configure-release --prefix=$CONDA_PREFIX --with-cuda=$CUDA_HOME --enable-mt CPPFLAGS="-I/$CUDA_HOME/include"
    # Debug build
    ../contrib/configure-devel --prefix=$CONDA_PREFIX --with-cuda=$CUDA_HOME --enable-mt CPPFLAGS="-I/$CUDA_HOME/include"
    make -j install

UCX-Py
~~~~~~

::

    conda activate ucx
    git clone git@github.com:rapidsai/ucx-py.git
    cd ucx-py
    python setup.py build_ext --inplace
    pip install .
    # or for develop build
    pip install -v -e .
