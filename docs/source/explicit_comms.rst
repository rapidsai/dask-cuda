Explicit-comms
==============

Communication and scheduling overhead can be a major bottleneck in Dask/Distributed. Dask-CUDA addresses this by introducing an API for explicit communication in Dask tasks.
The idea is that Dask/Distributed spawns workers and distribute data as usually while the user can submit tasks on the workers that communicate explicitly.



Usage
-----

In order to use explicit-comms in Dask/Distributed automatically, simply define the environment variable ``DASK_EXPLICIT_COMMS=True`` or setting the ``"explicit-comms"``
key in the `Dask configuration <https://docs.dask.org/en/latest/configuration.html>`_.

It is also possible to use explicit-comms in tasks manually, see the `API <api.html#explicit-comms>`_ and our `implementation of shuffle <https://github.com/rapidsai/dask-cuda/blob/branch-0.20/dask_cuda/explicit_comms/dataframe/shuffle.py>`_ for guidance.

