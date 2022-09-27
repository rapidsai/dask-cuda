Explicit-comms
==============

Communication and scheduling overhead can be a major bottleneck in Dask/Distributed. Dask-CUDA addresses this by introducing an API for explicit communication in Dask tasks.
The idea is that Dask/Distributed spawns workers and distributes the data as usual while the user can submit tasks on the workers that communicate explicitly.

This makes it possible to bypass Distributed's scheduler and write hand-tuned computation and communication patterns. Currently, Dask-CUDA includes an explicit-comms
implementation of the Dataframe `shuffle <https://github.com/rapidsai/dask-cuda/blob/d3c723e2c556dfe18b47b392d0615624453406a5/dask_cuda/explicit_comms/dataframe/shuffle.py#L210>`_ operation used for merging and sorting.


Usage
-----

In order to use explicit-comms in Dask/Distributed for shuffle-based DataFrame operations, simply pass the ``shuffle="explicit-comms"`` key-word argument to the appropriate DataFrame API (e.g. ``ddf.sort_values(..., shuffle="explicit-comms")```). You can also change the default shuffle algorithm to explicit-comms by define the ``DASK_SHUFFLE=explicit-comms`` environment variable, or by setting the ``"shuffle"`` key to ``"explicit-comms"`` in the `Dask configuration <https://docs.dask.org/en/latest/configuration.html>`_.

It is also possible to use explicit-comms in tasks manually, see the `API <api.html#explicit-comms>`_ and our `implementation of shuffle <https://github.com/rapidsai/dask-cuda/blob/branch-0.20/dask_cuda/explicit_comms/dataframe/shuffle.py>`_ for guidance.
