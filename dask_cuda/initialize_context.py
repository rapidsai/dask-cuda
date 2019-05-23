"""
It is sometimes convenient to initialize the CUDA context, particularly before
starting up Dask workers which create a variety of threads.

This module is intended to be used within a Worker preload script.
https://docs.dask.org/en/latest/setup/custom-startup.html

You can add it to your global config with the following yaml

    distributed:
      worker:
        preload:
          - dask_cuda.initialize_context

See https://docs.dask.org/en/latest/configuration.html for more information
about Dask configuration.
"""
import numba.cuda

numba.cuda.current_context()
