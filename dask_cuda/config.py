import yaml
import os

import dask

config = dask.config.config


fn = os.path.join(os.path.dirname(__file__), "cuda.yaml")
dask.config.ensure_file(source=fn)
with open(fn) as f:
    dask_cuda_defaults = yaml.safe_load(f)

dask.config.update_defaults(dask_cuda_defaults)
