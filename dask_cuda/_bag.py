import dask.bag as db

from .explicit_comms.dataframe.shuffle import get_default_shuffle_method

# We have to replace all modules that imports Dask's `get_default_shuffle_method()`
# TODO: introduce a shuffle-algorithm dispatcher in Dask so we don't need this hack
db.core.get_default_shuffle_method = get_default_shuffle_method
