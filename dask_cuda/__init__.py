import sys

if sys.platform != "linux":
    raise ImportError("Only Linux is supported by Dask-CUDA at this time")

import dask
import dask.utils
import dask.dataframe.shuffle
from .explicit_comms.dataframe.shuffle import patch_shuffle_expression
from distributed.protocol.cuda import cuda_deserialize, cuda_serialize
from distributed.protocol.serialize import dask_deserialize, dask_serialize

from ._version import __git_commit__, __version__
from .cuda_worker import CUDAWorker

from .local_cuda_cluster import LocalCUDACluster
from .proxify_device_objects import proxify_decorator, unproxify_decorator


# Monkey patching Dask to make use of explicit-comms when `DASK_EXPLICIT_COMMS=True`
patch_shuffle_expression()
# Monkey patching Dask to make use of proxify and unproxify in compatibility mode
dask.dataframe.shuffle.shuffle_group = proxify_decorator(
    dask.dataframe.shuffle.shuffle_group
)
dask.dataframe.core._concat = unproxify_decorator(dask.dataframe.core._concat)


def _register_cudf_spill_aware():
    import cudf

    # Only enable Dask/cuDF spilling if cuDF spilling is disabled, see
    # https://github.com/rapidsai/dask-cuda/issues/1363
    if not cudf.get_option("spill"):
        # This reproduces the implementation of `_register_cudf`, see
        # https://github.com/dask/distributed/blob/40fcd65e991382a956c3b879e438be1b100dff97/distributed/protocol/__init__.py#L106-L115
        from cudf.comm import serialize


for registry in [cuda_serialize, cuda_deserialize, dask_serialize, dask_deserialize]:
    for lib in ["cudf", "dask_cudf"]:
        if lib in registry._lazy:
            registry._lazy[lib] = _register_cudf_spill_aware
