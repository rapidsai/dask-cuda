from __future__ import annotations

import copy
from collections.abc import Sequence

import psutil

import dask.dataframe as dd
from dask import config
from dask.tokenize import tokenize
from distributed import get_client
from distributed.diagnostics.plugin import SchedulerPlugin
from distributed.scheduler import Scheduler, TaskState

from dask_cuda import LocalCUDACluster


class GPURestrictorPlugin(SchedulerPlugin):
    """Scheduler Plugin to restrict tasks to a 'gpu' resource

    TODO: Move to `plugins` module.

    The plugin will restrict all tasks to "gpu" resources
    unless those tasks start with a name that is included
    in `_free_task_names`.
    """

    scheduler: Scheduler
    _free_task_names: Sequence[str]

    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler
        self.scheduler.stream_handlers.update(
            {"add_free_task_names": self.add_free_task_names}
        )
        self.scheduler.add_plugin(self, name="gpu_restrictor")
        self._free_task_names = set()

    def add_free_task_names(self, *args, **kwargs) -> None:
        key_names = kwargs.pop("key_names", ())
        for k in key_names:
            self._free_task_names.add(k)

    def update_graph(self, *args, **kwargs) -> None:
        tasks = kwargs.pop("tasks", [])
        for key in tasks:
            ts: TaskState = self.scheduler.tasks[key]
            if isinstance(key, tuple) and key and key[0] in self._free_task_names:
                continue  # This is a "free" task
            else:
                ts.resource_restrictions = {"gpu": 1}


def _no_op(x):
    return x


def _combine(dfs: list):
    import pandas as pd

    import cudf

    return cudf.from_pandas(pd.concat(dfs))


class LocalHybridCluster(LocalCUDACluster):
    def __init__(
        self,
        n_workers=None,
        n_gpu_workers=None,
        n_cpu_workers=None,
        resources=None,
        **kwargs,
    ):
        if resources is not None:
            raise ValueError("resources is not a supported by LocalHybridCluster.")

        preloads = config.get("distributed.scheduler.preload")
        preloads.append("dask_cuda.hybrid.cluster")
        config.set({"distributed.scheduler.preload": preloads})

        if n_workers is not None:
            raise ValueError(
                f"Got n_workers={n_workers}. "
                "Please use n_gpu_workers and n_cpu_workers only."
            )

        super().__init__(
            n_workers=n_gpu_workers,
            resources={"gpu": 1},
            **kwargs,
        )

        if n_cpu_workers is None:
            # By default, add an extra CPU worker for every physical core
            n_cpu_workers = psutil.cpu_count(logical=False)

        if n_cpu_workers > 0:
            # Add cpu workers
            self.scale(n_cpu_workers + len(self.worker_spec))

    def new_worker_spec(self):
        try:
            # Add GPU workers until we have a worker
            # for every visible cuda device
            name = min(set(self.cuda_visible_devices) - set(self.worker_spec))
        except Exception:
            # Add a cpu-only worker
            name = max(self.worker_spec) + 1
            spec = copy.deepcopy(self.new_spec)
            spec["options"].update({"resources": {"cpu": 1}})
            # TODO: Make the CPU worker threaded?
            return {name: spec}
        return super().new_worker_spec()

    def read_parquet(self, *args, agg_factor=1, **kwargs):
        # TODO: Implement custom/optimized logic
        # (Avoid unnecessary pa.Table->pd.Dataframe)
        # TODO: Enable column projection, etc?
        assert dd.DASK_EXPR_ENABLED

        # Need a client to send "free" keys to the scheduler
        client = get_client()

        # Use arrow/pandas for IO
        with config.set({"dataframe.backend": "pandas"}):
            df0 = dd.read_parquet(*args, **kwargs)

        # "Hack" to enable small-file aggregation
        df0 = df0.map_partitions(
            _no_op,
            meta=df0._meta,
            enforce_metadata=False,
        ).optimize()

        # Let the scheduler know that these "IO"
        # tasks are free to run anywhere
        client._send_to_scheduler(
            {
                "op": "add_free_task_names",
                "key_names": [df0._name],
            }
        )

        # Use from_graph to make sure IO tasks don't change
        token = tokenize(df0, agg_factor)
        name = f"cpu-to-gpu-{token}"
        io_keys = [(df0._name, i) for i in range(df0.npartitions)]
        dsk = {
            (name, i): (_combine, io_keys[i : i + agg_factor])
            for i in range(0, len(io_keys), agg_factor)
        }
        output_keys = list(dsk.keys())
        dsk.update(df0.dask)
        meta = _combine([df0._meta])
        divisions = (None,) * (len(output_keys) + 1)
        name_prefix = "pq"
        df0 = dd.from_graph(dsk, meta, divisions, output_keys, name_prefix)
        return df0


def dask_setup(scheduler):
    GPURestrictorPlugin(scheduler)
