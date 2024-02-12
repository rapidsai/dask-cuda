import glob
import os
import pickle
from collections import defaultdict

from dask.blockwise import BlockIndex
from distributed import wait
from distributed.protocol import dask_deserialize, dask_serialize

from dask_cuda.explicit_comms import comms


class LazyLoad:
    def __init__(self, path, index, **kwargs):
        self.path = path
        self.index = index
        self.kwargs = kwargs

    def pre_serialize(self):
        """Make the unloaded partition serializable"""
        return self.load()

    def load(self):
        """Load the partition into memory"""
        import cudf

        fn = glob.glob(f"{self.path}/*.{self.index}.parquet")
        return cudf.read_parquet(fn, **self.kwargs)


@dask_serialize.register(LazyLoad)
def _serialize_unloaded(obj):
    return None, [pickle.dumps(obj.pre_serialize())]


@dask_deserialize.register(LazyLoad)
def _deserialize_unloaded(header, frames):
    return pickle.loads(frames[0])


def _prepare_dir(dirpath: str):
    os.makedirs(dirpath, exist_ok=True)


def _clean_worker_storage(dirpath: str):
    import shutil

    if os.path.isdir(dirpath):
        shutil.rmtree(dirpath)


def _write_partition(part, dirpath, index, token=None):
    if token is None:
        fn = f"{dirpath}/part.{index[0]}.parquet"
    else:
        fn = f"{dirpath}/part.{token}.{index[0]}.parquet"
    part.to_parquet(fn)
    return index


def _get_partition(dirpath, index):
    return LazyLoad(dirpath, index)


def _get_metadata(dirpath, index):
    import glob

    import pyarrow.parquet as pq

    import cudf

    fn = glob.glob(f"{dirpath}/*.{index}.parquet")[0]
    return cudf.DataFrame.from_arrow(
        pq.ParquetFile(fn).schema.to_arrow_schema().empty_table()
    )


def _load_partition(data):
    if isinstance(data, LazyLoad):
        data = data.load()
    return data


def to_worker_storage(df, dirpath, shuffle_on=None, overwrite=False, **kwargs):

    if shuffle_on:
        from dask_cuda.explicit_comms.dataframe.shuffle import shuffle_to_parquet

        if not isinstance(shuffle_on, list):
            shuffle_on = [shuffle_on]
        return shuffle_to_parquet(
            df, shuffle_on, dirpath, overwrite=overwrite, **kwargs
        )

    c = comms.default_comms()
    if overwrite:
        wait(c.client.run(_clean_worker_storage, dirpath))
    wait(c.client.run(_prepare_dir, dirpath))
    df.map_partitions(
        _write_partition,
        dirpath,
        BlockIndex((df.npartitions,)),
        **kwargs,
    ).compute()


def from_worker_storage(dirpath):
    import dask_cudf

    c = comms.default_comms()

    def get_indices(path):
        return {int(fn.split(".")[-2]) for fn in glob.glob(path + "/*.parquet")}

    worker_indices = c.client.run(get_indices, dirpath)

    summary = defaultdict(list)
    for worker, indices in worker_indices.items():
        for index in indices:
            summary[index].append(worker)

    assignments = {}
    futures = []
    meta = None
    for i, (worker, indices) in enumerate(summary.items()):
        assignments[worker] = indices[i % len(indices)]
        futures.append(
            c.client.submit(_get_partition, dirpath, i, workers=[assignments[i]])
        )
        if meta is None:
            meta = c.client.submit(_get_metadata, dirpath, i, workers=[assignments[i]])
            wait(meta)
            meta = meta.result()
    wait(futures)

    return dask_cudf.from_delayed(futures, meta=meta, verify_meta=False).map_partitions(
        _load_partition,
        meta=meta,
        enforce_metadata=False,
    )
