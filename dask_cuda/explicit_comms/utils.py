from collections import OrderedDict

import dask.dataframe as dd
from distributed import default_client, wait

from toolz import first
from tornado import gen


def workers_to_parts(futures):
    """
    Builds an ordered dict mapping each worker to their list
    of parts
    :param futures: list of (worker, part) tuples
    :return:
    """
    w_to_p_map = OrderedDict()
    for w, p in futures:
        if w not in w_to_p_map:
            w_to_p_map[w] = []
        w_to_p_map[w].append(p)
    return w_to_p_map


@gen.coroutine
def extract_ddf_partitions(ddf, client=None, agg=True):
    """
    Given a Dask dataframe, return an OrderedDict mapping
    'worker -> [list of futures]' for each partition in ddf.

    :param ddf: Dask.dataframe split dataframe partitions into a list of
               futures.
    :param client: dask.distributed.Client Optional client to use
    """
    client = default_client() if client is None else client

    delayed_ddf = ddf.to_delayed()
    parts = client.compute(delayed_ddf)
    yield wait(parts)

    key_to_part_dict = dict([(str(part.key), part) for part in parts])
    who_has = yield client.who_has(parts)

    worker_map = {}  # Map from part -> worker
    for key, workers in who_has.items():
        worker = first(workers)
        worker_map[key_to_part_dict[key]] = worker

    worker_to_parts = []
    for part in parts:
        worker = worker_map[part]
        worker_to_parts.append((worker, part))

    yield wait(worker_to_parts)
    raise gen.Return(worker_to_parts)


def get_meta(df):
    """
    Return the metadata from a single dataframe
    :param df: cudf.dataframe
    :return: Row data from the first row of the dataframe
    """
    ret = df.iloc[:0]
    return ret


def dataframes_to_dask_dataframe(futures, client=None):
    """
    Convert a list of futures containing Dataframes (pandas or cudf) into a
    Dask.Dataframe

    :param futures: list of futures containing dataframes
    :param client: dask.distributed.Client Optional client to use
    :return: dask.Dataframe a dask.Dataframe
    """
    c = default_client() if client is None else client
    # Convert a list of futures containing dfs back into a dask_cudf
    dfs = [d for d in futures if d.type != type(None)]  # NOQA
    meta = c.submit(get_meta, dfs[0]).result()
    return dd.from_delayed(dfs, meta=meta)
