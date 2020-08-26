from collections import defaultdict

from toolz import first

from dask import dataframe as dd
from distributed import default_client, get_client, wait


def extract_ddf_partitions(ddf):
    """ Returns the mapping: worker -> [list of futures]"""
    client = get_client()
    delayed_ddf = ddf.to_delayed()
    parts = client.compute(delayed_ddf)
    wait(parts)

    key_to_part = dict([(str(part.key), part) for part in parts])
    ret = defaultdict(list)  # Map worker -> [list of futures]
    for key, workers in client.who_has(parts).items():
        worker = first(
            workers
        )  # If multiple workers have the part, we pick the first worker
        ret[worker].append(key_to_part[key])
    return ret


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
