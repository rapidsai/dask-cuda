import asyncio
from typing import List

from dask.dataframe.core import DataFrame, _concat
from dask.dataframe.shuffle import partitioning_index, shuffle_group
from distributed.protocol import to_serialize

from . import comms


async def send_df(ep, df):
    if df is None:
        return await ep.write("empty")
    else:
        return await ep.write([to_serialize(df)])


async def recv_df(ep):
    ret = await ep.read()
    if ret == "empty":
        return None
    else:
        return ret[0]


async def barrier(rank, eps):
    if rank == 0:
        await asyncio.gather(*[ep.read() for ep in eps.values()])
    else:
        await eps[0].write("dummy")


async def send_bins(eps, bins):
    futures = []
    for rank, ep in eps.items():
        futures.append(send_df(ep, bins[rank]))
    await asyncio.gather(*futures)


async def recv_bins(eps, bins):
    futures = []
    for ep in eps.values():
        futures.append(recv_df(ep))
    bins.extend(await asyncio.gather(*futures))


async def exchange_and_concat_bins(rank, eps, bins):
    ret = [bins[rank]]
    await asyncio.gather(recv_bins(eps, ret), send_bins(eps, bins))
    return _concat([df for df in ret if df is not None])


def df_concat(df_parts):
    """Making sure df_parts is a single dataframe or None"""
    if len(df_parts) == 0:
        return None
    elif len(df_parts) == 1:
        return df_parts[0]
    else:
        return _concat(df_parts)


def partition_by_hash(df, columns, n_chunks, ignore_index=False):
    """Splits dataframe into partitions

    The partitions is determined by the hash value of the rows in `columns`.

    Parameters
    ----------
    df: DataFrame
    columns: label or list
        Column names on which to split the dataframe
    npartition: int
        Number of partitions
    ignore_index : bool, default False
        Set True to ignore the index of `df`

    Returns
    -------
    out: Dict[int, DataFrame]
        A dictionary mapping integers in {0..npartition} to dataframes.
    """
    if df is None:
        return [None] * n_chunks

    # Hashing `columns` in `df` and assing it to the "_partitions" column
    df["_partitions"] = partitioning_index(df[columns], n_chunks)
    # Split `df` based on the hash values in the "_partitions" column
    try:
        # For Dask < 2.17 compatibility
        ret = shuffle_group(df, "_partitions", 0, n_chunks, n_chunks, ignore_index)
    except TypeError:
        ret = shuffle_group(
            df, "_partitions", 0, n_chunks, n_chunks, ignore_index, n_chunks
        )

    # Let's remove the partition column and return the partitions
    del df["_partitions"]
    for df in ret.values():
        del df["_partitions"]
    return ret


async def shuffle(n_chunks, rank, eps, left_table, column):
    left_bins = partition_by_hash(left_table, column, n_chunks, ignore_index=True)
    return await exchange_and_concat_bins(rank, eps, left_bins)


async def _shuffle(s, workers, dfs_nparts, dfs_parts, column):
    """
    Parameters
    ----------
    s: dict
        Worker session state
    workers: set
        Set of ranks of all the participants
    dfs_nparts: list of dict
        List of dict that for each worker rank specifices the
        number of partitions that worker has. If the worker doesn't
        have any partitions, it is excluded from the dict.
        E.g. `dfs_nparts[0][1]` is how many partitions of the "left"
        dataframe worker 1 has.
    dfs_parts: list of lists of Dataframes
        List of inputs, which in this case are two dataframe lists.
    column : label or list, or array-like
        The bases of the rearrangement.
    """
    assert s["rank"] in workers
    df_parts = dfs_parts[0]

    # Trimming such that all participanting workers get a rank within 0..len(workers)
    trim_map = {}
    for i in range(s["nworkers"]):
        if i in workers:
            trim_map[i] = len(trim_map)

    rank = trim_map[s["rank"]]
    eps = {trim_map[i]: s["eps"][trim_map[i]] for i in workers if i != s["rank"]}

    df = df_concat(df_parts)

    return await shuffle(len(workers), rank, eps, df, column)


def dataframe_shuffle(df: DataFrame, column_names: List[str]) -> DataFrame:
    """Order divisions of DataFrame so that all values within column(s) align

    This enacts a task-based shuffle using explicit-comms. It requires a full
    dataset read, serialization and shuffle. This is expensive. If possible
    you should avoid shuffles.

    This does not preserve a meaningful index/partitioning scheme. This is not
    deterministic if done in parallel.

    Requires an activate client.

    Notice
    ------
    As a side effect, this operation concatenate all partitions located on
    the same worker thus npartitions of the returned dataframe equals number
    of available workers.

    Parameters
    ----------
    df: dask.dataframe.DataFrame
    column_names: list of strings
        List of column names on which we want to split.

    Returns
    -------
    df: dask.dataframe.DataFrame
        Shuffled dataframe
    """

    return comms.default_comms().dataframe_operation(
        _shuffle, df_list=(df,), extra_args=(column_names,),
    )
