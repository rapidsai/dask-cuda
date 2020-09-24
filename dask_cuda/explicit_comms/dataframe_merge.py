import asyncio

from . import comms
from .dataframe_shuffle import (
    df_concat,
    exchange_and_concat_bins,
    partition_by_hash,
    recv_df,
    send_df,
)


async def broadcast(rank, root_rank, eps, df=None):
    if rank == root_rank:
        await asyncio.gather(*[send_df(ep, df) for ep in eps.values()])
        return df
    else:
        return await recv_df(eps[root_rank])


async def hash_join(n_chunks, rank, eps, left_table, right_table, left_on, right_on):
    left_bins = partition_by_hash(left_table, left_on, n_chunks, ignore_index=True)
    left_df = exchange_and_concat_bins(rank, eps, left_bins)
    right_bins = partition_by_hash(right_table, right_on, n_chunks, ignore_index=True)
    left_df = await left_df
    right_df = await exchange_and_concat_bins(rank, eps, right_bins)
    return left_df.merge(right_df, left_on=left_on, right_on=right_on)


async def single_partition_join(
    n_chunks,
    rank,
    eps,
    left_table,
    right_table,
    left_on,
    right_on,
    single_table,
    single_rank,
):
    if single_table == "left":
        left_table = await broadcast(rank, single_rank, eps, left_table)
    else:
        assert single_table == "right"
        right_table = await broadcast(rank, single_rank, eps, right_table)

    return left_table.merge(right_table, left_on=left_on, right_on=right_on)


async def _dataframe_merge(s, workers, dfs_nparts, dfs_parts, left_on, right_on):
    """Worker job that merge local DataFrames

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
    left_on : str or list of str
        Column to join on in the left DataFrame.
    right_on : str or list of str
        Column to join on in the right DataFrame.

    Returns
    -------
        df: DataFrame
        Merged dataframe
    """
    assert s["rank"] in workers

    # Trimming such that all participanting workers get a rank within 0..len(workers)
    trim_map = {}
    for i in range(s["nworkers"]):
        if i in workers:
            trim_map[i] = len(trim_map)

    rank = trim_map[s["rank"]]
    eps = {trim_map[i]: s["eps"][trim_map[i]] for i in workers if i != s["rank"]}

    df1 = df_concat(dfs_parts[0])
    df2 = df_concat(dfs_parts[1])

    if len(dfs_nparts[0]) == 1 and len(dfs_nparts[1]) == 1:
        return df1.merge(df2, left_on=left_on, right_on=right_on)
    elif len(dfs_nparts[0]) == 1:
        return await single_partition_join(
            len(workers),
            rank,
            eps,
            df1,
            df2,
            left_on,
            right_on,
            "left",
            trim_map[
                next(iter(dfs_nparts[0]))
            ],  # Extracting the only key in `dfs_nparts[0]`
        )
    elif len(dfs_nparts[1]) == 1:
        return await single_partition_join(
            len(workers),
            rank,
            eps,
            df1,
            df2,
            left_on,
            right_on,
            "right",
            trim_map[
                next(iter(dfs_nparts[1]))
            ],  # Extracting the only key in `dfs_nparts[1]`
        )
    else:
        return await hash_join(len(workers), rank, eps, df1, df2, left_on, right_on)


def dataframe_merge(left, right, on=None, left_on=None, right_on=None):
    """Merge two DataFrames using explicit-comms.

    This is an explicit-comms version of Dask's Dataframe.merge() that
    only supports "inner" joins.

    Requires an activate client.

    Notice
    ------
    As a side effect, this operation concatenate all partitions located on
    the same worker thus npartitions of the returned dataframe equals number
    of workers.

    Parameters
    ----------
    left: dask.dataframe.DataFrame
    right: dask.dataframe.DataFrame
    on : str or list of str
        Column or index level names to join on. These must be found in both
        DataFrames.
    left_on : str or list of str
        Column to join on in the left DataFrame.
    right_on : str or list of str
        Column to join on in the right DataFrame.

    Returns
    -------
    df: dask.dataframe.DataFrame
        Merged dataframe
    """

    # Making sure that the "on" arguments are list of column names
    if on:
        on = [on] if isinstance(on, str) else list(on)
    if left_on:
        left_on = [left_on] if isinstance(left_on, str) else list(left_on)
    if right_on:
        right_on = [right_on] if isinstance(right_on, str) else list(right_on)

    if left_on is None:
        left_on = on
    if right_on is None:
        right_on = on

    if not (left_on and right_on):
        raise ValueError(
            "Some combination of the on, left_on, and right_on arguments must be set"
        )

    return comms.default_comms().dataframe_operation(
        _dataframe_merge, df_list=(left, right), extra_args=(left_on, right_on)
    )
