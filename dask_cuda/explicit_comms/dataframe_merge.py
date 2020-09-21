import asyncio

import pandas

from . import comms
from .dataframe_shuffle import (
    send_df,
    recv_df,
    partition_by_hash,
    exchange_and_concat_bins,
)


def concat(df_list):
    if len(df_list) == 0:
        return None
    else:
        typ = str(type(df_list[0]))
        if "cudf" in typ:
            # delay import of cudf to handle CPU only tests
            import cudf

            return cudf.concat(df_list)
        else:
            return pandas.concat(df_list)


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
    left_on : label or list, or array-like
        Column to join on in the left DataFrame. Other than in pandas
        arrays and lists are only support if their length is 1.
    right_on : label or list, or array-like
        Column to join on in the right DataFrame. Other than in pandas
        arrays and lists are only support if their length is 1.

    Returns
    -------
        merged_dataframe: DataFrame
    """

    def df_concat(df_parts):
        """Making sure df_parts is a single dataframe or None"""
        if len(df_parts) == 0:
            return None
        elif len(df_parts) == 1:
            return df_parts[0]
        else:
            return concat(df_parts)

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


def dataframe_merge(left, right, on=None, left_on=None, right_on=None, how="inner"):
    """Merge two Dask DataFrames

    This will merge the two datasets, either on the indices, a certain column
    in each dataset or the index in one dataset and the column in another.

    Requires an activate client.

    Parameters
    ----------
    left: dask.dataframe.DataFrame
    right: dask.dataframe.DataFrame
    how : {'left', 'right', 'outer', 'inner'}, default: 'inner'
        How to handle the operation of the two objects:

        - left: use calling frame's index (or column if on is specified)
        - right: use other frame's index
        - outer: form union of calling frame's index (or column if on is
            specified) with other frame's index, and sort it
            lexicographically
        - inner: form intersection of calling frame's index (or column if
            on is specified) with other frame's index, preserving the order
            of the calling's one

    on : label or list
        Column or index level names to join on. These must be found in both
        DataFrames. If on is None and not merging on indexes then this
        defaults to the intersection of the columns in both DataFrames.
    left_on : label or list, or array-like
        Column to join on in the left DataFrame. Other than in pandas
        arrays and lists are only support if their length is 1.
    right_on : label or list, or array-like
        Column to join on in the right DataFrame. Other than in pandas
        arrays and lists are only support if their length is 1.
    left_index : boolean, default False
        Use the index from the left DataFrame as the join key.
    right_index : boolean, default False
        Use the index from the right DataFrame as the join key.
    suffixes : 2-length sequence (tuple, list, ...)
        Suffix to apply to overlapping column names in the left and
        right side, respectively
    indicator : boolean or string, default False
        If True, adds a column to output DataFrame called "_merge" with
        information on the source of each row. If string, column with
        information on source of each row will be added to output DataFrame,
        and column will be named value of string. Information column is
        Categorical-type and takes on a value of "left_only" for observations
        whose merge key only appears in `left` DataFrame, "right_only" for
        observations whose merge key only appears in `right` DataFrame,
        and "both" if the observation’s merge key is found in both.

    Returns
    -------
        merged_dataframe: dask.dataframe.DataFrame

    Notes
    -----
    This function submits jobs the each available worker explicitly and the
    number of partitions of `left` and `right` might change (typically to the
    number of workers).
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

    if how != "inner":
        raise NotImplementedError('Only support `how="inner"`')

    return comms.default_comms().dataframe_operation(
        _dataframe_merge, df_list=(left, right), extra_args=(left_on, right_on)
    )
