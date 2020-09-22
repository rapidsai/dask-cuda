import asyncio

import pandas

from dask.base import is_dask_collection
from dask.dataframe.core import _Frame
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
    return concat([df for df in ret if df is not None])


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


def df_concat(df_parts):
    """Making sure df_parts is a single dataframe or None"""
    if len(df_parts) == 0:
        return None
    elif len(df_parts) == 1:
        return df_parts[0]
    else:
        return concat(df_parts)


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


async def rearrange_by_column(n_chunks, rank, eps, left_table, column):
    left_bins = partition_by_hash(left_table, column, n_chunks, ignore_index=True)
    return await exchange_and_concat_bins(rank, eps, left_bins)


async def _rearrange_by_column(s, workers, dfs_nparts, dfs_parts, column):
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

    return await rearrange_by_column(len(workers), rank, eps, df, column)


def dataframe_rearrange_by_column(df, column):
    """Order divisions of DataFrame so that all values within column(s) align

    This enacts a task-based shuffle.  It contains most of the tricky logic
    around the complex network of tasks.  Typically before this function is
    called a new column, ``"_partitions"`` has been added to the dataframe,
    containing the output partition number of every row.  This function
    produces a new dataframe where every row is in the proper partition.  It
    accomplishes this by splitting each input partition into several pieces,
    and then concatenating pieces from different input partitions into output
    partitions.

    Note that the `column` input may correspond to a list of columns (rather
    than just a single column name).  In this case, the `shuffle_group` and
    `shuffle_group_2` functions will use hashing to map each row to an output
    partition. This approach may require the same rows to be hashed multiple
    times, but avoids the need to assign a new "_partitions" column.

    Parameters
    ----------
    df: dask.dataframe.DataFrame
    column: str or list
        A column name on which we want to split, commonly ``"_partitions"``
        which is assigned by functions upstream.  This could also be a list of
        columns (in which case shuffle_group will create a hash array/column).

    Returns
    -------
    df3: dask.dataframe.DataFrame

    See also
    --------
    rearrange_by_column_disk: same operation, but uses partd
    rearrange_by_column: parent function that calls this or rearrange_by_column_disk
    shuffle_group: does the actual splitting per-partition
    """

    return comms.default_comms().dataframe_operation(
        _rearrange_by_column,
        df_list=(df,),
        extra_args=(column,),
    )


def dataframe_shuffle(df, index):
    """Group DataFrame by index

    Hash grouping of elements. After this operation all elements that have
    the same index will be in the same partition. Note that this requires
    full dataset read, serialization and shuffle. This is expensive. If
    possible you should avoid shuffles.

    This does not preserve a meaningful index/partitioning scheme. This is not
    deterministic if done in parallel.

    As a side effect, this operation concatenate all partitions located on
    the same worker thus npartitions of the returned dataframe equals number
    of workers.

    See Also
    --------
    set_index
    set_partition
    shuffle_disk
    """

    if isinstance(index, str) or (
        pandas.api.types.is_list_like(index) and not is_dask_collection(index)
    ):
        # Avoid creating the "_partitions" column if possible.
        # We currently do this if the user is passing in
        # specific column names (and shuffle == "tasks").
        if isinstance(index, str):
            index = [index]
        else:
            index = list(index)
        nset = set(index)
        if nset.intersection(set(df.columns)) == nset:
            return dataframe_rearrange_by_column(
                df,
                index,
            )

    if not isinstance(index, _Frame):
        index = df._select_columns_or_index(index)

    partitions = index.map_partitions(
        partitioning_index,
        npartitions=df.npartitions,
        meta=df._meta._constructor_sliced([0]),
        transform_divisions=False,
    )
    df2 = df.assign(_partitions=partitions)
    df2._meta.index.name = df._meta.index.name
    df3 = dataframe_rearrange_by_column(
        df2,
        "_partitions",
    )
    del df3["_partitions"]
    return df3
