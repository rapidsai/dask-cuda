import asyncio
import numpy as np
import pandas
import rmm
import cudf

from dask.dataframe.shuffle import shuffle_group, partitioning_index
from distributed.protocol import to_serialize

from . import comms, utils


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
    futures = []
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
    elif hasattr(df_list[0], "partition_by_hash"):
        return cudf.concat(df_list)
    else:
        return pandas.concat(df_list)


def partition_by_hash(df, columns, n_chunks):
    """ Splits dataframe into partitions

    The partitions is determined by the hash value of the rows in `columns`.

    Parameters
    ----------
    df: DataFrame
    columns: label or list
        Column names on which to split the dataframe
    npartition: int
        Number of partitions

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
    ret = shuffle_group(df, "_partitions", 0, n_chunks, n_chunks)

    # Let's remove the partition column and return the partitions
    del df["_partitions"]
    for df in ret.values():
        del df["_partitions"]
    return ret


async def distributed_join(
    n_chunks, rank, eps, left_table, right_table, left_on, right_on
):
    left_bins = partition_by_hash(left_table, left_on, n_chunks)
    left_df = exchange_and_concat_bins(rank, eps, left_bins)
    right_bins = partition_by_hash(right_table, right_on, n_chunks)
    left_df = await left_df
    right_df = await exchange_and_concat_bins(rank, eps, right_bins)
    return left_df.merge(right_df, left_on=left_on, right_on=right_on)


async def _dataframe_merge(s, df1_parts, df2_parts, left_on, right_on):
    def df_concat(df_parts):
        """Making sure df_parts is a single dataframe or None"""
        if len(df_parts) == 0:
            return None
        elif len(df_parts) == 1:
            return df_parts[0]
        else:
            return concat(df_parts)

    df1 = df_concat(df1_parts)
    df2 = df_concat(df2_parts)

    return await distributed_join(
        s["nworkers"], s["rank"], s["eps"], df1, df2, left_on, right_on
    )


def dataframe_merge(df1, df2, on=None, left_on=None, right_on=None, how="inner"):

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
        _dataframe_merge, df_list=(df1, df2), extra_args=(left_on, right_on)
    )
