import asyncio
import uuid
import numpy as np
import pandas

import rmm
import cudf
from distributed.protocol import to_serialize

from . import comms
from . import dask_df_utils


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
    """Partition the dataframe by the hashed value of data in column.
        Supports both Pandas and cuDF DataFrames
    """
    if df is None:
        return [None] * n_chunks
    elif hasattr(df, "partition_by_hash"):
        return df.partition_by_hash(columns, n_chunks)
    else:
        # Pandas doesn't have a partition_by_hash() method so we implement it here
        meta_col = "_partition_by_hash_%s" % uuid.uuid1()
        df[meta_col] = (
            pandas.util.hash_pandas_object(df[columns], index=False) % n_chunks
        )
        df_list = [None] * n_chunks
        for idx, group in df.groupby(meta_col):
            df_list[idx] = group
            del group[meta_col]
        del df[meta_col]
        header = dask_df_utils.get_meta(df)
        ret = []
        for df in df_list:
            if df is None:
                ret.append(header)
            else:
                ret.append(df)
        return ret


async def distributed_join(n_chunks, rank, eps, left_table, right_table):
    left_bins = partition_by_hash(left_table, ["key"], n_chunks)
    left_df = exchange_and_concat_bins(rank, eps, left_bins)
    right_bins = partition_by_hash(right_table, ["key"], n_chunks)
    left_df = await left_df
    right_df = await exchange_and_concat_bins(rank, eps, right_bins)
    return left_df.merge(right_df)


async def _dataframe_merge(s, df1_parts, df2_parts, r):
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

    return await distributed_join(s["nworkers"], s["rank"], s["eps"], df1, df2)


def dataframe_merge(df1, df2):
    return comms.default_comms().dataframe_operation(_dataframe_merge, (df1, df2))
