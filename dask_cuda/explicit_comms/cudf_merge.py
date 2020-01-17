import asyncio
import numpy as np

import rmm
import cudf
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
    futures = []
    if rank == 0:
        await asyncio.gather(*[ep.read() for ep in eps.values()])
    else:
        await eps[0].send("dummy")


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
    return cudf.concat([df for df in ret if df is not None])


def partition_by_hash(df, keys, n_chunks):
    if df is None:
        return [None] * n_chunks
    else:
        return df.partition_by_hash(keys, n_chunks)


async def distributed_join(n_chunks, rank, eps, left_table, right_table):
    left_bins = partition_by_hash(left_table, ["key"], n_chunks)
    left_df = exchange_and_concat_bins(rank, eps, left_bins)
    right_bins = partition_by_hash(right_table, ["key"], n_chunks)
    left_df = await left_df
    right_df = await exchange_and_concat_bins(rank, eps, right_bins)
    return left_df.merge(right_df)


async def _cudf_merge(s, df1_parts, df2_parts, r):
    def df_concat(df_parts):
        """Making sure df_parts is a single dataframe or None"""
        if len(df_parts) == 0:
            return None
        elif len(df_parts) == 1:
            return df_parts[0]
        else:
            return cudf.concat(df_parts)

    df1 = df_concat(df1_parts)
    df2 = df_concat(df2_parts)

    return await distributed_join(s["nworkers"], s["rank"], s["eps"], df1, df2)


def cudf_merge(df1, df2):
    return comms.default_comms().dataframe_operation(_cudf_merge, (df1, df2))
