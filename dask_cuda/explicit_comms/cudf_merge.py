import asyncio
import numpy as np

import rmm
import cudf
from distributed.protocol import to_serialize

from . import comms


async def send_df(ep, df):
    return await ep.write([to_serialize(df)])


async def recv_df(ep):
    ret = await ep.read()
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
    return cudf.concat(ret)


async def distributed_join(n_chunks, rank, eps, left_table, right_table):
    left_bins = left_table.partition_by_hash(["key"], n_chunks)
    right_bins = right_table.partition_by_hash(["key"], n_chunks)
    left_df = await exchange_and_concat_bins(rank, eps, left_bins)
    right_df = await exchange_and_concat_bins(rank, eps, right_bins)
    return left_df.merge(right_df)


async def _cudf_merge(s, df1_parts, df2_parts, r):
    # TODO: handle cases where df1_parts and df2_parts consist of multiple parts
    assert len(df1_parts) == 1
    assert len(df2_parts) == 1
    return await distributed_join(
        s["nworkers"], s["rank"], s["eps"], df1_parts[0], df2_parts[0]
    )


def cudf_merge(df1, df2):
    return comms.default_comms().dataframe_operation(_cudf_merge, df1, df2)
