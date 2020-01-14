import asyncio
import pickle
import numpy as np

import rmm
import cudf

from . import comms


async def send_df(ep, df):
    header, frames = df.serialize()
    header["frame_ifaces"] = [f.__cuda_array_interface__ for f in frames]
    header = pickle.dumps(header)
    header_nbytes = np.array([len(header)], dtype=np.uint64)
    await ep.send(header_nbytes)
    await ep.send(header)
    for frame in frames:
        await ep.send(frame)


async def recv_df(ep):
    header_nbytes = np.empty((1,), dtype=np.uint64)
    await ep.recv(header_nbytes)
    header = bytearray(header_nbytes[0])
    await ep.recv(header)
    header = pickle.loads(header)

    frames = [
        rmm.device_array(iface["shape"], dtype=np.dtype(iface["typestr"]))
        for iface in header["frame_ifaces"]
    ]
    for frame in frames:
        await ep.recv(frame)

    cudf_typ = pickle.loads(header["type"])
    return cudf_typ.deserialize(header, frames)


async def barrier(rank, eps):
    futures = []
    dummy_send = np.zeros(1, dtype="u1")

    if rank == 0:
        await asyncio.gather(*[ep.recv(np.empty(1, dtype="u1")) for ep in eps.values()])
    else:
        await eps[0].send(np.zeros(1, dtype="u1"))


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


async def exchange_and_concat_bins(rank, eps, bins, timings=None):
    ret = [bins[rank]]
    if timings is not None:
        t1 = clock()
    await asyncio.gather(recv_bins(eps, ret), send_bins(eps, bins))
    if timings is not None:
        t2 = clock()
        timings.append(
            (t2 - t1, sum([sys.getsizeof(b) for i, b in enumerate(bins) if i != rank]))
        )
    return cudf.concat(ret)


async def distributed_join(n_chunks, rank, eps, left_table, right_table, timings=None):
    left_bins = left_table.partition_by_hash(["key"], n_chunks)
    right_bins = right_table.partition_by_hash(["key"], n_chunks)
    left_df = await exchange_and_concat_bins(rank, eps, left_bins, timings)
    right_df = await exchange_and_concat_bins(rank, eps, right_bins, timings)
    return left_df.merge(right_df)


async def _cudf_merge(s, df1_parts, df2_parts, r):
    assert len(df1_parts) == 1
    assert len(df2_parts) == 1
    for _ in range(1):
        ret = await distributed_join(
            s["nworkers"], s["rank"], s["eps"], df1_parts[0], df2_parts[0]
        )
    return ret


def cudf_merge(df1, df2):
    return comms.default_comms().dataframe_operation(_cudf_merge, df1, df2)
