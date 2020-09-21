import asyncio

import pandas

from dask.dataframe.shuffle import partitioning_index, shuffle_group
from distributed.protocol import to_serialize


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
