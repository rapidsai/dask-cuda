import asyncio
from collections import defaultdict
from operator import getitem
from typing import Dict, List, Set

import dask
import distributed
from dask.dataframe.core import DataFrame, _concat
from dask.dataframe.shuffle import shuffle_group
from dask.delayed import delayed
from distributed.protocol import nested_deserialize, to_serialize

from . import comms, utils


async def _shuffle(
    s,
    workers: Set[int],
    npartitions: int,
    in_nparts: Dict[int, int],
    in_parts: List[DataFrame],
    rank_to_out_part_ids: Dict[int, List[int]],
    column_names: List[str],
    ignore_index: bool,
) -> List[DataFrame]:
    """
    Parameters
    ----------
    s: dict
        Worker session state
    workers: set
        Set of ranks of all the participants
    in_nparts: dict
        dict that for each worker rank specifices the
        number of partitions that worker has of the input dataframe.
        If the worker doesn't have any partitions, it is excluded from the dict.
    in_parts: list of dataframes
        List of input dataframes on this worker.
    rank_to_out_part_ids: dict
        dict that for each worker rank specifices a list of partition IDs that
        worker should return.
    column_names: list of strings
        List of column names on which we want to split.
    ignore_index: bool
        Ignore index during shuffle.  If ``True``, performance may improve,
        but index values will not be preserved.

    Returns
    -------
    partitions: list of DataFrames
        List of partitions
    """
    assert s["rank"] in workers

    # Trimming such that all participanting workers get a rank within 0..len(workers)
    trim_map = {}
    for i in range(s["nworkers"]):
        if i in workers:
            trim_map[i] = len(trim_map)
    myrank = trim_map[s["rank"]]
    eps = {trim_map[i]: s["eps"][trim_map[i]] for i in workers if i != s["rank"]}

    bins_list = []  # list of [part_id -> dataframe]
    for df in in_parts:
        bins_list.append(
            shuffle_group(
                df, column_names, 0, npartitions, npartitions, ignore_index, npartitions
            )
        )

    out_part_id_to_dataframes = defaultdict(list)  # part_id -> list of dataframes
    for bins in bins_list:
        for k, v in bins.items():
            out_part_id_to_dataframes[k].append(v)

    rank_to_out_parts_list: Dict[
        int, List[List[DataFrame]]
    ] = {}  # rank -> list of [list of dataframes]
    for rank, part_ids in rank_to_out_part_ids.items():
        rank_to_out_parts_list[rank] = [
            list(out_part_id_to_dataframes[i]) for i in part_ids
        ]

    debug_str = (
        f"[{myrank}] workers: {workers}, in_nparts: {in_nparts}, "
        f"rank_to_out_part_ids: {rank_to_out_part_ids}\n"
    )
    for rank, parts in rank_to_out_parts_list.items():
        debug_str += f"  {rank}: ["
        for chunks in parts:
            debug_str += "["
            for df in chunks:
                debug_str += (
                    f"{type(df).__name__}(id: {hex(id(df))}, nrows: {len(df)}), "
                )
                if df is chunks[-1]:
                    debug_str = debug_str[:-2]
            debug_str += "], "
            if chunks is parts[-1]:
                debug_str = debug_str[:-2]
        debug_str += "]\n"
    # print(debug_str)

    async def send(eps, rank_to_out_parts_list: Dict[int, List[List[DataFrame]]]):
        futures = []
        for rank, ep in eps.items():
            futures.append(ep.write(to_serialize(rank_to_out_parts_list[rank])))
        await asyncio.gather(*futures)

    async def recv(eps, out_parts_list: List[List[List[DataFrame]]]):
        out_parts_list.extend(
            nested_deserialize(
                await asyncio.gather(*[ep.read() for ep in eps.values()])
            )
        )

    # For each worker, for each output partition, list of dataframes
    out_parts_list: List[List[List[DataFrame]]] = []
    await asyncio.gather(recv(eps, out_parts_list), send(eps, rank_to_out_parts_list))

    debug_str = f"[{myrank}] out_parts_list:\n"
    for parts in out_parts_list:
        debug_str += " ["
        for chunks in parts:
            debug_str += "["
            for df in chunks:
                debug_str += (
                    f"{type(df).__name__}(id: {hex(id(df))}, nrows: {len(df)}), "
                )
                if df is chunks[-1]:
                    debug_str = debug_str[:-2]
            debug_str += "], "
            if chunks is parts[-1]:
                debug_str = debug_str[:-2]
        debug_str += "]\n"
    # print(debug_str)

    ret = []
    for i in range(len(rank_to_out_part_ids[myrank])):
        dfs = []
        for out_parts in out_parts_list:
            dfs.extend(out_parts[i])
        dfs.extend(rank_to_out_parts_list[myrank][i])
        if len(dfs) > 1:
            ret.append(dfs[0])
        else:
            ret.append(_concat(dfs, ignore_index=ignore_index))
    return ret


def dataframe_shuffle(
    df: DataFrame, column_names: List[str], npartitions=None, ignore_index=False
) -> DataFrame:
    """Order divisions of DataFrame so that all values within column(s) align

    This enacts a task-based shuffle using explicit-comms. It requires a full
    dataset read, serialization and shuffle. This is expensive. If possible
    you should avoid shuffles.

    This does not preserve a meaningful index/partitioning scheme. This is not
    deterministic if done in parallel.

    Requires an activate client.

    Parameters
    ----------
    df: dask.dataframe.DataFrame
        Dataframe to shuffle
    column_names: list of strings
        List of column names on which we want to split.

    Returns
    -------
    df: dask.dataframe.DataFrame
        Shuffled dataframe
    """

    c = comms.default_comms()
    in_parts = utils.extract_ddf_partitions(df)  # worker -> [list of futures]

    # Let's create a dict that specifices the number of partitions each worker has
    in_nparts = {}
    workers = set()  # All ranks that have a partition of `df`
    for rank, worker in enumerate(c.worker_addresses):
        nparts = len(in_parts.get(worker, ()))
        if nparts > 0:
            in_nparts[rank] = nparts
            workers.add(rank)

    # As default we preserve number of partitions
    if npartitions is None:
        npartitions = df.npartitions

    # Find the number of output partitions for each worker
    div = npartitions // len(workers)
    rank_to_out_part_ids = {}  # rank -> [list of partition id]
    for i, rank in enumerate(workers):
        rank_to_out_part_ids[rank] = list(range(div * rank, div * (rank + 1)))

    for rank, i in enumerate(range(div * len(workers), npartitions)):
        rank_to_out_part_ids[rank].append(i)

    # print(f"in_nparts: {in_nparts}, out_nparts: {rank_to_out_part_ids}")

    result_futures = {}
    for rank, worker in enumerate(c.worker_addresses):
        if rank in workers:
            result_futures[rank] = c.submit(
                worker,
                _shuffle,
                workers,
                npartitions,
                in_nparts,
                in_parts[worker],
                rank_to_out_part_ids,
                column_names,
                ignore_index,
            )
    distributed.wait(result_futures.values())

    ret = []
    for rank, parts in rank_to_out_part_ids.items():
        for i in range(len(parts)):
            ret.append(delayed(getitem)(result_futures[rank], i))
    ret = dask.dataframe.from_delayed(ret).persist()
    distributed.wait(ret)
    return ret
