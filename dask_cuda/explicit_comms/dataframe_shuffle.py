import asyncio
from collections import defaultdict
from operator import getitem
from typing import Dict, List, Optional, Set

import dask
import distributed
from dask.dataframe.core import DataFrame, _concat
from dask.dataframe.shuffle import shuffle_group
from dask.delayed import delayed
from distributed.protocol import nested_deserialize, to_serialize

from . import comms, utils


async def send(eps, rank_to_out_parts_list: Dict[int, List[List[DataFrame]]]):
    futures = []
    for rank, ep in eps.items():
        if rank in rank_to_out_parts_list:
            futures.append(
                ep.write([to_serialize(f) for f in rank_to_out_parts_list[rank]])
            )
    await asyncio.gather(*futures)


async def recv(
    eps, in_nparts: Dict[int, int], out_parts_list: List[List[List[DataFrame]]]
):
    futures = []
    for rank, ep in eps.items():
        if rank in in_nparts:
            futures.append(ep.read())
    out_parts_list.extend(nested_deserialize(await asyncio.gather(*futures)))


def partition_by_hash(
    in_parts: List[DataFrame],
    rank_to_out_part_ids: Dict[int, List[int]],
    column_names: List[str],
    npartitions: int,
    ignore_index: bool,
    concat_dfs_of_same_output_partition: bool,
) -> Dict[int, List[List[DataFrame]]]:
    """ Partition each dataframe in `in_parts`

    This local operation hash each dataframe in `in_parts` by hashing the
    values in the columns specified in `column_names`.

    It returns a dict that for each worker rank specifies the output partitions:
    '''
        for each worker:
            for each output partition:
                list of dataframes that makes of an output partition
    '''
    If `concat_dfs_of_same_output_partition` is True, all the dataframes of an
    output partition are concatinated.

    Parameters
    ----------
    in_parts: list of dataframes
        List of input dataframes to partition.
    rank_to_out_part_ids: dict
        dict that for each worker rank specifices a list of partition IDs that
        worker should return. If the worker shouldn't return any partitions,
        it is excluded from the dict.
    column_names: list of strings
        List of column names on which we want to split.
    npartitions: int
        Number of output partitions to produce
    ignore_index: bool
        Ignore index during shuffle.  If ``True``, performance may improve,
        but index values will not be preserved.
    concat_dfs_of_same_output_partition: bool
        Concatenate all dataframes of the same output partition.

    Returns
    -------
    rank_to_out_parts_list: dict of list of list of DataFrames
        Dict that maps each worker rank to its output partitions.
    """

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

    # Create mapping: rank -> list of [list of dataframes]
    rank_to_out_parts_list: Dict[int, List[List[DataFrame]]] = {}
    for rank, part_ids in rank_to_out_part_ids.items():
        rank_to_out_parts_list[rank] = [
            list(out_part_id_to_dataframes[i]) for i in part_ids
        ]

    if concat_dfs_of_same_output_partition:
        for rank in rank_to_out_part_ids.keys():
            for i in range(len(rank_to_out_parts_list[rank])):
                if len(rank_to_out_parts_list[rank][i]) > 1:
                    rank_to_out_parts_list[rank][i] = [
                        _concat(
                            rank_to_out_parts_list[rank][i], ignore_index=ignore_index
                        )
                    ]

    return rank_to_out_parts_list


async def local_shuffle(
    s,
    workers: Set[int],
    npartitions: int,
    in_nparts: Dict[int, int],
    in_parts: List[DataFrame],
    rank_to_out_part_ids: Dict[int, List[int]],
    column_names: List[str],
    ignore_index: bool,
) -> List[DataFrame]:
    """Local shuffle operation

    This function is running on each worker participating in the shuffle.

    Parameters
    ----------
    s: dict
        Worker session state
    workers: set
        Set of ranks of all the participants
    npartitions: int
        Number of output partitions this worker should produce
    in_nparts: dict
        dict that for each worker rank specifices the
        number of partitions that worker has of the input dataframe.
        If the worker doesn't have any partitions, it is excluded from the dict.
    in_parts: list of dataframes
        List of input dataframes on this worker.
    rank_to_out_part_ids: dict
        dict that for each worker rank specifices a list of partition IDs that
        worker should return. If the worker shouldn't return any partitions,
        it is excluded from the dict.
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
    myrank = s["rank"]
    eps = s["eps"]
    assert s["rank"] in workers

    rank_to_out_parts_list = partition_by_hash(
        in_parts,
        rank_to_out_part_ids,
        column_names,
        npartitions,
        ignore_index,
        concat_dfs_of_same_output_partition=True,
    )

    # For each worker, for each output partition, list of dataframes
    out_parts_list: List[List[List[DataFrame]]] = []
    futures = []
    if myrank in in_nparts:
        futures.append(send(eps, rank_to_out_parts_list))
    if myrank in rank_to_out_parts_list:
        futures.append(recv(eps, in_nparts, out_parts_list))
    await asyncio.gather(*futures)

    ret = []
    for i in range(len(rank_to_out_part_ids[myrank])):
        dfs = []
        for out_parts in out_parts_list:
            dfs.extend(out_parts[i])
        dfs.extend(rank_to_out_parts_list[myrank][i])
        if len(dfs) > 1:
            ret.append(_concat(dfs, ignore_index=ignore_index))
        else:
            ret.append(dfs[0])
    return ret


def dataframe_shuffle(
    df: DataFrame,
    column_names: List[str],
    npartitions: Optional[int] = None,
    ignore_index: bool = False,
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
    npartitions: int or None
        The desired number of output partitions. If None, the number of output
        partitions equals `df.npartitions`
    ignore_index: bool
        Ignore index during shuffle.  If True, performance may improve,
        but index values will not be preserved.

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

    # Find the output partitions for each worker
    div = npartitions // len(workers)
    rank_to_out_part_ids = {}  # rank -> [list of partition id]
    for i, rank in enumerate(workers):
        rank_to_out_part_ids[rank] = list(range(div * i, div * (i + 1)))
    for rank, i in zip(workers, range(div * len(workers), npartitions)):
        rank_to_out_part_ids[rank].append(i)

    result_futures = {}
    for rank, worker in enumerate(c.worker_addresses):
        if rank in workers:
            result_futures[rank] = c.submit(
                worker,
                local_shuffle,
                workers,
                npartitions,
                in_nparts,
                in_parts[worker],
                rank_to_out_part_ids,
                column_names,
                ignore_index,
            )
    distributed.wait(list(result_futures.values()))

    ret = []
    for rank, parts in rank_to_out_part_ids.items():
        for i in range(len(parts)):
            ret.append(delayed(getitem)(result_futures[rank], i))
    ret = dask.dataframe.from_delayed(ret).persist()
    distributed.wait(ret)
    return ret


def rearrange_by_column_tasks_wrapper(
    df, column, max_branch=32, npartitions=None, ignore_index=False
):
    if dask.config.get("explicit-comms", False):
        try:
            import distributed.worker

            distributed.worker.get_client()
        except (ImportError, ValueError):
            pass
        else:
            if isinstance(column, str):
                column = [column]
            return dataframe_shuffle(df, column, npartitions, ignore_index)

    from dask.dataframe.shuffle import rearrange_by_column_task_org

    return rearrange_by_column_task_org(
        df, column, max_branch, npartitions, ignore_index
    )
