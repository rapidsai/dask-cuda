from __future__ import annotations

import asyncio
import functools
import inspect
from collections import defaultdict
from operator import getitem
from typing import Callable, Dict, List, Optional

import dask
import dask.dataframe
from dask.base import compute_as_if_collection, tokenize
from dask.dataframe.core import DataFrame, _concat as dd_concat, new_dd_object
from dask.dataframe.shuffle import shuffle_group
from dask.dataframe.utils import make_meta
from dask.delayed import delayed
from distributed import wait
from distributed.protocol import nested_deserialize, to_serialize

from .. import comms


def get_concat(df: DataFrame) -> Callable:
    """Infer concatenate function to use"""

    try:
        manager = df._pxy_get().manager
        manager.proxify  # Raises if manager is disabled
    except AttributeError:
        return dd_concat

    def concat(args, ignore_index=False):
        if len(args) < 2:
            return args[0]
        return manager.proxify(dd_concat(args, ignore_index=ignore_index))[0]

    return concat


async def send(eps, rank_to_out_parts_list: Dict[int, List[List[DataFrame]]]):
    """Notice, items sent are removed from `rank_to_out_parts_list`"""
    futures = []
    for rank, ep in eps.items():
        out_parts_list = rank_to_out_parts_list.pop(rank, None)
        if out_parts_list is not None:
            futures.append(ep.write([to_serialize(f) for f in out_parts_list]))
    await asyncio.gather(*futures)


async def recv(
    eps, in_nparts: Dict[int, int], out_parts_list: List[List[List[DataFrame]]]
):
    """Notice, received items are appended to `out_parts_list`"""
    futures = []
    for rank, ep in eps.items():
        if rank in in_nparts:
            futures.append(ep.read())

    # Notice, since Dask may convert lists to tuples, we convert them back into lists
    out_parts_list.extend(
        [[y for y in x] for x in nested_deserialize(await asyncio.gather(*futures))]
    )


async def local_shuffle(
    s,
    in_parts: List[Dict[int, DataFrame]],
    rank_to_out_part_ids: Dict[int, List[int]],
    ignore_index: bool,
    concat_dfs_of_same_output_partition: bool = True,
) -> Dict[int, List[List[DataFrame]]]:
    """Local shuffle operation of the already grouped/partitioned dataframes

    This function is running on each worker participating in the shuffle. It
    does not do any communication instead it "shuffles" the local list of
    dataframes into groups -- one group for each output partition.

    It returns a dict that for each worker-rank specifies the output partitions:
    '''
        for each worker:
            for each output partition:
                list of dataframes that makes of an output partition
    '''

    If `concat_dfs_of_same_output_partition` is True, all the dataframes of an
    output partition are concatenated.

    Parameters
    ----------
    s: dict
        Worker session state
    in_parts: list of dict of dataframes
        List of dataframe groups that need to be shuffled.
    rank_to_out_part_ids: dict
        dict that for each worker rank specifices a list of partition IDs that
        worker should return. If the worker shouldn't return any partitions,
        it is excluded from the dict.
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

    concat = get_concat(next(iter(in_parts[0].values())))

    out_part_id_to_dataframes = defaultdict(list)  # part_id -> list of dataframes
    for bins in in_parts:
        for k, v in bins.items():
            out_part_id_to_dataframes[k].append(v)

    # Create mapping: rank -> list of [list of dataframes]
    rank_to_out_parts_list: Dict[int, List[List[DataFrame]]] = {}
    for rank, part_ids in rank_to_out_part_ids.items():
        rank_to_out_parts_list[rank] = [out_part_id_to_dataframes[i] for i in part_ids]

    # Concatenate all dataframes of the same output partition.
    if concat_dfs_of_same_output_partition:
        for rank in rank_to_out_part_ids.keys():
            for i in range(len(rank_to_out_parts_list[rank])):
                if len(rank_to_out_parts_list[rank][i]) > 1:
                    rank_to_out_parts_list[rank][i] = [
                        concat(
                            rank_to_out_parts_list[rank][i], ignore_index=ignore_index
                        )
                    ]
    return rank_to_out_parts_list


async def all_to_all(
    s,
    in_nparts: Dict[int, int],
    rank_to_out_part_ids: Dict[int, List[int]],
    rank_to_out_parts_list: Dict[int, List[List[DataFrame]]],
    ignore_index: bool,
) -> List[DataFrame]:
    """All-to-all communicate the dataframes returned from `local_shuffle()`

    This function is running on each worker participating in the shuffle.

    Parameters
    ----------
    s: dict
        Worker session state
    in_nparts: dict
        dict that for each worker rank specifices the
        number of partitions that worker has of the input dataframe.
        If the worker doesn't have any partitions, it is excluded from the dict.
    rank_to_out_part_ids: dict
        dict that for each worker rank specifices a list of partition IDs that
        worker should return. If the worker shouldn't return any partitions,
        it is excluded from the dict.
    rank_to_out_parts_list: dict of list of list of DataFrames
        Dict that maps each worker rank to its output partitions.
    ignore_index: bool
        Ignore index during shuffle.  If ``True``, performance may improve,
        but index values will not be preserved.

    Returns
    -------
    partitions: list of DataFrames
        List of dataframe-partitions
    """
    myrank = s["rank"]
    eps = s["eps"]

    # Communicate all the dataframe-partitions all-to-all. The result is
    # `out_parts_list` that for each worker and for each output partition
    # contains a list of dataframes received.
    out_parts_list: List[List[List[DataFrame]]] = []
    futures = []
    if myrank in rank_to_out_parts_list:
        futures.append(recv(eps, in_nparts, out_parts_list))
    if myrank in in_nparts:
        futures.append(send(eps, rank_to_out_parts_list))
    await asyncio.gather(*futures)

    # At this point `send()` should have pop'ed all output partitions
    # beside the partitions owned be `myrank`.
    assert len(rank_to_out_parts_list) == 1

    # Concatenate the received dataframes into the final output partitions
    concat = None
    ret = []
    for i in range(len(rank_to_out_part_ids[myrank])):
        dfs = []
        for out_parts in out_parts_list:
            dfs.extend(out_parts[i])
            out_parts[i] = None  # type: ignore
        dfs.extend(rank_to_out_parts_list[myrank][i])
        rank_to_out_parts_list[myrank][i] = None  # type: ignore
        if len(dfs) > 1:
            if concat is None:
                concat = get_concat(dfs[0])
            ret.append(concat(dfs, ignore_index=ignore_index))
        else:
            ret.append(dfs[0])
    return ret


def shuffle(
    df: DataFrame,
    column_names: str | List[str],
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

    Developer Notes
    ---------------
    The implementation consist of three steps:
      (a) Extend the dask graph of `df` with a call to `shuffle_group()` for each
          dataframe partition and submit the graph.
      (b) Submit a task on each worker that do a local shuffle.
      (c) Submit a task on each worker that shuffle (all-to-all communicate)
          the groups from (b) and return a list of dataframe-partitions.
      (d) Submit a dask graph that extract (using `getitem()`) individual
          dataframe-partitions from (c).
    """
    c = comms.default_comms()

    # As default we preserve number of partitions
    if npartitions is None:
        npartitions = df.npartitions

    # Step (a): partition/group each dataframe-partition
    name = (
        "explicit-comms-shuffle-group-"
        f"{tokenize(df, column_names, npartitions, ignore_index)}"
    )
    df = df.persist()  # Making sure optimizations are apply on the existing graph
    dsk = dict(df.__dask_graph__())
    output_keys = []
    for input_key in df.__dask_keys__():
        output_key = (name, input_key[1])
        dsk[output_key] = (
            shuffle_group,
            input_key,
            column_names,
            0,
            npartitions,
            npartitions,
            ignore_index,
            npartitions,
        )
        output_keys.append(output_key)

    # Compute `df_groups`, which is a list of futures, one future per partition in `df`.
    # Each future points to a dict of length `df.npartitions` that maps each
    # partition-id to a DataFrame.
    df_groups = compute_as_if_collection(type(df), dsk, output_keys, sync=False)
    wait(df_groups)
    for f in df_groups:  # Check for errors
        if f.status == "error":
            f.result()  # raise exception

    # Step (b): find out which workers has what part of `df_groups`,
    #           find the number of output each worker should have,
    #           and submit `local_shuffle()` on each worker.
    key_to_part = {str(part.key): part for part in df_groups}
    in_parts = defaultdict(list)  # Map worker -> [list of futures]
    for key, workers in c.client.who_has(df_groups).items():
        # Note, if multiple workers have the part, we pick the first worker
        in_parts[next(iter(workers))].append(key_to_part[key])

    # Let's create a dict that specifices the number of partitions each worker has
    in_nparts = {}
    workers = set()  # All ranks that have a partition of `df`
    for rank, worker in enumerate(c.worker_addresses):
        nparts = len(in_parts.get(worker, ()))
        if nparts > 0:
            in_nparts[rank] = nparts
            workers.add(rank)
    workers_sorted = sorted(workers)

    # Find the output partitions for each worker
    div = npartitions // len(workers)
    rank_to_out_part_ids = {}  # rank -> [list of partition id]
    for i, rank in enumerate(workers_sorted):
        rank_to_out_part_ids[rank] = list(range(div * i, div * (i + 1)))
    for rank, i in zip(workers_sorted, range(div * len(workers), npartitions)):
        rank_to_out_part_ids[rank].append(i)

    # Run `local_shuffle()` on each worker
    local_shuffle_result = {}
    for rank, worker in enumerate(c.worker_addresses):
        if rank in workers:
            local_shuffle_result[rank] = c.submit(
                worker,
                local_shuffle,
                in_parts[worker],
                rank_to_out_part_ids,
                ignore_index,
            )
    wait(list(local_shuffle_result.values()))

    # Release dataframes from step (a)
    del in_parts
    for fut in df_groups:
        fut.release()

    # Step (c): all-to-all communicate the result from step (a).
    all_to_all_result = {}
    for rank, worker in enumerate(c.worker_addresses):
        if rank in local_shuffle_result:
            all_to_all_result[rank] = c.submit(
                worker,
                all_to_all,
                in_nparts,
                rank_to_out_part_ids,
                local_shuffle_result[rank],
                ignore_index,
            )
    wait(list(all_to_all_result.values()))
    for fut in local_shuffle_result.values():
        fut.release()

    # Step (d): extract individual dataframe-partitions. We use `submit()`
    #           to control where the tasks are executed.
    # TODO: can we do this without using `submit()` to avoid the overhead
    #       of creating a Future for each dataframe partition?
    name = f"explicit-comms-shuffle-getitem-{tokenize(name)}"
    dsk = {}
    for rank, worker in enumerate(c.worker_addresses):
        if rank in workers:
            for i, part_id in enumerate(rank_to_out_part_ids[rank]):
                dsk[(name, part_id)] = c.client.submit(
                    getitem, all_to_all_result[rank], i, workers=[worker]
                )

    # Get the meta from the first output partition
    meta = delayed(make_meta)(next(iter(dsk.values()))).compute()

    # Create a distributed Dataframe from all the pieces
    divs = [None] * (len(dsk) + 1)
    ret = new_dd_object(dsk, name, meta, divs).persist()
    wait(ret)

    # Release all temporary dataframes
    for fut in [*all_to_all_result.values(), *dsk.values()]:
        fut.release()
    return ret


def get_rearrange_by_column_tasks_wrapper(func):
    """Returns a function wrapper that dispatch the shuffle to explicit-comms.

    Notice, this is monkey patched into Dask at dask_cuda import
    """

    func_sig = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if dask.config.get("explicit-comms", False):
            try:
                import distributed.worker

                # Make sure we have an activate client.
                distributed.worker.get_client()
            except (ImportError, ValueError):
                pass
            else:
                # Convert `*args, **kwargs` to a dict of `keyword -> values`
                kw = func_sig.bind(*args, **kwargs)
                kw.apply_defaults()
                kw = kw.arguments
                column = kw["column"]
                if isinstance(column, str):
                    column = [column]
                return shuffle(kw["df"], column, kw["npartitions"], kw["ignore_index"])
        return func(*args, **kwargs)

    return wrapper
