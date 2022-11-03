from __future__ import annotations

import asyncio
import functools
import inspect
from collections import defaultdict
from operator import getitem
from typing import Any, Dict, List, Optional

import numpy

import dask
import dask.dataframe
from dask.base import tokenize
from dask.dataframe.core import DataFrame, Series, _concat as dd_concat, new_dd_object
from dask.dataframe.shuffle import group_split_dispatch, hash_object_dispatch
from dask.dataframe.utils import make_meta
from dask.delayed import delayed
from distributed import wait
from distributed.protocol import nested_deserialize, to_serialize

from .. import comms


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


def get_proxify(worker):
    from dask_cuda.proxify_host_file import ProxifyHostFile

    if isinstance(worker.data, ProxifyHostFile):
        return lambda x: worker.data.manager.proxify(x)[0]
    return lambda x: x  # no-op


def compute_map_index(df: Any, column_names, npartitions) -> Series:
    if column_names[0] == "_partitions":
        ind = df[column_names[0]]
    else:
        ind = hash_object_dispatch(
            df[column_names] if column_names else df, index=False
        )
    typ = numpy.min_scalar_type(npartitions * 2)
    return (ind % npartitions).astype(typ, copy=False)


def single_shuffle_group(df: DataFrame, column_names, npartitions, ignore_index):
    map_index = compute_map_index(df, column_names, npartitions)
    return group_split_dispatch(df, map_index, npartitions, ignore_index=ignore_index)


def multi_shuffle_group(
    dfs: Dict[str, DataFrame],
    rank_to_out_part_ids: Dict[int, List[int]],
    column_names,
    npartitions,
    ignore_index,
    proxify,
) -> Dict[int, List[List[DataFrame]]]:

    # Hash into groups
    df_groups = []
    while dfs:
        df_groups.append(
            proxify(
                single_shuffle_group(
                    dfs.popitem()[1],  # pop dataframe in any order
                    column_names,
                    npartitions,
                    ignore_index,
                )
            )
        )
    out_part_id_to_dataframes = defaultdict(list)  # part_id -> list of dataframes
    for group in df_groups:
        for k, v in group.items():
            out_part_id_to_dataframes[k].append(v)
    del df_groups

    # Create mapping: rank -> list of [list of dataframes]
    rank_to_out_parts_list: Dict[int, List[List[DataFrame]]] = {}
    for rank, part_ids in rank_to_out_part_ids.items():
        rank_to_out_parts_list[rank] = [out_part_id_to_dataframes[i] for i in part_ids]
    del out_part_id_to_dataframes

    # Concatenate all dataframes of the same output partition.
    for rank in rank_to_out_part_ids.keys():
        for i in range(len(rank_to_out_parts_list[rank])):
            if len(rank_to_out_parts_list[rank][i]) > 1:
                rank_to_out_parts_list[rank][i] = [
                    proxify(
                        dd_concat(
                            rank_to_out_parts_list[rank][i], ignore_index=ignore_index
                        )
                    )
                ]
    return rank_to_out_parts_list


async def shuffle_task(
    s,
    stage_name,
    rank_to_inkeys: Dict[int, set],
    rank_to_out_part_ids: Dict[int, List[int]],
    column_names,
    npartitions,
    ignore_index,
):
    """Explicit-comms shuffle task

    This function is running on each worker participating in the shuffle.

    Parameters
    ----------
    s: dict
        Worker session state
    stage_name: str
        Name of the stage to retrieve the input keys from.
    rank_to_inkeys: dict
        dict that for each worker rank specifices the set of staged input keys.
    rank_to_out_part_ids: dict
        dict that for each worker rank specifices a list of partition IDs that
        worker should return. If the worker shouldn't return any partitions,
        it is excluded from the dict.
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
    partitions: list of DataFrames
        List of dataframe-partitions
    """

    proxify = get_proxify(s["worker"])
    myrank = s["rank"]
    eps = s["eps"]
    stage: dict = s["stages"].pop(stage_name)
    assert stage.keys() == rank_to_inkeys[myrank]

    rank_to_out_parts_list = multi_shuffle_group(
        dfs=stage,
        rank_to_out_part_ids=rank_to_out_part_ids,
        column_names=column_names,
        npartitions=npartitions,
        ignore_index=ignore_index,
        proxify=proxify,
    )

    # Communicate all the dataframe-partitions all-to-all. The result is
    # `out_parts_list` that for each worker and for each output partition
    # contains a list of dataframes received.
    in_nparts = {r: len(k) for r, k in rank_to_inkeys.items()}
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
    ret = []
    for i in range(len(rank_to_out_part_ids[myrank])):
        dfs = []
        for out_parts in out_parts_list:
            dfs.extend(out_parts[i])
            out_parts[i] = None  # type: ignore
        dfs.extend(rank_to_out_parts_list[myrank][i])
        rank_to_out_parts_list[myrank][i] = None  # type: ignore
        if len(dfs) > 1:
            ret.append(proxify(dd_concat(dfs, ignore_index=ignore_index)))
        else:
            ret.append(proxify(dfs[0]))
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
      (b) Submit a task on each worker that shuffle (all-to-all communicate)
          the groups from (a) and return a list of dataframe-partitions.
      (c) Submit a dask graph that extract (using `getitem()`) individual
          dataframe-partitions from (b).
    """
    c = comms.default_comms()

    # As default we preserve number of partitions
    if npartitions is None:
        npartitions = df.npartitions

    # Step (a):
    df = df.persist()  # Make sure optimizations are apply on the existing graph
    wait(df)  # Make sure all keys has been materialized on workers
    name = (
        "explicit-comms-shuffle-getitem-"
        f"{tokenize(df, column_names, npartitions, ignore_index)}"
    )

    # Stage all keys of `df` on the workers and cancel them, which makes it possible
    # for the shuffle to free memory as the partitions of `df` are consumed.
    rank_to_inkeys = c.stage_keys(name=name, keys=df.__dask_keys__())
    c.client.cancel(df)  # Notice, since `df` has been staged, nothing is freed here.

    # Find the output partition IDs for each worker
    ranks = sorted(rank_to_inkeys.keys())
    div = npartitions // len(ranks)
    rank_to_out_part_ids: Dict[int, List[int]] = {}  # rank -> [list of partition id]
    for i, rank in enumerate(ranks):
        rank_to_out_part_ids[rank] = list(range(div * i, div * (i + 1)))
    for rank, i in zip(ranks, range(div * len(ranks), npartitions)):
        rank_to_out_part_ids[rank].append(i)

    # Run `_shuffle()` on each worker
    shuffle_result = {}
    for rank in ranks:
        shuffle_result[rank] = c.submit(
            c.worker_addresses[rank],
            shuffle_task,
            name,
            rank_to_inkeys,
            rank_to_out_part_ids,
            column_names,
            npartitions,
            ignore_index,
        )
    wait(list(shuffle_result.values()))

    # Step (d): extract individual dataframe-partitions. We use `submit()`
    #           to control where the tasks are executed.
    # TODO: can we do this without using `submit()` to avoid the overhead
    #       of creating a Future for each dataframe partition?

    dsk = {}
    for rank in ranks:
        for i, part_id in enumerate(rank_to_out_part_ids[rank]):
            dsk[(name, part_id)] = c.client.submit(
                getitem, shuffle_result[rank], i, workers=[c.worker_addresses[rank]]
            )

    # Get the meta from the first output partition
    meta = delayed(make_meta)(next(iter(dsk.values()))).compute()

    # Create a distributed Dataframe from all the pieces
    divs = [None] * (len(dsk) + 1)
    ret = new_dd_object(dsk, name, meta, divs).persist()
    wait(ret)

    # Release all temporary dataframes
    for fut in [*shuffle_result.values(), *dsk.values()]:
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
