from __future__ import annotations

import asyncio
import functools
import inspect
from collections import defaultdict
from operator import getitem
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar

import numpy

import dask
import dask.dataframe
from dask.base import tokenize
from dask.dataframe.core import DataFrame, Series, _concat as dd_concat, new_dd_object
from dask.dataframe.shuffle import group_split_dispatch, hash_object_dispatch
from distributed import wait
from distributed.protocol import nested_deserialize, to_serialize
from distributed.worker import Worker

from .. import comms

T = TypeVar("T")


async def send(
    eps,
    myrank,
    rank_to_out_part_ids: Dict[int, Set[int]],
    out_part_id_to_dataframe: Dict[int, DataFrame],
) -> None:
    """Notice, items sent are removed from `out_part_id_to_dataframe`"""
    futures = []
    for rank, out_part_ids in rank_to_out_part_ids.items():
        if rank != myrank:
            msg = {
                i: to_serialize(out_part_id_to_dataframe.pop(i))
                for i in (out_part_ids & out_part_id_to_dataframe.keys())
            }
            futures.append(eps[rank].write(msg))
    await asyncio.gather(*futures)


async def recv(
    eps,
    myrank,
    rank_to_out_part_ids: Dict[int, Set[int]],
    out_part_id_to_dataframe_list: Dict[int, List[DataFrame]],
    proxify,
) -> None:
    """Notice, received items are appended to `out_parts_list`"""

    async def read_msg(rank: int) -> None:
        msg: Dict[int, DataFrame] = nested_deserialize(await eps[rank].read())
        for out_part_id, df in msg.items():
            out_part_id_to_dataframe_list[out_part_id].append(proxify(df))

    await asyncio.gather(
        *(read_msg(rank) for rank in rank_to_out_part_ids if rank != myrank)
    )


def get_proxify(worker: Worker) -> Callable[[T], T]:
    """Get function to proxify objects"""
    from dask_cuda.proxify_host_file import ProxifyHostFile

    if isinstance(worker.data, ProxifyHostFile):
        data = worker.data
        return lambda x: data.manager.proxify(x)[0]
    return lambda x: x  # no-op


def compute_map_index(df: Any, column_names, npartitions) -> Series:
    """Return a Series that maps each row `df` to a partition ID

    The partitions are determined by hashing the columns given by column_names
    unless if `column_names[0] == "_partitions"`, in which case the values of
    `column_names[0]` are used as index.

    Parameters
    ----------
    df: DataFrame
    column_names: list of strings
        List of column names on which we want to split.
    npartitions: int or None
        The desired number of output partitions.

    Returns
    -------
    out: Dict[int, DataFrame]
        A dictionary mapping integers in {0..k} to dataframes such that the
        hash values of `df[col]` are well partitioned.
    """

    if column_names[0] == "_partitions":
        ind = df[column_names[0]]
    else:
        ind = hash_object_dispatch(
            df[column_names] if column_names else df, index=False
        )
    typ = numpy.min_scalar_type(npartitions * 2)
    return (ind % npartitions).astype(typ, copy=False)


def single_shuffle_group(
    df: DataFrame, column_names, npartitions, ignore_index
) -> Dict[int, DataFrame]:
    """Split dataframe based on the indexes returned by `compute_map_index`"""
    map_index = compute_map_index(df, column_names, npartitions)
    return group_split_dispatch(df, map_index, npartitions, ignore_index=ignore_index)


def multi_shuffle_group(
    df_meta: DataFrame,
    dfs: Dict[str, DataFrame],
    column_names,
    npartitions,
    ignore_index,
    proxify,
) -> Dict[int, DataFrame]:
    """Split multiple dataframes such that each partition hashes to the same

    Since we concatenate dataframes belonging to the same partition, each
    partition ID maps to exactly one dataframe.

    Parameters
    ----------
    df_meta: DataFrame
        An empty dataframe matching the expected output
    dfs: dict of dataframes
        The dataframes to split given as a map of stage keys to dataframes
    column_names: list of strings
        List of column names on which we want to split.
    npartitions: int or None
        The desired number of output partitions.
    ignore_index: bool
        Ignore index during shuffle.  If True, performance may improve,
        but index values will not be preserved.
    proxify: callable
        Function to proxify object.

    Returns
    -------
    dict of DataFrames
        Mapping from partition ID to dataframe.
    """

    # Grouping each input dataframe, one part for each partition ID.
    dfs_grouped: List[Dict[int, DataFrame]] = []
    while dfs:
        dfs_grouped.append(
            proxify(
                single_shuffle_group(
                    # pop dataframe in any order, to free staged memory ASAP
                    dfs.popitem()[1],
                    column_names,
                    npartitions,
                    ignore_index,
                )
            )
        )

    # Maps each output partition ID to a dataframe. If the partition is empty,
    # an empty dataframe is used.
    ret: Dict[int, DataFrame] = {}
    for i in range(npartitions):  # Iterate over all possible output partition IDs
        t = [df_grouped[i] for df_grouped in dfs_grouped]
        if len(t) == 1:
            ret[i] = t[0]
        elif len(t) > 1:
            ret[i] = proxify(dd_concat(t, ignore_index=ignore_index))
        else:
            ret[i] = df_meta  # Empty dataframe
    return ret


async def shuffle_task(
    s,
    stage_name,
    df_meta,
    rank_to_inkeys: Dict[int, set],
    rank_to_out_part_ids: Dict[int, Set[int]],
    column_names,
    npartitions,
    ignore_index,
) -> List[DataFrame]:
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
        dict that for each worker rank specifices a set of output partition IDs.
        If the worker shouldn't return any partitions, it is excluded from the
        dict. Partition IDs are global integers `0..npartitions` and corresponds
        to the dict keys returned by `group_split_dispatch`.
    column_names: list of strings
        List of column names on which we want to split.
    npartitions: int or None
        The desired number of output partitions.
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
    stage = comms.pop_staging_area(s, stage_name)
    assert stage.keys() == rank_to_inkeys[myrank]

    out_part_id_to_dataframe = multi_shuffle_group(
        df_meta=df_meta,
        dfs=stage,
        column_names=column_names,
        npartitions=npartitions,
        ignore_index=ignore_index,
        proxify=proxify,
    )

    # Communicate all the dataframe-partitions all-to-all. The result is
    # `out_part_id_to_dataframe_list` that for each output partition maps
    # a list of dataframes received.
    out_part_id_to_dataframe_list: Dict[int, List[DataFrame]] = defaultdict(list)
    await asyncio.gather(
        recv(eps, myrank, rank_to_out_part_ids, out_part_id_to_dataframe_list, proxify),
        send(eps, myrank, rank_to_out_part_ids, out_part_id_to_dataframe),
    )

    # At this point `send()` should have pop'ed all output partitions
    # beside the partitions owned be `myrank`.
    assert rank_to_out_part_ids[myrank] == out_part_id_to_dataframe.keys()
    # We can now add them to the output dataframes.
    for out_part_id, dataframe in out_part_id_to_dataframe.items():
        out_part_id_to_dataframe_list[out_part_id].append(dataframe)
    del out_part_id_to_dataframe

    # Finally, we concatenate the output dataframes into the final output partitions
    return [
        proxify(dd_concat(dfs, ignore_index=ignore_index))
        for dfs in out_part_id_to_dataframe_list.values()
    ]


def shuffle(
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
        Ignore index during shuffle. If True, performance may improve,
        but index values will not be preserved.

    Returns
    -------
    df: dask.dataframe.DataFrame
        Shuffled dataframe

    Developer Notes
    ---------------
    The implementation consist of three steps:
      (a) Stage the partitions of `df` on all workers and then cancel them
          thus at this point the Dask Scheduler doesn't know about any of the
          the partitions.
      (b) Submit a task on each worker that shuffle (all-to-all communicate)
          the staged partitions and return a list of dataframe-partitions.
      (c) Submit a dask graph that extract (using `getitem()`) individual
          dataframe-partitions from (b).
    """
    c = comms.default_comms()

    # The ranks of the output workers
    ranks = list(range(len(c.worker_addresses)))

    # By default, we preserve number of partitions
    if npartitions is None:
        npartitions = df.npartitions

    # Step (a):
    df = df.persist()  # Make sure optimizations are apply on the existing graph
    wait(df)  # Make sure all keys has been materialized on workers
    name = (
        "explicit-comms-shuffle-"
        f"{tokenize(df, column_names, npartitions, ignore_index)}"
    )
    df_meta: DataFrame = df._meta

    # Stage all keys of `df` on the workers and cancel them, which makes it possible
    # for the shuffle to free memory as the partitions of `df` are consumed.
    # See CommsContext.stage_keys() for a description of staging.
    rank_to_inkeys = c.stage_keys(name=name, keys=df.__dask_keys__())
    c.client.cancel(df)

    # Find the output partition IDs for each worker
    div = npartitions // len(ranks)
    rank_to_out_part_ids: Dict[int, Set[int]] = {}  # rank -> set of partition id
    for i, rank in enumerate(ranks):
        rank_to_out_part_ids[rank] = set(range(div * i, div * (i + 1)))
    for rank, i in zip(ranks, range(div * len(ranks), npartitions)):
        rank_to_out_part_ids[rank].add(i)

    # Run `_shuffle()` on each worker
    shuffle_result = {}
    for rank in ranks:
        shuffle_result[rank] = c.submit(
            c.worker_addresses[rank],
            shuffle_task,
            name,
            df_meta,
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

    # Create a distributed Dataframe from all the pieces
    divs = [None] * (len(dsk) + 1)
    ret = new_dd_object(dsk, name, df_meta, divs).persist()
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
