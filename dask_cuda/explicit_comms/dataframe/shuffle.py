from __future__ import annotations

import asyncio
import functools
import inspect
from collections import defaultdict
from math import ceil
from operator import getitem
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar

import dask
import dask.config
import dask.dataframe
import dask.utils
import distributed.worker
from dask.base import tokenize
from dask.dataframe.core import DataFrame, Series, _concat as dd_concat, new_dd_object
from dask.dataframe.shuffle import group_split_dispatch, hash_object_dispatch
from distributed import wait
from distributed.protocol import nested_deserialize, to_serialize
from distributed.worker import Worker

from .. import comms

T = TypeVar("T")


Proxify = Callable[[T], T]


def get_proxify(worker: Worker) -> Proxify:
    """Get function to proxify objects"""
    from dask_cuda.proxify_host_file import ProxifyHostFile

    if isinstance(worker.data, ProxifyHostFile):
        # Notice, we know that we never call proxify() on the same proxied
        # object thus we can speedup the call by setting `duplicate_check=False`
        return lambda x: worker.data.manager.proxify(x, duplicate_check=False)[0]
    return lambda x: x  # no-op


def get_no_comm_postprocess(
    stage: Dict[str, Any], num_rounds: int, batchsize: int, proxify: Proxify
) -> Callable[[DataFrame], DataFrame]:
    """Get function for post-processing partitions not communicated

    In cuDF, the `group_split_dispatch` uses `scatter_by_map` to create
    the partitions, which is implemented by splitting a single base dataframe
    into multiple partitions. This means that memory are not freed until
    ALL partitions are deleted.

    In order to free memory ASAP, we can deep copy partitions NOT being
    communicated. We do this when `num_rounds != batchsize`.

    Parameters
    ----------
    stage
        The staged input dataframes.
    num_rounds
        Number of rounds of dataframe partitioning and all-to-all communication.
    batchsize
        Number of partitions each worker will handle in each round.
    proxify
        Function to proxify object.

    Returns
    -------
    Function to be called on partitions not communicated.

    """
    if num_rounds == batchsize:
        return lambda x: x

    # Check that we are shuffling a cudf dataframe
    try:
        import cudf
    except ImportError:
        return lambda x: x
    if not stage or not isinstance(next(iter(stage.values())), cudf.DataFrame):
        return lambda x: x

    # Deep copying a cuDF dataframe doesn't deep copy its index hence
    # we have to do it explicitly.
    return lambda x: proxify(
        x._from_data(
            x._data.copy(deep=True),
            x._index.copy(deep=True),
        )
    )


async def send(
    eps,
    myrank: int,
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
    myrank: int,
    rank_to_out_part_ids: Dict[int, Set[int]],
    out_part_id_to_dataframe_list: Dict[int, List[DataFrame]],
    proxify: Proxify,
) -> None:
    """Notice, received items are appended to `out_parts_list`"""

    async def read_msg(rank: int) -> None:
        msg: Dict[int, DataFrame] = nested_deserialize(await eps[rank].read())
        for out_part_id, df in msg.items():
            out_part_id_to_dataframe_list[out_part_id].append(proxify(df))

    await asyncio.gather(
        *(read_msg(rank) for rank in rank_to_out_part_ids if rank != myrank)
    )


def compute_map_index(
    df: DataFrame, column_names: List[str], npartitions: int
) -> Series:
    """Return a Series that maps each row `df` to a partition ID

    The partitions are determined by hashing the columns given by column_names
    unless if `column_names[0] == "_partitions"`, in which case the values of
    `column_names[0]` are used as index.

    Parameters
    ----------
    df
        The dataframe.
    column_names
        List of column names on which we want to split.
    npartitions
        The desired number of output partitions.

    Returns
    -------
    Series
        Series that maps each row `df` to a partition ID
    """

    if column_names[0] == "_partitions":
        ind = df[column_names[0]]
    else:
        ind = hash_object_dispatch(
            df[column_names] if column_names else df, index=False
        )
    return ind % npartitions


def partition_dataframe(
    df: DataFrame, column_names: List[str], npartitions: int, ignore_index: bool
) -> Dict[int, DataFrame]:
    """Partition dataframe to a dict of dataframes

    The partitions are determined by hashing the columns given by column_names
    unless `column_names[0] == "_partitions"`, in which case the values of
    `column_names[0]` are used as index.

    Parameters
    ----------
    df
        The dataframe to partition
    column_names
        List of column names on which we want to partition.
    npartitions
        The desired number of output partitions.
    ignore_index
        Ignore index during shuffle. If True, performance may improve,
        but index values will not be preserved.

    Returns
    -------
    partitions
        Dict of dataframe-partitions, mapping partition-ID to dataframe
    """
    if column_names[0] != "_partitions" and hasattr(df, "partition_by_hash"):
        return dict(
            zip(
                range(npartitions),
                df.partition_by_hash(
                    column_names, npartitions, keep_index=not ignore_index
                ),
            )
        )
    map_index = compute_map_index(df, column_names, npartitions)
    return group_split_dispatch(df, map_index, npartitions, ignore_index=ignore_index)


def create_partitions(
    stage: Dict[str, Any],
    batchsize: int,
    column_names: List[str],
    npartitions: int,
    ignore_index: bool,
    proxify: Proxify,
) -> Dict[int, DataFrame]:
    """Create partitions from one or more staged dataframes

    Parameters
    ----------
    stage
        The staged input dataframes
    column_names
        List of column names on which we want to split.
    npartitions
        The desired number of output partitions.
    ignore_index
        Ignore index during shuffle.  If True, performance may improve,
        but index values will not be preserved.
    proxify
        Function to proxify object.

    Returns
    -------
    partitions: list of DataFrames
        List of dataframe-partitions
    """

    if not stage:
        return {}
    batchsize = min(len(stage), batchsize)

    # Grouping each input dataframe, one part for each partition ID.
    dfs_grouped: List[Dict[int, DataFrame]] = []
    for _ in range(batchsize):
        dfs_grouped.append(
            proxify(
                partition_dataframe(
                    # pop dataframe in any order, to free staged memory ASAP
                    stage.popitem()[1],
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
        assert len(t) > 0
        if len(t) == 1:
            ret[i] = t[0]
        elif len(t) > 1:
            ret[i] = proxify(dd_concat(t, ignore_index=ignore_index))
    return ret


async def send_recv_partitions(
    eps: dict,
    myrank: int,
    rank_to_out_part_ids: Dict[int, Set[int]],
    out_part_id_to_dataframe: Dict[int, DataFrame],
    no_comm_postprocess: Callable[[DataFrame], DataFrame],
    proxify: Proxify,
    out_part_id_to_dataframe_list: Dict[int, List[DataFrame]],
) -> None:
    """Send and receive (all-to-all) partitions between all workers

    Parameters
    ----------
    eps
        Communication endpoints to the other workers.
    myrank
        The rank of this worker.
    rank_to_out_part_ids
        dict that for each worker rank specifies a set of output partition IDs.
        If the worker shouldn't return any partitions, it is excluded from the
        dict. Partition IDs are global integers `0..npartitions` and corresponds
        to the dict keys returned by `group_split_dispatch`.
    out_part_id_to_dataframe
        Mapping from partition ID to dataframe. This dict is cleared on return.
    no_comm_postprocess
        Function to post-process partitions not communicated.
        See `get_no_comm_postprocess`
    proxify
        Function to proxify object.
    out_part_id_to_dataframe_list
        The **output** of this function, which is a dict of the partitions owned by
        this worker.
    """
    await asyncio.gather(
        recv(
            eps,
            myrank,
            rank_to_out_part_ids,
            out_part_id_to_dataframe_list,
            proxify,
        ),
        send(eps, myrank, rank_to_out_part_ids, out_part_id_to_dataframe),
    )

    # At this point `send()` should have pop'ed all output partitions
    # beside the partitions owned be `myrank` (if any).
    assert (
        rank_to_out_part_ids[myrank] == out_part_id_to_dataframe.keys()
        or not out_part_id_to_dataframe
    )
    # We can now add them to the output dataframes.
    for out_part_id, dataframe in out_part_id_to_dataframe.items():
        out_part_id_to_dataframe_list[out_part_id].append(
            no_comm_postprocess(dataframe)
        )
    out_part_id_to_dataframe.clear()


async def shuffle_task(
    s,
    stage_name: str,
    rank_to_inkeys: Dict[int, set],
    rank_to_out_part_ids: Dict[int, Set[int]],
    column_names: List[str],
    npartitions: int,
    ignore_index: bool,
    num_rounds: int,
    batchsize: int,
) -> Dict[int, DataFrame]:
    """Explicit-comms shuffle task

    This function is running on each worker participating in the shuffle.

    Parameters
    ----------
    s: dict
        Worker session state
    stage_name: str
        Name of the stage to retrieve the input keys from.
    rank_to_inkeys: dict
        dict that for each worker rank specifies the set of staged input keys.
    rank_to_out_part_ids: dict
        dict that for each worker rank specifies a set of output partition IDs.
        If the worker shouldn't return any partitions, it is excluded from the
        dict. Partition IDs are global integers `0..npartitions` and corresponds
        to the dict keys returned by `group_split_dispatch`.
    column_names: list of strings
        List of column names on which we want to split.
    npartitions: int
        The desired number of output partitions.
    ignore_index: bool
        Ignore index during shuffle.  If True, performance may improve,
        but index values will not be preserved.
    num_rounds: int
        Number of rounds of dataframe partitioning and all-to-all communication.
    batchsize: int
        Number of partitions each worker will handle in each round.

    Returns
    -------
    partitions: dict
        dict that maps each Partition ID to a dataframe-partition
    """

    proxify = get_proxify(s["worker"])
    eps = s["eps"]
    myrank: int = s["rank"]
    stage = comms.pop_staging_area(s, stage_name)
    assert stage.keys() == rank_to_inkeys[myrank]
    no_comm_postprocess = get_no_comm_postprocess(stage, num_rounds, batchsize, proxify)

    out_part_id_to_dataframe_list: Dict[int, List[DataFrame]] = defaultdict(list)
    for _ in range(num_rounds):
        partitions = create_partitions(
            stage, batchsize, column_names, npartitions, ignore_index, proxify
        )
        await send_recv_partitions(
            eps,
            myrank,
            rank_to_out_part_ids,
            partitions,
            no_comm_postprocess,
            proxify,
            out_part_id_to_dataframe_list,
        )

    # Finally, we concatenate the output dataframes into the final output partitions
    ret = {}
    while out_part_id_to_dataframe_list:
        part_id, dataframe_list = out_part_id_to_dataframe_list.popitem()
        ret[part_id] = proxify(
            dd_concat(
                dataframe_list,
                ignore_index=ignore_index,
            )
        )
        # For robustness, we yield this task to give Dask a chance to do bookkeeping
        # such as letting the Worker answer heartbeat requests
        await asyncio.sleep(0)
    return ret


def shuffle(
    df: DataFrame,
    column_names: List[str],
    npartitions: Optional[int] = None,
    ignore_index: bool = False,
    batchsize: Optional[int] = None,
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
    batchsize: int
        A shuffle consist of multiple rounds where each worker partitions and
        then all-to-all communicates a number of its dataframe partitions. The batch
        size is the number of partitions each worker will handle in each round.
        If -1, each worker will handle all its partitions in a single round and
        all techniques to reduce memory usage are disabled, which might be faster
        when memory pressure isn't an issue.
        If None, the value of `DASK_EXPLICIT_COMMS_BATCHSIZE` is used or 1 if not
        set thus by default, we prioritize robustness over performance.

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
    wait([df])  # Make sure all keys has been materialized on workers
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

    # Get batchsize
    max_num_inkeys = max(len(k) for k in rank_to_inkeys.values())
    batchsize = batchsize or dask.config.get("explicit-comms-batchsize", 1)
    if batchsize == -1:
        batchsize = max_num_inkeys
    if not isinstance(batchsize, int) or batchsize < 0:
        raise ValueError(
            "explicit-comms-batchsize must be a "
            f"positive integer or -1 (was '{batchsize}')"
        )

    # Get number of rounds of dataframe partitioning and all-to-all communication.
    num_rounds = ceil(max_num_inkeys / batchsize)

    # Find the output partition IDs for each worker
    div = npartitions // len(ranks)
    rank_to_out_part_ids: Dict[int, Set[int]] = {}  # rank -> set of partition id
    for i, rank in enumerate(ranks):
        rank_to_out_part_ids[rank] = set(range(div * i, div * (i + 1)))
    for rank, i in zip(ranks, range(div * len(ranks), npartitions)):
        rank_to_out_part_ids[rank].add(i)

    # Run a shuffle task on each worker
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
            num_rounds,
            batchsize,
        )
    wait(list(shuffle_result.values()))

    # Step (d): extract individual dataframe-partitions. We use `submit()`
    #           to control where the tasks are executed.
    # TODO: can we do this without using `submit()` to avoid the overhead
    #       of creating a Future for each dataframe partition?

    dsk = {}
    for rank in ranks:
        for part_id in rank_to_out_part_ids[rank]:
            dsk[(name, part_id)] = c.client.submit(
                getitem,
                shuffle_result[rank],
                part_id,
                workers=[c.worker_addresses[rank]],
            )

    # Create a distributed Dataframe from all the pieces
    divs = [None] * (len(dsk) + 1)
    ret = new_dd_object(dsk, name, df_meta, divs).persist()
    wait([ret])

    # Release all temporary dataframes
    for fut in [*shuffle_result.values(), *dsk.values()]:
        fut.release()
    return ret


def _use_explicit_comms() -> bool:
    """Is explicit-comms and available?"""
    if dask.config.get("explicit-comms", False):
        try:
            # Make sure we have an activate client.
            distributed.worker.get_client()
        except (ImportError, ValueError):
            pass
        else:
            return True
    return False


def get_rearrange_by_column_wrapper(func):
    """Returns a function wrapper that dispatch the shuffle to explicit-comms.

    Notice, this is monkey patched into Dask at dask_cuda import
    """

    func_sig = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if _use_explicit_comms():
            # Convert `*args, **kwargs` to a dict of `keyword -> values`
            kw = func_sig.bind(*args, **kwargs)
            kw.apply_defaults()
            kw = kw.arguments
            # Notice, we only overwrite the default and the "tasks" shuffle
            # algorithm. The "disk" and "p2p" algorithm, we don't touch.
            if kw["shuffle"] in ("tasks", None):
                col = kw["col"]
                if isinstance(col, str):
                    col = [col]
                return shuffle(kw["df"], col, kw["npartitions"], kw["ignore_index"])
        return func(*args, **kwargs)

    return wrapper


def get_default_shuffle_method() -> str:
    """Return the default shuffle algorithm used by Dask

    This changes the default shuffle algorithm from "p2p" to "tasks"
    when explicit comms is enabled.
    """
    ret = dask.config.get("dataframe.shuffle.algorithm", None)
    if ret is None and _use_explicit_comms():
        return "tasks"
    return dask.utils.get_default_shuffle_method()
