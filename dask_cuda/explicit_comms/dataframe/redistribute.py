from __future__ import annotations

import asyncio
import collections
from collections import defaultdict
from operator import getitem
from typing import Dict, List, Optional, Set

from dask.base import tokenize
from dask.dataframe.core import DataFrame, new_dd_object
from distributed import wait
from distributed.protocol import nested_deserialize, to_serialize

from .. import comms


async def read_and_deserialize(ep, partitions: List[DataFrame]) -> None:
    """Notice, received items are appended to `partitions`"""
    partitions.extend(nested_deserialize(await ep.read()))


async def redistribute_task(
    s,
    stage_name,
    rank_to_send_count: Dict[int, int],
    ranks_to_recv_from: Set[int],
) -> List[DataFrame]:
    """Explicit-comms shuffle task

    This function is running on each worker participating in the shuffle.

    Parameters
    ----------
    s: dict
        Worker session state
    stage_name: str
        Name of the stage to retrieve the input keys from.

    Returns
    -------
    partitions: list of DataFrames
        List of dataframe-partitions
    """

    eps = s["eps"]

    partitions = list(comms.pop_staging_area(s, stage_name).values())

    f = []
    for rank, n in rank_to_send_count.items():
        f.append(eps[rank].write([to_serialize(partitions.pop()) for _ in range(n)]))
    for rank in ranks_to_recv_from:
        f.append(read_and_deserialize(eps[rank], partitions))
    await asyncio.gather(*f)

    return partitions


def redistribute(df: DataFrame, distribution: Optional[List[int]] = None) -> DataFrame:
    """Redistribute the partitions of a dataframe.

    Requires an activate client.

    Parameters
    ----------
    df
        Dataframe to redistribute
    distribution
        If not None, a list specifying the number of partition each worker
        should have after the redistribution. If None, the partitions will
        be distributed evenly between all workers.

    Returns
    -------
    df
        Redistributed dataframe

    """
    c = comms.default_comms()

    # Step (a):
    df = df.persist()  # Make sure optimizations are apply on the existing graph
    wait(df)  # Make sure all keys has been materialized on workers
    name = "explicit-comms-redistribute-" f"{tokenize(df, distribution)}"
    df_meta: DataFrame = df._meta

    rank_to_inkeys = c.stage_keys(name=name, keys=df.__dask_keys__())
    c.client.cancel(df)
    npartitions = sum(len(inkeys) for inkeys in rank_to_inkeys.values())

    if distribution is None:
        d = npartitions // len(rank_to_inkeys)
        rank_to_target_num_outkeys = {r: d for r in rank_to_inkeys}
    else:
        assert len(distribution) == len(rank_to_inkeys)
        rank_to_target_num_outkeys = {
            r: d for r, d in zip(rank_to_inkeys, distribution)
        }

    # Balance the partitions by recording what each rank needs to send and receive
    rank_to_send = defaultdict(list)
    rank_to_recv = defaultdict(list)
    rank_to_num_outkeys = {rank: len(inkeys) for rank, inkeys in rank_to_inkeys.items()}
    for src in rank_to_inkeys:
        target = rank_to_target_num_outkeys[src]
        # As long as `src` has more partitions than its target,
        # we assign partitions to other workers.
        while rank_to_num_outkeys[src] > target:
            surplus = rank_to_num_outkeys[src] - target
            # find destinations for the surplus keys
            candidates = [
                k
                for k, v in rank_to_num_outkeys.items()
                if v < rank_to_target_num_outkeys[k]
            ]
            if not candidates:
                break
            for dst in candidates[:surplus]:
                rank_to_send[src].append(dst)
                rank_to_recv[dst].append(src)
                rank_to_num_outkeys[src] -= 1
                rank_to_num_outkeys[dst] += 1

    # Run `redistribute_task()` on each worker
    task_result = {}
    for rank in rank_to_inkeys:
        task_result[rank] = c.submit(
            c.worker_addresses[rank],
            redistribute_task,
            name,
            dict(collections.Counter(rank_to_send[rank])),
            set(rank_to_recv[rank]),
        )
    wait(list(task_result.values()))

    # Step (d): extract individual dataframe-partitions. We use `submit()`
    #           to control where the tasks are executed.
    # TODO: can we do this without using `submit()` to avoid the overhead
    #       of creating a Future for each dataframe partition?
    dsk = {}
    part_id = 0
    for rank in rank_to_inkeys:
        for i in range(rank_to_num_outkeys[rank]):
            dsk[(name, part_id)] = c.client.submit(
                getitem, task_result[rank], i, workers=[c.worker_addresses[rank]]
            )
            part_id += 1

    # Create a distributed Dataframe from all the pieces
    divs = [None] * (len(dsk) + 1)
    ret = new_dd_object(dsk, name, df_meta, divs).persist()
    wait(ret)

    # Release all temporary dataframes
    for fut in [*task_result.values(), *dsk.values()]:
        fut.release()
    return ret
