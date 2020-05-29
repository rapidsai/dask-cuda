import asyncio
import concurrent.futures
import time
import uuid

import distributed.comm
from distributed import default_client, get_worker
from distributed.comm.addressing import (
    parse_address,
    parse_host_port,
    unparse_address,
)

from . import utils

_default_comms = None


def default_comms(client=None):
    """ Return a comms instance if one has been initialized.
        Otherwise, initialize a new comms instance.
    """
    global _default_comms
    if _default_comms is None:
        _default_comms = CommsContext(client=client)
    return _default_comms


def worker_state(sessionId=None):
    worker = get_worker()
    if not hasattr(worker, "_explicit_comm_state"):
        worker._explicit_comm_state = {}
    if sessionId is not None and sessionId not in worker._explicit_comm_state:
        worker._explicit_comm_state[sessionId] = {
            "ts": time.time(),
            "eps": {},
            "loop": worker.loop.asyncio_loop,
            "worker": worker,
        }

    if sessionId is not None:
        return worker._explicit_comm_state[sessionId]
    return worker._explicit_comm_state


def _run_coroutine_on_worker(sessionId, coroutine, args):
    session_state = worker_state(sessionId)

    def _run():
        future = asyncio.run_coroutine_threadsafe(
            coroutine(session_state, *args), session_state["loop"]
        )
        return future.result()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(_run).result()


async def _create_listeners(session_state, nworkers, rank):
    assert session_state["loop"] is asyncio.get_event_loop()
    assert "nworkers" not in session_state
    session_state["nworkers"] = nworkers
    assert "rank" not in session_state
    session_state["rank"] = rank

    async def server_handler(ep):
        peer_rank = await ep.read()
        session_state["eps"][peer_rank] = ep

    # We listen on the same protocol and address as the worker address
    protocol, address = parse_address(session_state["worker"].address)
    address = parse_host_port(address)[0]
    address = unparse_address(protocol, address)

    session_state["lf"] = distributed.comm.listen(address, server_handler)
    await session_state["lf"].start()
    return session_state["lf"].listen_address


async def _create_endpoints(session_state, peers):
    """ Each worker creates a UCX endpoint to all workers with greater rank"""
    assert session_state["loop"] is asyncio.get_event_loop()

    myrank = session_state["rank"]
    peers = list(enumerate(peers))

    # Create endpoints to workers with a greater rank the my rank
    for rank, address in peers[myrank + 1 :]:
        ep = await distributed.comm.connect(address)
        await ep.write(session_state["rank"])
        session_state["eps"][rank] = ep

    # Block until all endpoints has been created
    while len(session_state["eps"]) < session_state["nworkers"] - 1:
        await asyncio.sleep(0.1)


async def _stop_ucp_listeners(session_state):
    assert len(session_state["eps"]) == session_state["nworkers"] - 1
    assert session_state["loop"] is asyncio.get_event_loop()
    session_state["lf"].stop()
    del session_state["lf"]


class CommsContext:
    """Communication handler for explicit communication"""

    def __init__(self, client=None):
        self.client = client if client is not None else default_client()
        self.sessionId = uuid.uuid4().bytes

        # Get address of all workers (not Nanny addresses)
        self.worker_addresses = list(self.client.run(lambda: 42).keys())

        # Make all workers listen and get all listen addresses
        self.worker_direct_addresses = []
        for rank, address in enumerate(self.worker_addresses):
            self.worker_direct_addresses.append(
                self.submit(
                    address,
                    _create_listeners,
                    len(self.worker_addresses),
                    rank,
                    wait=True,
                )
            )

        # Each worker creates a UCX endpoint to all workers with greater rank
        self.run(_create_endpoints, self.worker_direct_addresses)

        # At this point all workers should have a rank and endpoints to
        # all other workers thus we can now stop the listening.
        self.run(_stop_ucp_listeners)

    def submit(self, worker, coroutine, *args, wait=False):
        """Run a coroutine on a single worker

        Parameters
        ----------
        worker: str
            Worker to run the `coroutine`
        coroutine: coroutine
            The function to run on the worker
        *args:
            Arguments for `coroutine`
        wait: boolean, optional
            If True, waits for the coroutine to finished before returning.
        Returns
        -------
        ret: object or Future
            If wait=True, the result of `coroutine`
            If wait=False, Future that can be waited on later.
        """
        ret = self.client.submit(
            _run_coroutine_on_worker,
            self.sessionId,
            coroutine,
            args,
            workers=[worker],
            pure=False,
        )
        return ret.result() if wait else ret

    def run(self, coroutine, *args, workers=None):
        """Run a coroutine on workers

        Parameters
        ----------
        coroutine: coroutine
            The function to run on each worker
        *args:
            Arguments for `coroutine`
        workers: list, optional
            List of workers. Default is all workers
        Returns
        -------
        ret: list
            List of the output from each worker
        """
        if workers is None:
            workers = self.worker_addresses
        ret = []
        for worker in workers:
            ret.append(
                self.client.submit(
                    _run_coroutine_on_worker,
                    self.sessionId,
                    coroutine,
                    args,
                    workers=[worker],
                    pure=False,
                )
            )
        return self.client.gather(ret)

    def dataframe_operation(self, coroutine, df_list, extra_args=tuple()):
        """Submit an operation on a list of Dask dataframe

        Parameters
        ----------
        coroutine: coroutine
            The function to run on each worker
        df_list: list of Dask.dataframe.Dataframe
            Input dataframes
        extra_args: tuple
            Extra function input
        Returns
        -------
        dataframe: Dask.dataframe.Dataframe
            The resulting dataframe
        """
        df_parts_list = []
        for df in df_list:
            df_parts_list.append(
                utils.workers_to_parts(
                    self.client.sync(utils.extract_ddf_partitions, df)
                )
            )

        # Let's create a dict for each dataframe that specifices the
        # number of partitions each worker has
        dfs_nparts = []
        for df_parts in df_parts_list:
            nparts = {}
            for rank, worker in enumerate(self.worker_addresses):
                npart = len(df_parts.get(worker, []))
                if npart > 0:
                    nparts[rank] = npart
            dfs_nparts.append(nparts)

        # Submit `coroutine` on each worker given the df_parts that
        # belong the specific worker as input
        ret = []
        for worker in self.worker_addresses:
            dfs = []
            for df_parts in df_parts_list:
                dfs.append(df_parts.get(worker, []))
            ret.append(self.submit(worker, coroutine, dfs_nparts, dfs, *extra_args))
        return utils.dataframes_to_dask_dataframe(ret)
