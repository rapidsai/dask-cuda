import asyncio
import concurrent.futures
import contextlib
import time
import uuid
from typing import Any, Dict, Hashable, Iterable, List, Optional

import distributed.comm
from distributed import Client, Worker, default_client, get_worker
from distributed.comm.addressing import parse_address, parse_host_port, unparse_address

_default_comms = None


def get_multi_lock_or_null_context(multi_lock_context, *args, **kwargs):
    """Return either a MultiLock or a NULL context

    Parameters
    ----------
    multi_lock_context: bool
        If True return MultiLock context else return a NULL context that
        doesn't do anything

    *args, **kwargs:
        Arguments parsed to the MultiLock creation

    Returns
    -------
    context: context
        Either `MultiLock(*args, **kwargs)` or a NULL context
    """
    if multi_lock_context:
        from distributed import MultiLock

        return MultiLock(*args, **kwargs)
    else:
        return contextlib.nullcontext()


def default_comms(client: Optional[Client] = None) -> "CommsContext":
    """Return the default comms object

    Creates a new default comms object if no one exist.

    Parameters
    ----------
    client: Client, optional
        If no default comm object exists, create the new comm on `client`
        are returned.

    Returns
    -------
    comms: CommsContext
        The default comms object
    """
    global _default_comms
    if _default_comms is None:
        _default_comms = CommsContext(client=client)
    return _default_comms


def worker_state(sessionId: Optional[int] = None) -> dict:
    """Retrieve the state(s) of the current worker

    Parameters
    ----------
    sessionId: int, optional
        Worker session state ID. If None, all states of the worker
        are returned.

    Returns
    -------
    state: dict
        Either a single state dict or a dict of state dict
    """
    worker: Any = get_worker()
    if not hasattr(worker, "_explicit_comm_state"):
        worker._explicit_comm_state = {}
    if sessionId is not None:
        if sessionId not in worker._explicit_comm_state:
            worker._explicit_comm_state[sessionId] = {
                "ts": time.time(),
                "eps": {},
                "loop": worker.loop.asyncio_loop,
                "worker": worker,
            }
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
    """Each worker creates a UCX endpoint to all workers with greater rank"""
    assert session_state["loop"] is asyncio.get_event_loop()

    myrank = session_state["rank"]
    peers = list(enumerate(peers))

    # Create endpoints to workers with a greater rank than my rank
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


async def _stage_keys(session_state: dict, name: str, keys: set):
    worker: Worker = session_state["worker"]
    data = worker.data
    my_keys = keys.intersection(data)

    stages = session_state.get("stages", {})
    stage = stages.get(name, {})
    for k in my_keys:
        stage[k] = data[k]
    stages[name] = stage
    session_state["stages"] = stages
    return (session_state["rank"], my_keys)


class CommsContext:
    """Communication handler for explicit communication

    Parameters
    ----------
    client: Client, optional
        Specify client to use for communication. If None, use the default client.
    """

    client: Client
    sessionId: int
    worker_addresses: List[str]

    def __init__(self, client: Optional[Client] = None):
        self.client = client if client is not None else default_client()
        self.sessionId = uuid.uuid4().int

        # Get address of all workers (not Nanny addresses)
        self.worker_addresses = list(self.client.scheduler_info()["workers"].keys())

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

        # Each worker creates an endpoint to all workers with greater rank
        self.run(_create_endpoints, self.worker_direct_addresses)

        # At this point all workers should have a rank and endpoints to
        # all other workers thus we can now stop the listening.
        self.run(_stop_ucp_listeners)

    def submit(self, worker, coroutine, *args, wait=False):
        """Run a coroutine on a single worker

        The coroutine is given the worker's state dict as the first argument
        and ``*args`` as the following arguments.

        Parameters
        ----------
        worker: str
            Worker to run the ``coroutine``
        coroutine: coroutine
            The function to run on the worker
        *args:
            Arguments for ``coroutine``
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

    def run(self, coroutine, *args, workers=None, lock_workers=False):
        """Run a coroutine on multiple workers

        The coroutine is given the worker's state dict as the first argument
        and ``*args`` as the following arguments.

        Parameters
        ----------
        coroutine: coroutine
            The function to run on each worker
        *args:
            Arguments for ``coroutine``
        workers: list, optional
            List of workers. Default is all workers
        lock_workers: bool, optional
            Use distributed.MultiLock to get exclusive access to the workers. Use
            this flag to support parallel runs.

        Returns
        -------
        ret: list
            List of the output from each worker
        """
        if workers is None:
            workers = self.worker_addresses

        with get_multi_lock_or_null_context(lock_workers, workers):
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

    def stage_keys(self, name: str, keys: Iterable[Hashable]) -> Dict[int, set]:
        """Staging keys on workers under the given name

        In an explicit-comms task, use `pop_staging_area(..., name)` to access
        the staged keys and the associated data.

        Notes
        -----
        In the context of explicit-comms, staging is the act of duplicating the
        responsibility of Dask keys. When staging a key, the worker owning the
        key (as assigned by the Dask scheduler) save a reference to the key and
        the associated data to its local staging area. From this point on, if
        the scheduler cancels the key, the worker (and the task running on the
        worker) now has exclusive access to the key and the associated data.
        This way, staging makes it possible for long running explicit-comms tasks
        to free input data ASAP.

        Parameters
        ----------
        name: str
            Name for the staging area
        keys: iterable
            The keys to stage

        Returns
        -------
        dict
            dict that maps each worker-rank to the workers set of staged keys
        """
        return dict(self.run(_stage_keys, name, set(keys)))


def pop_staging_area(session_state: dict, name: str) -> Dict[str, Any]:
    """Pop the staging area called `name`

    This function must be called within a running explicit-comms task.

    Parameters
    ----------
    session_state: dict
        Worker session state
    name: str
        Name for the staging area

    Returns
    -------
    dict
        The staging area, which is a dict that maps keys to their data.
    """
    return session_state["stages"].pop(name)
