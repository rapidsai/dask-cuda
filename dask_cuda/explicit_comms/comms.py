import time
import uuid
import random
import asyncio
import concurrent.futures

import ucp
import numpy as np
from distributed import get_worker, default_client, wait
from distributed.comm.addressing import parse_host_port

from .dask_df_utils import extract_ddf_partitions, to_dask_cudf
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


async def _create_ucp_listeners(session_state, nworkers, rank):
    assert session_state["loop"] is asyncio.get_event_loop()
    assert "nworkers" not in session_state
    session_state["nworkers"] = nworkers
    assert "rank" not in session_state
    session_state["rank"] = rank

    async def server_handler(ep):
        peer_rank = np.empty((1,), dtype=np.uint64)
        await ep.recv(peer_rank)
        assert peer_rank[0] not in session_state["eps"]
        session_state["eps"][peer_rank[0]] = ep

    session_state["lf"] = ucp.create_listener(server_handler)
    return session_state["lf"].port


async def _create_endpoints(session_state, peers):
    assert session_state["loop"] is asyncio.get_event_loop()

    for rank, address, ucx_port in peers:
        ep = await ucp.create_endpoint(parse_host_port(address)[0], ucx_port)
        await ep.send(np.array([session_state["rank"]], dtype=np.uint64))
        session_state["eps"][rank] = ep

    while len(session_state["eps"]) < session_state["nworkers"] - 1:
        await asyncio.sleep(0.1)


async def _stop_ucp_listeners(session_state):
    assert len(session_state["eps"]) == session_state["nworkers"] - 1
    assert session_state["loop"] is asyncio.get_event_loop()
    del session_state["lf"]


async def _get_worker_state(session_state):
    s = session_state.copy()
    s["eps"] = [(rank, hex(ep.uid)) for rank, ep in s["eps"].items()]
    return s["rank"]


class CommsContext:
    def __init__(self, client=None):
        self.client = client if client is not None else default_client()
        self.sessionId = uuid.uuid4().bytes

        # Get address of all workers (not Nanny addresses)
        self.worker_address = list(self.client.run(lambda: 42).keys())

        ucx_ports = []
        for rank, address in enumerate(self.worker_address):
            ucx_ports.append(
                self.submit(
                    address,
                    _create_ucp_listeners,
                    len(self.worker_address),
                    rank,
                    wait=True,
                )
            )

        # Create list of rank, address and ucx port
        workers = [
            (rank, address, port)
            for rank, (address, port) in enumerate(zip(self.worker_address, ucx_ports))
        ]

        # Each worker creates a UCX endpoint to all workers with greater rank
        futures = []
        for i in range(len(workers)):
            futures.append(
                self.submit(workers[i][1], _create_endpoints, workers[i + 1 :])
            )
        self.client.gather(futures)

        # At this point all workers should have a rank and endpoints to
        # all other workers thus we can now stop the listening.
        self.run(_stop_ucp_listeners)

    def submit(self, worker, coroutine, *args, wait=False):
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
        if workers is None:
            workers = self.worker_address
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

    def dataframe_operation(self, coroutine, df1, df2):
        key = uuid.uuid1()
        df1_fut = self.client.sync(extract_ddf_partitions, df1)
        df2_fut = self.client.sync(extract_ddf_partitions, df2)
        df1_parts = utils.workers_to_parts(df1_fut)
        df2_parts = utils.workers_to_parts(df2_fut)

        ret = []
        for i, worker in enumerate(df1_parts.keys()):
            ret.append(
                self.submit(
                    worker,
                    coroutine,
                    df1_parts[worker],
                    df2_parts[worker],
                    random.random(),
                )
            )
        return to_dask_cudf(ret)
