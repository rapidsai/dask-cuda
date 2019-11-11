import logging
import os
import threading

from multiprocessing.queues import Empty
from tornado import gen
from tornado.ioloop import IOLoop, TimeoutError
from tornado.locks import Event

from distributed.deploy import ProcessInterface
from distributed.process import AsyncProcess
from distributed.utils import mp_context, silence_logging
from distributed.comm.addressing import address_from_user_args
from distributed import Scheduler as _Scheduler

logger = logging.getLogger(__name__)


class Scheduler(ProcessInterface):
    def __init__(self, env=None, *args, **kwargs):
        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.proc_cls = _Scheduler
        self.process = None
        self.env = env or {}

    def __repr__(self):
        self.child_info_stop_q.put({"op": "info"})
        try:
            msg = self.parent_info_q.get(timeout=3000)
        except Empty:
            pass
        else:
            assert msg.pop("op") == "info"
            return "<Scheduler: '%s' processes: %d cores: %d>" % (
                self.address,
                msg.pop("workers"),
                msg.pop("total_nthreads"),
            )

    async def _wait_until_started(self):
        delay = 0.05
        while True:
            if self.status != "starting":
                return
            try:
                msg = self.init_result_q.get_nowait()
            except Empty:
                await gen.sleep(delay)
                continue

            if "exception" in msg:
                logger.error(
                    "Failed while trying to start scheduler process: %s",
                    msg["exception"],
                )
                await self.process.join()
                raise msg
            else:
                return msg

    async def start(self):
        if self.status == "running":
            return self.status
        if self.status == "starting":
            await self.running.wait()
            return self.status

        self.init_result_q = init_q = mp_context.Queue()
        self.child_info_stop_q = mp_context.Queue()
        self.parent_info_q = mp_context.Queue()

        self.process = AsyncProcess(
            target=self._run,
            name="Dask CUDA Scheduler process",
            kwargs=dict(
                proc_cls=self.proc_cls,
                kwargs=self.kwargs,
                silence_logs=False,
                init_result_q=self.init_result_q,
                child_info_stop_q=self.child_info_stop_q,
                parent_info_q=self.parent_info_q,
                env=self.env,
            ),
        )
        # self.process.daemon = dask.config.get("distributed.worker.daemon", default=True)
        self.process.set_exit_callback(self._on_exit)
        self.running = Event()
        self.stopped = Event()
        self.status = "starting"
        try:
            await self.process.start()
        except OSError:
            logger.exception("Failed to start CUDA Scheduler process", exc_info=True)
            self.process.terminate()
            return

        msg = await self._wait_until_started()
        if not msg:
            return self.status
        self.address = msg["address"]
        assert self.address
        self.status = "running"
        self.running.set()

        init_q.close()

        await super().start()

    def _on_exit(self, proc):
        if proc is not self.process:
            return
        self.mark_stopped()

    def _death_message(self, pid, exitcode):
        assert exitcode is not None
        if exitcode == 255:
            return "Scheduler process %d was killed by unknown signal" % (pid,)
        elif exitcode >= 0:
            return "Scheduler process %d exited with status %d" % (pid, exitcode)
        else:
            return "Scheduler process %d was killed by signal %d" % (pid, -exitcode)

    def mark_stopped(self):
        if self.status != "stopped":
            r = self.process.exitcode
            assert r is not None
            if r != 0:
                msg = self._death_message(self.process.pid, r)
                logger.info(msg)
            self.status = "stopped"
            self.stopped.set()
            # Release resources
            self.process.close()
            self.init_result_q = None
            self.child_info_stop_q = None
            self.parent_info_q = None
            self.process = None

    async def close(self):
        timeout = 2
        loop = IOLoop.current()
        deadline = loop.time() + timeout
        if self.status == "closing":
            await self.finished()
            assert self.status == "closed"

        if self.status == "closed":
            return

        try:
            if self.process is not None:
                # await self.kill()
                process = self.process
                self.child_info_stop_q.put(
                    {"op": "stop", "timeout": max(0, deadline - loop.time()) * 0.8}
                )
                self.child_info_stop_q.close()
                self.parent_info_q.close()

                while process.is_alive() and loop.time() < deadline:
                    await gen.sleep(0.05)

                if process.is_alive():
                    logger.warning(
                        "Scheduler process still alive after %d seconds, killing",
                        timeout,
                    )
                    try:
                        await process.terminate()
                    except Exception as e:
                        logger.error("Failed to kill scheduler process: %s", e)
        except Exception:
            pass
        self.process = None
        self.status = "closed"
        await super().close()

    @classmethod
    def _run(
        cls,
        silence_logs,
        init_result_q,
        child_info_stop_q,
        parent_info_q,
        proc_cls,
        kwargs,
        env,
    ):  # pragma: no cover
        os.environ.update(env)

        if silence_logs:
            logger.setLevel(silence_logs)

        IOLoop.clear_instance()
        loop = IOLoop()
        loop.make_current()
        scheduler = proc_cls(**kwargs)

        async def do_stop(timeout=5):
            try:
                await scheduler.close(comm=None, fast=False, close_workers=False)
            finally:
                loop.stop()

        def watch_stop_q():
            """
            Wait for an incoming stop message and then stop the
            scheduler cleanly.
            """
            while True:
                try:
                    msg = child_info_stop_q.get(timeout=1000)
                except Empty:
                    pass
                else:
                    op = msg.pop("op")
                    assert op == "stop" or op == "info"
                    if op == "stop":
                        child_info_stop_q.close()
                        loop.add_callback(do_stop, **msg)
                        break
                    elif op == "info":
                        parent_info_q.put(
                            {
                                "op": "info",
                                "workers": len(scheduler.workers),
                                "total_nthreads": scheduler.total_nthreads,
                            }
                        )

        t = threading.Thread(target=watch_stop_q, name="Scheduler stop queue watch")
        t.daemon = True
        t.start()

        async def run():
            """
            Try to start scheduler and inform parent of outcome.
            """
            try:
                await scheduler.start()
            except Exception as e:
                logger.exception("Failed to start scheduler")
                init_result_q.put({"exception": e})
                init_result_q.close()
            else:
                try:
                    assert scheduler.address
                except ValueError:
                    pass
                else:
                    init_result_q.put({"address": scheduler.address})
                    init_result_q.close()
                    await scheduler.finished()
                    logger.info("Scheduler closed")

        try:
            loop.run_sync(run)
        except TimeoutError:
            # Loop was stopped before wait_until_closed() returned, ignore
            pass
        except KeyboardInterrupt:
            pass
