import os

from distributed.utils_test import gen_test
from distributed import Client, SpecCluster

from dask_cuda.worker_spec import worker_spec
from dask_cuda import Scheduler


@gen_test(timeout=20)
async def test_scheduler_environment_variables():
    workers = worker_spec()
    scheduler = {
        "cls": Scheduler,
        "options": {"env": {"SCHEDULER_VAR1": "123", "SCHEDULER_VAR2": "aBc"}},
    }

    async with SpecCluster(
        workers=workers, scheduler=scheduler, asynchronous=True
    ) as cluster:
        async with Client(cluster, asynchronous=True) as client:
            env = await client.run_on_scheduler(lambda: os.environ)
            assert "SCHEDULER_VAR1" in env
            assert "SCHEDULER_VAR2" in env
            assert env["SCHEDULER_VAR1"] == "123"
            assert env["SCHEDULER_VAR2"] == "aBc"


@gen_test(timeout=20)
async def test_scheduler_repr():
    workers = worker_spec()
    scheduler = {"cls": Scheduler}

    async with SpecCluster(
        workers=workers, scheduler=scheduler, asynchronous=True
    ) as cluster:
        async with Client(cluster, asynchronous=True) as client:
            assert len(cluster.workers) == len(workers)
            assert "processes: %d " % len(workers) in str(cluster.scheduler)
            await cluster.workers[0].close()
            assert "processes: %d " % (len(workers) - 1) in str(cluster.scheduler)
