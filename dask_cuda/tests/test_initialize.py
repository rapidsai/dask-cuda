# SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap

import cuda.core.experimental
import numpy
import psutil
import pytest

from dask import array as da
from distributed import Client
from distributed.deploy.local import LocalCluster

from dask_cuda.initialize import initialize
from dask_cuda.utils import get_ucx_config
from dask_cuda.utils_test import IncreasedCloseTimeoutNanny

mp = mp.get_context("spawn")  # type: ignore

# Notice, all of the following tests is executed in a new process such
# that UCX options of the different tests doesn't conflict.
# Furthermore, all tests do some computation to trigger initialization
# of UCX before retrieving the current config.


def _test_initialize_ucx_tcp():
    ucxx = pytest.importorskip("ucxx")

    kwargs = {"enable_tcp_over_ucx": True}
    initialize(**kwargs)
    with LocalCluster(
        protocol="ucx",
        dashboard_address=None,
        n_workers=1,
        threads_per_worker=1,
        processes=True,
        worker_class=IncreasedCloseTimeoutNanny,
        config={"distributed-ucxx": get_ucx_config(**kwargs)},
    ) as cluster:
        with Client(cluster) as client:
            res = da.from_array(numpy.arange(10000), chunks=(1000,))
            res = res.sum().compute()
            assert res == 49995000

            def check_ucx_options():
                conf = ucxx.get_config()
                assert "TLS" in conf
                assert "tcp" in conf["TLS"]
                assert "cuda_copy" in conf["TLS"]
                assert "tcp" in conf["SOCKADDR_TLS_PRIORITY"]
                return True

            assert client.run_on_scheduler(check_ucx_options) is True
            assert all(client.run(check_ucx_options).values())


def test_initialize_ucx_tcp():
    pytest.importorskip("distributed_ucxx")

    p = mp.Process(target=_test_initialize_ucx_tcp)
    p.start()
    p.join()
    assert not p.exitcode


def _test_initialize_ucx_nvlink():
    ucxx = pytest.importorskip("ucxx")

    kwargs = {"enable_nvlink": True}
    initialize(**kwargs)
    with LocalCluster(
        protocol="ucx",
        dashboard_address=None,
        n_workers=1,
        threads_per_worker=1,
        processes=True,
        worker_class=IncreasedCloseTimeoutNanny,
        config={"distributed-ucxx": get_ucx_config(**kwargs)},
    ) as cluster:
        with Client(cluster) as client:
            res = da.from_array(numpy.arange(10000), chunks=(1000,))
            res = res.sum().compute()
            assert res == 49995000

            def check_ucx_options():
                conf = ucxx.get_config()
                assert "TLS" in conf
                assert "cuda_ipc" in conf["TLS"]
                assert "tcp" in conf["TLS"]
                assert "cuda_copy" in conf["TLS"]
                assert "tcp" in conf["SOCKADDR_TLS_PRIORITY"]
                return True

            assert client.run_on_scheduler(check_ucx_options) is True
            assert all(client.run(check_ucx_options).values())


def test_initialize_ucx_nvlink():
    pytest.importorskip("distributed_ucxx")

    p = mp.Process(target=_test_initialize_ucx_nvlink)
    p.start()
    p.join()
    assert not p.exitcode


def _test_initialize_ucx_infiniband():
    ucxx = pytest.importorskip("ucxx")

    kwargs = {"enable_infiniband": True}
    initialize(**kwargs)
    with LocalCluster(
        protocol="ucx",
        dashboard_address=None,
        n_workers=1,
        threads_per_worker=1,
        processes=True,
        worker_class=IncreasedCloseTimeoutNanny,
        config={"distributed-ucxx": get_ucx_config(**kwargs)},
    ) as cluster:
        with Client(cluster) as client:
            res = da.from_array(numpy.arange(10000), chunks=(1000,))
            res = res.sum().compute()
            assert res == 49995000

            def check_ucx_options():
                conf = ucxx.get_config()
                assert "TLS" in conf
                assert "rc" in conf["TLS"]
                assert "tcp" in conf["TLS"]
                assert "cuda_copy" in conf["TLS"]
                assert "tcp" in conf["SOCKADDR_TLS_PRIORITY"]
                return True

            assert client.run_on_scheduler(check_ucx_options) is True
            assert all(client.run(check_ucx_options).values())


@pytest.mark.skipif(
    "ib0" not in psutil.net_if_addrs(), reason="Infiniband interface ib0 not found"
)
def test_initialize_ucx_infiniband():
    pytest.importorskip("distributed_ucxx")

    p = mp.Process(target=_test_initialize_ucx_infiniband)
    p.start()
    p.join()
    assert not p.exitcode


def _test_initialize_ucx_all():
    ucxx = pytest.importorskip("ucxx")

    initialize()
    with LocalCluster(
        protocol="ucx",
        dashboard_address=None,
        n_workers=1,
        threads_per_worker=1,
        processes=True,
        worker_class=IncreasedCloseTimeoutNanny,
        config={"distributed-ucxx": get_ucx_config()},
    ) as cluster:
        with Client(cluster) as client:
            res = da.from_array(numpy.arange(10000), chunks=(1000,))
            res = res.sum().compute()
            assert res == 49995000

            def check_ucx_options():
                conf = ucxx.get_config()
                assert "TLS" in conf
                assert conf["TLS"] == "all"
                assert all(
                    [
                        p in conf["SOCKADDR_TLS_PRIORITY"]
                        for p in ["rdmacm", "tcp", "sockcm"]
                    ]
                )
                return True

            assert client.run_on_scheduler(check_ucx_options) is True
            assert all(client.run(check_ucx_options).values())


def test_initialize_ucx_all():
    pytest.importorskip("distributed_ucxx")

    p = mp.Process(target=_test_initialize_ucx_all)
    p.start()
    p.join()
    assert not p.exitcode


def _test_dask_cuda_import():
    # Check that importing `dask_cuda` does NOT
    # require `dask.dataframe` or `dask.array`.

    # Patch sys.modules so that `dask.dataframe`
    # and `dask.array` cannot be found.
    with pytest.MonkeyPatch.context() as monkeypatch:
        for k in list(sys.modules):
            if k.startswith("dask.dataframe") or k.startswith("dask.array"):
                monkeypatch.setitem(sys.modules, k, None)
        monkeypatch.delitem(sys.modules, "dask_cuda")

        # Check that top-level imports still succeed.
        import dask_cuda  # noqa: F401
        from dask_cuda import CUDAWorker  # noqa: F401
        from dask_cuda import LocalCUDACluster

        with LocalCUDACluster(
            dashboard_address=None,
            n_workers=1,
            threads_per_worker=1,
            processes=True,
            worker_class=IncreasedCloseTimeoutNanny,
        ) as cluster:
            with Client(cluster) as client:
                client.run(lambda *args: None)


def test_dask_cuda_import():
    p = mp.Process(target=_test_dask_cuda_import)
    p.start()
    p.join()
    assert not p.exitcode


def _test_cuda_context_warning_with_subprocess_warnings(protocol):
    """Test CUDA context warnings from both parent and worker subprocesses.

    This test creates a standalone script that imports a problematic library
    and creates LocalCUDACluster with processes=True. This should generate
    warnings from both the parent process and each worker subprocess, since
    they all inherit the CUDA context created at import time.
    """
    # Create temporary directory for our test files
    temp_dir = tempfile.mkdtemp()

    # Create the problematic library that creates CUDA context at import
    problematic_library_code = textwrap.dedent(
        """
        # Problematic library that creates CUDA context at import time
        import os

        import cuda.core.experimental

        try:
            # Create CUDA context at import time, this will be inherited by subprocesses
            cuda.core.experimental.Device().set_current()
            print("Problematic library: Created CUDA context at import time")
            os.environ['SUBPROCESS_CUDA_CONTEXT_CREATED'] = '1'
        except Exception as e:
            raise RuntimeError(
                f"Problematic library: Failed to create CUDA context({e})"
            )
            os.environ['SUBPROCESS_CUDA_CONTEXT_CREATED'] = '0'
        """
    )

    problematic_lib_path = os.path.join(temp_dir, "problematic_cuda_library.py")
    with open(problematic_lib_path, "w") as f:
        f.write(problematic_library_code)

    # Create the main test script that imports the problematic library
    # and creates LocalCUDACluster - this will run in a subprocess
    main_script_code = textwrap.dedent(
        f"""
        # Main script that demonstrates the real-world problematic scenario
        import os
        import sys
        import logging

        # Add the temp directory to path so we can import our problematic library
        sys.path.insert(0, '{temp_dir}')

        print("=== Starting subprocess warnings test ===")

        # This is the key part: import the problematic library BEFORE creating
        # LocalCUDACluster. This creates a CUDA context that will be inherited
        # by all worker subprocesses
        print("Importing problematic library...")
        import problematic_cuda_library

        context_mode = os.environ.get('SUBPROCESS_CUDA_CONTEXT_CREATED', None)
        if context_mode == "1":
            print(f"Context creation successful")
        else:
            raise RuntimeError("Context creation failed")

        if __name__ == "__main__":
            try:
                from dask_cuda import LocalCUDACluster
                from dask_cuda.utils_test import IncreasedCloseTimeoutNanny

                cluster = LocalCUDACluster(
                    dashboard_address=None,
                    worker_class=IncreasedCloseTimeoutNanny,
                    protocol=f"{protocol}",
                )
                print("LocalCUDACluster created successfully!")

                cluster.close()
                print("Cluster closed successfully")

            except Exception as e:
                raise RuntimeError(f"Cluster setup error: {{e}}")

        print("=== Subprocess warnings test completed ===")
    """
    )

    main_script_path = os.path.join(temp_dir, "test_subprocess_warnings.py")
    with open(main_script_path, "w") as f:
        f.write(main_script_code)

    try:
        # Run the main script in a subprocess
        result = subprocess.run(
            [sys.executable, main_script_path],
            capture_output=True,
            text=True,
            timeout=30,  # Reduced timeout for simpler test
            cwd=os.getcwd(),
        )

        # Check for successful test execution regardless of warnings
        assert "Context creation successful" in result.stdout, (
            "Test did not create a CUDA context"
        )
        assert (
            "Creating LocalCUDACluster" in result.stdout
            or "LocalCUDACluster created successfully" in result.stdout
        ), "LocalCUDACluster was not created"

        # Check the log file for warnings from multiple processes
        warnings_found = []
        warnings_assigned_device_found = []

        # Look for CUDA context warnings from different processes
        lines = result.stderr.split("\n")
        for line in lines:
            if "A CUDA context for device" in line and "already exists" in line:
                warnings_found.append(line)
            if (
                "should have a CUDA context assigned to device" in line
                and "but instead the CUDA context is on device" in line
            ):
                warnings_assigned_device_found.append(line)

        num_devices = cuda.core.experimental.system.num_devices

        # Every worker raises the warning once. With protocol="ucx" the warning is
        # raised once more by the parent process.
        expected_warnings = num_devices if protocol == "tcp" else num_devices + 1
        assert len(warnings_found) == expected_warnings, (
            f"Expected {expected_warnings} CUDA context warnings, "
            f"but found {len(warnings_assigned_device_found)}"
        )

        # Can only be tested in multi-GPU test environment, device 0 can never raise
        # this warning (because it's where all CUDA contexts are created), thus one
        # warning is raised by every device except 0.
        expected_assigned_device_warnings = num_devices - 1
        assert (
            len(warnings_assigned_device_found) == expected_assigned_device_warnings
        ), (
            f"Expected {expected_assigned_device_warnings} warnings assigned to "
            f"device, but found {len(warnings_assigned_device_found)}"
        )

        # Verify warnings contents
        for warning in warnings_found:
            assert (
                "This is often the result of a CUDA-enabled library calling a "
                "CUDA runtime function before Dask-CUDA" in warning
            ), f"Warning missing explanatory text: {warning}"
        for warning in warnings_assigned_device_found:
            assert (
                "This is often the result of a CUDA-enabled library calling a "
                "CUDA runtime function before Dask-CUDA" in warning
            ), f"Warning missing explanatory text: {warning}"
    finally:
        # Clean up temporary files
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Cleanup error: {e}")


@pytest.mark.parametrize("protocol", ["tcp", "ucx"])
def test_cuda_context_warning_with_subprocess_warnings(protocol):
    """Test CUDA context warnings from parent and worker subprocesses.

    This test creates a standalone script that imports a problematic library at the top
    level and then creates LocalCUDACluster with processes=True. This replicates the
    exact real-world scenario where:

    1. User imports a problematic library that creates CUDA context at import time
    2. User creates LocalCUDACluster with multiple workers
    3. Each worker subprocess inherits the CUDA context and emits warnings
    4. Multiple warnings are generated (parent process + each worker subprocess)

    This is the ultimate test as it demonstrates the distributed warning scenario
    that users actually encounter in production.
    """
    p = mp.Process(
        target=_test_cuda_context_warning_with_subprocess_warnings, args=(protocol,)
    )
    p.start()
    p.join()
    assert not p.exitcode
