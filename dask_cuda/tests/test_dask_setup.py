# SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import tempfile
import time
from contextlib import contextmanager
from unittest.mock import Mock, patch

import pytest

from distributed import Client
from distributed.utils import open_port
from distributed.utils_test import popen

from dask_cuda.initialize import dask_setup
from dask_cuda.utils import wait_workers


@pytest.mark.parametrize("protocol", ["tcp", "ucx", "ucxx"])
def test_dask_setup_function_with_mock_worker(protocol):
    """Test the dask_setup function directly with mock worker."""
    # Create a mock worker object
    mock_worker = Mock()
    mock_worker._protocol = protocol

    with patch("dask_cuda.initialize._create_cuda_context") as mock_create_context:
        # Test with create_cuda_context=True
        # Call the underlying function directly (the Click decorator wraps the real
        # function)
        dask_setup.callback(
            worker=mock_worker,
            create_cuda_context=True,
        )

        mock_create_context.assert_called_once_with(protocol=protocol)

        mock_create_context.reset_mock()

        # Test with create_cuda_context=False
        dask_setup.callback(
            worker=mock_worker,
            create_cuda_context=False,
        )

        mock_create_context.assert_not_called()


@contextmanager
def start_dask_scheduler(protocol: str, max_attempts: int = 5, timeout: int = 10):
    """Start Dask scheduler in subprocess.

    Attempts to start a Dask scheduler in subprocess, if the port is not available
    retry on a different port up to a maximum of `max_attempts` attempts. The stdout
    and stderr of the process is read to determine whether the scheduler failed to
    bind to port or succeeded, and ensures no more than `timeout` seconds are awaited
    for between reads.

    This is primarily useful because UCX does not release TCP ports immediately. A
    workaround without the need for this function is setting `UCX_TCP_CM_REUSEADDR=y`,
    but that requires to be explicitly set when running tests, and that is not very
    friendly.

    Parameters
    ----------
    protocol: str
        Communication protocol to use.
    max_attempts: int
        Maximum attempts to try to open scheduler.
    timeout: int
        Time to wait while reading stdout/stderr of subprocess.
    """
    port = open_port()
    for _ in range(max_attempts):
        with popen(
            [
                "dask",
                "scheduler",
                "--no-dashboard",
                "--protocol",
                protocol,
                "--port",
                str(port),
            ],
            capture_output=True,  # Capture stdout and stderr
        ) as scheduler_process:
            # Check if the scheduler process started successfully by streaming output
            try:
                start_time = time.monotonic()
                while True:
                    if time.monotonic() - start_time > timeout:
                        raise TimeoutError("Timeout while waiting for scheduler output")

                    # Use select to wait for data with a timeout
                    line = scheduler_process.stdout.readline()
                    if not line:
                        break  # End of output
                    print(
                        line.decode(), end=""
                    )  # Since capture_output=True, print the line here
                    if b"Scheduler at:" in line:
                        # Scheduler is now listening
                        break
                    elif b"UCXXBusyError" in line:
                        raise Exception("UCXXBusyError detected in scheduler output")
            except Exception:
                port += 1
            else:
                yield scheduler_process, port
                return
    else:
        pytest.fail(f"Failed to start dask scheduler after {max_attempts} attempts.")


@pytest.mark.timeout(20)
@patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"})
@pytest.mark.parametrize("protocol", ["tcp", "ucx", "ucxx"])
def test_dask_cuda_worker_cli_integration(protocol):
    """Test that dask cuda worker CLI correctly passes arguments to dask_setup.

    Verifies the end-to-end integration where the CLI tool actually launches and calls
    dask_setup with correct args.
    """

    # Use a global file path to ensure cleanup
    capture_file_path = tempfile.NamedTemporaryFile(
        delete=False, suffix="_dask_setup_integration_test.json"
    ).name

    # Create a simple capture script that works reliably
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".py") as f:
        f.write(
            f'''
import json
import os

def capture_dask_setup_call(worker, create_cuda_context):
    """Capture dask_setup arguments and write to file."""
    result = {{
        'worker_protocol': getattr(worker, '_protocol', 'unknown'),
        'create_cuda_context': create_cuda_context,
        'test_success': True
    }}

    # Write immediately to ensure it gets captured
    with open('{capture_file_path}', 'w') as f:
        json.dump(result, f)

# Patch dask_setup callback
from dask_cuda.initialize import dask_setup
dask_setup.callback = capture_dask_setup_call
'''
        )
        preload_file = f.name

    try:
        # Clean up any existing capture file
        if os.path.exists(capture_file_path):
            os.unlink(capture_file_path)

        with start_dask_scheduler(protocol=protocol) as scheduler_process_port:
            scheduler_process, scheduler_port = scheduler_process_port
            sched_addr = f"{protocol}://127.0.0.1:{scheduler_port}"
            print(f"{sched_addr=}", flush=True)

            # Build dask cuda worker args
            dask_cuda_worker_args = [
                "dask",
                "cuda",
                "worker",
                sched_addr,
                "--host",
                "127.0.0.1",
                "--no-dashboard",
                "--preload",
                preload_file,
                "--death-timeout",
                "10",
            ]

            with popen(dask_cuda_worker_args):
                # Wait and check for worker connection
                with Client(sched_addr) as client:
                    assert wait_workers(client, n_gpus=1)

                    # Give extra time for preload execution
                    time.sleep(3)

                    # Check if dask_setup was called and captured correctly
                    if os.path.exists(capture_file_path):
                        with open(capture_file_path, "r") as cf:
                            captured_args = json.load(cf)

                        # Verify the critical arguments were passed correctly
                        assert (
                            captured_args["create_cuda_context"] is True
                        ), "create_cuda_context should be True"

                        # Verify worker has a protocol set
                        assert (
                            captured_args["worker_protocol"] == protocol
                        ), "Worker should have a protocol"
                    else:
                        pytest.fail(
                            "capture file not found: dask_setup was not called or "
                            "failed to write to file"
                        )

    finally:
        # Cleanup
        try:
            os.unlink(preload_file)
            os.unlink(capture_file_path)
        except FileNotFoundError:
            pass
