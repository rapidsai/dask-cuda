# SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import tempfile
import time
from unittest.mock import Mock, patch

import pytest

from distributed import Client
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

        # Build CLI args
        cli_args = [
            "dask",
            "cuda",
            "worker",
            f"{protocol}://127.0.0.1:9358",
            "--host",
            "127.0.0.1",
            "--no-dashboard",
            "--preload",
            preload_file,
            "--death-timeout",
            "10",
        ]

        # Start scheduler and worker
        with popen(
            [
                "dask",
                "scheduler",
                "--protocol",
                protocol,
                "--port",
                "9358",
                "--no-dashboard",
            ]
        ):
            time.sleep(2)  # Give scheduler time to start

            with popen(cli_args):
                # Wait and check for worker connection
                with Client(f"{protocol}://127.0.0.1:9358") as client:
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
