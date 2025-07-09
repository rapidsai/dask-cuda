# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0


import importlib
import io
import multiprocessing as mp
import sys

import pytest

from dask_cuda import LocalCUDACluster

mp = mp.get_context("spawn")  # type: ignore


def _has_distributed_ucxx() -> bool:
    return bool(importlib.util.find_spec("distributed_ucxx"))


def _test_protocol_ucx():
    with LocalCUDACluster(protocol="ucx") as cluster:
        assert cluster.scheduler_comm.address.startswith("ucx://")

        if _has_distributed_ucxx():
            import distributed_ucxx

            assert all(
                isinstance(batched_send.comm, distributed_ucxx.ucxx.UCXX)
                for batched_send in cluster.scheduler.stream_comms.values()
            )
        else:
            import rapids_dask_dependency

            assert all(
                isinstance(
                    batched_send.comm,
                    rapids_dask_dependency.patches.distributed.comm.__rdd_patch_ucx.UCX,
                )
                for batched_send in cluster.scheduler.stream_comms.values()
            )


def _test_protocol_ucxx():
    if _has_distributed_ucxx():
        with LocalCUDACluster(protocol="ucxx") as cluster:
            assert cluster.scheduler_comm.address.startswith("ucxx://")
            import distributed_ucxx

            assert all(
                isinstance(batched_send.comm, distributed_ucxx.ucxx.UCXX)
                for batched_send in cluster.scheduler.stream_comms.values()
            )
    else:
        with pytest.raises(RuntimeError, match="Cluster failed to start"):
            LocalCUDACluster(protocol="ucxx")


def _test_protocol_ucx_old():
    with LocalCUDACluster(protocol="ucx-old") as cluster:
        assert cluster.scheduler_comm.address.startswith("ucx-old://")

        import rapids_dask_dependency

        assert all(
            isinstance(
                batched_send.comm,
                rapids_dask_dependency.patches.distributed.comm.__rdd_patch_ucx.UCX,
            )
            for batched_send in cluster.scheduler.stream_comms.values()
        )


def _run_test_with_output_capture(test_func_name, conn):
    """Run a test function in a subprocess and capture stdout/stderr."""
    # Redirect stdout and stderr to capture output
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    captured_output = io.StringIO()
    sys.stdout = sys.stderr = captured_output

    try:
        # Import and run the test function
        if test_func_name == "_test_protocol_ucx":
            _test_protocol_ucx()
        elif test_func_name == "_test_protocol_ucxx":
            _test_protocol_ucxx()
        elif test_func_name == "_test_protocol_ucx_old":
            _test_protocol_ucx_old()
        else:
            raise ValueError(f"Unknown test function: {test_func_name}")

        output = captured_output.getvalue()
        conn.send((True, output))  # True = success
    except Exception as e:
        output = captured_output.getvalue()
        output += f"\nException: {e}"
        import traceback

        output += f"\nTraceback:\n{traceback.format_exc()}"
        conn.send((False, output))  # False = failure
    finally:
        # Restore original stdout/stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        conn.close()


@pytest.mark.parametrize("protocol", ["ucx", "ucxx", "ucx-old"])
def test_rdd_protocol(protocol):
    """Test rapids-dask-dependency protocol selection"""
    if protocol == "ucx":
        test_func_name = "_test_protocol_ucx"
    elif protocol == "ucxx":
        test_func_name = "_test_protocol_ucxx"
    else:
        test_func_name = "_test_protocol_ucx_old"

    # Create a pipe for communication between parent and child processes
    parent_conn, child_conn = mp.Pipe()
    p = mp.Process(
        target=_run_test_with_output_capture, args=(test_func_name, child_conn)
    )

    p.start()
    p.join(timeout=60)

    if p.is_alive():
        p.kill()
        p.close()
        raise TimeoutError("Test process timed out")

    # Get the result from the child process
    success, output = parent_conn.recv()

    # Check that the test passed
    assert success, f"Test failed in subprocess. Output:\n{output}"

    # For the ucx protocol, check if warnings are printed when distributed_ucxx is not
    # available
    if protocol == "ucx" and not _has_distributed_ucxx():
        # Check if the warning about protocol='ucx' is printed
        print(f"Output for {protocol} protocol:\n{output}")
        assert (
            "you have requested protocol='ucx'" in output
        ), f"Expected warning not found in output: {output}"
        assert (
            "'distributed-ucxx' is not installed" in output
        ), f"Expected warning about distributed-ucxx not found in output: {output}"
    elif protocol == "ucx" and _has_distributed_ucxx():
        # When distributed_ucxx is available, the warning should NOT be printed
        assert "you have requested protocol='ucx'" not in output, (
            "Warning should not be printed when distributed_ucxx is available: "
            f"{output}"
        )
    elif protocol == "ucx-old":
        # The ucx-old protocol should not generate warnings
        assert (
            "you have requested protocol='ucx'" not in output
        ), f"Warning should not be printed for ucx-old protocol: {output}"
