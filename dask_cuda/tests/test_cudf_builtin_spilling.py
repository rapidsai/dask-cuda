import pytest

from distributed.sizeof import safe_sizeof

from dask_cuda.device_host_file import DeviceHostFile
from dask_cuda.is_spillable_object import is_spillable_object
from dask_cuda.proxify_host_file import ProxifyHostFile

cupy = pytest.importorskip("cupy")
pandas = pytest.importorskip("pandas")

pytest.importorskip(
    "cudf.core.buffer.spill_manager",
    reason="Current version of cudf doesn't support built-in spilling",
)

import cudf  # noqa: E402
from cudf.core.buffer.spill_manager import (  # noqa: E402
    SpillManager,
    get_global_manager,
    set_global_manager,
)
from cudf.testing._utils import assert_eq  # noqa: E402

if get_global_manager() is not None:
    pytest.skip(
        reason=(
            "cannot test cudf built-in spilling, if already enabled globally. "
            "Please set the `CUDF_SPILL=off` environment variable."
        ),
        allow_module_level=True,
    )


@pytest.fixture
def manager(request):
    """Fixture to enable and make a spilling manager available"""
    kwargs = dict(getattr(request, "param", {}))
    set_global_manager(manager=SpillManager(**kwargs))
    yield get_global_manager()
    set_global_manager(manager=None)


def test_is_spillable_object_when_cudf_spilling_disabled():
    pdf = pandas.DataFrame({"a": [1, 2, 3]})
    cdf = cudf.DataFrame({"a": [1, 2, 3]})
    assert not is_spillable_object(pdf)
    assert not is_spillable_object(cdf)


def test_is_spillable_object_when_cudf_spilling_enabled(manager):
    pdf = pandas.DataFrame({"a": [1, 2, 3]})
    cdf = cudf.DataFrame({"a": [1, 2, 3]})
    assert not is_spillable_object(pdf)
    assert is_spillable_object(cdf)


def test_device_host_file_when_cudf_spilling_is_disabled(tmp_path):
    tmpdir = tmp_path / "storage"
    tmpdir.mkdir()
    dhf = DeviceHostFile(
        device_memory_limit=1024 * 16,
        memory_limit=1024 * 16,
        worker_local_directory=tmpdir,
    )
    dhf["pandas"] = pandas.DataFrame({"a": [1, 2, 3]})
    dhf["cudf"] = cudf.DataFrame({"a": [1, 2, 3]})

    assert set(dhf.others.keys()) == set()
    assert set(dhf.device.keys()) == set(["cudf"])
    assert set(dhf.host.keys()) == set(["pandas"])
    assert set(dhf.disk.keys()) == set()


def test_device_host_file_step_by_step(tmp_path, manager: SpillManager):
    tmpdir = tmp_path / "storage"
    tmpdir.mkdir()
    pdf = pandas.DataFrame({"a": [1, 2, 3]})
    cdf = cudf.DataFrame({"a": [1, 2, 3]})

    # Pandas will cache the result of probing this attribute.
    # We trigger it here, to get consistent results from `safe_sizeof()`
    hasattr(pdf, "__cuda_array_interface__")

    dhf = DeviceHostFile(
        device_memory_limit=safe_sizeof(pdf),
        memory_limit=safe_sizeof(pdf),
        worker_local_directory=tmpdir,
    )
    dhf["pa1"] = pdf
    dhf["cu1"] = cdf

    assert set(dhf.others.keys()) == set(["cu1"])
    assert set(dhf.device.keys()) == set()
    assert set(dhf.host.keys()) == set(["pa1"])
    assert set(dhf.disk.keys()) == set()
    assert_eq(dhf["pa1"], dhf["cu1"])

    dhf["pa2"] = pdf
    assert set(dhf.others.keys()) == set(["cu1"])
    assert set(dhf.device.keys()) == set()
    assert set(dhf.host.keys()) == set(["pa2"])
    assert set(dhf.disk.keys()) == set(["pa1"])

    dhf["cu2"] = cdf
    assert set(dhf.others.keys()) == set(["cu1", "cu2"])
    assert set(dhf.device.keys()) == set()
    assert set(dhf.host.keys()) == set(["pa2"])
    assert set(dhf.disk.keys()) == set(["pa1"])

    del dhf["cu1"]
    assert set(dhf.others.keys()) == set(["cu2"])
    assert set(dhf.device.keys()) == set()
    assert set(dhf.host.keys()) == set(["pa2"])
    assert set(dhf.disk.keys()) == set(["pa1"])

    del dhf["pa2"]
    assert set(dhf.others.keys()) == set(["cu2"])
    assert set(dhf.device.keys()) == set()
    assert set(dhf.host.keys()) == set()
    assert set(dhf.disk.keys()) == set(["pa1"])

    del dhf["pa1"]
    assert set(dhf.others.keys()) == set(["cu2"])
    assert set(dhf.device.keys()) == set()
    assert set(dhf.host.keys()) == set()
    assert set(dhf.disk.keys()) == set()

    del dhf["cu2"]
    assert set(dhf.others.keys()) == set()
    assert set(dhf.device.keys()) == set()
    assert set(dhf.host.keys()) == set()
    assert set(dhf.disk.keys()) == set()


def test_proxify_host_file(tmp_path_factory, manager: SpillManager):
    # Reuse the spill-to-disk dir, if it exist
    if ProxifyHostFile._spill_to_disk is None:
        tmpdir = tmp_path_factory.mktemp("jit-unspill")
    else:
        tmpdir = ProxifyHostFile._spill_to_disk.root_dir / ".."

    with pytest.warns(
        UserWarning,
        match="JIT-Unspill and cuDF's built-in spilling don't work together",
    ):
        dhf = ProxifyHostFile(
            device_memory_limit=1000,
            memory_limit=1000,
            worker_local_directory=str(tmpdir),
        )
    dhf["cu1"] = cudf.DataFrame({"a": [1, 2, 3]})
    del dhf["cu1"]
