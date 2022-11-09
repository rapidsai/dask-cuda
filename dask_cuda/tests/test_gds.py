import tempfile

import pytest

from distributed.protocol.serialize import deserialize, serialize

from dask_cuda.proxify_host_file import ProxifyHostFile

# Make the "disk" serializer available and use a directory that is
# removed on exit.
if ProxifyHostFile._spill_to_disk is None:
    tmpdir = tempfile.TemporaryDirectory()
    ProxifyHostFile(
        worker_local_directory=tmpdir.name,
        device_memory_limit=1024,
        memory_limit=1024,
    )


@pytest.mark.parametrize("cuda_lib", ["cupy", "cudf", "numba.cuda"])
@pytest.mark.parametrize("gds_enabled", [True, False])
def test_gds(gds_enabled, cuda_lib):
    lib = pytest.importorskip(cuda_lib)
    if cuda_lib == "cupy":
        data_create = lambda: lib.arange(10)
        data_compare = lambda x, y: all(x == y)
    elif cuda_lib == "cudf":
        data_create = lambda: lib.Series(range(10))
        data_compare = lambda x, y: all((x == y).values_host)
    elif cuda_lib == "numba.cuda":
        data_create = lambda: lib.to_device(range(10))
        data_compare = lambda x, y: all(x.copy_to_host() == y.copy_to_host())

    try:
        if gds_enabled and not ProxifyHostFile._spill_to_disk.gds_enabled:
            pytest.skip("GDS not available")

        a = data_create()
        header, frames = serialize(a, serializers=("disk",))
        b = deserialize(header, frames)
        assert type(a) == type(b)
        assert data_compare(a, b)
    finally:
        ProxifyHostFile.register_disk_spilling()  # Reset disk spilling options
