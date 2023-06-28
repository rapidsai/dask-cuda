import itertools
import os
import os.path
import pathlib
import tempfile
import threading
import weakref
from typing import Callable, Iterable, Mapping, Optional, Union

import numpy as np

import dask
from distributed.utils import nbytes

_new_cuda_buffer: Optional[Callable[[int], object]] = None


def get_new_cuda_buffer() -> Callable[[int], object]:
    """Return a function to create an empty CUDA buffer"""
    global _new_cuda_buffer
    if _new_cuda_buffer is not None:
        return _new_cuda_buffer
    try:
        import rmm

        _new_cuda_buffer = lambda n: rmm.DeviceBuffer(size=n)
        return _new_cuda_buffer
    except ImportError:
        pass

    try:
        import cupy

        _new_cuda_buffer = lambda n: cupy.empty((n,), dtype="u1")
        return _new_cuda_buffer
    except ImportError:
        pass

    try:
        import numba.cuda

        def numba_device_array(n):
            a = numba.cuda.device_array((n,), dtype="u1")
            weakref.finalize(a, numba.cuda.current_context)
            return a

        _new_cuda_buffer = numba_device_array
        return _new_cuda_buffer
    except ImportError:
        pass

    raise RuntimeError("GPUDirect Storage requires RMM, CuPy, or Numba")


class SpillToDiskFile:
    """File path the gets removed on destruction

    When spilling to disk, we have to delay the removal of the file
    until no more proxies are pointing to the file.
    """

    path: str

    def __init__(self, path: str) -> None:
        self.path = path

    def __del__(self):
        os.remove(self.path)

    def __str__(self) -> str:
        return self.path

    def exists(self):
        return os.path.exists(self.path)

    def __deepcopy__(self, memo) -> str:
        """A deep copy is simply the path as a string.

        In order to avoid multiple instance of SpillToDiskFile pointing
        to the same file, we do not allow a direct copy.
        """
        return self.path

    def __copy__(self):
        raise RuntimeError("Cannot copy or pickle a SpillToDiskFile")

    def __reduce__(self):
        self.__copy__()


class SpillToDiskProperties:
    gds_enabled: bool
    shared_filesystem: bool
    root_dir: pathlib.Path
    tmpdir: tempfile.TemporaryDirectory

    def __init__(
        self,
        root_dir: Union[str, os.PathLike],
        shared_filesystem: Optional[bool] = None,
        gds: Optional[bool] = None,
    ):
        """
        Parameters
        ----------
        root_dir : os.PathLike
            Path to the root directory to write serialized data.
        shared_filesystem: bool or None, default None
            Whether the `root_dir` above is shared between all workers or not.
            If ``None``, the "jit-unspill-shared-fs" config value are used, which
            defaults to False.
        gds: bool
            Enable the use of GPUDirect Storage. If ``None``, the "gds-spilling"
            config value are used, which defaults to ``False``.
        """
        self.lock = threading.Lock()
        self.counter = 0
        self.root_dir = pathlib.Path(root_dir)
        os.makedirs(self.root_dir, exist_ok=True)
        self.tmpdir = tempfile.TemporaryDirectory(dir=self.root_dir)

        self.shared_filesystem = shared_filesystem or dask.config.get(
            "jit-unspill-shared-fs", default=False
        )
        self.gds_enabled = gds or dask.config.get("gds-spilling", default=False)

        if self.gds_enabled:
            try:
                import kvikio  # noqa F401
            except ImportError:
                raise ImportError(
                    "GPUDirect Storage requires the kvikio Python package"
                )
            else:
                self.gds_enabled = kvikio.DriverProperties().is_gds_available

    def gen_file_path(self) -> str:
        """Generate an unique file path"""
        with self.lock:
            self.counter += 1
            return str(
                pathlib.Path(self.tmpdir.name) / pathlib.Path("%04d" % self.counter)
            )


def disk_write(path: str, frames: Iterable, shared_filesystem: bool, gds=False) -> dict:
    """Write frames to disk

    Parameters
    ----------
    path: str
        File path
    frames: Iterable
        The frames to write to disk
    shared_filesystem: bool
        Whether the target filesystem is shared between all workers or not.
        If True, the filesystem must support the `os.link()` operation.
    gds: bool
        Enable the use of GPUDirect Storage. Notice, the consecutive
        `disk_read()` must enable GDS as well.

    Returns
    -------
    header: dict
        A dict of metadata
    """
    cuda_frames = tuple(hasattr(f, "__cuda_array_interface__") for f in frames)

    if gds and any(cuda_frames):
        import kvikio

        # Write each frame consecutively into `path` in parallel
        with kvikio.CuFile(path, "w") as f:
            file_offsets = itertools.accumulate(map(nbytes, frames), initial=0)
            futures = [f.pwrite(b, file_offset=o) for b, o in zip(frames, file_offsets)]
            for each_fut in futures:
                each_fut.get()
    else:
        with open(path, "wb") as f:
            os.writev(f.fileno(), frames)  # type: ignore
    return {
        "method": "stdio",
        "path": SpillToDiskFile(path),
        "frame-lengths": tuple(map(nbytes, frames)),
        "shared-filesystem": shared_filesystem,
        "cuda-frames": cuda_frames,
    }


def disk_read(header: Mapping, gds=False) -> list:
    """Read frames from disk

    Parameters
    ----------
    header: Mapping
        The metadata of the frames to read
    gds: bool
        Enable the use of GPUDirect Storage. Notice, this must
        match the GDS option set by the prior `disk_write()` call.

    Returns
    -------
    frames: list
        List of read frames
    """
    ret: list = [
        get_new_cuda_buffer()(length)
        if gds and is_cuda
        else np.empty((length,), dtype="u1")
        for length, is_cuda in zip(header["frame-lengths"], header["cuda-frames"])
    ]
    if gds:
        import kvikio  # isort:skip

        with kvikio.CuFile(str(header["path"]), "r") as f:
            # Read each frame consecutively from `path` in parallel
            file_offsets = itertools.accumulate((b.nbytes for b in ret), initial=0)
            futures = [f.pread(b, file_offset=o) for b, o in zip(ret, file_offsets)]
            for each_fut in futures:
                each_fut.get()
    else:
        with open(str(header["path"]), "rb") as f:
            os.readv(f.fileno(), ret)  # type: ignore
    return ret
