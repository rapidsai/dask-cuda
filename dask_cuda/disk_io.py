import weakref
from typing import Callable, Iterable, Mapping, Optional

import numpy as np

from distributed.utils import Any, nbytes

_new_cuda_buffer: Optional[Callable[[int], Any]] = None


def get_new_cuda_buffer() -> Callable[[int], Any]:
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
    frame_lengths = tuple(map(nbytes, frames))
    if gds and any(cuda_frames):
        import cucim.clara.filesystem as cucim_fs

        with cucim_fs.open(path, "w") as f:
            for frame, length in zip(frames, frame_lengths):
                f.pwrite(buf=frame, count=length, file_offset=0, buf_offset=0)

    else:
        with open(path, "wb") as f:
            for frame in frames:
                f.write(frame)
    return {
        "method": "stdio",
        "path": path,
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
    ret = []
    if gds:
        import cucim.clara.filesystem as cucim_fs  # isort:skip

        with cucim_fs.open(header["path"], "rb") as f:
            file_offset = 0
            for length, is_cuda in zip(header["frame-lengths"], header["cuda-frames"]):
                if is_cuda:
                    buf = get_new_cuda_buffer()(length)
                else:
                    buf = np.empty((length,), dtype="u1")
                f.pread(buf=buf, count=length, file_offset=file_offset, buf_offset=0)
                file_offset += length
                ret.append(buf)
    else:
        with open(header["path"], "rb") as f:
            for length in header["frame-lengths"]:
                ret.append(f.read(length))
    return ret
