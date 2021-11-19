from typing import Iterable, Mapping

import numpy as np

from distributed.utils import nbytes


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
        import rmm  # isort:skip

        with cucim_fs.open(header["path"], "rb") as f:
            file_offset = 0
            for length, is_cuda in zip(header["frame-lengths"], header["cuda-frames"]):
                if is_cuda:
                    buf = rmm.DeviceBuffer(size=length)
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
