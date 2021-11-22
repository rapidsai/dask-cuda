from typing import Iterable, Mapping

from distributed.utils import nbytes


def disk_write(path: str, frames: Iterable, shared_filesystem: bool) -> dict:
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

    Returns
    -------
    header: dict
        A dict of metadata
    """

    with open(path, "wb") as f:
        for frame in frames:
            f.write(frame)
    return {
        "method": "stdio",
        "path": path,
        "frame-lengths": tuple(map(nbytes, frames)),
        "shared-filesystem": shared_filesystem,
    }


def disk_read(header: Mapping) -> list:
    """Read frames from disk

    Parameters
    ----------
    header: Mapping
        The metadata of the frames to read

    Returns
    -------
    frames: list
        List of read frames
    """

    ret = []
    with open(header["path"], "rb") as f:
        for length in header["frame-lengths"]:
            ret.append(f.read(length))
    return ret
