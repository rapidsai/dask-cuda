from typing import Mapping

from distributed.utils import nbytes


def disk_write(path, frames, shared_filesystem: bool) -> dict:
    frame_lengths = tuple(map(nbytes, frames))
    with open(path, "wb") as f:
        for frame in frames:
            f.write(frame)
    return {
        "method": "stdio",
        "path": path,
        "frame-lengths": frame_lengths,
        "shared-filesystem": shared_filesystem,
    }


def disk_read(header: Mapping) -> list:
    ret = []
    with open(header["path"], "rb") as f:
        for length in header["frame-lengths"]:
            ret.append(f.read(length))
    return ret
