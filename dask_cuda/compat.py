import pickle

import msgpack
from packaging.version import Version

import dask
import distributed
import distributed.comm.utils
import distributed.protocol
from distributed.comm.utils import OFFLOAD_THRESHOLD, nbytes, offload
from distributed.protocol.core import (
    Serialized,
    decompress,
    logger,
    merge_and_deserialize,
    msgpack_decode_default,
    msgpack_opts,
)

if Version(distributed.__version__) >= Version("2023.8.1"):
    # Monkey-patch protocol.core.loads (and its users)
    async def from_frames(
        frames, deserialize=True, deserializers=None, allow_offload=True
    ):
        """
        Unserialize a list of Distributed protocol frames.
        """
        size = False

        def _from_frames():
            try:
                # Patched code
                return loads(
                    frames, deserialize=deserialize, deserializers=deserializers
                )
                # end patched code
            except EOFError:
                if size > 1000:
                    datastr = "[too large to display]"
                else:
                    datastr = frames
                # Aid diagnosing
                logger.error("truncated data stream (%d bytes): %s", size, datastr)
                raise

        if allow_offload and deserialize and OFFLOAD_THRESHOLD:
            size = sum(map(nbytes, frames))
        if (
            allow_offload
            and deserialize
            and OFFLOAD_THRESHOLD
            and size > OFFLOAD_THRESHOLD
        ):
            res = await offload(_from_frames)
        else:
            res = _from_frames()

        return res

    def loads(frames, deserialize=True, deserializers=None):
        """Transform bytestream back into Python value"""

        allow_pickle = dask.config.get("distributed.scheduler.pickle")

        try:

            def _decode_default(obj):
                offset = obj.get("__Serialized__", 0)
                if offset > 0:
                    sub_header = msgpack.loads(
                        frames[offset],
                        object_hook=msgpack_decode_default,
                        use_list=False,
                        **msgpack_opts,
                    )
                    offset += 1
                    sub_frames = frames[offset : offset + sub_header["num-sub-frames"]]
                    if deserialize:
                        if "compression" in sub_header:
                            sub_frames = decompress(sub_header, sub_frames)
                        return merge_and_deserialize(
                            sub_header, sub_frames, deserializers=deserializers
                        )
                    else:
                        return Serialized(sub_header, sub_frames)

                offset = obj.get("__Pickled__", 0)
                if offset > 0:
                    sub_header = msgpack.loads(frames[offset])
                    offset += 1
                    sub_frames = frames[offset : offset + sub_header["num-sub-frames"]]
                    # Patched code
                    if "compression" in sub_header:
                        sub_frames = decompress(sub_header, sub_frames)
                    # end patched code
                    if allow_pickle:
                        return pickle.loads(
                            sub_header["pickled-obj"], buffers=sub_frames
                        )
                    else:
                        raise ValueError(
                            "Unpickle on the Scheduler isn't allowed, "
                            "set `distributed.scheduler.pickle=true`"
                        )

                return msgpack_decode_default(obj)

            return msgpack.loads(
                frames[0], object_hook=_decode_default, use_list=False, **msgpack_opts
            )

        except Exception:
            logger.critical("Failed to deserialize", exc_info=True)
            raise

    distributed.protocol.loads = loads
    distributed.protocol.core.loads = loads
    distributed.comm.utils.from_frames = from_frames
