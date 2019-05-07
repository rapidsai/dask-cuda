import os
from zict.file import _safe_key as safe_key
import string
from random import choice


def assert_device_host_file_size(dhf, total_bytes, chunk_overhead=1024):
    byte_sum = dhf.device.total_weight + dhf.host.total_weight
    file_path = [
        os.path.join(dhf.disk.d.directory, safe_key(k))
        for k in dhf.disk.keys()
    ]
    file_size = [os.path.getsize(f) for f in file_path]
    byte_sum += sum(file_size)

    # Allow up to chunk_overhead bytes overhead per chunk on disk
    host_overhead = len(dhf.host) * chunk_overhead
    disk_overhead = len(dhf.disk) * chunk_overhead
    assert (
        byte_sum >= total_bytes
        and byte_sum <= total_bytes + host_overhead + disk_overhead
    )


def gen_random_key(key_length):
    chars = string.ascii_letters + string.digits + "%_-"
    return "".join(choice(chars) for _ in range(key_length))
