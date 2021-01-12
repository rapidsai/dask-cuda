from collections import defaultdict

from toolz import first

from distributed import get_client, wait


def extract_ddf_partitions(ddf):
    """ Returns the mapping: worker -> [list of futures]"""
    client = get_client()
    delayed_ddf = ddf.to_delayed()
    parts = client.compute(delayed_ddf)
    wait(parts)

    key_to_part = dict([(str(part.key), part) for part in parts])
    ret = defaultdict(list)  # Map worker -> [list of futures]
    for key, workers in client.who_has(parts).items():
        worker = first(
            workers
        )  # If multiple workers have the part, we pick the first worker
        ret[worker].append(key_to_part[key])
    return ret
