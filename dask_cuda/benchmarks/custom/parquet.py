import math
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import dataset

import dask
import dask.dataframe as dd
from dask.base import apply, tokenize
from dask.distributed import get_worker
from dask.utils import parse_bytes

# NOTE: The pyarrow component of this code was mostly copied
# from dask-expr (dask_expr/io/parquet.py)


_CPU_COUNT_SET = False


def _maybe_adjust_cpu_count():
    global _CPU_COUNT_SET
    if not _CPU_COUNT_SET:
        # Set the number of threads to the number of cores
        # This is a default for pyarrow, but it's not set by default in
        # dask/distributed
        pa.set_cpu_count(os.cpu_count())
        _CPU_COUNT_SET = True


def fragment_to_table(fragment, filters=None, columns=None, schema=None):
    _maybe_adjust_cpu_count()

    if isinstance(filters, list):
        filters = pq.filters_to_expression(filters)

    return fragment.to_table(
        schema=schema,
        columns=columns,
        filter=filters,
        # Batch size determines how many rows are read at once and will
        # cause the underlying array to be split into chunks of this size
        # (max). We'd like to avoid fragmentation as much as possible and
        # and to set this to something like inf but we have to set a finite,
        # positive number.
        # In the presence of row groups, the underlying array will still be
        # chunked per rowgroup
        batch_size=10_000_000,
        fragment_scan_options=pa.dataset.ParquetFragmentScanOptions(
            pre_buffer=True,
            cache_options=pa.CacheOptions(
                hole_size_limit=parse_bytes("4 MiB"),
                range_size_limit=parse_bytes("32.00 MiB"),
            ),
        ),
        use_threads=True,
    )


def tables_to_frame(tables):
    import cudf

    return cudf.DataFrame.from_arrow(
        pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    )


def read_parquet_fragments(
    fragments,
    columns=None,
    filters=None,
    fragment_parallelism=None,
):

    kwargs = {"columns": columns, "filters": filters}
    if not isinstance(fragments, list):
        fragments = [fragments]

    if len(fragments) > 1:
        # Read multiple fragments
        token = tokenize(fragments, columns, filters)
        chunk_name = f"read-chunk-{token}"
        dsk = {
            (chunk_name, i): (apply, fragment_to_table, [fragment], kwargs)
            for i, fragment in enumerate(fragments)
        }
        dsk[chunk_name] = (tables_to_frame, list(dsk.keys()))

        try:
            worker = get_worker()
        except ValueError:
            return dask.threaded.get(dsk, chunk_name)

        if not hasattr(worker, "_rapids_executor"):
            fragment_parallelism = fragment_parallelism or 8
            num_threads = min(
                fragment_parallelism,
                len(os.sched_getaffinity(0)),
            )
            worker._rapids_executor = ThreadPoolExecutor(num_threads)
        with dask.config.set(pool=worker._rapids_executor):
            return dask.threaded.get(dsk, chunk_name)

    else:
        # Read single fragment
        return tables_to_frame([fragment_to_table(fragments[0], **kwargs)])


def mean_file_size(fragments, n=10):
    n_frags = len(fragments)
    if n < n_frags:
        indices = np.random.choice(np.arange(n_frags), size=n, replace=False)
    else:
        indices = np.arange(n_frags)

    sizes = []
    for f in indices:
        size = 0
        frag = fragments[f]
        for row_group in frag.row_groups:
            size += row_group.total_byte_size
        sizes.append(size)

    return np.mean(sizes)


def aggregate_fragments(fragments, blocksize):
    size = mean_file_size(fragments)
    blocksize = parse_bytes(blocksize)
    stride = int(math.floor(blocksize / size))

    if stride < 1:
        pass  # Not implemented yet!

    stride = max(stride, 1)
    return [fragments[i : i + stride] for i in range(0, len(fragments), stride)]


def read_parquet(
    urlpath,
    columns=None,
    filters=None,
    blocksize="256MB",
    fragment_parallelism=None,
):

    # Use pyarrow dataset API to get fragments and meta
    ds = dataset.dataset(urlpath, format="parquet")
    meta = tables_to_frame([ds.schema.empty_table()])
    if columns is not None:
        meta = meta[columns]
    fragments = list(ds.get_fragments())

    # Aggregate fragments together if necessary
    if blocksize:
        fragments = aggregate_fragments(fragments, blocksize)

    # Construct collection
    return dd.from_map(
        read_parquet_fragments,
        fragments,
        columns=columns,
        filters=filters,
        fragment_parallelism=fragment_parallelism,
        meta=meta,
        enforce_metadata=False,
    )
