Worker
======

Dask-CUDA workers extend the standard Dask worker in two ways:

1) Advanced networking configuration
2) GPU Memory Pool configuration

These configurations can be defined in the single cluster use case with ``LocalCUDACluster`` or passed to workers on the cli with ``dask-cuda-worker``

Single Cluster configuration
----------------------------


Command Line Tool
-----------------

New configuration options::

    --device-memory-limit
    --rmm-pool-size
    --enable-tcp-over-ucx / --disable-tcp-over-ucx
    -enable-infiniband / --disable-infiniband
    --enable-nvlink / --disable-nvlink
    --net-devices

Full details ``dask-cuda-worker`` options
::

    $ dask-cuda-worker --help
    Options:
    --tls-ca-file PATH              CA cert(s) file for TLS (in PEM format)
    --tls-cert PATH                 certificate file for TLS (in PEM format)
    --tls-key PATH                  private key file for TLS (in PEM format)
    --dashboard-address TEXT        dashboard address
    --dashboard / --no-dashboard    Launch dashboard  [default: True]
    --host TEXT                     Serving host. Should be an ip address that
                                    is visible to the scheduler and other
                                    workers. See --listen-address and --contact-
                                    address if you need different listen and
                                    contact addresses. See --interface.

    --interface TEXT                The external interface used to connect to
                                    the scheduler, usually an ethernet interface
                                    is used for connection, and not an
                                    InfiniBand interface (if one is available).

    --nthreads INTEGER              Number of threads per process.
    --name TEXT                     A unique name for this worker like
                                    'worker-1'. If used with --nprocs then the
                                    process number will be appended like name-0,
                                    name-1, name-2, ...

    --memory-limit TEXT             Bytes of memory per process that the worker
                                    can use. This can be an integer (bytes),
                                    float (fraction of total system memory),
                                    string (like 5GB or 5000M), 'auto', or zero
                                    for no memory management

    --device-memory-limit TEXT      Bytes of memory per CUDA device that the
                                    worker can use. This can be an integer
                                    (bytes), float (fraction of total device
                                    memory), string (like 5GB or 5000M), 'auto',
                                    or zero for no memory management (i.e.,
                                    allow full device memory usage).

    --rmm-pool-size TEXT            If specified, initialize each worker with an
                                    RMM pool of the given size, otherwise no RMM
                                    pool is created. This can be an integer
                                    (bytes) or string (like 5GB or 5000M).

    --reconnect / --no-reconnect    Reconnect to scheduler if disconnected
    --pid-file TEXT                 File to write the process PID
    --local-directory TEXT          Directory to place worker files
    --resources TEXT                Resources for task constraints like "GPU=2
                                    MEM=10e9". Resources are applied separately
                                    to each worker process (only relevant when
                                    starting multiple worker processes with '--
                                    nprocs').

    --scheduler-file TEXT           Filename to JSON encoded scheduler
                                    information. Use with dask-scheduler
                                    --scheduler-file

    --death-timeout TEXT            Seconds to wait for a scheduler before
                                    closing

    --dashboard-prefix TEXT         Prefix for the Dashboard
    --preload TEXT                  Module that should be loaded by each worker
                                    process like "foo.bar" or "/path/to/foo.py"

    --enable-tcp-over-ucx / --disable-tcp-over-ucx
                                    Enable TCP communication over UCX
    --enable-infiniband / --disable-infiniband
                                    Enable InfiniBand communication
    --enable-nvlink / --disable-nvlink
                                    Enable NVLink communication
    --net-devices TEXT              When None (default), 'UCX_NET_DEVICES' will
                                    be left to its default. Otherwise, it must
                                    be a non-empty string with the interface
                                    name. Normally used only with --enable-
                                    infiniband to specify the interface to be
                                    used by the worker, such as 'mlx5_0:1' or
                                    'ib0'.

    --help                          Show this message and exit.