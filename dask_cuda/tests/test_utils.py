import os

import pytest
from numba import cuda

from dask_cuda.utils import (
    _ucx_111,
    cuda_visible_devices,
    get_cpu_affinity,
    get_device_total_memory,
    get_gpu_count,
    get_n_gpus,
    get_preload_options,
    get_ucx_config,
    get_ucx_net_devices,
    nvml_device_index,
    parse_cuda_visible_device,
    parse_device_memory_limit,
    unpack_bitmask,
)


def test_get_n_gpus():
    assert isinstance(get_n_gpus(), int)

    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
        assert get_n_gpus() == 3
    finally:
        del os.environ["CUDA_VISIBLE_DEVICES"]


@pytest.mark.parametrize(
    "params",
    [
        {
            "input": [1152920405096267775, 0],
            "output": [i for i in range(20)] + [i + 40 for i in range(20)],
        },
        {
            "input": [17293823668613283840, 65535],
            "output": [i + 20 for i in range(20)] + [i + 60 for i in range(20)],
        },
        {"input": [18446744073709551615, 0], "output": [i for i in range(64)]},
        {"input": [0, 18446744073709551615], "output": [i + 64 for i in range(64)]},
    ],
)
def test_unpack_bitmask(params):
    assert unpack_bitmask(params["input"]) == params["output"]


def test_unpack_bitmask_single_value():
    with pytest.raises(TypeError):
        unpack_bitmask(1)


def test_cpu_affinity():
    for i in range(get_n_gpus()):
        affinity = get_cpu_affinity(i)
        os.sched_setaffinity(0, affinity)
        assert os.sched_getaffinity(0) == set(affinity)


def test_cpu_affinity_and_cuda_visible_devices():
    affinity = dict()
    for i in range(get_n_gpus()):
        # The negative here would be `device = 0` as required for CUDA runtime
        # calls.
        device = nvml_device_index(0, cuda_visible_devices(i))
        affinity[device] = get_cpu_affinity(device)

    for i in range(get_n_gpus()):
        assert get_cpu_affinity(i) == affinity[i]


def test_get_device_total_memory():
    for i in range(get_n_gpus()):
        with cuda.gpus[i]:
            total_mem = get_device_total_memory(i)
            assert type(total_mem) is int
            assert total_mem > 0


@pytest.mark.parametrize("enable_tcp", [True, False])
@pytest.mark.parametrize(
    "enable_infiniband_netdev",
    [(True, lambda i: "mlx5_%d:1" % (i // 2)), (True, "eth0"), (True, ""), (False, "")],
)
@pytest.mark.parametrize("enable_nvlink", [True, False])
def test_get_preload_options(enable_tcp, enable_infiniband_netdev, enable_nvlink):
    pytest.importorskip("ucp")

    enable_infiniband, net_devices = enable_infiniband_netdev

    opts = get_preload_options(
        protocol="ucx",
        create_cuda_context=True,
        enable_tcp_over_ucx=enable_tcp,
        enable_infiniband=enable_infiniband,
        enable_nvlink=enable_nvlink,
        ucx_net_devices=net_devices,
        cuda_device_index=5,
    )

    assert "preload" in opts
    assert opts["preload"] == ["dask_cuda.initialize"]
    assert "preload_argv" in opts
    assert "--create-cuda-context" in opts["preload_argv"]

    if enable_tcp:
        assert "--enable-tcp-over-ucx" in opts["preload_argv"]
    if enable_infiniband:
        assert "--enable-infiniband" in opts["preload_argv"]
        if callable(net_devices):
            dev = net_devices(5)
            assert str("--net-devices=" + dev) in opts["preload_argv"]
        elif isinstance(net_devices, str) and net_devices != "":
            assert str("--net-devices=" + net_devices) in opts["preload_argv"]
    if enable_nvlink:
        assert "--enable-nvlink" in opts["preload_argv"]


@pytest.mark.skipif(
    _ucx_111, reason="`ucx_net_devices='auto'` is deprecated for UCX >= 1.11.0",
)
def test_get_ucx_net_devices_raises():
    pytest.importorskip("ucp")

    with pytest.raises(ValueError):
        get_ucx_net_devices(None, "auto")


def test_get_ucx_net_devices_callable():
    pytest.importorskip("ucp")

    net_devices = [
        "mlx5_0:1",
        "mlx5_0:1",
        "mlx5_1:1",
        "mlx5_1:1",
        "mlx5_2:1",
        "mlx5_2:1",
        "mlx5_3:1",
        "mlx5_3:1",
    ]

    for idx in range(8):
        dev = get_ucx_net_devices(idx, lambda i: "mlx5_%d:1" % (i // 2))
        assert dev == net_devices[idx]


def test_get_ucx_net_devices_auto():
    pytest.importorskip("ucp")

    for idx in range(get_n_gpus()):
        # Since the actual device is system-dependent, we just check that
        # this function call doesn't fail. If any InfiniBand devices are
        # available, it will return that, otherwise an empty string.
        get_ucx_net_devices(idx, "auto")


@pytest.mark.parametrize("enable_tcp_over_ucx", [True, False])
@pytest.mark.parametrize("enable_infiniband", [True, False])
@pytest.mark.parametrize("net_devices", ["eth0", "auto", ""])
def test_get_ucx_config(enable_tcp_over_ucx, enable_infiniband, net_devices):
    pytest.importorskip("ucp")

    kwargs = {
        "enable_tcp_over_ucx": enable_tcp_over_ucx,
        "enable_infiniband": enable_infiniband,
        "net_devices": net_devices,
        "cuda_device_index": 0,
    }
    if net_devices == "auto" and enable_infiniband is False:
        with pytest.raises(ValueError):
            get_ucx_config(**kwargs)
        return
    else:
        ucx_config = get_ucx_config(**kwargs)

    if enable_tcp_over_ucx is True:
        assert ucx_config["tcp"] is True
        assert ucx_config["cuda_copy"] is True
    else:
        assert ucx_config["tcp"] is None

    if enable_infiniband is True:
        assert ucx_config["infiniband"] is True
        assert ucx_config["cuda_copy"] is True
    else:
        assert ucx_config["infiniband"] is None

    if enable_tcp_over_ucx is False and enable_infiniband is False:
        assert ucx_config["cuda_copy"] is None

    if net_devices == "eth0":
        assert ucx_config["net-devices"] == "eth0"
    elif net_devices == "auto":
        # Since the actual device is system-dependent, we don't do any
        # checks at the moment. If any InfiniBand devices are available,
        # that will be the value of "net-devices", otherwise an empty string.
        pass
    elif net_devices == "":
        assert "net-device" not in ucx_config


def test_parse_visible_devices():
    pynvml = pytest.importorskip("pynvml")
    pynvml.nvmlInit()
    indices = []
    uuids = []
    for index in range(get_gpu_count()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        uuid = pynvml.nvmlDeviceGetUUID(handle).decode("utf-8")

        assert parse_cuda_visible_device(index) == index
        assert parse_cuda_visible_device(uuid) == uuid

        indices.append(str(index))
        uuids.append(pynvml.nvmlDeviceGetUUID(handle).decode("utf-8"))

    index_devices = ",".join(indices)
    os.environ["CUDA_VISIBLE_DEVICES"] = index_devices
    for index in range(get_gpu_count()):
        visible = cuda_visible_devices(index)
        assert visible.split(",")[0] == str(index)

    uuid_devices = ",".join(uuids)
    os.environ["CUDA_VISIBLE_DEVICES"] = uuid_devices
    for index in range(get_gpu_count()):
        visible = cuda_visible_devices(index)
        assert visible.split(",")[0] == str(uuids[index])

    with pytest.raises(ValueError):
        parse_cuda_visible_device("Foo")

    with pytest.raises(TypeError):
        parse_cuda_visible_device(None)
        parse_cuda_visible_device([])


def test_parse_device_memory_limit():
    total = get_device_total_memory(0)

    assert parse_device_memory_limit(None) == total
    assert parse_device_memory_limit(0) == total
    assert parse_device_memory_limit("auto") == total

    assert parse_device_memory_limit(0.8) == int(total * 0.8)
    assert parse_device_memory_limit(1000000000) == 1000000000
    assert parse_device_memory_limit("1GB") == 1000000000


def test_parse_visible_mig_devices():
    pynvml = pytest.importorskip("pynvml")
    pynvml.nvmlInit()
    for index in range(get_gpu_count()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        try:
            mode = pynvml.nvmlDeviceGetMigMode(handle)[0]
        except pynvml.NVMLError:
            # if not a MIG device, i.e. a normal GPU, skip
            continue
        if mode:
            # Just checks to see if there are any MIG enabled GPUS.
            # If there is one, check if the number of mig instances
            # in that GPU is <= to count, where count gives us the
            # maximum number of MIG devices/instances that can exist
            # under a given parent NVML device.
            count = pynvml.nvmlDeviceGetMaxMigDeviceCount(handle)
            miguuids = []
            for i in range(count):
                try:
                    mighandle = pynvml.nvmlDeviceGetMigDeviceHandleByIndex(
                        device=handle, index=i
                    )
                    miguuids.append(mighandle)
                except pynvml.NVMLError:
                    pass
            assert len(miguuids) <= count
