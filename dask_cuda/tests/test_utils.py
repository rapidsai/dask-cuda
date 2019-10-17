import pytest

import os

from numba import cuda

from dask_cuda.utils import (
    get_cpu_affinity,
    get_device_total_memory,
    get_n_gpus,
    get_preload_options,
    get_ucx_env,
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
        assert list(os.sched_getaffinity(0)) == affinity


def test_get_device_total_memory():
    for i in range(get_n_gpus()):
        with cuda.gpus[i]:
            assert (
                get_device_total_memory(i)
                == cuda.current_context().get_memory_info()[1]
            )


@pytest.mark.parametrize("enable_tcp", [True, False])
@pytest.mark.parametrize("enable_infiniband", [True, False])
@pytest.mark.parametrize("enable_nvlink", [True, False])
def test_get_ucx_env(enable_tcp, enable_infiniband, enable_nvlink):

    env = get_ucx_env(
        enable_tcp=enable_tcp,
        enable_infiniband=enable_infiniband,
        enable_nvlink=enable_nvlink,
    )

    if enable_tcp or enable_infiniband or enable_nvlink:
        assert "UCX_TLS" in env
        assert "UCX_SOCKADDR_TLS_PRIORITY" in env
    else:
        assert env == {}
        return

    assert "tcp" in env["UCX_TLS"]
    assert "cuda_copy" in env["UCX_TLS"]
    assert "sockcm" in env["UCX_TLS"]
    assert "sockcm" in env["UCX_SOCKADDR_TLS_PRIORITY"]

    if enable_infiniband:
        assert "rc" in env["UCX_TLS"]
    if enable_nvlink:
        assert "cuda_ipc" in env["UCX_TLS"]


@pytest.mark.parametrize("enable_tcp", [True, False])
@pytest.mark.parametrize(
    "enable_infiniband_netdev",
    [(True, lambda i: "mlx5_%d:1" % (i // 2)), (True, "eth0"), (True, ""), (False, "")],
)
@pytest.mark.parametrize("enable_nvlink", [True, False])
def test_get_preload_options(enable_tcp, enable_infiniband_netdev, enable_nvlink):
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
