import pytest

from distributed import Nanny

from dask_cuda.worker_spec import worker_spec


def _check_option(spec, k, v):
    assert all([s["options"][k] == v for s in spec.values()])


def _check_env_key(spec, k, enable):
    if enable:
        assert all([k in s["options"]["env"] for s in spec.values()])
    else:
        assert all([k not in s["options"]["env"] for s in spec.values()])


def _check_env_value(spec, k, v):
    if not isinstance(v, list):
        v = [v]

    for i in v:
        assert all([i in set(s["options"]["env"][k].split(",")) for s in spec.values()])


@pytest.mark.filterwarnings("ignore:Cannot get CPU affinity")
@pytest.mark.parametrize("num_devices", [1, 4])
@pytest.mark.parametrize("cls", [Nanny])
@pytest.mark.parametrize("interface", [None, "eth0", "enp1s0f0"])
@pytest.mark.parametrize("protocol", [None, "tcp", "ucx"])
@pytest.mark.parametrize("dashboard_address", [None, ":0", ":8787"])
@pytest.mark.parametrize("threads_per_worker", [1, 8])
@pytest.mark.parametrize("silence_logs", [False, True])
@pytest.mark.parametrize("enable_infiniband", [False, True])
@pytest.mark.parametrize("enable_nvlink", [False, True])
def test_worker_spec(
    num_devices,
    cls,
    interface,
    protocol,
    dashboard_address,
    threads_per_worker,
    silence_logs,
    enable_infiniband,
    enable_nvlink,
):
    def _test():
        return worker_spec(
            CUDA_VISIBLE_DEVICES=list(range(num_devices)),
            cls=cls,
            interface=interface,
            protocol=protocol,
            dashboard_address=dashboard_address,
            threads_per_worker=threads_per_worker,
            silence_logs=silence_logs,
            enable_infiniband=enable_infiniband,
            enable_nvlink=enable_nvlink,
        )

    if (enable_infiniband or enable_nvlink) and protocol != "ucx":
        with pytest.raises(
            TypeError, match="Enabling InfiniBand or NVLink requires protocol='ucx'"
        ):
            _test()
        return
    else:
        spec = _test()

    assert len(spec) == num_devices
    assert all([s["cls"] == cls for s in spec.values()])

    _check_option(spec, "interface", interface)
    _check_option(spec, "protocol", protocol)
    _check_option(spec, "dashboard_address", dashboard_address)
    _check_option(spec, "nthreads", threads_per_worker)
    _check_option(spec, "silence_logs", silence_logs)
