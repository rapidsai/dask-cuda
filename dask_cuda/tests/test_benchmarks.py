# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import sys

from dask_cuda.benchmarks.utils import (
    get_cluster_options,
    parse_benchmark_args,
    ucx_config_to_dict,
    ucx_transport_selection,
)


def parse_args(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["benchmark", *argv])
    return parse_benchmark_args()


def test_benchmark_ucx_defaults_to_automatic_transport_selection(monkeypatch):
    args = parse_args(monkeypatch, "--protocol", "ucx")

    assert args.enable_tcp_over_ucx is None
    assert args.enable_infiniband is None
    assert args.enable_nvlink is None
    assert ucx_transport_selection(args) == "automatic"
    assert ucx_config_to_dict(args) == {
        "ucx_transport_selection": "automatic",
        "tcp": None,
        "ib": None,
        "nvlink": None,
    }

    cluster_kwargs = get_cluster_options(args)["kwargs"]
    assert cluster_kwargs["enable_tcp_over_ucx"] is None
    assert cluster_kwargs["enable_infiniband"] is None
    assert cluster_kwargs["enable_nvlink"] is None


def test_benchmark_ucx_transport_overrides_are_manual(monkeypatch):
    args = parse_args(monkeypatch, "--protocol", "ucx", "--disable-nvlink")

    assert args.enable_tcp_over_ucx is None
    assert args.enable_infiniband is None
    assert args.enable_nvlink is False
    assert ucx_transport_selection(args) == "manual"
