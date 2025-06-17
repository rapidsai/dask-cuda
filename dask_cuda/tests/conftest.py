# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dask_cuda.utils import has_device_memory_resource


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "skip_if_no_device_memory: mark test to skip if device has no dedicated memory "
        "resource",
    )
    config.addinivalue_line(
        "markers",
        "skip_if_device_memory: mark test to skip if device has dedicated memory "
        "resource",
    )


def pytest_collection_modifyitems(items):
    """Handle skip_if_no_device_memory marker."""
    for item in items:
        if item.get_closest_marker("skip_if_no_device_memory"):
            skip_marker = item.get_closest_marker("skip_if_no_device_memory")
            reason = skip_marker.kwargs.get(
                "reason", "Test requires device with dedicated memory resource"
            )
            item.add_marker(
                pytest.mark.skipif(not has_device_memory_resource(), reason=reason)
            )
        if item.get_closest_marker("skip_if_device_memory"):
            skip_marker = item.get_closest_marker("skip_if_device_memory")
            reason = skip_marker.kwargs.get(
                "reason", "Test requires device without dedicated memory resource"
            )
            item.add_marker(
                pytest.mark.skipif(has_device_memory_resource(), reason=reason)
            )
