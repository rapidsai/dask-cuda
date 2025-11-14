#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

wheel_dir_relative_path=$1

# Pure wheels can be installed on any OS and we want to avoid users being able
# to begin installing them on Windows or OSX when we know that the dependencies
# won't work / be available.
for wheel in "${wheel_dir_relative_path}"/*-py3-none-any.whl; do
    if [ -f "${wheel}" ]; then
        rapids-logger "Retagging pure Python wheel: ${wheel}"

        # Retag for manylinux x86_64 and manylinux aarch64
        wheel tags --platform-tag=manylinux_2_28_x86_64.manylinux_2_28_aarch64 --remove "${wheel}"

        rapids-logger "Successfully retagged wheel for manylinux platforms"
    fi
done

rapids-logger "validate packages with 'pydistcheck'"

pydistcheck \
    --inspect \
    "$(echo "${wheel_dir_relative_path}"/*.whl)"

rapids-logger "validate packages with 'twine'"

twine check \
    --strict \
    "$(echo "${wheel_dir_relative_path}"/*.whl)"
