# Copyright (c) 2022, NVIDIA CORPORATION.

from packaging import version

import distributed

DISTRIBUTED_VERSION = version.parse(distributed.__version__)
DISTRIBUTED_GT_2022_11_1 = DISTRIBUTED_VERSION > version.parse("2022.11.1")
