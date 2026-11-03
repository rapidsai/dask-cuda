# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pynvml
from cuda.core import Device
from distributed.diagnostics.nvml import has_cuda_context

print("pid", os.getpid(), "has_ctx before", has_cuda_context())
Device().set_current()
print("has_ctx after", has_cuda_context())
pynvml.nvmlInit()
print("driver", pynvml.nvmlSystemGetDriverVersion())
h = pynvml.nvmlDeviceGetHandleByIndex(0)
for name in [
    "nvmlDeviceGetComputeRunningProcesses_v2",
    "nvmlDeviceGetComputeRunningProcesses_v3",
    "nvmlDeviceGetGraphicsRunningProcesses",
]:
    try:
        procs = getattr(pynvml, name)(h)
        print(name, [(p.pid, p.usedGpuMemory) for p in procs])
    except Exception as e:
        print(name, repr(e))
print(open("/proc/self/status").read().split("NSpid")[-1].split("\n")[0])
