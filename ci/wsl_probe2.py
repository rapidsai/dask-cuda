# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import subprocess
import sys
import time

import pynvml
from cuda.core import Device


def procs():
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    return [p.pid for p in pynvml.nvmlDeviceGetComputeRunningProcesses_v2(h)]


def child():
    Device().set_current()
    print("  child", os.getpid(), "sees", procs(), flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["child"]:
        child()
        sys.exit(0)
    print("step1 parent", os.getpid(), "no ctx, sees", procs(), flush=True)
    subprocess.run([sys.executable, __file__, "child"])
    Device().set_current()
    print("step2 parent", os.getpid(), "with ctx, sees", procs(), flush=True)
    for _ in range(3):
        subprocess.run([sys.executable, __file__, "child"])
    kids = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import time; from cuda.core import Device; Device().set_current(); time.sleep(5)",
            ]
        )
        for _ in range(3)
    ]
    time.sleep(3)
    print(
        "step3 parent with 3 live ctx children",
        [k.pid for k in kids],
        "sees",
        procs(),
        flush=True,
    )
    [k.wait() for k in kids]
    print("step4 after children exit, sees", procs(), flush=True)
