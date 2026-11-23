#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Run `cosmos_framework.scripts.inference` unchanged and print torch's peak GPU memory when it exits.

Use it in place of `-m cosmos_framework.scripts.inference` in the torchrun command:
    torchrun --standalone --nproc-per-node=1 tools/run_with_peak_memory.py -i ... -o ... (same flags)
The last line is `PEAK_TORCH_MEMORY_GB allocated=<x> reserved=<y>` (rank 0). Nothing else changes.
"""
import atexit
import runpy

import torch


def _report() -> None:
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        gb = 1024**3
        print(f"PEAK_TORCH_MEMORY_GB allocated={torch.cuda.max_memory_allocated() / gb:.2f} "
              f"reserved={torch.cuda.max_memory_reserved() / gb:.2f}", flush=True)


atexit.register(_report)
runpy.run_module("cosmos_framework.scripts.inference", run_name="__main__")
