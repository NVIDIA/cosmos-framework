#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# Diagnostic: sustained BF16 matmul TFLOP/s. About 40 TFLOP/s or more is a healthy RTX Spark N1X GPU under load;
# a reading several times lower means a clock limiter or a latched low-power state (see the thermal reference).
# Diagnostic only: keep the GPU under steady compute load for N seconds (matmul loop) while another process decodes.
import sys, time, torch
secs = float(sys.argv[1]) if len(sys.argv) > 1 else 120
a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16); b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
t0 = time.time(); n = 0
while time.time() - t0 < secs:
    c = a @ b; n += 1
    if n % 50 == 0: torch.cuda.synchronize()
torch.cuda.synchronize(); print(f"spinner done: {n} matmuls in {time.time()-t0:.0f}s ({n*2*4096**3/1e12/(time.time()-t0):.1f} TFLOP/s sustained)")
