#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# gpu_probe.py [seconds] - the load half of gpu_probe.ps1: a bf16 8192^2 matmul loop on the otherwise idle GPU, sampled by
# nvidia-smi once a second while it runs. Prints TFLOP/s and a verdict line. Healthy RTX Spark N1X: about 70 TFLOP/s at a
# 1600 MHz cap (1599 MHz, 95 % utilisation, 39 W). Latched low-power state: about 7 to 10 TFLOP/s at 663 MHz and 5 to 7 W,
# cleared only by a reboot. Exit code 0 healthy, 3 degraded.
import subprocess, sys, time, torch
secs = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0
floor = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16); b = torch.randn_like(a)
c = a @ b; torch.cuda.synchronize()
smi = subprocess.Popen(["nvidia-smi", "-i", "0", "--query-gpu=timestamp,temperature.gpu,clocks.sm,pstate,utilization.gpu,power.draw",
                        "--format=csv,noheader", "-lms", "1000"])
t0 = time.time(); n = 0
while time.time() - t0 < secs:
    c = a @ b; torch.cuda.synchronize(); n += 1
dt = time.time() - t0
smi.terminate(); smi.wait()
tflops = n * 2 * 8192**3 / dt / 1e12
print(f"probe: {n} bf16 8192^2 matmuls in {dt:.1f} s = {tflops:.1f} TFLOP/s (healthy ~70 at a 1600 MHz cap, latched ~7)", flush=True)
print(f"probe verdict: {'HEALTHY' if tflops >= floor else 'DEGRADED: reboot before any run'} (floor {floor:.0f} TFLOP/s)", flush=True)
sys.exit(0 if tflops >= floor else 3)
