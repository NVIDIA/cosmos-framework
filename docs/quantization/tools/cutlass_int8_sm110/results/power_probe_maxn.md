# FP8 vs INT8 probe, power mode: MAXN, GPU max_freq 1575 MHz, 2026-09-14 08:28

cfg3 (2SM 256x256x128 cluster2x1, swizzle auto) for both dtypes. warm = --flush=0 --nw=1 (L2-resident), hotcold = --flush=0 --nw=8 (activations hot, 8 rotating weights), cold = --flush=1. normal = per-tensor-quantized Gaussian operands.

| case | dtype | TFLOPS | median us | gpu clk med/min MHz | emc clk med/min MHz | Tj end C | GPU power mean/max W (tegrastats VDD_GPU) | TOPS per W |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| warm 4096^3 normal 3000it | fp8 | 245.9 | 559.01 | 1575/1575 | 4266/4266 | 54.1 | 85.9/83.2 | 2.9 |
| warm 4096^3 normal 3000it | int8 | 332.3 | 413.62 | 1575/1575 | 4266/4266 | 55.5 | 88.7/98.9 | 3.7 |
| warm 4096^3 zeros 3000it | fp8 | 389.1 | 353.25 | 1575/1575 | 4266/4266 | 54.1 | 58.0/60.5 | 6.7 |
| warm 4096^3 zeros 3000it | int8 | 392.4 | 350.24 | 1575/1575 | 4266/4266 | 54.0 | 43.5/46.7 | 9.0 |
| warm 4096^3 normal 3000it (repeat) | fp8 | 245.9 | 559.01 | 1575/1575 | 4266/4266 | 60.8 | 91.0/94.4 | 2.7 |
| warm 4096^3 normal 3000it (repeat) | int8 | 319.6 | 430.08 | 1575/1575 | 4266/4266 | 59.1 | 91.0/98.9 | 3.5 |
| hotcold 4096x4096 M=901 normal 200it | fp8 | 216.4 | 139.71 | 1575/1575 | 4266/4266 | 59.5 | 69.5/80.0 | 3.1 |
| hotcold 4096x4096 M=901 normal 200it | int8 | 314.0 | 96.29 | 1575/1575 | 4266/4266 | 59.1 | 63.6/75.6 | 4.9 |
| hotcold 4096x4096 M=1517 normal 200it | fp8 | 252.5 | 201.63 | 1575/1575 | 4266/4266 | 60.0 | 71.3/93.8 | 3.5 |
| hotcold 4096x4096 M=1517 normal 200it | int8 | 360.1 | 141.34 | 1575/1575 | 4266/4266 | 59.5 | 66.3/73.9 | 5.4 |
| hotcold 4096x4096 M=1802 normal 200it | fp8 | 234.3 | 258.02 | 1575/1575 | 4266/4266 | 59.3 | 68.8/81.7 | 3.4 |
| hotcold 4096x4096 M=1802 normal 200it | int8 | 327.9 | 184.38 | 1575/1575 | 4266/4266 | 60.5 | 62.0/75.6 | 5.3 |
| hotcold 1024x4096 M=901 normal 200it | fp8 | 217.5 | 34.75 | 1575/1575 | 4266/4266 | 58.8 | 67.7/79.6 | 3.2 |
| hotcold 1024x4096 M=901 normal 200it | int8 | 217.5 | 34.75 | 1575/1575 | 4266/4266 | 57.9 | 53.6/57.0 | 4.1 |
| hotcold 1024x4096 M=1517 normal 200it | fp8 | 267.2 | 47.63 | 1575/1575 | 4266/4266 | 58.0 | 50.6/59.6 | 5.3 |
| hotcold 1024x4096 M=1517 normal 200it | int8 | 270.0 | 47.14 | 1575/1575 | 4266/4266 | 57.9 | 49.2/59.2 | 5.5 |
| cold 4096x4096 M=1517 normal 100it | fp8 | 241.3 | 210.96 | 1575/1575 | 4266/4266 | 58.0 | 60.8/86.9 | 4.0 |
| cold 4096x4096 M=1517 normal 100it | int8 | 242.5 | 209.92 | 1575/1575 | 4266/4266 | 58.2 | 57.2/71.1 | 4.2 |

GPU power source: tegrastats VDD_GPU from /tmp/tegrastats_maxn.log (root tegrastats; the non-root reading is 0 mW). Idle before probe: VDD_GPU 4357mW.

NOTE: the "max" half of the power column in this file is unreliable (string comparison bug in the first version of the script, fixed since); the mean is correct. Means of the 200-iteration cases include ~1 s of process start-up; see power_probe_maxn_long.md for kernel-dominated means.
