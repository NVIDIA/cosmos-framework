# FP8 vs INT8 tensor-core throughput on Thor is data/power dependent (probe, 2026-09-14, 120 W mode)

Kernel: CUTLASS cfg3 (2SM 256x256x128, cluster 2x1, swizzle 16) for both dtypes, 4096x4096x4096 unless noted. GPU clock (devfreq
gpu-gpc-0) and EMC clock (bwmgr) were sampled every 50 ms during every run: always 1386 MHz / 4266 MHz (min sample >= 1314 MHz).
Tj = tj-thermal at the end of the run. "warm" = --flush=0 --nw=1 (everything L2-resident, pure MMA + L2), "hot-A/cold-W" =
--flush=0 --nw=8, "cold" = --flush=1. uniform = full-range uniform operands; normal = per-tensor-quantized Gaussian (absmax = 5 sigma);
zeros = all-zero operands.

| case | FP8 TFLOPS (us) | INT8 TFLOPS (us) | INT8/FP8 |
| --- | ---: | ---: | ---: |
| warm, uniform, 30 iters | 264 (521) | 368 (373) | 1.39 |
| warm, uniform, 3000 iters | 217 (634) | 342 (401) | 1.58 |
| warm, uniform, 3000 iters (repeat) | 217 (634) | 344 (399) | 1.59 |
| warm, normal, 3000 iters | 217 (633) | 346 (397) | 1.59 |
| **warm, zeros, 3000 iters** | **379 (363)** | **382 (360)** | 1.01 |
| warm, zeros, 30 iters | 385 (357) | 344 (399) | 0.89 |
| hot-A/cold-W, M=1517, normal, 200 iters | 233 (218), Tj 80 C | 355 (144), Tj 75 C | 1.52 |
| hot-A/cold-W, M=1517, uniform, 200 iters | 239 (213) | 355 (143) | 1.49 |
| hot-A/cold-W, M=901, normal, 200 iters | 213 (142) | 274 (110) | 1.29 |
| cold, M=1517, normal, 100 iters | 245 (208) | 244 (209) | 1.00 |
| cold, 4096^3, uniform, 50 iters (earlier, Tj ~45 C) | 286 (481) | 286 (481) | 1.00 |
| cold, 4096^3, uniform, 50 iters (later, Tj ~75 C) | 287 (479) | 266 (517) | 0.93 |

Reading: with all-zero operands FP8 and INT8 both reach ~380 TOPS (84 % of the 454 TOPS peak at 1386 MHz), so the FP8 MMA path is
not intrinsically slower. With non-trivial operands FP8 drops to ~217 TOPS while INT8 stays at ~345, at identical GPU/EMC clocks and
with a *higher* junction temperature for FP8. The consistent explanation is a power/current limiter acting on the tensor-core datapath
(FP8 e4m3 multiply-add toggles more logic than s8 x s8), not SM-clock DVFS. In the DRAM-bound cold regime (M <= 1802 with cold
activations) the tensor cores idle enough that the limiter never engages and FP8 == INT8. Chip temperature drifts +/-8 % into the
cold numbers over a long session (Tj 45 -> 75 C); clocks were never observed below 1314 MHz. Not testable here: the MAXN power mode
(needs sudo), and a direct GPU-rail power reading (VDD_GPU reports 0 mW on this board).

Board power attempt (hwmon6 ina3221, carrier-board rails CVB_ATX_12V/3V3/5V summed, 10 Hz): idle 34-41 W, FP8 normal 40-41 W,
INT8 normal 42-43 W, both zeros 42 W, cuBLASLt bf16 45-46 W. The readings barely move and lag the load (idle *after* the runs read
41 W), and the GPU rail (VDD_GPU) reports 0 mW, so these sensors do not resolve GPU power on this board. The power/current-limiter
explanation therefore rests on: identical clocks, zero-data parity, and the higher Tj under FP8 -- inferred, not measured.
