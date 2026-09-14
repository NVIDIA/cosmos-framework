# Ad-hoc probes quoted in the README (run interactively on 2026-09-14, 120 W mode unless noted; numbers copied from the console)

## Memory system (`build/membw`)
NVIDIA Thor, 20 SMs, L2 32 MB. L2 read (all SMs, 8/16 MB L2-resident buffer, 4/8/16 blocks per SM): 1501-1524 GB/s. DRAM read (1 GB, flushed): 232-235 GB/s.
cudaMemcpy D2D 1 GB: 214 GB/s counting read+write.

## Cluster co-residency (`build/clusterocc`, cudaOccupancyMaxActiveClusters, 1 CTA/SM with 200 KB smem)
cluster size 1: 20 clusters (20 SMs); 2: 10 (20 SMs); 4: 4 (16 SMs); 8: 1 (8 SMs); 16: 0.

## Rasterization swizzle sweep (int8 cfg3, --flush=1, uniform operands, 10 iterations, TFLOPS)
| shape | swz 0 | 1 | 2 | 4 | 8 | 16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192x8192x8192 | 107.0 | 106.2 | 189.1 | 296.5 | 356.5 | 114.8 |
| M=16384 N=12288 K=4096 | 111.9 | | 200.6 | 316.2 | 329.6 | 334.0 |
Raster order with swizzle auto (int8 cfg3, 10 it): 8192^3 H 353 / M 325 / N 356; 16384x12288x4096 H 335 / M 310 / N 334; 42240x4096x12288 (swz 4) H 323 / M 346 / N 324.
On the target shapes (W <= 16 MB) swizzle 16 vs 0 is within +/-3 % at M <= 4096 and 7-11 % slower at M >= 16384 (results/full_v1_*.csv), hence `--swizzle=auto` now turns swizzle off when N*K <= 16 MiB.

## DRAM sensitivity (int8 cfg3, uniform, before the swizzle rule)
4096^3: flush=1 280 TFLOPS (490 us), flush=0 375 (366 us, 30 iterations = before the power limiter engages; 342-368 in the later probes).
2048x2048x16384: flush=1 120, flush=0 142. 8192x8192x1024: 208. 8192^3 (swizzle 0): 106.

## cuBLASLt autotune at 4096^3 (subagent run, 20 iterations, heuristic candidates ranked by 5-iteration timing)
FP8: heuristic0 260.9 TFLOPS (algo 66 tile 513) -> autotune 284.6 (algo 66 tile 23). INT8: heuristic0 261.4 (algo 71) -> autotune 249.5 (no gain). In the later full sweeps with --autotune=1 cuBLASLt FP8 4096^3 read 281-290.

## torch cross-check (venv torch 2.14.0+cu130, 4096^3, warm-up 300 ms, L2 flushed, run while a build was using the CPUs)
torch bf16 matmul 114 TFLOPS; torch._scaled_mm fp8 per-tensor -> bf16 242; torch._int_mm int8 -> int32 217.
