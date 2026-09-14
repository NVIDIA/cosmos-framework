# CUTLASS per-tensor INT8 vs FP8 GEMM on Thor (sm_110a), 2026-09-14

Context: handoff section 9 aligned per-tensor INT8 with per-tensor FP8 on H100 (SM90). This directory repeats that first step on
NVIDIA Thor (Jetson AGX Thor dev kit, compute capability 11.0, Blackwell-family tcgen05 tensor cores, 20 SMs, 32 MB L2, unified
LPDDR5X). Everything here is standalone CUDA C++ (no torch dependency): a CUTLASS 3.x template that instantiates the *same*
kernel structure for FP8 (e4m3, fp32 accumulate) and INT8 (s8, int32 accumulate), a cuBLASLt baseline program, a memory-system
microbenchmark and a sweep script.

Template: CUTLASS `examples/70_blackwell_gemm/70_blackwell_fp8_gemm.cu` (CollectiveBuilder, `Sm100` arch tag also serves
sm_110a, TMA warp-specialized, CLC persistent scheduler). The example's heavy fusion (bias / ReLU / aux / amax) is replaced by a
plain per-tensor scale EVT so that FP8 and INT8 differ only in the MMA kind:

    D[M,N] = bf16( (A[M,K] . W[N,K]^T) * sa * sb )        A row-major, W = nn.Linear weight [N,K] row-major, D row-major bf16
    EVT   = Sm90EVT< Sm90Compute<multiplies, bf16, f32>, Sm90ScalarBroadcast<f32, ., 2, multiplies>{sa, sb}, Sm90AccFetch >

`sa`/`sb` can be host scalars or device pointers (dynamic activation scale without a host sync). Per-row x per-col scaling
(per-token / per-channel) is the same EVT with `Sm90ColBroadcast` / `Sm90RowBroadcast` instead of the scalar node (see the SM90
tool in `../cutlass_int8_sm90/pertensor/`); not instantiated here yet.

## Files

| file | what |
| --- | --- |
| `pertensor_gemm.cuh` | `thor::PerTensorGemm<ElementAB, ElementAcc, MmaTile, Cluster, KernelSchedule, EpilogueSchedule>` + type-erased `GemmHandle` |
| `configs.h`, `cfg_select.h`, `cfg_{int8,fp8}_N.cu` | 14 tile/cluster/schedule configurations, one translation unit each (parallel build) |
| `pertensor_gemm.cu` | driver: correctness vs a naive device reference (rows [0,512) and the last 128 rows), timing, `--swizzle=auto`, `--raster`, `--nw` (rotating weight copies), `--dist=uniform|normal`, `--zeros` |
| `cublaslt_bench.cu` | cuBLASLt baselines: bf16, FP8 per-tensor (A/B scale pointers, bf16 out), INT8 (s32 out), INT8 + separate int32->bf16 rescale kernel |
| `bench_util.cuh` | timing (L2 flush before every iteration, >= 300 ms warm-up for the devfreq governor), operand init, reference GEMM, CLI |
| `membw.cu` | L2 / DRAM read bandwidth microbenchmark |
| `bench_thor.py` | sweep over the Nano gen-tower shapes, CSV + markdown with INT8/FP8/bf16 ratios |
| `power_probe.sh`, `power_probe_long.sh` | FP8 vs INT8 probes under the current nvpmodel mode with clock / Tj / tegrastats VDD_GPU sampling per case (long = >= 3 s of kernels per case, energy per GEMM) |
| `results/` | raw logs / CSV / markdown of every run quoted below: `full_smallM_*` (mode A), `full_hotA_coldW_*` (mode B, 50 it), `full_modeB_sustained_*` (mode B, 1.5 s warm-up = headline), `modeB_sustained_120w.md`, `kv_nw32_sustained_120w.md`, `power_probe_*` (GPU power), `fp8_limiter_onset_*` (time series), `adhoc_probes_*` (membw, cluster residency, swizzle sweep, autotune, torch), `full_v1_*` (large-M / MLP shapes, partial) |

## Build and run

```bash
export CUTLASS=/home/pzeren/thor/cutlass          # CUTLASS >= 4.8 checkout (headers only), nvcc 13.0
make -j14                                          # ~2 min: 28 kernel TUs + driver + cublaslt_bench + build/membw
./pertensor_gemm --list
./pertensor_gemm --dtype=int8 --cfg=3 --m=4096 --n=4096 --k=4096          # verify + time one kernel (swizzle=auto)
./cublaslt_bench --dtype=fp8 --m=4096 --n=4096 --k=4096 --autotune=1
python3 bench_thor.py --stage full --int8-cfg 3 --fp8-cfg 3 --flush 0 --nw 8 --warmup-ms 1500 --dist normal   # headline mode-B table
# FP8 on Thor is power-limited after ~0.5 s of sustained load: always warm up >= 1.5 s (--warmup_ms) before timing FP8.
```

Compile flags: `-arch=sm_110a` (needed: `CUTLASS_ARCH_MMA_SM110A_ENABLED` gates tcgen05 incl. `kind::i8`; plain `sm_110`
disables the MMA atoms). Every kernel verifies to rel-L2 1.65e-3 against the fp32/int32 reference = the bf16 output rounding
floor; INT8 through cuBLASLt (int32 out) is bit-exact.

## Thor facts that decide the kernel design (all measured here)

| item | value | consequence |
| --- | --- | --- |
| SMs / clock | 20 SMs, 1386 MHz cap in the default 120 W power mode (`nvpmodel` mode 1; MAXN would allow 1575 MHz), idles at 315 MHz | dense INT8 = FP8 peak ~ 454 TOPS at 1386 MHz (8192 MAC/clk/SM, same per-SM rate as B200); warm-up must last >= 300 ms or the governor has not ramped |
| cluster residency (`cudaOccupancyMaxActiveClusters`, 1 CTA/SM) | size 1: 20, size 2: 10 clusters (20 SMs), **size 4: 4 clusters (16 SMs), size 8: 1 cluster (8 SMs)** | cluster shapes with more than 2 CTAs leave 20-60 % of the SMs idle. For the 256x256 2SM tile the TMA multicast configs (2x2, 4x1, 4x2: cfg 9/10/11) are therefore *slower* than 2x1 (cfg3) even though they cut L2 traffic; on the smaller tiles 2x2 multicast does help (2SM 256x128: cfg5 250 vs cfg2 198; 1SM 128x256: cfg13 245 vs cfg1 210) but stays below cfg3. Use 2SM MMA with cluster 2x1x1. |
| L2 read bandwidth (all SMs, L2-resident buffer) | ~1.52 TB/s aggregate (76 GB/s per SM, ~55 B/clk/SM) | a 128x128x128 1SM tile (128 FLOP per byte loaded into SMEM) tops out around 170 TFLOPS; 2SM 256x256 (256 FLOP/B) is needed to approach the MMA peak |
| DRAM read bandwidth | ~235 GB/s (D2D copy 214 GB/s counting both directions) | a 4096x4096 int8/fp8 weight (16 MB) costs ~70 us to stream; at M=901 the GEMM is memory-bound whatever the dtype |
| L2 size | 32 MB | the gate/up (12288x4096) and down (4096x12288) weights are 48 MB: without tile rasterization swizzle, W is re-streamed from DRAM once per 256-row M block (3x slowdown, see below) |

### Tile / cluster sweep (4096x4096x4096, L2 flushed before every iteration, median of 30)

| cfg | tile (MMA) | cluster | INT8 TFLOPS | FP8 TFLOPS |
| --- | --- | --- | ---: | ---: |
| 0 | 1SM 128x128x128 | 1x1 | 156 | 172 |
| 1 | 1SM 128x256x128 | 1x1 | 210 | 196 |
| 2 | 2SM 256x128x128 | 2x1 | 198 | 215 |
| **3** | **2SM 256x256x128** | **2x1** | **286** | **286** |
| 4 | 1SM 128x128x128 | 1x2 (A multicast) | 198 | 217 |
| 5 | 2SM 256x128x128 | 2x2 | 250 | 242 |
| 6 | 1SM 128x128x64 | 1x1 | 149 | 166 |
| 7 | 1SM 64x128x128 | 1x1 | 123 | 109 |
| 8 | 1SM 128x128x128 | 2x1 (B multicast) | 197 | 217 |
| 9 | 2SM 256x256x128 | 2x2 (A multicast, 4 CTAs) | 264 | 264 |
| 10 | 2SM 256x256x128 | 4x1 (B multicast, 4 CTAs) | 265 | 266 |
| 11 | 2SM 256x256x128 | 4x2 (8 CTAs) | 166 | 164 |
| 12 | 2SM 256x256x256 | 2x1 | 282 | 285 |
| 13 | 1SM 128x256x128 | 2x2 | 245 | 247 |

(cfg 0-8 from `results/tune_shape4096_*`, cfg 3 and 9-13 from `results/tune_bigtiles_*`; the two runs differ by ~8 % on cfg3 FP8
because the first one overlapped with the cuBLASLt build/test on the same GPU. Same-run comparisons only.)
Reference in the same run: cuBLASLt bf16 143, FP8 per-tensor 271 (heuristic; 285 with autotune in an ad-hoc run and 281-290 in the full
sweeps), INT8 s32-out 263, INT8 + separate rescale 149. These tile sweeps ran with a 300 ms warm-up and 30 iterations, i.e. before the
FP8 power limiter (below) engages; they rank tiles, they are not sustained FP8 throughput.

### Rasterization swizzle (cfg3 INT8, L2 flushed; raw numbers in `results/adhoc_probes_20260914.md`)

| shape | swizzle 0 (default) | 2 | 4 | 8 | 16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8192x8192x8192 (A = W = 64 MB) | 107 | 189 | 297 | **357** | 115 |
| M=16384, N=12288, K=4096 (W = 48 MB) | 112 | 201 | 316 | 330 | **334** |

With the default order the scheduler walks all N tiles of one 256-row M block before moving on, so the whole weight streams from
DRAM once per M block (8192^3: 32 x 64 MiB = 2 GiB ~ 9.1 ms of the 10.3 ms). `max_swizzle_size = g` groups g M blocks, cutting the
weight traffic by g, as long as the group's A rows (g x 256 x K bytes) stay in L2: at K = 8192, g = 16 needs 32 MB = the whole L2 and
collapses. On the target shapes, whose weights fit in L2 (16 MB / 4 MB), swizzle is a no-op or slightly harmful (within +/-3 % at
M <= 4096, 7-11 % slower at M >= 16384 on 4096x4096, `results/full_v1_*.csv`), so `--swizzle=auto` keeps it off when N x K <= 16 MiB and
otherwise picks the largest power of two g <= 16 with g x TileM x K <= 16 MiB (K=4096 -> 16, 8192 -> 8, 12288 -> 4).
The same 4096^3 INT8 kernel reaches 342-375 TOPS with a warm L2 (`--flush=0`; 375 in a 30-iteration burst, 342-368 sustained) vs
280 cold, i.e. the remaining gap to the 454 peak is DRAM traffic (A + W + D = 64 MiB ~ 285 us of a 480 us kernel), not the tensor cores.

## Results on the target shapes (q/o_proj 4096x4096 and k/v_proj 1024x4096, M = 901 / 1517 / 1802 / 4096)

M = 901 is the t2i single branch, 1802 = cond + uncond, 1517 = the policy (robot action) gen-token count; the video-size M (16384,
42240) and the MLP shapes (12288x4096, 4096x12288) were dropped from the final grid on request. Partial data for them (incl. the
large-M rows) is in `results/full_v1_*.md` / `results/full_smallM_*.md`. All rows verified (rel-L2 1.65e-3 = bf16 rounding; cuBLASLt
INT8 bit-exact). cuBLASLt runs with `--autotune=1` (best of up to 16 heuristic candidates); in tables A and B "CUTLASS" is the best of
cfg 3/2/1/0/7 taken from the same 50-iteration measurement (a winner's-curse bias of a few % where cfgs are within noise; cfg3 alone is
within 8 % of the best everywhere: it is the best in every 4096x4096 cell and every mode-B cell and trails cfg1/cfg2 by up to 7.6 % only on 1024x4096 in mode A), in the sustained table it is cfg3 only. TFLOPS = 2MNK / median time; INT8 ops counted as "TFLOPS".
Operands: uniform full-range in A and B, per-tensor-quantized Gaussian in the sustained table and the power probes (the FP8/INT8
ratios do not depend on this choice, see the probe).

### A. All operands cold, GEMMs separated by ~1.2 ms of idle (256 MB L2 flush memset before every iteration) -- `results/full_smallM_20260914_080500.md`

| N x K | M | cuBLASLt bf16 | cuBLASLt FP8 pt | cuBLASLt INT8 (s32 out) | cuBLASLt INT8 + rescale | CUTLASS FP8 pt | CUTLASS INT8 pt | INT8/FP8 (CUTLASS) | INT8/cuBLASLt FP8 | INT8/bf16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096x4096 | 901 | 88 (343 us) | 158 (191 us) | 168 (180 us) | 139 (218 us) | 178 (169 us; cfg3) | 178 (169 us; cfg3) | 1.00 | 1.13 | 2.02 |
| 4096x4096 | 1517 | 118 (430 us) | 218 (233 us) | 211 (241 us) | 151 (336 us) | 246 (207 us; cfg3) | 238 (214 us; cfg3) | 0.97 | 1.09 | 2.01 |
| 4096x4096 | 1802 | 135 (448 us) | 246 (246 us) | 215 (282 us) | 145 (416 us) | 237 (255 us; cfg3) | 236 (256 us; cfg3) | 1.00 | 0.96 | 1.75 |
| 4096x4096 | 4096 | 158 (872 us) | 290 (473 us) | 254 (540 us) | 146 (940 us) | 288 (477 us; cfg3) | 288 (477 us; cfg3) | 1.00 | 0.99 | 1.83 |
| 1024x4096 | 901 | 53 (142 us) | 93 (81 us) | 80 (94 us) | 72 (105 us) | 86 (88 us; cfg3) | 87 (87 us; cfg2) | 1.01 | 0.93 | 1.63 |
| 1024x4096 | 1517 | 72 (176 us) | 117 (109 us) | 106 (120 us) | 98 (130 us) | 111 (114 us; cfg3) | 114 (112 us; cfg1) | 1.02 | 0.97 | 1.58 |
| 1024x4096 | 1802 | 74 (204 us) | 131 (116 us) | 113 (134 us) | 103 (146 us) | 127 (119 us; cfg1) | 124 (122 us; cfg1) | 0.97 | 0.95 | 1.67 |
| 1024x4096 | 4096 | 110 (312 us) | 192 (179 us) | 183 (187 us) | 151 (228 us) | 198 (173 us; cfg3) | 189 (181 us; cfg3) | 0.96 | 0.99 | 1.72 |

Per-config detail (TFLOPS, same run): cfg3 (2SM 256x256) wins everywhere on 4096x4096 (M=1517: c3 246 / c2 199 / c1 198 / c0 161 /
c7 105); on 1024x4096 the tiles barely matter (M=901: c3 81-86, c2 84-87, c1 81-85, c0 75-77; cfg1/cfg2 beat cfg3 by 3-8 % in 4 of the
8 dtype cells) because the whole problem is a 9.7 MB DRAM stream. The best-of INT8/FP8 ratio is 0.96-1.02 in every row; individual
small-tile cells differ by up to 10 %.

DRAM-bound regime: the cold floor (A + W + D bytes at 235 GB/s) is 119 us for 4096x4096 M=901 (measured 169), 166 us for M=1802
(measured 255), 41 us for 1024x4096 M=901 (measured 87), while the MMA floor at 454 TOPS is 67 / 133 / 17 us. Both formats stream the
same bytes and buy ~1.6-2x over bf16, which streams twice as many. The FP8 == INT8 parity of this mode is, however, NOT evidence that
the GEMM is DRAM-bound: the 256 MB flush memset between iterations idles the tensor cores for ~1.2 ms per ~1.4-1.7 ms period, so the FP8
power limiter (section below, ~0.5 s time constant) never engages -- the 4096^3 row (477 us vs a 286 us DRAM floor and 303 us MMA floor)
is equal for the two formats only because of that gap. A production layer loop has cold weights and no gaps; it behaves like mode B.

### B. Activations hot in L2, weights cold (`--flush=0 --nw=8`: 8 rotating weight copies, no flush) -- `results/full_hotA_coldW_*.md`

This is the production layer loop: the activation was just written by the previous kernel and is L2-resident (strictly true while
A + W + D <= 32 MiB, i.e. M <= 1365 on 4096x4096 -- of the M grid only 901 qualifies, 1517 is already 33.8 MiB; at larger M part of A is re-read), every layer's weight is a fresh 16 MB (4 MB) read from
DRAM (checked with 32 rotating copies = 128 MB on 1024x4096: identical numbers, `results/kv_nw32_sustained_120w.md`). The table below
is the first pass: 300 ms warm-up + 50 back-to-back iterations. Its FP8 cells are partly *pre-limiter* -- FP8 times are bimodal (the
fastest iterations equal the INT8 median, the slow ones are 30 % longer) -- so the sustained table further down is the headline.

| N x K | M | cuBLASLt bf16 | cuBLASLt FP8 pt | cuBLASLt INT8 (s32 out) | cuBLASLt INT8+rescale | CUTLASS FP8 pt | CUTLASS INT8 pt | INT8/FP8 (CUTLASS) | INT8/cuBLASLt FP8 | INT8/bf16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096x4096 | 901 | 108 (281 us) | 217 (139 us) | 211 (143 us) | 169 (179 us) | 229 (132 us; cfg3) | 274 (110 us; cfg3) | 1.20 | 1.26 | 2.54 |
| 4096x4096 | 1517 | 136 (373 us) | 258 (197 us) | 276 (184 us) | 162 (313 us) | 269 (189 us; cfg3) | 355 (143 us; cfg3) | 1.32 | 1.38 | 2.60 |
| 4096x4096 | 1802 | 128 (471 us) | 209 (290 us) | 271 (223 us) | 155 (390 us) | 239 (253 us; cfg3) | 290 (209 us; cfg3) | 1.21 | 1.39 | 2.26 |
| 4096x4096 | 4096 | 123 (1117 us) | 212 (649 us) | 266 (517 us) | 151 (909 us) | 271 (508 us; cfg3) | 329 (418 us; cfg3) | 1.22 | 1.55 | 2.67 |
| 1024x4096 | 901 | 142 (53 us) | 246 (31 us) | 195 (39 us) | 161 (47 us) | 205 (37 us; cfg3) | 206 (37 us; cfg3) | 1.00 | 0.83 | 1.45 |
| 1024x4096 | 1517 | 156 (82 us) | 283 (45 us) | 239 (53 us) | 200 (64 us) | 259 (49 us; cfg3) | 249 (51 us; cfg3) | 0.96 | 0.88 | 1.60 |
| 1024x4096 | 1802 | 152 (99 us) | 295 (51 us) | 224 (67 us) | 180 (84 us) | 243 (62 us; cfg3) | 232 (65 us; cfg3) | 0.95 | 0.79 | 1.52 |
| 1024x4096 | 4096 | 137 (252 us) | 309 (111 us) | 210 (164 us) | 168 (205 us) | 260 (132 us; cfg3) | 304 (113 us; cfg3) | 1.17 | 0.98 | 2.23 |

Per-config detail (TFLOPS, same run), 4096x4096: M=901 fp8 c3 229 / c1 214 / c2 197 / c0 152 / c7 124, int8 c3 274 / c2 220 / c1 192 /
c0 168 / c7 109; M=1517 fp8 c3 269 / c1 236 / c2 226, int8 c3 355 / c1 251 / c2 224. 1024x4096: M=901 fp8 c3 205 / c1 169 / c2 161,
int8 c3 206 / c2 176 / c1 160. cfg3 (2SM 256x256x128, cluster 2x1) is the best configuration in every mode-B cell and in every
4096x4096 cell of mode A; on 1024x4096 in mode A cfg1/cfg2 win by 3-8 %, so a smaller-tile or stream-K instance may still pay off there.

### B (sustained). Same mode, 1.5 s warm-up so FP8 is in its power-limited steady state; cfg3 only; Gaussian operands; cuBLASLt autotune ranked in the same regime -- `results/full_modeB_sustained_20260914_084644.md`

| N x K | M | cuBLASLt bf16 | cuBLASLt FP8 pt | cuBLASLt INT8 (s32 out) | cuBLASLt INT8 + rescale | CUTLASS FP8 pt | CUTLASS INT8 pt | INT8/FP8 (CUTLASS) | INT8/cuBLASLt FP8 | INT8/bf16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096x4096 | 901 | 114 (266 us) | 231 (131 us) | 215 (141 us) | 173 (175 us) | 264 (115 us) | 274 (110 us) | 1.04 | 1.19 | 2.41 |
| 4096x4096 | 1517 | 132 (387 us) | 249 (204 us) | 278 (183 us) | 164 (311 us) | 204 (250 us) | 323 (158 us) | 1.58 | 1.30 | 2.45 |
| 4096x4096 | 1802 | 137 (442 us) | 237 (255 us) | 275 (220 us) | 156 (387 us) | 187 (324 us) | 290 (209 us) | 1.55 | 1.22 | 2.12 |
| 4096x4096 | 4096 | 148 (926 us) | 243 (566 us) | 264 (521 us) | 151 (909 us) | 204 (674 us) | 294 (467 us) | 1.44 | 1.21 | 1.98 |
| 1024x4096 | 901 | 142 (53 us) | 246 (31 us) | 194 (39 us) | 161 (47 us) | 205 (37 us) | 205 (37 us) | 1.00 | 0.83 | 1.44 |
| 1024x4096 | 1517 | 160 (80 us) | 282 (45 us) | 238 (53 us) | 197 (64 us) | 249 (51 us) | 248 (51 us) | 1.00 | 0.88 | 1.56 |
| 1024x4096 | 1802 | 172 (88 us) | 295 (51 us) | 224 (67 us) | 185 (82 us) | 232 (65 us) | 232 (65 us) | 1.00 | 0.79 | 1.35 |
| 1024x4096 | 4096 | 138 (250 us) | 221 (156 us) | 212 (162 us) | 167 (206 us) | 191 (180 us) | 302 (114 us) | 1.58 | 1.37 | 2.20 |

Observations (sustained mode B):
- q/o_proj (4096x4096): CUTLASS INT8 is 1.44-1.58x the *same-structure* CUTLASS FP8 (1.04x at M=901, where the weight stream leaves
  the tensor cores idle enough that FP8 is barely limited), 1.19-1.30x cuBLASLt FP8 per-tensor and 2.0-2.45x cuBLASLt bf16. cuBLASLt's
  FP8 kernel sustains better than the CUTLASS FP8 twin (249 vs 204 at M=1517): it is the more power-efficient FP8 kernel, so the
  INT8-vs-vendor-FP8 gain is 1.2-1.3x, not the 1.5x of the same-structure comparison.
- k/v_proj (1024x4096): a 4 MB weight + 3.7-17 MB activation problem of 16-64 output tiles on 10 CTA pairs; INT8 = FP8 at M <= 1802
  (memory-bound, no limiter), both 0.79-0.88x cuBLASLt FP8 (its autotuned kernel handles the 1.6-3 wave tail better; a stream-K /
  split-K instance is the obvious follow-up); at M=4096 INT8 is 1.58x CUTLASS FP8 and 1.37x cuBLASLt FP8.
- cuBLASLt INT8 without a scaled epilogue (s32 out + separate rescale) is 0.62-0.76x cuBLASLt FP8: the fused bf16 epilogue is what makes
  INT8 usable, exactly as on H100.

## FP8 tensor-core throughput on Thor is data/power dependent; INT8 is not (`results/fp8_power_probe_20260914.md`)

Same cfg3 kernel, 4096^3, everything L2-resident (`--flush=0 --nw=1`), GPU 1386 MHz and EMC 4266 MHz sampled every 50 ms during
every run (never below 1314 MHz):

| operands | FP8 TFLOPS | INT8 TFLOPS |
| --- | ---: | ---: |
| all zeros | 379-385 | 344-382 |
| uniform random (full range) | 217 (264 in a 30-iteration burst) | 342-368 |
| per-tensor-quantized Gaussian (absmax = 5 sigma) | 217 | 346 |

With zero operands both formats reach ~380 TOPS = 84 % of the 454 TOPS peak, so the `kind::f8f6f4` MMA is not intrinsically slower
than `kind::i8`. With real data FP8 loses ~43 % (385 -> 217) while INT8 loses ~10 % (382 -> 344), at identical clocks and with a higher
junction temperature for FP8 (80 vs 75 C). The consistent explanation is a power/current limiter on the tensor-core datapath, invisible in the devfreq clock
(confirmed with a real GPU-rail power reading under MAXN, next section). It is why sustained mode B shows INT8 at 1.44-1.58x FP8 on
q/o_proj while mode A -- whose 1.2 ms flush gaps let the limiter recover -- shows 1.00x. The chip also drifts: cold-mode INT8 4096^3 read
286 TFLOPS at Tj ~45 C and 266 at ~75 C late in the session, so compare within a run.

Time constant (`results/fp8_limiter_onset_20260914.md`: per-iteration times from an idle GPU, 4096^3 L2-resident, Gaussian operands):
after the ~20 ms clock ramp FP8 runs at 272 TFLOPS for ~0.5 s and then steps down to 219-228 for good; INT8 runs at 315-336 throughout.
A step at 0.5 s with unchanged GPU/EMC clocks and +10 C is a power controller with a ~0.5 s averaging window -- neither thermal
throttling (seconds) nor an instantaneous current limiter. Any FP8 benchmark shorter than ~0.5 s of sustained load (50 iterations after
a 300 ms warm-up, or any mode with >= 1 ms gaps between kernels) is therefore ~20-25 % optimistic; hence `--warmup_ms=1500` for the
headline table.

### MAXN check with real GPU power (`power_probe.sh`, `results/power_probe_maxn.md`, `results/power_probe_120w.md`)

`sudo nvpmodel -m 0` + `sudo jetson_clocks` (GPU pinned at 1575 MHz, EMC 4266 MHz) and `sudo tegrastats --interval 200 --logfile`
(as root the VDD_GPU rail reads real values; idle 4.4 W). Per case: TFLOPS and the mean/max VDD_GPU over the process lifetime.

| case (Gaussian operands) | FP8 TFLOPS @ GPU W | INT8 TFLOPS @ GPU W | INT8/FP8 | TOPS/W FP8 -> INT8 |
| --- | ---: | ---: | ---: | ---: |
| warm 4096^3, 3000 it | 246 @ 86-91 | 332 @ 89-91 | 1.35 | 2.8 -> 3.6 |
| warm 4096^3, zeros | 389 @ 58 | 392 @ 44 | 1.01 | 6.7 -> 9.0 |
| hot-A/cold-W 4096x4096 M=901 | 216 @ 70 | 314 @ 64 | 1.45 | 3.1 -> 4.9 |
| hot-A/cold-W 4096x4096 M=1517 | 253 @ 71 | 360 @ 66 | 1.43 | 3.5 -> 5.4 |
| hot-A/cold-W 4096x4096 M=1802 | 234 @ 69 | 328 @ 62 | 1.40 | 3.4 -> 5.3 |
| hot-A/cold-W 1024x4096 M=901 | 218 @ 68 | 218 @ 54 | 1.00 | 3.2 -> 4.1 |
| cold 4096x4096 M=1517 | 241 @ 61 | 243 @ 57 | 1.00 | 4.0 -> 4.2 |

With the clock pinned, FP8 still loses 26-31 % to INT8 on real data (INT8 1.35-1.45x faster) while both reach ~390 TOPS (76 % of the 516 TOPS peak at 1575 MHz)
on zeros: the GPU rail has a power limiter that throttles tensor-core issue without touching the clock, and FP8 e4m3 MMAs cost more
energy per op than s8 MMAs. MAXN barely helps INT8 (332 vs 343 TFLOPS in the 120 W mode: power-bound either way) and helps FP8 a
little (246 vs 217). The short 200-iteration rows above include ~1 s of process start-up in the power mean; the kernel-dominated
measurement (>= 3 s of back-to-back kernels per case, `results/power_probe_maxn_long.md`) is:

| case (Gaussian operands, MAXN) | FP8 W | INT8 W | FP8 us | INT8 us | FP8 mJ/GEMM | INT8 mJ/GEMM | INT8 energy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hot-A/cold-W 4096x4096 M=901 | 88.8 | 93.0 | 157.7 | 97.3 | 14.0 | 9.1 | -35 % |
| hot-A/cold-W 4096x4096 M=1517 | 103.8 | 94.5 | 221.2 | 145.2 | 23.0 | 13.7 | -40 % |
| hot-A/cold-W 4096x4096 M=1802 | 104.3 | 94.7 | 286.7 | 189.5 | 29.9 | 17.9 | -40 % |
| hot-A/cold-W 1024x4096 M=901 (memory-bound) | 91.7 | 71.1 | 34.8 | 34.7 | 3.2 | 2.5 | -23 % |
| hot-A/cold-W 1024x4096 M=1802 | 92.8 | 83.4 | 94.2 | 59.1 | 8.7 | 4.9 | -44 % |
| all cold 4096x4096 M=901 (memory-bound) | 44.4 | 34.7 | 169.0 | 173.1 | 7.5 | 6.0 | -20 % |
| warm 4096^3 | 103.9 | 94.7 | 558.0 | 472.0 | 58.0 | 44.7 | -23 % |
| cuBLASLt bf16, hot-A/cold-W 4096x4096 M=1517 | 95.2 | | 397.4 | | 37.8 | | (INT8 = 36 % of bf16) |

The same measurement back in the default 120 W mode (`results/power_probe_120w_long.md`, GPU cap 1386 MHz, DVFS active):

| case (Gaussian operands, 120 W mode) | FP8 W | INT8 W | FP8 us | INT8 us | FP8 mJ/GEMM | INT8 mJ/GEMM | INT8 energy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hot-A/cold-W 4096x4096 M=901 | 88.3 | 86.4 | 165.7 | 110.5 | 14.6 | 9.5 | -35 % |
| hot-A/cold-W 4096x4096 M=1517 | 95.2 | 93.1 | 249.8 | 155.7 | 23.8 | 14.5 | -39 % |
| hot-A/cold-W 4096x4096 M=1802 | 95.4 | 91.4 | 325.5 | 210.7 | 31.1 | 19.3 | -38 % |
| hot-A/cold-W 1024x4096 M=901 (memory-bound) | 89.2 | 63.1 | 36.9 | 36.7 | 3.3 | 2.3 | -29 % |
| hot-A/cold-W 1024x4096 M=1802 (memory-bound) | 89.2 | 72.8 | 65.3 | 65.4 | 5.8 | 4.8 | -18 % |
| all cold 4096x4096 M=901 (flush gaps, memory-bound) | 39.2 | 31.7 | 171.1 | 170.2 | 6.7 | 5.4 | -20 % |
| warm 4096^3 | 92.0 | 92.3 | 634.6 | 398.2 | 58.4 | 36.8 | -37 % |
| warm 4096^3, zeros | 56.2 | 41.3 | 397.2 | 399.3 | 22.3 | 16.5 | -26 % |
| cuBLASLt bf16 / FP8, hot-A/cold-W 4096x4096 M=1517 | 91.1 / 91.7 | | 366.5 / 197.6 | | 33.4 / 18.1 | | INT8 = 43 % / 80 % |

**What INT8 saves is energy per GEMM, not watts.** Compute-bound, both formats run into the same ~95-104 W GPU power cap and INT8
finishes 1.5-1.6x sooner: 35-40 % less energy per GEMM at unchanged power. Memory-bound (same time), INT8 draws 18-29 % less power
(120 W mode: 63 vs 89 W on k/v_proj M=901, 32 vs 39 W all-cold M=901). Equivalently 1.3-1.7x TOPS/W. The picture is the same in
both power modes; MAXN only raises FP8 by ~10 %. Caveat from the review: the all-cold mode idles the tensor cores for ~1 ms per
iteration (the 256 MB flush), so its FP8 == INT8 result is partly the limiter never engaging, not only DRAM-boundness.

## Conclusions (per-tensor alignment step)

1. **INT8 per-tensor is aligned with (and under sustained load ahead of) FP8 per-tensor on Thor.** With the identical CUTLASS structure
   (2SM 256x256x128 tcgen05 MMA, cluster 2x1x1, CLC scheduler, bf16 epilogue with the fused sa*sb scale) INT8 is 0.96-1.02x FP8 when
   the GEMMs are cold and separated by idle gaps (mode A) or memory-bound (k/v_proj at M <= 1802), and 1.44-1.58x FP8 in the sustained
   production-like loop on q/o_proj (1.04x at M=901), because the GPU power limiter throttles FP8 MMAs harder than INT8 MMAs (measured:
   same cap for both formats in both power modes (~86-95 W in the 120 W mode, ~89-104 W in MAXN), INT8 35-40 % less energy per GEMM; memory-bound: same time, 18-29 % less power). Against the
   vendor library (sustained, autotuned): INT8 CUTLASS = 1.19-1.30x cuBLASLt FP8 and 2.0-2.45x cuBLASLt bf16 on q/o_proj; on k/v_proj
   0.79-0.88x cuBLASLt FP8 at M <= 1802 (tail-wave problem) and 1.37x at M=4096. Mode A (cold, gapped): 0.93-1.13x cuBLASLt FP8.
2. **Thor-specific kernel rules**: cluster size <= 2 (4-CTA clusters strand 4 of 20 SMs, 8-CTA clusters 12); the biggest MMA tile
   (2SM 256x256) wins everywhere because L2->SMEM delivery (~1.5 TB/s) is the ceiling; rasterization swizzle only when the weight
   exceeds the 32 MB L2 (then it is 3x; with W <= 16 MB it is a no-op or slightly negative -- the auto rule encodes this); warm up
   >= 300 ms for the clock and >= 1.5 s for FP8's power limiter before timing; expect +/-8 % thermal drift.
3. **At the target M (901-1802) everything is a weight stream.** 4096x4096 M=901: 16 MB weight = 70 us at 235 GB/s, measured 110 us
   (INT8, mode B) / 169 us (all cold); bf16 streams twice the bytes, hence the 1.35-2.45x over cuBLASLt bf16 in the sustained run. Fusing more work per
   weight read (e.g. q/k/v as one 6144x4096 GEMM, gate+up as one) is worth more than any further mainloop tuning at these sizes.
4. cuBLASLt INT8 on Thor has no scaled bf16 epilogue; the int32 output plus a separate rescale costs 1.1-1.8x the GEMM itself (1.1-1.25x
   on 1024x4096, up to 1.75x on 4096x4096), so
   the CUTLASS kernel (or any fused-epilogue INT8 kernel) is required for INT8 to be usable at all -- same conclusion as on H100.

## Next steps (not done here)

- Per-row x per-col scaled variant (Sm90ColBroadcast / Sm90RowBroadcast EVT) and a torch extension binding, so the cosmos-framework
  per-token/per-channel path can call it; then the g64/g128 blockwise INT8 port of the SM100 blockwise collective (its builder accepts
  int8 but ties the scale type to the int32 accumulator, so the same decoupling as the SM90 port is needed).
- Stream-K / split-K instance for the 1024x4096 shape at M <= 1802 (1.6 waves of 256x256 tiles) to close the 10-20 % gap to cuBLASLt.
- Fused QKV / gate-up GEMMs to amortize the weight stream at small M.
- (done) MAXN re-run: the FP8 limiter is power-mode independent; see the MAXN section.
- Review follow-ups not done: separate the flush-gap and DRAM effects in mode A (smaller read-based flush + `--gap_us`); percentiles /
  bimodality flag in the timing output; stream-K instance for k/v_proj.
