# kernels/deepgemm_int8 -- DeepGEMM SM90 "1D2D" FP8 kernel ported to INT8

Self-contained torch `cpp_extension` (no dependency on the DeepGEMM build; CUTLASS headers come from
`third_party/DeepGEMM/third-party/cutlass`). Upstream: DeepGEMM 2.8.0 @ 66081d4, MIT license (`LICENSE.DeepGEMM`).

    from kernels.deepgemm_int8 import int8_gemm_g128, prepare_sfa, sfa_from_kb_major, configs
    Y = int8_gemm_g128(A, W, sfa_kernel, sfb, cfg=None)      # bf16 [M, N]

* `A` int8 `[M, K]`, `W` int8 `[N, K]` (nn.Linear weight), both row-major (K-major). `K % 128 == 0`, `N % 128 == 0`, any `M`
  (no padding of A: TMA zero-fills out-of-range rows on load and clips the bf16 TMA store).
* `sfa_kernel` fp32 `[K/128, round_up(M, 4)]` contiguous = the plain per-(row, 128-K-block) scales `[M, K/128]` transposed
  (MN-major, as TMA wants: `sfa_kernel[kb, m] = sfa[m, kb]`) with the row stride padded to a multiple of 4 floats (16 B TMA
  stride rule). Build once with `prepare_sfa(sfa_plain)` or `sfa_from_kb_major(sfa_kb)` (the `[K/128, M]` layout of
  `kernels/cutlass_int8_bw`; identical when `M % 4 == 0`). Padding columns may hold anything.
* `sfb` fp32 `[N/128, K/128]` contiguous (128x128 weight block scales, K-major) -- used as-is.
* math: `Y[m,n] = sum_kb sfa[m,kb] * sfb[n/128,kb] * (int32 dot of A[m, kb-block] . W[n, kb-block])`; int32 accumulate inside a
  128-K block (`wgmma.mma_async.m64nNk32.s32.s8.s8`, exact), promoted `final += scale * float(acc)` in fp32 across blocks
  (exact conversion: |acc| <= 128*127*127 < 2^22), bf16 output.

## Files
* `gen_kernel.py` -> `include/deep_gemm/impls/sm90_int8_gemm_1d2d.cuh`: checked string edits of upstream
  `sm90_fp8_gemm_1d2d.cuh` (element type int8_t, `S8MMASelector`, int32 `accum`, promotion). Everything else (TMA warp,
  mbarrier pipeline, persistent scheduler with L2 swizzle, 2-CTA TMA multicast, STSM + TMA-store epilogue) is upstream code.
* `include/deep_gemm/mma/sm90.cuh`: upstream + `S8MMA`/`S8MMASelector` wrapping CUTLASS `MMA_64xNx32_S32S8S8_SS_TN`
  (N in {8,16,24,32,48,...,256}: the PTX-legal integer wgmma shapes). `ptx/wgmma.cuh`: + int32 `warpgroup_fence_operand`.
* other `include/deep_gemm/**`: verbatim copies (scheduler, TMA copy, ld/st, math, types, epilogue transform); `comm/barrier.cuh` trimmed.
* `launcher.cuh`: host side = `csrc/jit_kernels/impls/runtime_utils.hpp` TMA descriptors (A/B 128B swizzle, D swizzle by
  BLOCK_N*2 bytes, SFA no swizzle) + `heuristics/sm90.hpp` smem formula + `cudaLaunchKernelEx` with cluster dims.
* `gen.py`: CONFIGS table -> `cfg_XX.cu` (one instantiation per TU) + `configs.h`. Template knobs exposed: BLOCK_M, BLOCK_N,
  num_stages (None = max fitting 232448 B), num_tma_multicast (1/2), multicast on A (cluster 1x2) or B (2x1), compiled N/K
  (0 = dynamic); BLOCK_K = 128, swizzles derived, math threads = 128 (BLOCK_M <= 64) or 256.
* `__init__.py`: build (`load()`), layout helpers, `pick_config` heuristic (+ `BEST` table from the sweep), `int8_gemm_g128`.
* nvcc flags mirror deep_jit: -O3, --expt-relaxed-constexpr, --expt-extended-lambda, --ptxas-options=--register-usage-level=10,
  no fast-math, sm_90a. `DGINT8_PROMOTE_MODE=1` builds the magic-number (IADD+FADD) promotion variant instead of cvt (I2FP).

Probes: `probes/dgint8_smoke.py` (correctness vs fp64 + vs cutlass_int8_bw), `probes/dgint8_sweep.py` (per-config timing),
`probes/bench_deepgemm_int8.py` (final table -> `results/deepgemm_int8_bench.{md,json}`).

## Performance notes (H100 SXM, see results/deepgemm_int8_bench.md)
* Best exact variant: mode 0 promotion, 256x128 tiles, 3 stages, cluster 1x2 (A multicast), N/K compiled in: 1060-1116 TFLOPS at
  M >= 4096 (0.81-0.86x upstream DeepGEMM FP8, 1.07-1.32x kernels/cutlass_int8_bw); M=901: 793-953 TFLOPS (N >= 4096), 425 (N=1024, 64x128 tile).
* Bottleneck = the per-element int32->fp32 conversion (I2FP) in the promotion: the same pipeline without it (DGINT8_PROMOTE_MODE=2,
  wrong numerics) reaches 1405-1527 TFLOPS. Mode 1 (magic-number IADD+FADD) is exact but slower (970-1027). `-DDG_INT8_OVERLAP=1`
  (double-buffered accumulators, BLOCK_M <= 128) is exact but slower: ptxas serializes the wgmma pipeline (C7514) when the
  ping-pong buffer is read while the next k-block's WGMMAs are in flight. Both knobs are kept for experiments; defaults are the fast exact path.
* Dynamic-shape 256x128 kernels spill (2.4 KB for cluster 1x1) -> always prefer the shape-compiled configs (ids 8-16) for the 4 production (N,K).
