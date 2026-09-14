# CUTLASS SM90 INT8 GEMM kernels (H100), 2026-09-14

Context: handoff section 9. cuBLASLt has no Hopper-native INT8 kernel (`torch._int_mm` runs a CUTLASS 2.x sm80 mma.sync kernel), so these
two kernels are built from CUTLASS 3.x (headers from a CUTLASS >= 4.2 checkout, `export CUTLASS_DIR=...`) via `torch.utils.cpp_extension`
(nvcc 13.x, `sm_90a`, build ~1 min each). Both take A[M,K] int8 row-major and W[N,K] int8 row-major (= nn.Linear weight), output bf16.

- `pertensor/`: `int8_scaled_mm(A, W, sa[M], sb[N], cfg=0) = bf16(A@W^T * sa[:,None] * sb[None,:])`. CollectiveBuilder int8->int32,
  TMA warp-specialized cooperative, 128x256x128 tile, cluster 2x1, Sm90 EVT epilogue. Matches cuBLASLt FP8 per-tensor speed at M>=4096
  (1400-1540 TFLOPS); weak at M=901 with N=1024 (needs a 64x128 pingpong instance, see `configs.txt` for the 7 configs tried).
- `blockwise/`: `int8_blockwise_mm(A, W, sfa[K/128, M], sfb[N/128, K/128])` -- DeepSeek-style (1,128,128) block scaling applied INSIDE the
  mainloop: int32 wgmma accumulation per 128-K block, then fp32 promotion with `sfa*sfb` (exact, |sum| < 2^24). Ported from CUTLASS's FP8
  blockwise collective by `gen_int8_bw.py` (regenerates `sm90_mma_int8_blockwise.cuh` from `$CUTLASS_DIR`). ~1000 TFLOPS = 1.35x bf16,
  0.70x per-tensor. M is padded to a multiple of 4 (TMA-loaded A scales). 128x128x128 tile, cluster 1x2, 6 stages.

Build + test + benchmark (inside a container with torch >= 2.10, triton, nvcc):
```bash
export CUTLASS_DIR=/path/to/cutlass   # e.g. DeepGEMM/third-party/cutlass @ 4.2.1
python bench_cutlass_int8.py --quick      # per-tensor: correctness vs torch._int_mm reference + timing
python bench_cutlass_int8_bw.py --quick   # blockwise: correctness vs fp64 exact block-scaled math + timing (DeepGEMM column optional)
python why_int8_slow.py; python kernel_names.py   # the cuBLASLt diagnosis (kernel names, K sweep, clocks)
```

Pitfall that cost an hour: a custom mainloop `DispatchPolicy` for a block-scaled collective MUST specialize
`cutlass::gemm::kernel::detail::HasAuxiliaryLoad<MyPolicy<...>> : cute::true_type`. The SM90 warp-specialized kernels only run the extra
`MainloopAux` producer warp (which cp.async-loads the scales and arrives on the mainloop barrier) when that trait is true; CUTLASS specializes
it only for its own FP8 blockwise policy. Without it the kernel hangs silently. Also: the CUTLASS 4.x scheduler tag is `PersistentScheduler`.
