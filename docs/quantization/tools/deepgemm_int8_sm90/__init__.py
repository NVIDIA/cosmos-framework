"""DeepGEMM SM90 FP8 "1D2D" kernel ported to INT8 (torch cpp_extension build).

    int8_gemm_g128(A int8 [M,K], W int8 [N,K], sfa, sfb) -> bf16 [M,N]
    Y[m, n] = sum_kb sfa[m, kb] * sfb[n // 128, kb] * (int32 dot of A[m, kb-block] . W[n, kb-block])   (fp32 across blocks)

Scale layouts the kernel wants (same as DeepGEMM's SM90 FP8 path, see kernels/deepgemm_wrapper.py):
  * sfa: fp32 [K//128, Mp] contiguous with Mp = round_up(M, 4) (TMA needs a 16-byte row stride), i.e. MN-major /
    transposed: sfa_kernel[kb, m] = sfa_plain[m, kb]; columns M..Mp-1 are padding (any value; the matching A rows are never
    stored). `prepare_sfa(sfa_plain [M, K//128])` builds it; `sfa_from_kb_major(sfa [K//128, M])` accepts the layout used by
    kernels/cutlass_int8_bw (identical when M % 4 == 0, otherwise zero-padded columns are appended).
  * sfb: fp32 [N//128, K//128] contiguous (plain 128x128 block scales, K-major) -- used as-is.
A itself needs NO padding: TMA zero-fills rows >= M on load and clips the bf16 TMA store, so M=901 runs directly.
Requirements: K % 128 == 0, N % 128 == 0, A/W row-major contiguous (K-major), any M.
"""
import os, glob, torch

_HERE = os.path.dirname(os.path.abspath(__file__)); _BENCH = os.path.dirname(os.path.dirname(_HERE))
CUTLASS = os.environ.get("CUTLASS_DIR", os.path.join(_BENCH, "third_party", "DeepGEMM", "third-party", "cutlass"))
BUILD_DIR = os.environ.get("DGINT8_BUILD_DIR", os.path.join(_BENCH, "third_party", "deepgemm_int8_build"))
PROMOTE_MODE = int(os.environ.get("DGINT8_PROMOTE_MODE", "0"))   # 0: cvt (I2FP), 1: magic-number IADD+FADD, 2: EXPERIMENT no-convert (wrong numerics)
INCLUDE_DIR = os.environ.get("DGINT8_INCLUDE_DIR", os.path.join(_HERE, "include"))   # override for kernel-header experiments
_ext = None

# nvcc flags mirror DeepGEMM's deep_jit defaults (-O3, --expt-relaxed-constexpr, --expt-extended-lambda,
# --ptxas-options=--register-usage-level=10, no fast-math) plus its diag-suppress list; sm_90a only.
NVCC_FLAGS = ["-O3", "-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda",
              "-gencode", "arch=compute_90a,code=sm_90a",
              "--ptxas-options=--register-usage-level=10", "--ptxas-options=--verbose",
              "--diag-suppress=39,161,174,177,186,940", "-Xcompiler", "-Wno-psabi", "-Xcompiler", "-Wno-deprecated-declarations",
              f"-DDG_INT8_PROMOTE_MODE={PROMOTE_MODE}"] + os.environ.get("DGINT8_EXTRA_NVCC_FLAGS", "").split()

def load(verbose=False):
    global _ext
    if _ext is not None:
        return _ext
    from torch.utils.cpp_extension import load as _load
    bd = BUILD_DIR if PROMOTE_MODE == 0 else BUILD_DIR + f"_pm{PROMOTE_MODE}"
    os.makedirs(bd, exist_ok=True); os.environ.setdefault("MAX_JOBS", "8"); os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
    _ext = _load(name="deepgemm_int8_ext", sources=[os.path.join(_HERE, "bindings.cpp")] + sorted(glob.glob(os.path.join(_HERE, "cfg_*.cu"))),
                 build_directory=bd, verbose=verbose,
                 extra_include_paths=[os.path.join(CUTLASS, "include"), os.path.join(CUTLASS, "tools", "util", "include"),
                                      INCLUDE_DIR, _HERE],
                 extra_cflags=["-O3", "-std=c++17"], extra_cuda_cflags=NVCC_FLAGS)
    return _ext

def configs():
    """list of dicts describing the compiled kernel instantiations (index = cfg id)."""
    ext = load(); names = ext.config_names()
    keys = ("block_m", "block_n", "stages", "mcast", "on_a", "shape_n", "shape_k", "math_threads")
    return [dict(zip(keys, c), name=n, id=i) for i, (c, n) in enumerate(zip(ext.configs(), names))]

# ---------------------------------------------------------------- scale layout helpers (run once, outside timing)
def sfa_ld(M):
    return (M + 3) // 4 * 4

def prepare_sfa(sfa_plain: torch.Tensor) -> torch.Tensor:
    """fp32 [M, K//128] (one scale per (row, 128-K block)) -> kernel layout fp32 [K//128, round_up(M, 4)] contiguous."""
    assert sfa_plain.dtype == torch.float32 and sfa_plain.dim() == 2
    M, G = sfa_plain.shape; Mp = sfa_ld(M)
    out = torch.zeros((G, Mp), dtype=torch.float32, device=sfa_plain.device)
    out[:, :M] = sfa_plain.t()
    return out

def sfa_from_kb_major(sfa_kb: torch.Tensor) -> torch.Tensor:
    """fp32 [K//128, M] (kernels/cutlass_int8_bw layout) -> kernel layout [K//128, round_up(M, 4)] (a no-op view when M % 4 == 0)."""
    assert sfa_kb.dtype == torch.float32 and sfa_kb.dim() == 2
    G, M = sfa_kb.shape; Mp = sfa_ld(M)
    if Mp == M:
        return sfa_kb.contiguous()
    return torch.cat([sfa_kb, sfa_kb.new_zeros(G, Mp - M)], dim=1).contiguous()

def prepare_sfb(sfb: torch.Tensor, N: int, K: int) -> torch.Tensor:
    """fp32 [N//128, K//128] -> contiguous (no transform). Also accepts block-constant per-channel [N, K//128]."""
    G = K // 128
    if sfb.shape == (N // 128, G):
        return sfb.contiguous()
    if sfb.shape == (N, G):
        blk = sfb.view(N // 128, 128, G)
        assert torch.equal(blk, blk[:, :1, :].expand_as(blk)), "per-channel W scales are not constant inside 128-row blocks"
        return blk[:, 0, :].contiguous()
    raise ValueError(f"sfb shape {tuple(sfb.shape)} incompatible with N={N}, K={K}")

# ---------------------------------------------------------------- config heuristic (DeepGEMM's picks for these shapes, refined by the sweep)
NUM_SMS = 132

def _find(bm, bn, mc, on_a=True):
    for c in configs():
        if (c["block_m"], c["block_n"], c["mcast"], bool(c["on_a"])) == (bm, bn, mc, on_a) and c["shape_n"] == 0 and c["shape_k"] == 0:
            return c["id"]
    return None

# best config measured per (N, K, M) by probes/dgint8_sweep.py; filled in after the sweep (see results/deepgemm_int8_bench.md)
BEST = {  # (N, K, M): cfg id -- measured best in results/deepgemm_int8_sweep.json (probes/dgint8_sweep.py, r2 build)
    (1024, 4096, 901): 6,   # 64x128 s8 c1x1 (423 TFLOPS)
    (1024, 4096, 4096): 9,   # 256x128 s3 c1x2 N1024K4096 (893 TFLOPS)
    (1024, 4096, 42240): 9,   # 256x128 s3 c1x2 N1024K4096 (1115 TFLOPS)
    (4096, 4096, 901): 8,   # 256x128 s3 c1x2 N4096K4096 (792 TFLOPS)
    (4096, 4096, 4096): 8,   # 256x128 s3 c1x2 N4096K4096 (1060 TFLOPS)
    (4096, 4096, 42240): 8,   # 256x128 s3 c1x2 N4096K4096 (1103 TFLOPS)
    (4096, 12288, 901): 11,   # 256x128 s3 c1x2 N4096K12288 (939 TFLOPS)
    (4096, 12288, 4096): 11,   # 256x128 s3 c1x2 N4096K12288 (1116 TFLOPS)
    (4096, 12288, 42240): 11,   # 256x128 s3 c1x2 N4096K12288 (1106 TFLOPS)
    (12288, 4096, 901): 10,   # 256x128 s3 c1x2 N12288K4096 (919 TFLOPS)
    (12288, 4096, 4096): 10,   # 256x128 s3 c1x2 N12288K4096 (1081 TFLOPS)
    (12288, 4096, 42240): 10,   # 256x128 s3 c1x2 N12288K4096 (1078 TFLOPS)
}

def pick_config(M, N, K):
    key = (N, K, M)
    if key in BEST:
        cid = BEST[key]
        if cid is not None and cid < len(configs()):
            return cid
    # DeepGEMM-heuristic fallback: 256x128 tiles; multicast only if the grid has more than one wave; 128x64 for tiny M x N
    if M <= 1024 and N <= 1024:
        return _find(128, 64, 1)
    blocks = -(-M // 256) * -(-N // 128)
    cid = _find(256, 128, 2) if blocks > NUM_SMS else _find(256, 128, 1)
    return cid if cid is not None else 0

# ---------------------------------------------------------------- main entry point
def int8_gemm_g128(A: torch.Tensor, W: torch.Tensor, sfa: torch.Tensor, sfb: torch.Tensor, out: torch.Tensor = None, cfg: int = None) -> torch.Tensor:
    """sfa must already be in kernel layout [K//128, round_up(M,4)] (prepare_sfa / sfa_from_kb_major); sfb [N//128, K//128]."""
    ext = load()
    M, K = A.shape; N = W.shape[0]
    if cfg is None:
        cfg = pick_config(M, N, K)
    return ext.int8_gemm_g128(A, W, sfa, sfb, out, cfg)

__all__ = ["load", "configs", "int8_gemm_g128", "prepare_sfa", "sfa_from_kb_major", "prepare_sfb", "pick_config", "sfa_ld", "BEST"]
