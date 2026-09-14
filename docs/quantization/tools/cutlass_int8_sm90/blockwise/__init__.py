"""CUTLASS SM90 INT8 GEMM with (1,128,128) block scaling in the mainloop (torch cpp_extension build).
int8_blockwise_mm(A[M,K] int8, W[N,K] int8, sfa[K//128, M] fp32, sfb[N//128, K//128] fp32) -> bf16 [M,N]; pads M to a multiple of 4."""
import os, glob, torch
_HERE = os.path.dirname(os.path.abspath(__file__)); _BENCH = os.path.dirname(os.path.dirname(_HERE))
CUTLASS = os.environ.get("CUTLASS_DIR")  # path to a CUTLASS >= 4.2 checkout (headers only)
if not CUTLASS: raise RuntimeError("set CUTLASS_DIR to a CUTLASS 4.2+ checkout")
BUILD_DIR = os.environ.get("INT8BW_BUILD_DIR", os.path.join(_HERE, "build"))
_ext = None
def load(verbose=False):
    global _ext
    if _ext is not None: return _ext
    from torch.utils.cpp_extension import load as _load
    os.makedirs(BUILD_DIR, exist_ok=True); os.environ.setdefault("MAX_JOBS", "8"); os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
    _ext = _load(name="int8sm90_bw_ext", sources=[os.path.join(_HERE, "bindings.cpp")] + sorted(glob.glob(os.path.join(_HERE, "cfg_*.cu"))),
                 build_directory=BUILD_DIR, verbose=verbose,
                 extra_include_paths=[os.path.join(CUTLASS, "include"), os.path.join(CUTLASS, "tools", "util", "include"), _HERE],
                 extra_cflags=["-O3", "-std=c++17"],
                 extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-DNDEBUG",
                                    "-gencode", "arch=compute_90a,code=sm_90a", "--use_fast_math", "-Xcompiler", "-Wno-psabi", "-Xptxas", "-v"])
    return _ext
def int8_blockwise_mm(A, W, sfa, sfb):
    """sfa: [K//128, M] (scale of row m, k-block kb at sfa[kb, m]); sfb: [N//128, K//128]."""
    ext = load(); M = A.shape[0]; Mp = (M + 3) // 4 * 4
    if Mp != M:
        A = torch.cat([A, A.new_zeros(Mp - M, A.shape[1])], 0); sfa = torch.cat([sfa, sfa.new_zeros(sfa.shape[0], Mp - M)], 1)
    out = ext.int8_blockwise_mm(A, W, sfa.contiguous(), sfb.contiguous())
    return out[:M] if Mp != M else out
