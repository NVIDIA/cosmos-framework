"""Wave-interleave experiment of the DeepGEMM INT8 port (see ../gen_kernel_wi.py): isolated torch extension `deepgemm_int8_wi_ext`
with its own config table (wi/gen_wi.py -> wi_configs.h, wicfg_XX.cu) and build dir, so the default kernel / cfg ids stay intact.

    from kernels.deepgemm_int8.wi import load, configs, int8_gemm_g128_wi
    Y = int8_gemm_g128_wi(A, W, sfa_kernel, sfb, cfg=6)     # same arguments / layouts as kernels.deepgemm_int8.int8_gemm_g128

cfg ids: 0-5 reference instantiations of the unchanged base kernel (== main cfg 8, 11, 10, 9, 12, 0), 6-10 & 12 wave mode 2
(quarter-wave interleave), 11 wave mode 1 (literal two-wave interleave, spills). Build flags identical to the main extension.
"""
import os, glob, torch
from .. import NVCC_FLAGS, CUTLASS, INCLUDE_DIR, PROMOTE_MODE

_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE); _BENCH = os.path.dirname(os.path.dirname(_PKG))
BUILD_DIR = os.environ.get("DGINT8_WI_BUILD_DIR", os.path.join(_BENCH, "third_party", "deepgemm_int8_wi_build"))
_ext = None

def load(verbose=False):
    global _ext
    if _ext is not None:
        return _ext
    from torch.utils.cpp_extension import load as _load
    bd = BUILD_DIR if PROMOTE_MODE == 0 else BUILD_DIR + f"_pm{PROMOTE_MODE}"
    os.makedirs(bd, exist_ok=True); os.environ.setdefault("MAX_JOBS", "8"); os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
    _ext = _load(name="deepgemm_int8_wi_ext", sources=[os.path.join(_HERE, "bindings_wi.cpp")] + sorted(glob.glob(os.path.join(_HERE, "wicfg_*.cu"))),
                 build_directory=bd, verbose=verbose,
                 extra_include_paths=[os.path.join(CUTLASS, "include"), os.path.join(CUTLASS, "tools", "util", "include"), INCLUDE_DIR, _PKG, _HERE],
                 extra_cflags=["-O3", "-std=c++17"], extra_cuda_cflags=NVCC_FLAGS)
    return _ext

def configs():
    ext = load(); names = ext.config_names()
    keys = ("block_m", "block_n", "stages", "mcast", "on_a", "shape_n", "shape_k", "math_threads", "wave_mode")
    return [dict(zip(keys, c), name=n, id=i) for i, (c, n) in enumerate(zip(ext.configs(), names))]

def int8_gemm_g128_wi(A, W, sfa, sfb, out=None, cfg=6):
    """sfa in kernel layout [K//128, round_up(M,4)] (kernels.deepgemm_int8.prepare_sfa / sfa_from_kb_major); sfb [N//128, K//128]."""
    return load().int8_gemm_g128_wi(A, W, sfa, sfb, out, cfg)

# best reference / new cfg per compiled (N, K): (ref id, wi2 id)
PAIRS = {(4096, 4096): (0, 6), (4096, 12288): (1, 7), (12288, 4096): (2, 8), (1024, 4096): (3, 9)}

__all__ = ["load", "configs", "int8_gemm_g128_wi", "PAIRS", "BUILD_DIR"]
