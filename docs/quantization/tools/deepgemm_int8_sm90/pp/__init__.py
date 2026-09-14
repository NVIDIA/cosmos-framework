"""Math-warpgroup ping-pong variant of the DeepGEMM INT8 port (see ../gen_kernel_pp.py): isolated torch extension `deepgemm_int8_pp_ext`
with its own config table (pp/gen_pp.py -> pp_configs.h, ppcfg_XX.cu) and build dir, so the default kernel / cfg ids stay intact.
Only `sm90_int8_gemm_1d2d_pp_impl` instantiations are compiled here, so it can be loaded next to the main extension in one process.

    from kernels.deepgemm_int8.pp import load, configs, int8_gemm_g128_pp
    Y = int8_gemm_g128_pp(A, W, sfa_kernel, sfb, cfg=0)     # same arguments / layouts as kernels.deepgemm_int8.int8_gemm_g128
"""
import os, glob, torch
from .. import NVCC_FLAGS, CUTLASS, INCLUDE_DIR, PROMOTE_MODE

_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE); _BENCH = os.path.dirname(os.path.dirname(_PKG))
BUILD_DIR = os.environ.get("DGINT8_PP_BUILD_DIR", os.path.join(_BENCH, "third_party", "deepgemm_int8_pp_build"))
_ext = None

def load(verbose=False):
    global _ext
    if _ext is not None:
        return _ext
    from torch.utils.cpp_extension import load as _load
    bd = BUILD_DIR if PROMOTE_MODE == 0 else BUILD_DIR + f"_pm{PROMOTE_MODE}"
    os.makedirs(bd, exist_ok=True); os.environ.setdefault("MAX_JOBS", "8"); os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
    _ext = _load(name="deepgemm_int8_pp_ext", sources=[os.path.join(_HERE, "bindings_pp.cpp")] + sorted(glob.glob(os.path.join(_HERE, "ppcfg_*.cu"))),
                 build_directory=bd, verbose=verbose,
                 extra_include_paths=[os.path.join(CUTLASS, "include"), os.path.join(CUTLASS, "tools", "util", "include"), INCLUDE_DIR, _PKG, _HERE],
                 extra_cflags=["-O3", "-std=c++17"], extra_cuda_cflags=NVCC_FLAGS)
    return _ext

def configs():
    ext = load(); names = ext.config_names()
    keys = ("block_m", "block_n", "stages", "mcast", "on_a", "shape_n", "shape_k", "math_threads", "pp_mode")
    return [dict(zip(keys, c), name=n, id=i) for i, (c, n) in enumerate(zip(ext.configs(), names))]

def int8_gemm_g128_pp(A, W, sfa, sfb, out=None, cfg=0):
    """sfa in kernel layout [K//128, round_up(M,4)] (kernels.deepgemm_int8.prepare_sfa / sfa_from_kb_major); sfb [N//128, K//128]."""
    return load().int8_gemm_g128_pp(A, W, sfa, sfb, out, cfg)

__all__ = ["load", "configs", "int8_gemm_g128_pp", "BUILD_DIR"]
