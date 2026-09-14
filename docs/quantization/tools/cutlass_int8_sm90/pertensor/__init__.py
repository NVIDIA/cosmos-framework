"""Build/load the CUTLASS SM90 INT8 GEMM extension. Usage: from kernels.cutlass_int8 import load; ext = load(); ext.int8_scaled_mm(A, W, sa, sb, cfg)"""
import os, glob
_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.dirname(os.path.dirname(_HERE))
CUTLASS = os.environ.get("CUTLASS_DIR")  # path to a CUTLASS >= 4.2 checkout (headers only)
if not CUTLASS: raise RuntimeError("set CUTLASS_DIR to a CUTLASS 4.2+ checkout")
BUILD_DIR = os.environ.get("INT8SM90_BUILD_DIR", os.path.join(_HERE, "build"))
_ext = None
def load(verbose=False):
    global _ext
    if _ext is not None: return _ext
    from torch.utils.cpp_extension import load as _load
    os.makedirs(BUILD_DIR, exist_ok=True)
    os.environ.setdefault("MAX_JOBS", "8")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
    sources = [os.path.join(_HERE, "bindings.cpp")] + sorted(glob.glob(os.path.join(_HERE, "cfg_*.cu")))
    _ext = _load(name="int8sm90_ext", sources=sources, build_directory=BUILD_DIR, verbose=verbose,
                 extra_include_paths=[os.path.join(CUTLASS, "include"), os.path.join(CUTLASS, "tools", "util", "include"), _HERE],
                 extra_cflags=["-O3", "-std=c++17"],
                 extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-DNDEBUG",
                                    "-gencode", "arch=compute_90a,code=sm_90a", "--use_fast_math", "-Xcompiler", "-Wno-psabi"])
    return _ext
