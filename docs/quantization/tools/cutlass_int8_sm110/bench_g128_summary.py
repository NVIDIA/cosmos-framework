#!/usr/bin/env python3
"""Same-condition comparison of the g128 kernels against the baselines, with speedup ratios.
Conditions: activations L2-hot / 8 rotating cold weights (--flush=0 --nw=8), >= 1.5 s sustained warm-up, Gaussian operands for the
CUTLASS kernels (cuBLASLt fills uniform), 50 timed iterations, M padded to a multiple of 4 (g128 A-scale layout requirement).
Writes results/g128_summary_<stamp>.md and prints it."""
import subprocess, re, datetime, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
SHAPES = [(4096, 4096), (1024, 4096)]
MS = [904, 1520, 1804]
COMMON = ["--iters=50", "--warmup_ms=1500", "--flush=0", "--nw=8", "--verify=0"]
def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE)
    m = re.search(r"median_us=([0-9.]+).*?tflops=([0-9.]+)", p.stdout)
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)
ROWS = [  # (label, command builder)
    ("cuBLASLt bf16",                      lambda M,N,K: ["./cublaslt_bench", "--dtype=bf16", "--autotune=1"] + COMMON),
    ("cuBLASLt FP8 per-tensor",            lambda M,N,K: ["./cublaslt_bench", "--dtype=fp8", "--autotune=1"] + COMMON),
    ("CUTLASS FP8 per-tensor (256x256)",   lambda M,N,K: ["./pertensor_gemm", "--dtype=fp8", "--cfg=3", "--dist=normal"] + COMMON),
    ("CUTLASS INT8 per-tensor (256x256)",  lambda M,N,K: ["./pertensor_gemm", "--dtype=int8", "--cfg=3", "--dist=normal"] + COMMON),
    ("CUTLASS INT8 per-tensor (256x128)",  lambda M,N,K: ["./pertensor_gemm", "--dtype=int8", "--cfg=2", "--dist=normal"] + COMMON),
    ("FP8 g128 W-block (cfg8)",            lambda M,N,K: ["./blockwise_gemm", "--dtype=fp8", "--cfg=8", "--dist=normal"] + COMMON),
    ("FP8 g128 per-col (cfg9)",            lambda M,N,K: ["./blockwise_gemm", "--dtype=fp8", "--cfg=9", "--dist=normal"] + COMMON),
    ("INT8 g128 W-block (cfg8)",           lambda M,N,K: ["./blockwise_gemm", "--dtype=int8", "--cfg=8", "--dist=normal"] + COMMON),
    ("INT8 g128 per-col (cfg9)",           lambda M,N,K: ["./blockwise_gemm", "--dtype=int8", "--cfg=9", "--dist=normal"] + COMMON),
]
res = {}
for (N, K) in SHAPES:
    for M in MS:
        for label, mk in ROWS:
            res[(label, N, K, M)] = run(mk(M, N, K) + [f"--m={M}", f"--n={N}", f"--k={K}"])
            print(f"{label:36s} {N}x{K} M={M}: {res[(label,N,K,M)]}", file=sys.stderr, flush=True)
stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
L = [f"# g128 vs baselines, same conditions ({stamp})", "",
     "Thor, 120 W mode, activations L2-hot / weights cold (--flush=0 --nw=8), >= 1.5 s sustained warm-up, 50 iterations, K=4096, M padded to a multiple of 4. "
     "CUTLASS kernels use per-tensor-quantized Gaussian operands, cuBLASLt uniform. TFLOPS = 2MNK/median; INT8 counted as TFLOPS.", ""]
for (N, K) in SHAPES:
    L += [f"## {N}x{K} (N x K)", "", "| kernel | " + " | ".join(f"M={M} us (TFLOPS)" for M in MS) + " |", "|---|" + "---:|" * len(MS)]
    for label, _ in ROWS:
        cells = []
        for M in MS:
            us, tf = res[(label, N, K, M)]
            cells.append(f"{us:.0f} ({tf:.0f})" if us else "n/a")
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    L += ["", f"### Speedups (time ratio baseline / kernel; >1 = kernel faster)", "",
          "| kernel | vs | " + " | ".join(f"M={M}" for M in MS) + " |", "|---|---|" + "---:|" * len(MS)]
    for label in ["INT8 g128 W-block (cfg8)", "INT8 g128 per-col (cfg9)", "FP8 g128 W-block (cfg8)", "CUTLASS INT8 per-tensor (256x256)"]:
        for base in ["cuBLASLt bf16", "cuBLASLt FP8 per-tensor", "CUTLASS INT8 per-tensor (256x256)"]:
            if base == label: continue
            cells = []
            for M in MS:
                a, b = res[(base, N, K, M)][0], res[(label, N, K, M)][0]
                cells.append(f"{a / b:.2f}" if a and b else "n/a")
            L.append(f"| {label} | {base} | " + " | ".join(cells) + " |")
    L.append("")
out = os.path.join(HERE, "results", f"g128_summary_{stamp}.md")
open(out, "w").write("\n".join(L) + "\n"); print("\n".join(L)); print("wrote", out, file=sys.stderr)
