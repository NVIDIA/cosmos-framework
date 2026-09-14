"""Build + verify + time the CUTLASS SM90 INT8 (1,128,128) block-scaled GEMM against bf16 / FP8 per-tensor / CUTLASS INT8 per-tensor / DeepGEMM FP8 g128."""
import sys, time, subprocess, torch
from triton.testing import do_bench
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from blockwise import load as load_bw, int8_blockwise_mm
from pertensor import load as load_pt
t0 = time.time(); ext_bw = load_bw(verbose=False); print(f"blockwise ext build/load: {time.time()-t0:.0f} s, mainloop stages = {ext_bw.stages()}")
ext_pt = load_pt()
try:
    # optional DeepGEMM column: needs users/pzeren/int8_bench/kernels/deepgemm_wrapper.py (+ a built DeepGEMM) on sys.path
    from deepgemm_wrapper import deepgemm_fp8_g128, quantize_activation_g128, quantize_weight_128x128
    HAVE_DG = True
except Exception as e:
    print("DeepGEMM unavailable:", type(e).__name__, str(e)[:120]); HAVE_DG = False
dev = "cuda"
def clk(): return subprocess.run(["nvidia-smi","--query-gpu=clocks.sm,power.draw","--format=csv,noheader,nounits"],capture_output=True,text=True).stdout.strip()
def tf(fn, M, N, K, rep=200):
    t = do_bench(fn, warmup=25, rep=rep, return_mode="median"); return t*1e3, 2*M*N*K/(t*1e-3)/1e12
def ref_blockwise(Aq, Wq, sfa, sfb):
    M, K = Aq.shape; N = Wq.shape[0]; nb = K // 128
    sfb_full = sfb.repeat_interleave(128, dim=0)            # [N, K/128]
    Y = torch.zeros(M, N, dtype=torch.float64, device=dev)
    for kb in range(nb):
        d = Aq[:, kb*128:(kb+1)*128].double() @ Wq[:, kb*128:(kb+1)*128].double().t()
        Y += d * sfa[kb].double()[:, None] * sfb_full[:, kb].double()[None, :]
    return Y
torch.manual_seed(0)
print("\n== correctness (rel-L2 vs fp64 exact block-scaled math; pass < 3e-3 = bf16 output rounding) ==")
for (N, K) in [(1024, 4096), (4096, 12288)]:
    for M in (901, 1802, 4096):
        Aq = torch.randint(-127, 128, (M, K), device=dev, dtype=torch.int8); Wq = torch.randint(-127, 128, (N, K), device=dev, dtype=torch.int8)
        sfa = torch.rand(K // 128, M, device=dev) * 9e-3 + 1e-3; sfb = torch.rand(N // 128, K // 128, device=dev) * 9e-3 + 1e-3
        out = int8_blockwise_mm(Aq, Wq, sfa, sfb).double(); ref = ref_blockwise(Aq, Wq, sfa, sfb)
        err = ((out - ref).norm() / ref.norm()).item(); mx = ((out - ref).abs().max() / ref.abs().max()).item()
        print(f"  M={M:5d} N={N:5d} K={K:5d}: rel-L2 {err:.2e}  max|d|/max|ref| {mx:.2e}  {'OK' if err < 3e-3 else 'FAIL'}")
print("\n== timing: TFLOPS (do_bench median, L2 flushed) ==")
shapes = [(4096, 4096), (1024, 4096), (12288, 4096), (4096, 12288)]
Ms = [901, 4096, 42240] if "--quick" not in sys.argv else [4096]
for (N, K) in shapes:
    for M in Ms:
        A = torch.randn(M, K, device=dev, dtype=torch.bfloat16); W = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
        Aq = torch.randint(-127, 128, (M, K), device=dev, dtype=torch.int8); Wq = torch.randint(-127, 128, (N, K), device=dev, dtype=torch.int8)
        Af = A.to(torch.float8_e4m3fn); Wf = W.to(torch.float8_e4m3fn); one = torch.ones((), device=dev)
        sa = torch.full((M,), 1e-3, device=dev); sb = torch.full((N,), 1e-3, device=dev)
        Mp = (M + 3) // 4 * 4; Aq_p = torch.cat([Aq, Aq.new_zeros(Mp - M, K)], 0) if Mp != M else Aq   # pre-padded so timing excludes the pad copy
        sfa = torch.rand(K // 128, Mp, device=dev) * 9e-3 + 1e-3; sfb = torch.rand(N // 128, K // 128, device=dev) * 9e-3 + 1e-3
        rows = {"bf16 cuBLAS": lambda: A @ W.t(),
                "FP8 per-tensor cuBLASLt": lambda: torch._scaled_mm(Af, Wf.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16, use_fast_accum=True),
                "CUTLASS INT8 per-tensor (128x256 c2x1)": lambda: ext_pt.int8_scaled_mm(Aq, Wq, sa, sb, 0),
                "CUTLASS INT8 g128 blockwise (128x128 c1x2)": lambda: ext_bw.int8_blockwise_mm(Aq_p, Wq, sfa, sfb)}
        if HAVE_DG:
            try:
                Af8, As = quantize_activation_g128(A if Mp == M else torch.cat([A, A.new_zeros(Mp - M, K)], 0)); Wf8, Ws = quantize_weight_128x128(W)
                deepgemm_fp8_g128(Af8, As, Wf8, Ws); rows["DeepGEMM FP8 g128"] = lambda: deepgemm_fp8_g128(Af8, As, Wf8, Ws)
            except Exception as e:
                print("  (DeepGEMM skipped:", type(e).__name__, str(e)[:100], ")")
        print(f"\n=== N={N} K={K} M={M} ===")
        for name, fn in rows.items():
            try: us, tflops = tf(fn, M, N, K)
            except Exception as e: print(f"  {name:44s} ERROR {type(e).__name__}: {str(e)[:120]}"); continue
            print(f"  {name:44s} {us:9.1f} us {tflops:7.0f} TFLOPS   [{clk()}]")
