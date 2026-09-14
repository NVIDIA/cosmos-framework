"""Build the CUTLASS SM90 INT8 extension, check correctness vs torch._int_mm, then time every config against
cuBLASLt FP8 per-tensor / INT8 _int_mm / bf16 on the Nano gen-tower shapes."""
import sys, time, subprocess, torch
from triton.testing import do_bench
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from pertensor import load
t0 = time.time(); ext = load(verbose=False); print(f"build/load: {time.time()-t0:.0f} s; configs:")
for i, d in enumerate(ext.configs()): print(f"  cfg{i}: {d}")
dev = "cuda"
def clk(): return subprocess.run(["nvidia-smi","--query-gpu=clocks.sm,power.draw","--format=csv,noheader,nounits"],capture_output=True,text=True).stdout.strip()
def tf(fn, M, N, K, rep=200):
    t = do_bench(fn, warmup=25, rep=rep, return_mode="median")  # ms
    return t*1e3, 2*M*N*K/(t*1e-3)/1e12
shapes = [(4096,4096),(1024,4096),(12288,4096),(4096,12288)]
Ms = [901, 4096, 42240] if "--quick" not in sys.argv else [4096]
torch.manual_seed(0)
# correctness first (all configs, one awkward shape)
for (N,K) in [(1024,4096),(4096,12288)]:
    M = 901
    Aq = torch.randint(-127,128,(M,K),device=dev,dtype=torch.int8); Wq = torch.randint(-127,128,(N,K),device=dev,dtype=torch.int8)
    sa = torch.rand(M,device=dev)*1e-2+1e-3; sb = torch.rand(N,device=dev)*1e-2+1e-3
    ref = (torch._int_mm(Aq, Wq.t()).double() * sa.double()[:,None] * sb.double()[None,:])
    for cfg in range(len(ext.configs())):
        try:
            out = ext.int8_scaled_mm(Aq, Wq, sa, sb, cfg).double()
            err = ((out-ref).norm()/ref.norm()).item()
            print(f"  correctness M={M} N={N} K={K} cfg{cfg}: rel-L2 {err:.2e} {'OK' if err < 3e-3 else 'FAIL'}")
        except Exception as e:
            print(f"  correctness cfg{cfg}: ERROR {type(e).__name__}: {str(e)[:200]}")
print("\nTFLOPS (do_bench median, L2 flushed); clocks sampled after each row")
for (N,K) in shapes:
    for M in Ms:
        A = torch.randn(M,K,device=dev,dtype=torch.bfloat16); W = torch.randn(N,K,device=dev,dtype=torch.bfloat16)
        Aq = torch.randint(-127,128,(M,K),device=dev,dtype=torch.int8); Wq = torch.randint(-127,128,(N,K),device=dev,dtype=torch.int8)
        Af = A.to(torch.float8_e4m3fn); Wf = W.to(torch.float8_e4m3fn); one = torch.ones((),device=dev)
        sa = torch.full((M,),1e-3,device=dev); sb = torch.full((N,),1e-3,device=dev)
        rows = {"bf16 cuBLAS": lambda: A@W.t(),
                "FP8 per-tensor cuBLASLt": lambda: torch._scaled_mm(Af,Wf.t(),scale_a=one,scale_b=one,out_dtype=torch.bfloat16,use_fast_accum=True),
                "INT8 _int_mm cuBLASLt(int32)": lambda: torch._int_mm(Aq,Wq.t())}
        for cfg in range(len(ext.configs())):
            rows[f"CUTLASS int8 cfg{cfg}"] = (lambda c: (lambda: ext.int8_scaled_mm(Aq,Wq,sa,sb,c)))(cfg)
        print(f"\n=== N={N} K={K} M={M} ===")
        best = None
        for name, fn in rows.items():
            try:
                us, tflops = tf(fn, M, N, K)
            except Exception as e:
                print(f"  {name:34s} ERROR {type(e).__name__}"); continue
            if name.startswith("CUTLASS") and (best is None or tflops > best[1]): best = (name, tflops)
            print(f"  {name:34s} {us:9.1f} us {tflops:7.0f} TFLOPS   [{clk()}]")
        print(f"  -> best CUTLASS: {best}")
