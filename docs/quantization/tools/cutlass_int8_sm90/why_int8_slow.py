"""Why is cuBLASLt per-tensor INT8 (_int_mm) slower than per-tensor FP8 (_scaled_mm) on this H100?
Tests: (1) which cuBLASLt kernels get selected (sm80 mma.sync vs sm90 wgmma), (2) int32-output bandwidth
hypothesis via K sweep, (3) SM clock/power during each measurement, (4) Triton INT8 wgmma with bf16 output as
an 'achievable INT8' proxy."""
import subprocess, threading, time, torch, triton, triton.language as tl
from triton.testing import do_bench
from torch.profiler import profile, ProfilerActivity

dev = "cuda"
def smi():
    out = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,power.draw,clocks_throttle_reasons.active",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    return out

class Sampler:
    def __init__(self): self.samples=[]; self.stop=False
    def run(self):
        while not self.stop:
            self.samples.append(smi()); time.sleep(0.05)
    def __enter__(self):
        self.t=threading.Thread(target=self.run); self.t.start(); return self
    def __exit__(self,*a):
        self.stop=True; self.t.join()
    def summary(self):
        clocks=[int(s.split(',')[0]) for s in self.samples if s]; pw=[float(s.split(',')[1]) for s in self.samples if s]
        if not clocks: return "n/a"
        return f"SM {min(clocks)}-{max(clocks)} MHz (median {sorted(clocks)[len(clocks)//2]}), power {max(pw):.0f} W"

def bench(fn, flops):
    torch.cuda.synchronize()
    with Sampler() as s:
        t = do_bench(fn, warmup=50, rep=400, return_mode="median")
    return t, flops/t/1e6, s.summary()

def kernel_names(fn, n=3):
    fn(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(n): fn()
        torch.cuda.synchronize()
    names = {}
    for e in p.key_averages():
        if e.device_time_total > 0 and ("gemm" in e.key.lower() or "xmma" in e.key.lower() or "cutlass" in e.key.lower() or "kernel" in e.key.lower()):
            names[e.key] = e.device_time_total / n
    return sorted(names.items(), key=lambda kv: -kv[1])[:3]

@triton.jit
def _int8_mm_bf16out(A, B, C, M, N, K, sam, sak, sbk, sbn, scm, scn, scale,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr):
    pid = tl.program_id(0); npm = tl.cdiv(M, BM); npn = tl.cdiv(N, BN)
    g = pid // (GM * npn); first = g * GM; gs = min(npm - first, GM)
    pm = first + (pid % (GM * npn)) % gs; pn = (pid % (GM * npn)) // gs
    om = pm * BM + tl.arange(0, BM); on = pn * BN + tl.arange(0, BN); ok = tl.arange(0, BK)
    ap = A + om[:, None] * sam + ok[None, :] * sak; bp = B + ok[:, None] * sbk + on[None, :] * sbn
    acc = tl.zeros((BM, BN), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BK)):
        a = tl.load(ap); b = tl.load(bp); acc += tl.dot(a, b, out_dtype=tl.int32)
        ap += BK * sak; bp += BK * sbk
    c = (acc.to(tl.float32) * scale).to(tl.bfloat16)
    cp = C + om[:, None] * scm + on[None, :] * scn
    tl.store(cp, c, mask=(om[:, None] < M) & (on[None, :] < N))

def triton_int8(Aq, Wq, cfg):
    M, K = Aq.shape; N = Wq.shape[0]; C = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
    Bt = Wq.t()  # K-major B view
    grid = (triton.cdiv(M, cfg["BM"]) * triton.cdiv(N, cfg["BN"]),)
    _int8_mm_bf16out[grid](Aq, Bt, C, M, N, K, Aq.stride(0), Aq.stride(1), Bt.stride(0), Bt.stride(1), C.stride(0), C.stride(1), 1e-3,
                           BM=cfg["BM"], BN=cfg["BN"], BK=cfg["BK"], GM=8, num_warps=cfg["w"], num_stages=cfg["s"])
    return C

print("device", torch.cuda.get_device_name(0), "| torch", torch.__version__, "| triton", triton.__version__)
print("idle:", smi())
torch.manual_seed(0)
shapes = [(4096, 4096, 4096), (4096, 4096, 16384), (4096, 4096, 32768), (16384, 4096, 4096)]
for (M, N, K) in shapes:
    flops = 2 * M * N * K
    A = torch.randn(M, K, device=dev, dtype=torch.bfloat16); W = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
    Aq = torch.randint(-127, 128, (M, K), device=dev, dtype=torch.int8); Wq = torch.randint(-127, 128, (N, K), device=dev, dtype=torch.int8)
    Af = A.to(torch.float8_e4m3fn); Wf = W.to(torch.float8_e4m3fn); one = torch.ones((), device=dev)
    fns = {
        "bf16 cuBLAS": lambda: A @ W.t(),
        "INT8 cuBLASLt _int_mm (int32 out)": lambda: torch._int_mm(Aq, Wq.t()),
        "FP8 cuBLASLt _scaled_mm per-tensor fast_accum (bf16 out)": lambda: torch._scaled_mm(Af, Wf.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16, use_fast_accum=True),
        "FP8 cuBLASLt _scaled_mm per-tensor (fp32 out)": lambda: torch._scaled_mm(Af, Wf.t(), scale_a=one, scale_b=one, out_dtype=torch.float32, use_fast_accum=True),
    }
    print(f"\n=== M={M} N={N} K={K}  ({flops/1e9:.0f} GFLOP; int32 out = {M*N*4/1e6:.0f} MB, bf16 out = {M*N*2/1e6:.0f} MB) ===")
    for name, fn in fns.items():
        t, tf, clk = bench(fn, flops)
        print(f"  {name:58s} {t:8.1f} us {tf:6.0f} TFLOPS | {clk}")
        if M == 4096 and K == 4096:
            for kn, kt in kernel_names(fn): print(f"        kernel: {kn[:110]}  ({kt:.0f} us/call)")
    # Triton INT8 wgmma with bf16 output, a few configs
    best = None
    for cfg in [dict(BM=128, BN=128, BK=128, w=8, s=3), dict(BM=128, BN=256, BK=64, w=8, s=3), dict(BM=128, BN=256, BK=128, w=8, s=3), dict(BM=256, BN=128, BK=64, w=8, s=4)]:
        try:
            t, tf, clk = bench(lambda: triton_int8(Aq, Wq, cfg), flops)
        except Exception as e:
            print(f"  triton {cfg}: {type(e).__name__}"); continue
        if best is None or t < best[0]: best = (t, tf, clk, cfg)
        print(f"  Triton INT8 wgmma bf16-out {str(cfg):48s} {t:8.1f} us {tf:6.0f} TFLOPS | {clk}")
    ref = torch._int_mm(Aq, Wq.t()).float() * 1e-3
    out = triton_int8(Aq, Wq, best[3]).float()
    print(f"  Triton best rel-L2 vs _int_mm*scale: {((out-ref).norm()/ref.norm()).item():.2e}")
