"""Build the DeepGEMM INT8 port and check every compiled config against the fp64 exact block-scaled reference
(imitates probes/bench_cutlass_int8_bw.py) and against kernels/cutlass_int8_bw. Usage: python dgint8_smoke.py [--build-only] [--cfgs 0,1,2] [--quick]"""
import sys, time, torch
sys.path.insert(0, "/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/users/pzeren/int8_bench")
from kernels.deepgemm_int8 import load, configs, int8_gemm_g128, prepare_sfa, sfa_from_kb_major
t0 = time.time(); ext = load(verbose=("--verbose" in sys.argv)); print(f"deepgemm_int8 ext build/load: {time.time()-t0:.0f} s", flush=True)
cfgs = configs(); print("configs:", [(c["id"], c["name"]) for c in cfgs], flush=True)
if "--build-only" in sys.argv: sys.exit(0)
sel = [int(x) for x in sys.argv[sys.argv.index("--cfgs") + 1].split(",")] if "--cfgs" in sys.argv else [c["id"] for c in cfgs]
dev = "cuda"
def ref_blockwise(Aq, Wq, sfa_kb, sfb):
    M, K = Aq.shape; N = Wq.shape[0]; nb = K // 128
    sfb_full = sfb.repeat_interleave(128, dim=0)
    Y = torch.zeros(M, N, dtype=torch.float64, device=dev)
    for kb in range(nb):
        d = Aq[:, kb*128:(kb+1)*128].double() @ Wq[:, kb*128:(kb+1)*128].double().t()
        Y += d * sfa_kb[kb].double()[:, None] * sfb_full[:, kb].double()[None, :]
    return Y
try:
    from kernels.cutlass_int8_bw import int8_blockwise_mm; HAVE_BW = True
except Exception as e:
    print("cutlass_int8_bw unavailable:", e); HAVE_BW = False
torch.manual_seed(0)
shapes = [(1024, 4096), (4096, 12288)] if "--quick" not in sys.argv else [(1024, 4096)]
Ms = (901, 1802, 4096) if "--quick" not in sys.argv else (901, 4096)
nfail = 0
print("\n== correctness (rel-L2 vs fp64 exact block-scaled math; pass <= 3e-3, expect ~1.66e-3 = bf16 output rounding) ==", flush=True)
for (N, K) in shapes:
    for M in Ms:
        Aq = torch.randint(-127, 128, (M, K), device=dev, dtype=torch.int8); Wq = torch.randint(-127, 128, (N, K), device=dev, dtype=torch.int8)
        sfa_kb = torch.rand(K // 128, M, device=dev) * 9e-3 + 1e-3; sfb = torch.rand(N // 128, K // 128, device=dev) * 9e-3 + 1e-3
        sfa_k = sfa_from_kb_major(sfa_kb); assert torch.equal(prepare_sfa(sfa_kb.t().contiguous()), sfa_k)
        ref = ref_blockwise(Aq, Wq, sfa_kb, sfb)
        bw = int8_blockwise_mm(Aq, Wq, sfa_kb, sfb).double() if HAVE_BW else None
        for cid in sel:
            c = cfgs[cid]
            if (c["shape_n"] and c["shape_n"] != N) or (c["shape_k"] and c["shape_k"] != K): continue
            try:
                out = int8_gemm_g128(Aq, Wq, sfa_k, sfb, cfg=cid); torch.cuda.synchronize(); out = out.double()
                err = ((out - ref).norm() / ref.norm()).item(); mx = ((out - ref).abs().max() / ref.abs().max()).item()
                vs_bw = ((out - bw).abs().max()).item() if bw is not None else float("nan")
                ok = err <= 3e-3 and torch.isfinite(out).all().item(); nfail += (not ok)
                print(f"  cfg {cid} {c['name']:22s} M={M:5d} N={N:5d} K={K:5d}: rel-L2 {err:.2e}  max|d|/max|ref| {mx:.2e}  max|d| vs cutlass_bw {vs_bw:.2e}  {'OK' if ok else 'FAIL'}", flush=True)
            except Exception as e:
                nfail += 1; print(f"  cfg {cid} {c['name']:22s} M={M:5d} N={N:5d} K={K:5d}: ERROR {type(e).__name__}: {str(e)[:200]}", flush=True)
print(f"\nRESULT: {'PASS' if nfail == 0 else f'{nfail} FAILURES'}")
