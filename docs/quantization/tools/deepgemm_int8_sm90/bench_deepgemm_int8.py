"""Final timing table for the DeepGEMM INT8 port: 4 (N,K) shapes x M in {901, 4096, 42240}, vs deepgemm_fp8_g128 (upstream) and
kernels/cutlass_int8_bw (+ CUTLASS INT8 per-tensor as the wgmma ceiling reference). Best config per shape comes from
results/deepgemm_int8_sweep.json when present (else the built-in heuristic). Writes results/deepgemm_int8_bench.{md,json}."""
import sys, os, json, time, subprocess, datetime, torch
from triton.testing import do_bench
B = "/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/users/pzeren/int8_bench"; sys.path.insert(0, B)
import kernels.deepgemm_int8 as dgi
from kernels.deepgemm_int8 import load, configs, int8_gemm_g128, sfa_from_kb_major, pick_config
from kernels.cutlass_int8_bw import int8_blockwise_mm
from kernels.cutlass_int8 import load as load_pt
from kernels.deepgemm_wrapper import deepgemm_fp8_g128, quantize_activation_g128, quantize_weight_128x128, prepare_activation_scales
ext = load(); cfgs = configs(); pt = load_pt()
sweep = {}
sp = f"{B}/results/deepgemm_int8_sweep.json"
if os.path.exists(sp):
    sweep = json.load(open(sp))
    for k, v in sweep.items():
        if "best_cfg" in v: dgi.BEST[(v["N"], v["K"], v["M"])] = v["best_cfg"]
dev = "cuda"
def clk(): return subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
def tf(fn, M, N, K):
    t = do_bench(fn, warmup=25, rep=200, return_mode="median"); return t * 1e3, 2 * M * N * K / (t * 1e-3) / 1e12
def ref_blockwise(Aq, Wq, sfa_kb, sfb):
    M, K = Aq.shape; N = Wq.shape[0]; sfb_full = sfb.repeat_interleave(128, dim=0)
    Y = torch.zeros(M, N, dtype=torch.float64, device=dev)
    for kb in range(K // 128):
        Y += (Aq[:, kb*128:(kb+1)*128].double() @ Wq[:, kb*128:(kb+1)*128].double().t()) * sfa_kb[kb].double()[:, None] * sfb_full[:, kb].double()[None, :]
    return Y
torch.manual_seed(0)
rows = []; shapes = [(4096, 4096), (1024, 4096), (12288, 4096), (4096, 12288)]; Ms = [901, 4096, 42240]
for (N, K) in shapes:
    for M in Ms:
        Mp = (M + 3) // 4 * 4
        Aq = torch.randint(-127, 128, (M, K), device=dev, dtype=torch.int8); Wq = torch.randint(-127, 128, (N, K), device=dev, dtype=torch.int8)
        sfa_kb = torch.rand(K // 128, M, device=dev) * 9e-3 + 1e-3; sfb = torch.rand(N // 128, K // 128, device=dev) * 9e-3 + 1e-3
        sfa_k = sfa_from_kb_major(sfa_kb)                       # kernel layout [K/128, Mp] (prepared once, outside timing)
        Aq_p = torch.cat([Aq, Aq.new_zeros(Mp - M, K)], 0) if Mp != M else Aq; sfa_bw = sfa_from_kb_major(sfa_kb)   # cutlass_bw needs M % 4 == 0 (pre-padded, untimed)
        A = torch.randn(M, K, device=dev, dtype=torch.bfloat16); W = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
        Af8, As = quantize_activation_g128(A); Wf8, Ws = quantize_weight_128x128(W); As = prepare_activation_scales(As)
        sa = torch.full((M,), 1e-3, device=dev); sb = torch.full((N,), 1e-3, device=dev)
        cid = pick_config(M, N, K); cname = cfgs[cid]["name"]
        out = int8_gemm_g128(Aq, Wq, sfa_k, sfb, cfg=cid); torch.cuda.synchronize()
        rel = ((out.double() - ref_blockwise(Aq, Wq, sfa_kb, sfb)).norm() / ref_blockwise(Aq, Wq, sfa_kb, sfb).norm()).item() if M <= 4096 else float("nan")
        r = {"N": N, "K": K, "M": M, "config": cname, "cfg_id": cid, "rel_l2": rel}
        for name, fn, k in [("deepgemm_int8", lambda: int8_gemm_g128(Aq, Wq, sfa_k, sfb, cfg=cid), "int8"),
                            ("deepgemm_fp8", lambda: deepgemm_fp8_g128(Af8, As, Wf8, Ws), "fp8"),
                            ("cutlass_int8_bw", lambda: int8_blockwise_mm(Aq_p, Wq, sfa_bw, sfb), "bw"),
                            ("cutlass_int8_pertensor", lambda: pt.int8_scaled_mm(Aq_p, Wq, torch.full((Mp,), 1e-3, device=dev), sb, 0), "pt")]:
            try:
                us, tflops = tf(fn, M, N, K); r[name + "_us"] = us; r[name + "_tflops"] = tflops; r[name + "_sm_mhz"] = clk()
            except Exception as e:
                r[name + "_error"] = f"{type(e).__name__}: {str(e)[:120]}"
        if "cfgs" in sweep.get(f"{N}x{K}x{M}", {}): r["sweep"] = {v["name"]: round(v["tflops"]) for v in sweep[f"{N}x{K}x{M}"]["cfgs"].values() if "tflops" in v}
        rows.append(r)
        print(f"N={N:5d} K={K:5d} M={M:5d} | INT8 port {cname:22s} {r.get('deepgemm_int8_tflops', 0):6.0f} TF ({r.get('deepgemm_int8_us', 0):7.1f} us, rel-L2 {rel:.2e}) | "
              f"DG FP8 {r.get('deepgemm_fp8_tflops', 0):6.0f} TF | CUTLASS INT8 bw {r.get('cutlass_int8_bw_tflops', 0):6.0f} TF | INT8 per-tensor {r.get('cutlass_int8_pertensor_tflops', 0):6.0f} TF | SM {r.get('deepgemm_int8_sm_mhz')} MHz", flush=True)
meta = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "timing": "triton.testing.do_bench(warmup=25, rep=200, median, L2 flushed); TFLOPS = 2MNK/t; clocks unlocked (~5% noise); other agents may share the GPU",
        "configs": [c["name"] for c in cfgs], "promote_mode": dgi.PROMOTE_MODE}
json.dump({"meta": meta, "rows": rows}, open(f"{B}/results/deepgemm_int8_bench.json", "w"), indent=1)
with open(f"{B}/results/deepgemm_int8_bench.md", "w") as f:
    f.write("# DeepGEMM SM90 1D2D kernel ported to INT8 (1,128,128) block scaling -- timing\n\n")
    f.write(f"- {meta['gpu']}, torch {meta['torch']}, {meta['time']}; {meta['timing']}\n")
    f.write("- kernel: `kernels/deepgemm_int8` (DeepGEMM 2.8.0 @66081d4 `sm90_fp8_gemm_1d2d.cuh` with int8 operands, `wgmma.m64nNk32.s32.s8.s8`, int32 per-128-K-block accumulator promoted as `final += sfa*sfb * float(acc)`); "
            "scale layouts: sfa fp32 [K/128, round_up(M,4)] (transposed / TMA-aligned, built once outside timing), sfb fp32 [N/128, K/128]. A is NOT padded (TMA OOB zero-fill + clipped store).\n")
    f.write("- references: `deepgemm_fp8_g128` = upstream DeepGEMM FP8 (recipe (1,128,128), pre-aligned SFA); `cutlass_int8_bw` = kernels/cutlass_int8_bw (128x128 c1x2, M pre-padded to a multiple of 4 outside timing); `cutlass_int8_pertensor` = per-tensor INT8 CUTLASS 128x256 c2x1 (wgmma ceiling reference).\n")
    f.write("- rel-L2 = INT8 port output vs fp64 exact block-scaled math (bf16 output floor ~1.66e-3), measured for M <= 4096.\n\n")
    f.write("| N | K | M | INT8 port config | INT8 port us | INT8 port TFLOPS | DeepGEMM FP8 TFLOPS | CUTLASS INT8 bw TFLOPS | INT8 per-tensor TFLOPS | INT8/FP8 | INT8/cutlass_bw | rel-L2 | SM MHz |\n|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
    for r in rows:
        g = lambda k: r.get(k, float("nan"))
        f.write(f"| {r['N']} | {r['K']} | {r['M']} | {r['config']} | {g('deepgemm_int8_us'):.1f} | {g('deepgemm_int8_tflops'):.0f} | {g('deepgemm_fp8_tflops'):.0f} | {g('cutlass_int8_bw_tflops'):.0f} | {g('cutlass_int8_pertensor_tflops'):.0f} | "
                f"{g('deepgemm_int8_tflops')/g('deepgemm_fp8_tflops'):.2f} | {g('deepgemm_int8_tflops')/g('cutlass_int8_bw_tflops'):.2f} | {r['rel_l2']:.2e} | {r.get('deepgemm_int8_sm_mhz')} |\n")
    if any("sweep" in r for r in rows):
        f.write("\n## Config sweep (TFLOPS per compiled config, from probes/dgint8_sweep.py)\n\n")
        names = [c["name"] for c in cfgs]
        f.write("| N | K | M | " + " | ".join(names) + " |\n|---|---|---|" + "---:|" * len(names) + "\n")
        for r in rows:
            if "sweep" in r: f.write(f"| {r['N']} | {r['K']} | {r['M']} | " + " | ".join(str(r["sweep"].get(n, "-")) for n in names) + " |\n")
print("wrote results/deepgemm_int8_bench.md/.json")
