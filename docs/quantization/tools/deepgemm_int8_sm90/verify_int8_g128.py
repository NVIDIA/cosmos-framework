"""Independent verifier for kernels/deepgemm_int8 (DeepGEMM 1D2D kernel ported to INT8, (1,128,128) block scaling).

(a) output vs an fp64 exact reference written here (NOT the porter's): per 128-K block, int8 dot in fp64 (exact: |dot| <= 2^21),
    scaled by sfa[m,kb]*sfb[n//128,kb] in fp64, summed over blocks in fp64, compared to the kernel's bf16 output.
    Cases per shape: random scales U[1e-3,1e-2]; adversarial scales (rows/blocks 1e-4 vs 1.0); int8 values at +-127 (and -128)
    extremes with adversarial scales (max |int32 acc| = 128*127*127 = 2064512 < 2^24 -> the int32->fp32 promotion must be exact);
    for M % 4 != 0 the sfa padding columns are filled with NaN to check the "padding may hold anything" claim.
    pass = rel-L2 <= 3e-3 and max|d|/max|ref| <= 5e-3.
(b) bf16 bit-equality (or tiny difference) vs kernels/cutlass_int8_bw and vs the CUTLASS tuner's best variants
    (third_party/cutlass_int8_bw_tune_build_best: v0 coop128x128 c2x1 DB, v2 pp64x128 c1x2 DB) on the same inputs.
Also sweeps every compiled deepgemm_int8 config compatible with (N, K) at the smaller shapes.
"""
import sys, os, json, time, importlib.util, torch
B = "/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/users/pzeren/int8_bench"; sys.path.insert(0, B)
from kernels.deepgemm_int8 import int8_gemm_g128, prepare_sfa, configs, pick_config, sfa_ld
from kernels.cutlass_int8_bw import load as load_bw

dev = "cuda"; torch.manual_seed(1234)
ext_bw = load_bw()
tuner = None
so = f"{B}/third_party/cutlass_int8_bw_tune_build_best/int8bw_tune_best.so"
if os.path.exists(so):
    spec = importlib.util.spec_from_file_location("int8bw_tune_best", so); tuner = importlib.util.module_from_spec(spec); spec.loader.exec_module(tuner)
TUNER_V = {0: "tuner coop128x128 c2x1 DB", 2: "tuner pp64x128 c1x2 DB"}

def ref_fp64(A, W, sfa_plain, sfb):
    """exact block-scaled math: sfa_plain fp32 [M, K/128], sfb fp32 [N/128, K/128] -> fp64 [M, N]."""
    M, K = A.shape; N = W.shape[0]; G = K // 128
    Y = torch.zeros(M, N, dtype=torch.float64, device=dev)
    sfb_rows = sfb.double().repeat_interleave(128, dim=0)   # [N, G]
    for kb in range(G):
        s = slice(kb * 128, (kb + 1) * 128)
        dot = A[:, s].double() @ W[:, s].double().t()        # exact integers in fp64
        Y += dot * sfa_plain[:, kb].double()[:, None] * sfb_rows[:, kb][None, :]
    return Y

def make_case(case, M, N, K):
    G = K // 128
    if case == "random":
        A = torch.randint(-127, 128, (M, K), device=dev, dtype=torch.int8); W = torch.randint(-127, 128, (N, K), device=dev, dtype=torch.int8)
        sfa = torch.rand(M, G, device=dev) * 9e-3 + 1e-3; sfb = torch.rand(N // 128, G, device=dev) * 9e-3 + 1e-3
    elif case == "adversarial_scales":
        A = torch.randint(-127, 128, (M, K), device=dev, dtype=torch.int8); W = torch.randint(-127, 128, (N, K), device=dev, dtype=torch.int8)
        sfa = torch.where(torch.rand(M, G, device=dev) < 0.5, torch.full((M, G), 1e-4, device=dev), torch.ones(M, G, device=dev))
        sfa[::7] = 1.0; sfa[3::11] = 1e-4                       # whole rows at the extremes too
        sfb = torch.where(torch.rand(N // 128, G, device=dev) < 0.5, torch.full((N // 128, G), 1e-4, device=dev), torch.ones(N // 128, G, device=dev))
    elif case == "extremes_pm127":
        A = torch.where(torch.rand(M, K, device=dev) < 0.5, -127, 127).to(torch.int8); W = torch.where(torch.rand(N, K, device=dev) < 0.5, -127, 127).to(torch.int8)
        A[:64] = 127; W[:256] = 127                             # rows/cols where every block dot hits +128*127*127 exactly
        A[64:128] = -127
        sfa = torch.where(torch.rand(M, G, device=dev) < 0.5, torch.full((M, G), 1e-4, device=dev), torch.ones(M, G, device=dev))
        sfb = torch.where(torch.rand(N // 128, G, device=dev) < 0.5, torch.full((N // 128, G), 1e-4, device=dev), torch.ones(N // 128, G, device=dev))
    elif case == "extremes_m128":
        A = torch.full((M, K), -128, device=dev, dtype=torch.int8); W = torch.full((N, K), -128, device=dev, dtype=torch.int8)
        A[1::2] = 127; W[:, 1::3] = 127                         # mixes -128 with 127 -> block dots up to 2^21
        sfa = torch.rand(M, G, device=dev) * 9e-3 + 1e-3; sfb = torch.rand(N // 128, G, device=dev) * 9e-3 + 1e-3
    return A, W, sfa, sfb

def metrics(out, ref):
    d = (out.double() - ref)
    return (d.norm() / ref.norm()).item(), (d.abs().max() / ref.abs().max()).item(), int(torch.isnan(out).sum().item())

SHAPES = [(1024, 4096, 901), (1024, 4096, 1802), (4096, 12288, 901), (4096, 12288, 4096), (12288, 4096, 4096)]
CASES = ["random", "adversarial_scales", "extremes_pm127", "extremes_m128"]
results = []; all_ok = True
cfgs = configs()
for (N, K, M) in SHAPES:
    Mp = sfa_ld(M); G = K // 128
    compatible = [c["id"] for c in cfgs if (c["shape_n"] in (0, N)) and (c["shape_k"] in (0, K))]
    sweep_cfgs = compatible if M <= 1802 else [pick_config(M, N, K)]
    for case in CASES:
        A, W, sfa, sfb = make_case(case, M, N, K)
        t0 = time.time(); ref = ref_fp64(A, W, sfa, sfb); torch.cuda.synchronize(); tref = time.time() - t0
        sfa_k = prepare_sfa(sfa)                                   # [G, Mp]
        if Mp != M: sfa_k[:, M:] = float("nan")                     # padding columns: anything goes (claimed)
        # cutlass_int8_bw wants sfa [G, M]; its wrapper pads A/sfa to Mp itself
        sfa_kb = sfa.t().contiguous()
        out_bw = ext_bw.int8_blockwise_mm(*( (A, W, sfa_kb, sfb) if Mp == M else
                 (torch.cat([A, A.new_zeros(Mp - M, K)]), W, torch.cat([sfa_kb, sfa_kb.new_zeros(G, Mp - M)], 1), sfb)))[:M]
        e_bw, mx_bw, _ = metrics(out_bw, ref)
        row = dict(N=N, K=K, M=M, case=case, ref_seconds=round(tref, 1), cutlass_bw=dict(rel_l2=e_bw, max_rel=mx_bw), port={}, tuner={})
        if tuner is not None:
            Ap = A if Mp == M else torch.cat([A, A.new_zeros(Mp - M, K)]); sfa_p = sfa_kb if Mp == M else torch.cat([sfa_kb, sfa_kb.new_zeros(G, Mp - M)], 1)
            for v, name in TUNER_V.items():
                o = tuner.mm(Ap, W, sfa_p, sfb, v, 0, 1)[:M]; e, mx, nn = metrics(o, ref)
                row["tuner"][name] = dict(rel_l2=e, max_rel=mx, bits_equal_bw=bool(torch.equal(o, out_bw)), max_abs_diff_bw=(o.float() - out_bw.float()).abs().max().item())
        for cid in sweep_cfgs:
            out = int8_gemm_g128(A, W, sfa_k, sfb, cfg=cid); torch.cuda.synchronize()
            e, mx, nn = metrics(out, ref)
            ok = (e <= 3e-3) and (mx <= 5e-3) and nn == 0 and out.shape == (M, N)
            all_ok &= ok
            dbw = (out.float() - out_bw.float()); n_diff = int((dbw != 0).sum().item())
            # a 1-ulp bf16 difference relative to the element magnitude
            ulp = (dbw.abs() / out_bw.float().abs().clamp_min(1e-30)).max().item()
            row["port"][cfgs[cid]["name"]] = dict(cfg=cid, rel_l2=e, max_rel=mx, nan=nn, ok=ok, n_elems_differ_from_bw=n_diff,
                                                 frac_differ_from_bw=n_diff / (M * N), max_abs_diff_bw=dbw.abs().max().item(), max_rel_diff_bw=ulp,
                                                 bits_equal_bw=n_diff == 0)
            flag = "OK " if ok else "FAIL"
            print(f"{flag} N={N:5d} K={K:5d} M={M:5d} {case:18s} cfg{cid:2d} {cfgs[cid]['name']:28s} rel-L2 {e:.2e} max|d|/max|ref| {mx:.2e} nan={nn} "
                  f"| vs cutlass_bw: {n_diff} elems differ ({100*n_diff/(M*N):.3f}%), max|d| {dbw.abs().max().item():.3e}, max rel {ulp:.2e}  [bw rel-L2 {e_bw:.2e}]", flush=True)
        results.append(row)
        del ref, A, W
os.makedirs(f"{B}/results", exist_ok=True)
json.dump(dict(all_ok=all_ok, results=results), open(f"{B}/results/verify_int8_g128.json", "w"), indent=1)
print("\nALL OK" if all_ok else "\nSOME FAILURES", "-> results/verify_int8_g128.json")
