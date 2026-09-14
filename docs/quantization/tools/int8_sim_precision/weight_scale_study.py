#!/usr/bin/env python
"""Weight-scale layout study for the Cosmos3-Nano generation tower (INT8, symmetric, qmax=127).

Compares per-col g128 (S0), 128x128 blockwise, two separable s_w[n]*c[kb] layouts, per-col g64,
per-channel and per-tensor on all 252 gen-tower linears (36 layers x {q,k,v,o,gate,up,down}).

Math (int8_handoff.md 2.1 / 3.6): scale = absmax/127 in fp32, q = torch.round(W/scale) (half-even),
clamp to +-127, dequant = q*scale. W is nn.Linear layout [N=out, K=in]; groups are along K.

Self-contained: only torch + safetensors. Reads the HF export read-only, processes one shard at a time,
moves each weight to the GPU as fp32, writes JSON + Markdown next to this file.
"""
import json
import math
import os
import re
import sys
import time
from collections import defaultdict

import torch
from safetensors import safe_open

CKPT = ("/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/users/pzeren/hf_cache/hub/"
        "models--nvidia--Cosmos3-Nano/snapshots/411f42a8fdfb8c5b2583cb8786e0938f49796eaa/transformer")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(OUT_DIR, "weight_scale_study.json")
OUT_MD = os.path.join(OUT_DIR, "weight_scale_study.md")

QMAX = 127.0
GEMM_M = 2048
TYPES = ["q", "k", "v", "o", "gate", "up", "down"]
# diffusers name -> gen-tower type (mapping verified in cosmos_framework/inference2/_convert_model_to_diffusers.py)
NAME_RE = re.compile(r"^layers\.(\d+)\.(self_attn\.(add_q_proj|add_k_proj|add_v_proj|to_add_out)"
                     r"|mlp_moe_gen\.(gate_proj|up_proj|down_proj))\.weight$")
TYPE_OF = {"add_q_proj": "q", "add_k_proj": "k", "add_v_proj": "v", "to_add_out": "o",
           "gate_proj": "gate", "up_proj": "up", "down_proj": "down"}
CANON = {"q": "q_proj_moe_gen", "k": "k_proj_moe_gen", "v": "v_proj_moe_gen", "o": "o_proj_moe_gen",
         "gate": "mlp_moe_gen.gate_proj", "up": "mlp_moe_gen.up_proj", "down": "mlp_moe_gen.down_proj"}

SCHEMES = [
    ("a_percol_g128", "(a) per-col g128 [S0]"),
    ("b_block128x128", "(b) W 128x128 blockwise"),
    ("c_sep_noclip", "(c) separable s_w[n]*c[kb], max-ratio (no clip)"),
    ("d_sep_log_raw", "(d0) separable log rank-1 fit, unscaled"),
    ("d_sep_log_clip1e-4", "(d1) separable log fit, rescaled clip<=0.01%"),
    ("d_sep_log_clip1e-3", "(d2) separable log fit, rescaled clip<=0.1%"),
    ("d_sep_log_noclip", "(d3) separable log fit, rescaled exact no-clip"),
    ("h_sep_logc_rownoclip", "(h) separable: c[kb] from log fit, s_w[n] per-row no-clip"),
    ("e_percol_g64", "(e) per-col g64 [reference]"),
    ("f_perchannel", "(f) per-channel"),
    ("g_pertensor", "(g) per-tensor"),
    ("x_sep_noclip_g64", "(extra) separable no-clip, g64"),
    ("x_block64x64", "(extra) W 64x64 blockwise"),
]

DEV = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
_T0 = time.time()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - _T0:7.1f}s] {msg}", flush=True)


# ----------------------------------------------------------------------------- quantization helpers
def absmax_groups(W, g):
    """A[n, kb] = absmax(W[n, kb*g:(kb+1)*g])."""
    N, K = W.shape
    assert K % g == 0
    return W.abs().view(N, K // g, g).amax(dim=-1)


def expand_scale(scale_nk, N, K, g_n, g_k):
    """scale_nk [N/g_n, K/g_k] -> full [N, K]."""
    return scale_nk.repeat_interleave(g_n, dim=0).repeat_interleave(g_k, dim=1)


def quant_dequant(W, scale_full):
    """Return (Wdeq, n_clipped). scale_full is broadcastable to W."""
    s = torch.where(scale_full > 0, scale_full, torch.ones_like(scale_full))
    q = torch.round(W / s)
    clipped = (q.abs() > QMAX).sum()
    q = q.clamp_(-QMAX, QMAX)
    return q * s, int(clipped.item())


def log_rank1_fit(A):
    """Additive rank-1 fit in log domain: log A[n,kb] ~ u[n] + v[kb], least squares.
    Balanced two-way layout -> closed form (row means + col means - grand mean); this is the fixed point
    of alternating means. Returns (u, v, resid) in natural log. Zero entries are floored."""
    L = torch.log(A.clamp_min(1e-30))
    gm = L.mean()
    u = L.mean(dim=1)             # [N]
    v = L.mean(dim=0) - gm        # [KB]
    fit = u[:, None] + v[None, :]
    return u, v, L - fit


def sep_scale_noclip(A):
    """(c): s_w[n] = max_kb A; c[kb] = max_n A[n,kb]/s_w[n]; scale = s_w*c/127 >= A/127."""
    s_w = A.amax(dim=1)                                    # [N]
    ratio = A / torch.where(s_w > 0, s_w, torch.ones_like(s_w))[:, None]
    c = ratio.amax(dim=0)                                  # [KB]
    return s_w[:, None] * c[None, :] / QMAX


def kth_threshold(r_flat, frac_allowed):
    """Smallest gamma such that #(r > gamma) <= frac_allowed * numel."""
    n = r_flat.numel()
    allowed = int(math.floor(frac_allowed * n))
    k = n - allowed                     # k-th smallest (1-indexed); elements strictly above it: <= allowed
    k = max(1, min(n, k))
    return torch.kthvalue(r_flat, k).values


# ----------------------------------------------------------------------------- per-weight evaluation
_X_CACHE = {}


def get_X(K):
    if K not in _X_CACHE:
        gen = torch.Generator(device=DEV).manual_seed(1234 + K)
        X = torch.randn(GEMM_M, K, generator=gen, device=DEV, dtype=torch.float32)
        _X_CACHE[K] = X.to(torch.bfloat16).to(torch.float32)   # bf16-representable values, fp32 GEMM
    return _X_CACHE[K]


def eval_weight(W):
    """W fp32 [N,K] on GPU. Returns dict scheme -> metrics plus separability stats."""
    N, K = W.shape
    out = {}
    W_norm2 = float((W * W).sum().item())
    W_absmax = float(W.abs().max().item())
    X = get_X(K)
    Y_ref = X @ W.t()
    Y_ref_norm2 = float((Y_ref * Y_ref).sum().item())

    def record(name, scale_full, extra=None):
        Wd, n_clip = quant_dequant(W, scale_full)
        E = W - Wd
        err2 = float((E * E).sum().item())
        max_err = float(E.abs().max().item())
        Yq = X @ Wd.t()
        D = Yq - Y_ref
        gemm_err2 = float((D * D).sum().item())
        m = {
            "snr_db": 10.0 * math.log10(W_norm2 / max(err2, 1e-300)),
            "rel_l2": math.sqrt(err2 / W_norm2),
            "max_err_over_absmax": max_err / W_absmax,
            "clip_frac": n_clip / (N * K),
            "n_clipped": n_clip,
            "gemm_rel_l2": math.sqrt(gemm_err2 / Y_ref_norm2),
            "_err2": err2, "_gemm_err2": gemm_err2,
        }
        if extra:
            m.update(extra)
        out[name] = m
        del Wd, E, Yq, D

    # --- absmax matrices
    A128 = absmax_groups(W, 128)               # [N, K/128]
    A64 = absmax_groups(W, 64)                 # [N, K/64]
    KB = K // 128

    # (a) per-col g128
    record("a_percol_g128", expand_scale(A128 / QMAX, N, K, 1, 128))
    # (b) 128x128 blockwise
    B128 = A128.view(N // 128, 128, KB).amax(dim=1)       # [N/128, KB]
    record("b_block128x128", expand_scale(B128 / QMAX, N, K, 128, 128))
    # (c) separable no-clip
    Sc = sep_scale_noclip(A128)
    record("c_sep_noclip", expand_scale(Sc, N, K, 1, 128),
           extra={"scale_over_absmax_mean": float((Sc * QMAX / A128.clamp_min(1e-30)).mean().item())})
    # (d) separable log rank-1 fit
    u, v, resid = log_rank1_fit(A128)
    S0 = torch.exp(u[:, None] + v[None, :]) / QMAX        # [N, KB]
    ratio_blk = A128 / (S0 * QMAX)                          # >1 => that block would clip
    gamma_noclip = float(ratio_blk.max().item())
    record("d_sep_log_raw", expand_scale(S0, N, K, 1, 128), extra={"gamma": 1.0})
    # element-wise ratio |W| / (127*S0) to find global rescale factors for clip budgets
    r = (W.abs() / expand_scale(S0 * QMAX, N, K, 1, 128)).flatten()
    g1 = float(kth_threshold(r, 1e-4).item())
    g2 = float(kth_threshold(r, 1e-3).item())
    del r
    # note: gamma < 1 is allowed (fit looser than needed for the clip budget -> shrink for resolution)
    record("d_sep_log_clip1e-4", expand_scale(S0 * g1, N, K, 1, 128), extra={"gamma": g1})
    record("d_sep_log_clip1e-3", expand_scale(S0 * g2, N, K, 1, 128), extra={"gamma": g2})
    record("d_sep_log_noclip", expand_scale(S0 * gamma_noclip, N, K, 1, 128), extra={"gamma": gamma_noclip})
    # (h) separable with the fitted per-K-block factor c[kb] = exp(v), and s_w[n] = max_kb A[n,kb]/c[kb]
    #     (per-row tightest no-clip factor). scale = s_w[n]*c[kb]/127 >= A/127 everywhere -> no clipping.
    c_fit = torch.exp(v)                                    # [KB]
    s_row = (A128 / c_fit[None, :]).amax(dim=1)             # [N]
    Sh = s_row[:, None] * c_fit[None, :] / QMAX
    record("h_sep_logc_rownoclip", expand_scale(Sh, N, K, 1, 128),
           extra={"scale_over_absmax_mean": float((Sh * QMAX / A128.clamp_min(1e-30)).mean().item())})
    # (e) per-col g64
    record("e_percol_g64", expand_scale(A64 / QMAX, N, K, 1, 64))
    # (f) per-channel
    record("f_perchannel", (W.abs().amax(dim=1, keepdim=True) / QMAX).expand(N, K))
    # (g) per-tensor
    record("g_pertensor", torch.full((1, 1), W_absmax / QMAX, device=DEV).expand(N, K))
    # extras
    record("x_sep_noclip_g64", expand_scale(sep_scale_noclip(A64), N, K, 1, 64))
    B64 = A64.view(N // 64, 64, K // 64).amax(dim=1)
    record("x_block64x64", expand_scale(B64 / QMAX, N, K, 64, 64))

    # --- separability of A128 (bits)
    R = resid / math.log(2.0)                               # log2 residual [N, KB]
    Rabs = R.abs()
    sep = {
        "resid_std_bits": float(R.std().item()),
        "resid_mean_abs_bits": float(Rabs.mean().item()),
        "resid_max_bits": float(Rabs.max().item()),
        "frac_entries_gt1bit": float((Rabs > 1.0).float().mean().item()),
        "n_entries_gt1bit": int((Rabs > 1.0).sum().item()),
        "n_entries": int(R.numel()),
        "n_channels_gt1bit": int((Rabs.amax(dim=1) > 1.0).sum().item()),
        "n_channels": N,
        "n_kblocks_gt1bit": int((Rabs.amax(dim=0) > 1.0).sum().item()),
        "n_kblocks": KB,
        "log2A_std_bits": float((torch.log2(A128.clamp_min(1e-30))).std().item()),
        "log2A_row_effect_std_bits": float((u / math.log(2.0)).std().item()),
        "log2A_col_effect_std_bits": float((v / math.log(2.0)).std().item()),
        "c_noclip_over_fit_max_bits": math.log2(gamma_noclip),
        # spread of the (c) per-K-block factor c[kb] = Sc[n,kb]/s_w[n] (same for every n), in bits
        "c_kblock_factor_range_bits": float(torch.log2(Sc[0].max() / Sc[0].min().clamp_min(1e-30)).item()),
    }
    out["_sep"] = sep
    out["_meta"] = {"N": N, "K": K, "W_norm2": W_norm2, "W_absmax": W_absmax, "Y_ref_norm2": Y_ref_norm2,
                    "numel": N * K}
    return out


# ----------------------------------------------------------------------------- aggregation
def aggregate(per_weight, keys):
    """Pooled (energy-weighted) metrics over the given weight keys, plus plain means."""
    agg = {}
    for sname, _ in SCHEMES:
        W2 = sum(per_weight[k]["_meta"]["W_norm2"] for k in keys)
        Y2 = sum(per_weight[k]["_meta"]["Y_ref_norm2"] for k in keys)
        E2 = sum(per_weight[k][sname]["_err2"] for k in keys)
        G2 = sum(per_weight[k][sname]["_gemm_err2"] for k in keys)
        numel = sum(per_weight[k]["_meta"]["numel"] for k in keys)
        nclip = sum(per_weight[k][sname]["n_clipped"] for k in keys)
        snrs = [per_weight[k][sname]["snr_db"] for k in keys]
        agg[sname] = {
            "snr_db_pooled": 10.0 * math.log10(W2 / max(E2, 1e-300)),
            "snr_db_mean": sum(snrs) / len(snrs),
            "snr_db_min": min(snrs),
            "rel_l2_pooled": math.sqrt(E2 / W2),
            "gemm_rel_l2_pooled": math.sqrt(G2 / Y2),
            "gemm_rel_l2_mean": sum(per_weight[k][sname]["gemm_rel_l2"] for k in keys) / len(keys),
            "max_err_over_absmax_max": max(per_weight[k][sname]["max_err_over_absmax"] for k in keys),
            "max_err_over_absmax_mean": sum(per_weight[k][sname]["max_err_over_absmax"] for k in keys) / len(keys),
            "clip_frac": nclip / numel,
            "n_weights": len(keys),
        }
        if "gamma" in per_weight[keys[0]][sname]:
            gs = [per_weight[k][sname]["gamma"] for k in keys]
            agg[sname]["gamma_mean"] = sum(gs) / len(gs)
            agg[sname]["gamma_min"] = min(gs)
            agg[sname]["gamma_max"] = max(gs)
    seps = [per_weight[k]["_sep"] for k in keys]
    agg["_sep"] = {
        "resid_std_bits_mean": sum(s["resid_std_bits"] for s in seps) / len(seps),
        "resid_std_bits_max": max(s["resid_std_bits"] for s in seps),
        "resid_mean_abs_bits_mean": sum(s["resid_mean_abs_bits"] for s in seps) / len(seps),
        "resid_max_bits_max": max(s["resid_max_bits"] for s in seps),
        "frac_entries_gt1bit": sum(s["n_entries_gt1bit"] for s in seps) / sum(s["n_entries"] for s in seps),
        "n_entries_gt1bit": sum(s["n_entries_gt1bit"] for s in seps),
        "n_entries": sum(s["n_entries"] for s in seps),
        "n_channels_gt1bit": sum(s["n_channels_gt1bit"] for s in seps),
        "n_channels": sum(s["n_channels"] for s in seps),
        "n_kblocks_gt1bit": sum(s["n_kblocks_gt1bit"] for s in seps),
        "n_kblocks": sum(s["n_kblocks"] for s in seps),
        "log2A_std_bits_mean": sum(s["log2A_std_bits"] for s in seps) / len(seps),
        "log2A_row_effect_std_bits_mean": sum(s["log2A_row_effect_std_bits"] for s in seps) / len(seps),
        "log2A_col_effect_std_bits_mean": sum(s["log2A_col_effect_std_bits"] for s in seps) / len(seps),
        "c_noclip_over_fit_max_bits_mean": sum(s["c_noclip_over_fit_max_bits"] for s in seps) / len(seps),
        "c_noclip_over_fit_max_bits_max": max(s["c_noclip_over_fit_max_bits"] for s in seps),
    }
    return agg


# ----------------------------------------------------------------------------- markdown
def fmt_table(rows, header):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


def write_md(res):
    ov = res["overall"]
    bt = res["by_type"]
    L = []
    L.append("# Cosmos3-Nano gen-tower weight scale layout study (INT8 symmetric, qmax 127)\n")
    L.append(f"Checkpoint: `{CKPT}`  ")
    L.append(f"Weights: {res['n_weights']} gen-tower linears (36 layers x q,k,v,o,gate,up,down); "
             f"diffusers names `self_attn.add_{{q,k,v}}_proj`, `self_attn.to_add_out` (= `{{q,k,v,o}}_proj_moe_gen`, "
             f"q/k/v are NOT fused), `mlp_moe_gen.{{gate,up,down}}_proj`. Shapes (N x K): q/o 4096x4096, k/v 1024x4096, "
             f"gate/up 12288x4096, down 4096x12288. Total {res['n_params'] / 1e9:.3f} B params.  ")
    L.append("Math: scale = absmax/127 (fp32), q = torch.round(W/scale) (half-even), clamp +-127, deq = q*scale. "
             "Groups along K. GEMM proxy: X ~ N(0,1) [2048,K] bf16-representable, fp32 matmul (TF32 off), "
             "rel-L2 of X@Wdeq^T vs X@W^T; same X for every scheme of a given K.  ")
    L.append("Aggregation: SNR/rel-L2/GEMM are *pooled* (energy-weighted: sum of squared errors over sum of squared "
             "signal across the weights in the group); `mean` columns are plain means of per-weight numbers. "
             "clip frac = clipped elements / all elements.\n")

    L.append("## Overall (all 252 weights)\n")
    rows = []
    for s, desc in SCHEMES:
        m = ov[s]
        g = f" (gamma mean {m['gamma_mean']:.3f}, range {m['gamma_min']:.3f}-{m['gamma_max']:.3f})" if "gamma_mean" in m else ""
        rows.append([desc + g, f"{m['snr_db_pooled']:.2f}", f"{m['snr_db_mean']:.2f}", f"{m['snr_db_min']:.2f}",
                     f"{m['rel_l2_pooled']:.4f}", f"{m['gemm_rel_l2_pooled']:.4f}",
                     f"{m['max_err_over_absmax_max']:.4f}", f"{m['clip_frac']:.2e}"])
    L.append(fmt_table(rows, ["scheme", "SNR dB (pooled)", "SNR dB (mean)", "SNR dB (min weight)", "rel-L2",
                              "GEMM rel-L2", "max err / max|W| (worst)", "clip frac"]))
    L.append("")

    L.append("## Per layer type: weight SNR dB (pooled)\n")
    rows = []
    for s, desc in SCHEMES:
        rows.append([desc] + [f"{bt[t][s]['snr_db_pooled']:.2f}" for t in TYPES] + [f"{ov[s]['snr_db_pooled']:.2f}"])
    L.append(fmt_table(rows, ["scheme"] + TYPES + ["all"]))
    L.append("")
    L.append("## Per layer type: GEMM-proxy rel-L2 (pooled)\n")
    rows = []
    for s, desc in SCHEMES:
        rows.append([desc] + [f"{bt[t][s]['gemm_rel_l2_pooled']:.4f}" for t in TYPES] + [f"{ov[s]['gemm_rel_l2_pooled']:.4f}"])
    L.append(fmt_table(rows, ["scheme"] + TYPES + ["all"]))
    L.append("")
    L.append("## Per layer type: relative L2 error (pooled)\n")
    rows = []
    for s, desc in SCHEMES:
        rows.append([desc] + [f"{bt[t][s]['rel_l2_pooled']:.4f}" for t in TYPES] + [f"{ov[s]['rel_l2_pooled']:.4f}"])
    L.append(fmt_table(rows, ["scheme"] + TYPES + ["all"]))
    L.append("")
    L.append("## Per layer type: max element error / max|W| (worst weight)\n")
    rows = []
    for s, desc in SCHEMES:
        rows.append([desc] + [f"{bt[t][s]['max_err_over_absmax_max']:.4f}" for t in TYPES] + [f"{ov[s]['max_err_over_absmax_max']:.4f}"])
    L.append(fmt_table(rows, ["scheme"] + TYPES + ["all"]))
    L.append("")
    L.append("## Per layer type: clip fraction (only d-variants clip; a/b/c/e/f/g are 0 by construction)\n")
    rows = []
    for s, desc in SCHEMES:
        if not s.startswith("d_"):
            continue
        rows.append([desc] + [f"{bt[t][s]['clip_frac']:.2e}" for t in TYPES] + [f"{ov[s]['clip_frac']:.2e}"])
    L.append(fmt_table(rows, ["scheme"] + TYPES + ["all"]))
    L.append("")
    L.append("## Per layer type: rescale factor gamma for the (d) log-fit variants (mean over weights; scale = gamma * exp(u+v)/127)\n")
    rows = []
    for s, desc in SCHEMES:
        if "gamma_mean" not in ov[s]:
            continue
        rows.append([desc] + [f"{bt[t][s]['gamma_mean']:.3f}" for t in TYPES] + [f"{ov[s]['gamma_mean']:.3f}"])
    L.append(fmt_table(rows, ["scheme"] + TYPES + ["all"]))
    L.append("")

    L.append("## Separability of the absmax matrix A[n,kb] (g=128), additive rank-1 fit in log2 domain\n")
    L.append("log2 A[n,kb] ~ u[n] + v[kb] (least squares; closed form = row means + col means - grand mean, "
             "the fixed point of alternating means). Residual in bits; '> 1 bit' = a block whose absmax is more than 2x "
             "off the separable prediction. `row effect std` / `col effect std` = spread of the per-channel / per-K-block "
             "log2 factors; `total log2A std` = spread before the fit.\n")
    rows = []
    for t in TYPES + ["all"]:
        s = (bt[t] if t != "all" else ov)["_sep"]
        rows.append([t, f"{s['log2A_std_bits_mean']:.3f}", f"{s['log2A_row_effect_std_bits_mean']:.3f}",
                     f"{s['log2A_col_effect_std_bits_mean']:.3f}", f"{s['resid_std_bits_mean']:.3f}",
                     f"{s['resid_std_bits_max']:.3f}", f"{s['resid_max_bits_max']:.2f}",
                     f"{s['frac_entries_gt1bit']:.2e}", f"{s['n_entries_gt1bit']}/{s['n_entries']}",
                     f"{s['n_channels_gt1bit']}/{s['n_channels']}", f"{s['n_kblocks_gt1bit']}/{s['n_kblocks']}",
                     f"{s['c_noclip_over_fit_max_bits_mean']:.2f}"])
    L.append(fmt_table(rows, ["type", "total log2A std (bits)", "row effect std", "col effect std",
                              "resid std (bits, mean)", "resid std (max weight)", "max |resid| (bits)",
                              "frac entries >1 bit", "entries >1 bit", "channels with any block >1 bit",
                              "K-blocks with any channel >1 bit", "log2(gamma_noclip) mean (bits)"]))
    L.append("")

    L.append("## Reading\n")
    L.append(res["reading"])
    L.append("")
    L.append("## Worst weights per scheme (lowest SNR)\n")
    rows = []
    for s, desc in SCHEMES:
        worst = sorted(res["per_weight"].items(), key=lambda kv: kv[1][s]["snr_db"])[:3]
        rows.append([desc, "; ".join(f"{k} {v[s]['snr_db']:.2f}" for k, v in worst)])
    L.append(fmt_table(rows, ["scheme", "3 lowest-SNR weights"]))
    L.append("")
    with open(OUT_MD, "w") as f:
        f.write("\n".join(L))


def make_reading(res):
    ov = res["overall"]
    a, b, c = ov["a_percol_g128"], ov["b_block128x128"], ov["c_sep_noclip"]
    d0, d1, d2, d3 = ov["d_sep_log_raw"], ov["d_sep_log_clip1e-4"], ov["d_sep_log_clip1e-3"], ov["d_sep_log_noclip"]
    e, f_, g = ov["e_percol_g64"], ov["f_perchannel"], ov["g_pertensor"]
    sep = ov["_sep"]
    best_d = max([("d1 clip<=0.01%", d1), ("d2 clip<=0.1%", d2), ("d3 no-clip", d3)], key=lambda kv: kv[1]["snr_db_pooled"])
    h = ov["h_sep_logc_rownoclip"]
    cr = res["c_kblock_factor_range_bits"]
    lines = [
        f"- **(c) as specified degenerates to per-channel.** With N >> K/128 (4096..12288 rows vs 32..96 K-blocks) every "
        f"K-block contains at least one row whose block absmax is that row's global absmax, so c[kb] = max_n A[n,kb]/s_w[n] = 1 "
        f"for (almost) every kb: the spread of c[kb] is {cr['mean']:.3f} bits on average (max {cr['max']:.2f} bits, exactly 0 for "
        f"q/k/down). Hence (c) == (f) per-channel to within {cr['max_snr_diff_db']:.2f} dB on any weight, and identical pooled.",
        f"- **(h) separable with the fitted c[kb]** (c[kb]=exp(v[kb]) from the log rank-1 fit, s_w[n] set per row for exact no-clip): "
        f"{h['snr_db_pooled']:.2f} dB, GEMM proxy {h['gemm_rel_l2_pooled']:.4f}, scale/absmax {res['h_scale_over_absmax_mean']:.2f}x. "
        f"This is the best no-clip separable layout, {a['snr_db_pooled'] - h['snr_db_pooled']:.2f} dB below (a) and "
        f"{h['snr_db_pooled'] - b['snr_db_pooled']:+.2f} dB vs (b); it beats per-channel by only "
        f"{h['snr_db_pooled'] - f_['snr_db_pooled']:.2f} dB because the per-K-block (column) effect is tiny "
        f"({sep['log2A_col_effect_std_bits_mean']:.3f} bits std vs {sep['log2A_row_effect_std_bits_mean']:.3f} bits for the per-channel effect).",
        f"- max err / max|W| for the clipped (d0/d1/d2) variants is dominated by the clipped outliers (up to ~0.9 = an outlier "
        f"quantized to +-127 at a scale that is far too small), not by rounding; for every non-clipping scheme it is 1/(2*127) = 0.0039.",
        f"- Bottom line: what per-col g128 buys over per-channel ({a['snr_db_pooled'] - f_['snr_db_pooled']:.2f} dB) is the "
        f"*non-separable* intra-row variation of absmax along K (residual {sep['resid_std_bits_mean']:.3f} bits std after the rank-1 fit); "
        f"a separable s_w[n]*c[kb] layout cannot recover it. Separable no-clip layouts land at the per-channel level "
        f"(~{f_['snr_db_pooled']:.1f} dB), which is still {f_['snr_db_pooled'] - b['snr_db_pooled']:.2f} dB better than 128x128 blockwise "
        f"(b) - i.e. per-channel scale beats 128-row sharing, consistent with handoff 3.6 (per-output-channel scale is the pillar). "
        f"Clipping-based rank-1 fits (d1/d2) are worse than (b).",
        f"- Baselines (pooled weight SNR over all 252 gen-tower weights): per-col g128 (a) {a['snr_db_pooled']:.2f} dB, "
        f"128x128 blockwise (b) {b['snr_db_pooled']:.2f} dB, per-col g64 (e) {e['snr_db_pooled']:.2f} dB, "
        f"per-channel (f) {f_['snr_db_pooled']:.2f} dB, per-tensor (g) {g['snr_db_pooled']:.2f} dB. "
        f"The a-vs-b gap is {a['snr_db_pooled'] - b['snr_db_pooled']:.2f} dB in weight SNR "
        f"(GEMM proxy rel-L2 {a['gemm_rel_l2_pooled']:.4f} vs {b['gemm_rel_l2_pooled']:.4f}).",
        f"- (c) separable no-clip: {c['snr_db_pooled']:.2f} dB, i.e. {a['snr_db_pooled'] - c['snr_db_pooled']:.2f} dB below (a) and "
        f"{c['snr_db_pooled'] - b['snr_db_pooled']:+.2f} dB vs (b). GEMM proxy {c['gemm_rel_l2_pooled']:.4f}. "
        f"Zero clipping by construction; the price is that scale is on average "
        f"{res['c_scale_over_absmax_mean']:.2f}x the per-block absmax.",
        f"- (d) separable log rank-1 fit: unscaled it clips {d0['clip_frac']:.2e} of elements ({d0['snr_db_pooled']:.2f} dB, "
        f"clipping dominates); rescaled to clip<=0.01%: {d1['snr_db_pooled']:.2f} dB (gamma mean {d1['gamma_mean']:.3f}); "
        f"clip<=0.1%: {d2['snr_db_pooled']:.2f} dB (gamma {d2['gamma_mean']:.3f}); exact no-clip: {d3['snr_db_pooled']:.2f} dB "
        f"(gamma {d3['gamma_mean']:.3f}). Best (d) variant is {best_d[0]} at {best_d[1]['snr_db_pooled']:.2f} dB, "
        f"{a['snr_db_pooled'] - best_d[1]['snr_db_pooled']:.2f} dB below (a) and {best_d[1]['snr_db_pooled'] - b['snr_db_pooled']:+.2f} dB vs (b).",
        f"- Separability: after removing the per-channel and per-K-block log2 factors the residual std of log2(A) is "
        f"{sep['resid_std_bits_mean']:.3f} bits (mean over weights; max {sep['resid_std_bits_max']:.3f}), from a total spread of "
        f"{sep['log2A_std_bits_mean']:.3f} bits (row effect {sep['log2A_row_effect_std_bits_mean']:.3f}, col effect "
        f"{sep['log2A_col_effect_std_bits_mean']:.3f}). {sep['frac_entries_gt1bit']:.2e} of (n,kb) blocks deviate by >1 bit "
        f"({sep['n_entries_gt1bit']} of {sep['n_entries']}); {sep['n_channels_gt1bit']}/{sep['n_channels']} channels and "
        f"{sep['n_kblocks_gt1bit']}/{sep['n_kblocks']} K-blocks have at least one such block. The no-clip rescale of the fit costs "
        f"log2(gamma) = {sep['c_noclip_over_fit_max_bits_mean']:.2f} bits on average (max {sep['c_noclip_over_fit_max_bits_max']:.2f}), "
        f"which is the resolution lost to the single worst block of each weight.",
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------------------- main
def main():
    log(f"torch {torch.__version__}, device {torch.cuda.get_device_name(0)}")
    with open(os.path.join(CKPT, "diffusion_pytorch_model.safetensors.index.json")) as f:
        wm = json.load(f)["weight_map"]
    shard_to_keys = defaultdict(list)
    names = []
    for k, sh in wm.items():
        m = NAME_RE.match(k)
        if m:
            shard_to_keys[sh].append(k)
            names.append(k)
    log(f"found {len(names)} gen-tower linears in {len(shard_to_keys)} shards")
    assert len(names) == 252, len(names)

    per_weight = {}
    n_params = 0
    for sh in sorted(shard_to_keys):
        keys = sorted(shard_to_keys[sh], key=lambda k: (int(NAME_RE.match(k).group(1)), k))
        log(f"shard {sh}: {len(keys)} weights")
        with safe_open(os.path.join(CKPT, sh), framework="pt", device="cpu") as f:
            for k in keys:
                m = NAME_RE.match(k)
                layer = int(m.group(1))
                leaf = m.group(3) or m.group(4)
                t = TYPE_OF[leaf]
                Wb = f.get_tensor(k)
                assert Wb.dtype == torch.bfloat16, Wb.dtype
                W = Wb.to(DEV, non_blocking=False).to(torch.float32)
                del Wb
                r = eval_weight(W)
                r["_meta"].update({"layer": layer, "type": t, "canonical": f"layers.{layer}.{CANON[t]}.weight",
                                   "shape": list(W.shape)})
                per_weight[k] = r
                n_params += W.numel()
                del W
                torch.cuda.empty_cache()
                if leaf == "down_proj":
                    a = r["a_percol_g128"]["snr_db"]; b = r["b_block128x128"]["snr_db"]
                    c = r["c_sep_noclip"]["snr_db"]; d = r["d_sep_log_clip1e-3"]["snr_db"]
                    log(f"  layer {layer:2d} done (down_proj: a {a:.2f} b {b:.2f} c {c:.2f} d1e-3 {d:.2f} dB; "
                        f"sep resid {r['_sep']['resid_std_bits']:.3f} bits)")

    log("aggregating")
    by_type = {t: aggregate(per_weight, [k for k in per_weight if per_weight[k]["_meta"]["type"] == t]) for t in TYPES}
    overall = aggregate(per_weight, list(per_weight))
    c_ratio = sum(per_weight[k]["c_sep_noclip"]["scale_over_absmax_mean"] for k in per_weight) / len(per_weight)
    h_ratio = sum(per_weight[k]["h_sep_logc_rownoclip"]["scale_over_absmax_mean"] for k in per_weight) / len(per_weight)
    cranges = [per_weight[k]["_sep"]["c_kblock_factor_range_bits"] for k in per_weight]
    c_kblock_factor_range_bits = {
        "mean": sum(cranges) / len(cranges), "max": max(cranges),
        "n_weights_exactly_zero": sum(1 for x in cranges if x == 0.0),
        "max_snr_diff_db": max(abs(per_weight[k]["c_sep_noclip"]["snr_db"] - per_weight[k]["f_perchannel"]["snr_db"])
                               for k in per_weight),
        "by_type_max": {t: max(per_weight[k]["_sep"]["c_kblock_factor_range_bits"] for k in per_weight
                               if per_weight[k]["_meta"]["type"] == t) for t in TYPES},
    }

    # per-weight output: drop private accumulators
    pw_out = {}
    for k, r in per_weight.items():
        pw_out[k] = {s: {kk: vv for kk, vv in r[s].items() if not kk.startswith("_")} for s, _ in SCHEMES}
        pw_out[k]["sep"] = r["_sep"]
        pw_out[k]["meta"] = {kk: vv for kk, vv in r["_meta"].items() if not kk.startswith("_")}

    res = {
        "checkpoint": CKPT,
        "n_weights": len(per_weight),
        "n_params": n_params,
        "qmax": QMAX, "gemm_M": GEMM_M,
        "schemes": {s: d for s, d in SCHEMES},
        "types": TYPES,
        "name_mapping": {"self_attn.add_q_proj": "q_proj_moe_gen", "self_attn.add_k_proj": "k_proj_moe_gen",
                         "self_attn.add_v_proj": "v_proj_moe_gen", "self_attn.to_add_out": "o_proj_moe_gen",
                         "mlp_moe_gen.gate_proj": "gate", "mlp_moe_gen.up_proj": "up", "mlp_moe_gen.down_proj": "down"},
        "shapes_by_type": {t: per_weight[next(k for k in per_weight if per_weight[k]["_meta"]["type"] == t)]["_meta"]["shape"] for t in TYPES},
        "overall": overall,
        "by_type": by_type,
        "c_scale_over_absmax_mean": c_ratio,
        "h_scale_over_absmax_mean": h_ratio,
        "c_kblock_factor_range_bits": c_kblock_factor_range_bits,
        "per_weight": pw_out,
    }
    res["reading"] = make_reading(res)
    res["per_weight"] = {k: {**v} for k, v in pw_out.items()}
    # write_md needs per-weight snr; pass the full per_weight table
    res_md = dict(res)
    res_md["per_weight"] = per_weight
    write_md(res_md)
    with open(OUT_JSON, "w") as f:
        json.dump(res, f, indent=1)
    log(f"wrote {OUT_JSON} and {OUT_MD}")
    print()
    print(res["reading"])


if __name__ == "__main__":
    main()
