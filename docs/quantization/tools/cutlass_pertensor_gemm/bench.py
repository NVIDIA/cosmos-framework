# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Drive the CUTLASS per-tensor INT8/FP8 GEMM binaries over the Cosmos3-Nano gen-tower shapes and
time the cuBLASLt baselines (bf16, FP8 per-tensor `_scaled_mm`, INT8 `_int_mm`) with the same
flush-L2 / median methodology. Writes results/<tag>.json and results/<tag>.md.

Usage (inside a container with torch + the built binaries):
  python bench.py --bin-dir build --tag gb200_$(date +%F) [--ms 901,1802,4096,16384,42240] [--iters 50]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import socket
import statistics
import subprocess
import sys

import torch

SHAPES = {  # name: (N, K)  -- Y[M,N] = A[M,K] @ W[N,K]^T
    "q/o_proj 4096x4096": (4096, 4096),
    "k/v_proj 1024x4096": (1024, 4096),
    "gate/up 12288x4096": (12288, 4096),
    "down 4096x12288": (4096, 12288),
}
FLUSH_BYTES = 512 << 20


def smi():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=clocks.sm,power.draw,temperature.gpu", "--format=csv,noheader,nounits"],
            text=True).strip().splitlines()[0]
        return out.replace(" ", "")
    except Exception:  # noqa: BLE001
        return "n/a"


def time_fn(fn, iters: int, warmup: int, flush: torch.Tensor | None):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for i in range(iters):
        if flush is not None:
            flush.fill_(i & 0xFF)
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e) * 1000.0)
    return statistics.median(times), statistics.fmean(times)


def baselines(m: int, n: int, k: int, iters: int, warmup: int, flush: torch.Tensor | None):
    dev = "cuda"
    rows = []
    flops = 2.0 * m * n * k
    a_bf = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
    w_bf = torch.randn(n, k, device=dev, dtype=torch.bfloat16)
    a_i8 = torch.randint(-127, 128, (m, k), device=dev, dtype=torch.int8)
    w_i8 = torch.randint(-127, 128, (n, k), device=dev, dtype=torch.int8)
    a_f8 = (a_bf * 30).to(torch.float8_e4m3fn)
    w_f8 = (w_bf * 30).to(torch.float8_e4m3fn)
    sa = torch.tensor(1 / 30, device=dev, dtype=torch.float32)
    sb = torch.tensor(1 / 30, device=dev, dtype=torch.float32)

    cands = {
        "bf16 cuBLAS": lambda: torch.matmul(a_bf, w_bf.t()),
        "fp8 per-tensor cuBLASLt (_scaled_mm)": lambda: torch._scaled_mm(
            a_f8, w_f8.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16),
        "int8 cuBLASLt (_int_mm, int32 out, no scale)": lambda: torch._int_mm(a_i8, w_i8.t()),
    }
    for name, fn in cands.items():
        try:
            fn()
            torch.cuda.synchronize()
            med, mean = time_fn(fn, iters, warmup, flush)
            rows.append(dict(kernel=name, m=m, n=n, k=k, us_med=med, us_mean=mean,
                             tflops=flops / (med * 1e-6) / 1e12, verify="n/a", clocks=smi()))
        except Exception as ex:  # noqa: BLE001
            rows.append(dict(kernel=name, m=m, n=n, k=k, us_med=None, us_mean=None, tflops=None,
                             verify=f"error: {str(ex).splitlines()[0][:120]}", clocks=smi()))
    del a_bf, w_bf, a_i8, w_i8, a_f8, w_f8
    torch.cuda.empty_cache()
    return rows


RESULT_RE = re.compile(r"RESULT (.*)")


def run_binary(path: str, m: int, n: int, k: int, iters: int, warmup: int, flush: bool, verify: int, extra: str = ""):
    cmd = [path, f"--m={m}", f"--n={n}", f"--k={k}", f"--iterations={iters}", f"--warmup={warmup}",
           f"--flush_l2={int(flush)}", f"--verify_samples={verify}"] + extra.split()
    label = os.path.basename(path) + (f" [{extra}]" if extra else "")
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        return dict(kernel=label, m=m, n=n, k=k, us_med=None, us_mean=None, tflops=None, verify="TIMEOUT (hang?)", clocks=smi())
    mm = RESULT_RE.search(p.stdout)
    if not mm:
        return dict(kernel=label, m=m, n=n, k=k, us_med=None, us_mean=None, tflops=None,
                    verify=f"error rc={p.returncode}: {(p.stderr or p.stdout).strip().splitlines()[-1:] }", clocks=smi())
    kv = dict(tok.split("=", 1) for tok in mm.group(1).split())
    sched = "" if kv.get("sched", "default") == "default" else f" {kv['sched']}"
    return dict(kernel=f"cutlass {kv['kind']} {kv['tile']} c{kv['cluster']}{sched}", m=m, n=n, k=k,
                us_med=float(kv["us_med"]), us_mean=float(kv["us_mean"]), tflops=float(kv["tflops"]),
                verify=f"{kv['verify']} (max_rel {kv['max_rel']})", clocks=smi())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-dir", default="build")
    ap.add_argument("--tag", default=f"{socket.gethostname()}_{dt.date.today().isoformat()}")
    ap.add_argument("--ms", default="901,1802,4096,16384,42240")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--no-flush", action="store_true")
    ap.add_argument("--verify-samples", type=int, default=2048)
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--extra", action="append", default=None,
                    help="extra CLI arg string, repeatable; each binary is run once per entry. Use the '=' form so "
                         "argparse does not eat it: --extra='' --extra='--swizzle=4' --extra='--decomposition=SplitK --splits=4'")
    args = ap.parse_args()

    ms = [int(x) for x in args.ms.split(",")]
    if not args.extra:
        args.extra = [""]
    bins = sorted(os.path.join(args.bin_dir, f) for f in os.listdir(args.bin_dir) if f.startswith("pertensor_gemm_"))
    flush = None if args.no_flush else torch.empty(FLUSH_BYTES, device="cuda", dtype=torch.uint8)
    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    print(f"# device: {torch.cuda.get_device_name()}  torch {torch.__version__}  binaries: {len(bins)}", flush=True)
    for shape_name, (n, k) in SHAPES.items():
        for m in ms:
            if not args.skip_baselines:
                for r in baselines(m, n, k, args.iters, args.warmup, flush):
                    r["shape"] = shape_name
                    rows.append(r)
                    print(f"{shape_name:22s} M={m:6d} {r['kernel']:48s} {r['us_med'] or 0:9.1f} us "
                          f"{r['tflops'] or 0:7.1f} TF/s {r['verify']} {r['clocks']}", flush=True)
            for b in bins:
              for extra in args.extra:
                r = run_binary(b, m, n, k, args.iters, args.warmup, flush is not None, args.verify_samples, extra)
                r["shape"] = shape_name
                rows.append(r)
                print(f"{shape_name:22s} M={m:6d} {r['kernel']:48s} {r['us_med'] or 0:9.1f} us "
                      f"{r['tflops'] or 0:7.1f} TF/s {r['verify']} {r['clocks']}", flush=True)

    meta = dict(device=torch.cuda.get_device_name(), torch=torch.__version__, host=socket.gethostname(),
                date=dt.datetime.now().isoformat(timespec="seconds"), iters=args.iters, warmup=args.warmup,
                flush_l2=flush is not None, ms=ms)
    with open(os.path.join(args.out_dir, f"{args.tag}.json"), "w") as fh:
        json.dump(dict(meta=meta, rows=rows), fh, indent=1)

    # markdown: one table per shape, kernels as rows, M as columns (TFLOPS)
    kernels = list(dict.fromkeys(r["kernel"] for r in rows))
    lines = [f"# per-tensor GEMM bench {args.tag}", "", f"{meta}", ""]
    for shape_name in SHAPES:
        lines += [f"## {shape_name}  (TFLOPS, median of {args.iters}, flush_l2={meta['flush_l2']})", "",
                  "| kernel | " + " | ".join(f"M={m}" for m in ms) + " |",
                  "|---|" + "---|" * len(ms)]
        for kname in kernels:
            cells = []
            for m in ms:
                hit = [r for r in rows if r["shape"] == shape_name and r["kernel"] == kname and r["m"] == m]
                if not hit or hit[0]["tflops"] is None:
                    cells.append("err")
                else:
                    v = hit[0]["verify"]
                    flag = "" if (v.startswith("PASS") or v == "n/a") else " ❌"
                    cells.append(f"{hit[0]['tflops']:.0f}{flag}")
            lines.append(f"| {kname} | " + " | ".join(cells) + " |")
        lines.append("")
        base = "fp8 per-tensor cuBLASLt (_scaled_mm)"
        lines += [f"### {shape_name}: speed relative to {base} (>1 = faster)", "",
                  "| kernel | " + " | ".join(f"M={m}" for m in ms) + " |",
                  "|---|" + "---|" * len(ms)]
        for kname in kernels:
            if kname == base:
                continue
            cells = []
            for m in ms:
                hit = [r for r in rows if r["shape"] == shape_name and r["kernel"] == kname and r["m"] == m]
                ref = [r for r in rows if r["shape"] == shape_name and r["kernel"] == base and r["m"] == m]
                if not hit or not ref or hit[0]["us_med"] is None or ref[0]["us_med"] is None:
                    cells.append("err")
                else:
                    cells.append(f"{ref[0]['us_med'] / hit[0]['us_med']:.2f}")
            lines.append(f"| {kname} | " + " | ".join(cells) + " |")
        lines.append("")
    with open(os.path.join(args.out_dir, f"{args.tag}.md"), "w") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
