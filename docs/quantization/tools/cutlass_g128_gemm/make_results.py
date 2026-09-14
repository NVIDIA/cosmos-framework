# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Assemble results/<tag>.md from RESULT lines in bench logs plus per-tensor reference rows.
Usage: python3 make_results.py --tag gb200_g128_2026-09-14 LABEL=path/to/log [LABEL=path ...]
  LABEL is prefixed to the kernel description parsed from the RESULT line (kind/tile/sfn)."""
import argparse, json, os, re, sys

SHAPES = {(4096, 4096): "q/o_proj 4096x4096", (1024, 4096): "k/v_proj 1024x4096",
          (12288, 4096): "gate/up 12288x4096", (4096, 12288): "down 4096x12288"}
MS = [901, 1802, 4096, 16384, 42240]
REF = os.path.join(os.path.dirname(__file__), "..", "cutlass_pertensor_gemm", "results", "gb200_2026-09-13_2158916.json")
REF_ROWS = [("bf16 cuBLAS", "bf16 cuBLAS"), ("FP8 per-tensor cuBLASLt", "fp8 per-tensor cuBLASLt (_scaled_mm)"),
            ("INT8 per-tensor cuBLASLt _int_mm", "int8 cuBLASLt (_int_mm, int32 out, no scale)"),
            ("FP8 per-tensor CUTLASS 256x128x128 c2x1", "cutlass fp8 256x128x128 c2x1"),
            ("INT8 per-tensor CUTLASS 256x128x128 c2x1", "cutlass int8 256x128x128 c2x1")]

def parse(label, path):
    rows = {}
    for line in open(path):
        m = re.search(r"RESULT (.*)", line)
        if not m:
            continue
        kv = dict(t.split("=", 1) for t in m.group(1).split())
        key = f"{label} {kv['kind']} {kv['tile']} c{kv['cluster']} sfn{kv['sfn']}"
        rows[(key, int(kv["n"]), int(kv["k"]), int(kv["m"]))] = (float(kv["tflops"]), kv["verify"])
    return rows

ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("logs", nargs="+")
a = ap.parse_args()
rows = {}
order = []
for spec in a.logs:
    label, path = spec.split("=", 1)
    r = parse(label, path)
    rows.update(r)
    for k in r:
        if k[0] not in order:
            order.append(k[0])
ref = json.load(open(REF))["rows"]
def refv(shape, m, kernel):
    x = [r for r in ref if r["shape"] == shape and r["m"] == m and r["kernel"] == kernel]
    return x[0]["tflops"] if x else None
out = [f"# g128 GEMM bench {a.tag} (TFLOPS, GB200, flush-L2 median of 50; g128 rows pad M to a multiple of 4, TFLOPS on true M)", ""]
for (n, k), shape in SHAPES.items():
    out += [f"## {shape} (N x K)", "", "| kernel | " + " | ".join(f"M={m}" for m in MS) + " |", "|---|" + "---|" * len(MS)]
    for lab, kern in REF_ROWS:
        out.append(f"| {lab} | " + " | ".join(f"{refv(shape, m, kern):.0f}" if refv(shape, m, kern) else "-" for m in MS) + " |")
    for key in order:
        cells = []
        for m in MS:
            v = rows.get((key, n, k, m))
            cells.append("-" if v is None else (f"{v[0]:.0f}" + ("" if v[1] == "PASS" else " ❌")))
        out.append(f"| {key} | " + " | ".join(cells) + " |")
    out.append("")
os.makedirs("results", exist_ok=True)
open(f"results/{a.tag}.md", "w").write("\n".join(out)); print("\n".join(out))
