#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Summarize a gpu_sampler.ps1 CSV: peaks, temperature range, clock-event reasons seen. Prints JSON (and writes <csv>.summary.json)."""
import csv, json, sys
path = sys.argv[1]
rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
def col(name, cast=float):
    out = []
    for r in rows:
        try: out.append(cast(r[name]))
        except Exception: pass
    return out
def mx(name, cast=float): v = col(name, cast); return max(v) if v else None
def mn(name): v = col(name); return min(v) if v else None
reasons = sorted(set(r["reasons"] for r in rows if r.get("reasons")))
summary = {
    "csv": path, "samples": len(rows), "first_ts": rows[0]["ts"] if rows else None, "last_ts": rows[-1]["ts"] if rows else None,
    "temp_c_min": mn("temp_c"), "temp_c_max": mx("temp_c"), "sm_mhz_max": mx("sm_mhz"), "sm_mhz_min_nonzero": min([v for v in col("sm_mhz") if v > 0], default=None),
    "power_w_max": mx("power_w"), "util_pct_max": mx("util_pct"), "clock_event_reasons_seen": reasons,
    "peak_smi_mem_used_mib": mx("smi_mem_used_mib"), "peak_sys_used_gib": mx("sys_used_gib"), "baseline_sys_used_gib": col("sys_used_gib")[0] if rows else None,
    "peak_proc_ws_gib": mx("ws_gib"), "peak_proc_private_gib": mx("private_gib"), "peak_threads": mx("threads", int), "max_procs": mx("n_procs", int),
    "peak_gpu_proc_dedicated_gib": mx("gpu_dedicated_gib"), "peak_gpu_proc_shared_gib": mx("gpu_shared_gib"), "peak_gpu_proc_committed_gib": mx("gpu_committed_gib"),
    "peak_adapter_dedicated_gib": mx("adapter_dedicated_gib"), "peak_adapter_shared_gib": mx("adapter_shared_gib"),
}
if rows:
    base_ded = col("adapter_dedicated_gib")[0]; base_sha = col("adapter_shared_gib")[0]
    summary["adapter_dedicated_delta_gib"] = round((summary["peak_adapter_dedicated_gib"] or 0) - base_ded, 3)
    summary["adapter_shared_delta_gib"] = round((summary["peak_adapter_shared_gib"] or 0) - base_sha, 3)
    summary["sys_used_delta_gib"] = round((summary["peak_sys_used_gib"] or 0) - summary["baseline_sys_used_gib"], 3)
json.dump(summary, open(path + ".summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
