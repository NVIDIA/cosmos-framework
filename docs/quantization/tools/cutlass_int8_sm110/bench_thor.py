#!/usr/bin/env python3
"""Sweep driver for the Thor GEMM benchmarks. Runs ./pertensor_gemm (CUTLASS fp8/int8, all or selected cfgs) and
./cublaslt_bench (bf16 / fp8 / int8 baselines) over the Cosmos3-Nano gen-tower shapes, collects the tab-separated result
lines, and writes a CSV plus a markdown summary.

  python bench_thor.py --stage tune          # all CUTLASS cfgs, 4 shapes x M in {901, 4096}: pick the best cfg per dtype
  python bench_thor.py --stage full --int8-cfg 2 --fp8-cfg 2   # chosen cfgs + cuBLASLt baselines over the full M grid
  python bench_thor.py --stage full --int8-cfg 2,7 --fp8-cfg 2,7   # several cfgs (best-of reported per row)
Results go to results/<stage>_<timestamp>.{csv,md}; every raw line is also appended to results/<stage>_<timestamp>.log.
"""
import argparse, csv, datetime, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
SHAPES = [(4096, 4096), (1024, 4096), (12288, 4096), (4096, 12288)]  # (N, K): q/o, k/v, gate/up, down
MS_FULL = [901, 1802, 4096, 16384, 42240]
MS_TUNE = [901, 4096]


def parse_line(line):
    d = {}
    for tok in line.strip().split("\t"):
        if "=" in tok:
            k, v = tok.split("=", 1)
            if k == "verify":  # "verify=PASS rel_l2=1.6e-3"
                m = re.match(r"(\w+) rel_l2=(\S+)", v)
                d["verify"], d["rel_l2"] = m.group(1), m.group(2)
            else:
                d[k] = v
    return d


def run(cmd, log):
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE)
    dt = time.time() - t0
    with open(log, "a") as f:
        f.write(f"$ {' '.join(cmd)}  ({dt:.1f}s)\n{p.stdout}{p.stderr}\n")
    rows = [parse_line(l) for l in p.stdout.splitlines() if l.startswith("lib=")]
    if p.returncode != 0 and not rows:
        rows = [{"lib": cmd[0], "ERROR": (p.stderr.strip().splitlines() or ["rc=%d" % p.returncode])[-1][:120]}]
    return rows


def gpu_clock():
    try:
        return int(open("/sys/class/devfreq/gpu-gpc-0/cur_freq").read()) // 1000000
    except Exception:
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["tune", "full"], default="tune")
    ap.add_argument("--int8-cfg", default="all")
    ap.add_argument("--fp8-cfg", default="all")
    ap.add_argument("--Ms", default=None, help="comma list, overrides stage default")
    ap.add_argument("--shapes", default=None, help="comma list of NxK")
    ap.add_argument("--swizzles", default="auto", help="comma list passed as --swizzle= to pertensor_gemm (e.g. auto,0,8,16)")
    ap.add_argument("--raster", default="H", help="H|M|N raster order for the CUTLASS CLC scheduler")
    ap.add_argument("--autotune", type=int, default=1, help="cuBLASLt --autotune")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--flush", type=int, default=1)
    ap.add_argument("--nw", type=int, default=1, help="rotate through N weight copies (--flush=0 --nw=8 = activations hot, weights cold)")
    ap.add_argument("--warmup-ms", type=int, default=300, help="min warm-up wall time; >=1500 puts FP8 into its sustained power-limited regime")
    ap.add_argument("--dist", default="uniform", help="CUTLASS operand distribution: uniform | normal")
    ap.add_argument("--no-cublaslt", action="store_true")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--summarize", default=None, help="do not run anything; rebuild csv/md from an existing results/*.log")
    a = ap.parse_args()
    if a.summarize:
        rows = []
        for l in open(a.summarize):
            if l.startswith("lib="):
                rows.append(parse_line(l))
        cases = []
        for r in rows:
            if all(k in r for k in ("N", "K", "M")):
                c = ((int(r["N"]), int(r["K"])), int(r["M"]))
                if c not in cases: cases.append(c)
        shapes = [c[0] for c in cases if c[0] not in [x[0] for x in cases[:cases.index(c)]]]
        Ms = sorted(set(c[1] for c in cases))
        base = a.summarize[:-4] if a.summarize.endswith(".log") else a.summarize
        write_outputs(rows, shapes, Ms, base + ".csv", base + ".md", a, os.path.basename(base))
        return

    ncfg = len([l for l in subprocess.run(["./pertensor_gemm", "--list"], capture_output=True, text=True, cwd=HERE).stdout.splitlines() if l.startswith("cfg")])
    def cfgs(s):
        return list(range(ncfg)) if s == "all" else [int(x) for x in s.split(",")]
    int8_cfgs, fp8_cfgs = cfgs(a.int8_cfg), cfgs(a.fp8_cfg)
    Ms = [int(x) for x in a.Ms.split(",")] if a.Ms else (MS_TUNE if a.stage == "tune" else MS_FULL)
    shapes = [tuple(int(v) for v in s.split("x")) for s in a.shapes.split(",")] if a.shapes else SHAPES
    use_lt = (a.stage == "full" or a.int8_cfg != "all") and not a.no_cublaslt and os.path.exists(os.path.join(HERE, "cublaslt_bench"))

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(HERE, "results", f"{a.stage}{('_' + a.tag) if a.tag else ''}_{stamp}")
    log, csvp, mdp = base + ".log", base + ".csv", base + ".md"
    common = [f"--iters={a.iters}", f"--warmup={a.warmup}", f"--warmup_ms={a.warmup_ms}", f"--flush={a.flush}", f"--nw={a.nw}", f"--verify={0 if a.no_verify else 1}"]
    rows = []
    for (N, K) in shapes:
        for M in Ms:
            shp = [f"--m={M}", f"--n={N}", f"--k={K}"]
            if use_lt:
                for dt in ["bf16", "fp8", "int8"]:
                    rows += run(["./cublaslt_bench", f"--dtype={dt}", f"--autotune={a.autotune}"] + shp + common, log)
            for dt, cl in (("fp8", fp8_cfgs), ("int8", int8_cfgs)):
                for c in cl:
                    for sw in a.swizzles.split(","):
                        rows += run(["./pertensor_gemm", f"--dtype={dt}", f"--cfg={c}", f"--swizzle={sw}", f"--raster={a.raster}", f"--dist={a.dist}"] + shp + common, log)
            done = [r for r in rows if r.get("M") == str(M) and r.get("N") == str(N)]
            print(f"[{N}x{K} M={M}] clk={gpu_clock()} MHz  " + "  ".join(
                f"{r.get('lib')}/{r.get('dtype')}{('/c' + r['cfg'] + '/s' + r.get('swizzle','?')) if 'cfg' in r else ''}={r.get('tflops', r.get('ERROR', '?'))}" for r in done), flush=True)

    write_outputs(rows, shapes, Ms, csvp, mdp, a, stamp)
    print(f"\nwrote {csvp}\n      {mdp}\n      {log}")


def write_outputs(rows, shapes, Ms, csvp, mdp, a, stamp):
    keys = ["lib", "dtype", "cfg", "swizzle", "raster", "flush", "nw", "warmup_ms", "dist", "M", "N", "K", "median_us", "mean_us", "min_us", "tflops", "clk_before", "clk_after", "verify", "rel_l2", "algo", "desc", "ERROR"]
    with open(csvp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Markdown: one row per (shape, M); columns = each (lib, dtype[, cfg]) with TFLOPS (median us); best CUTLASS cfg per dtype marked.
    def key(r):
        return (r.get("lib"), r.get("dtype"), r.get("cfg", ""), r.get("swizzle", ""))
    cols = []
    for r in rows:
        if key(r) not in cols and "tflops" in r:
            cols.append(key(r))
    def colname(c):
        return f"{c[0]} {c[1]}" + (f" cfg{c[2]} swz{c[3]}" if c[2] != "" else "")
    mode = "all operands cold (L2 flushed before every iteration)" if a.flush else (f"activations hot, weights cold ({a.nw} rotating weight copies)" if a.nw > 1 else "everything L2-hot (no flush, single weight)")
    lines = [f"# {a.stage} run {stamp}", "", f"iters={a.iters} warmup={a.warmup} warmup_ms={a.warmup_ms} flush_L2={a.flush} nw={a.nw} dist={a.dist} -> {mode}; cell = TFLOPS (median us); GPU clock sampled after each cell in the CSV.", "",
             "| N x K | M | " + " | ".join(colname(c) for c in cols) + " |", "|---|---:|" + "---:|" * len(cols)]
    for (N, K) in shapes:
        for M in Ms:
            cells = []
            for c in cols:
                r = next((x for x in rows if key(x) == c and x.get("M") == str(M) and x.get("N") == str(N) and x.get("K") == str(K)), None)
                if r is None:
                    cells.append("")
                elif "tflops" in r:
                    flag = "" if r.get("verify", "PASS") in ("PASS", "SKIP") else " **FAIL**"
                    cells.append(f"{float(r['tflops']):.0f} ({float(r['median_us']):.0f}){flag}")
                else:
                    cells.append("ERR")
            lines.append(f"| {N}x{K} | {M} | " + " | ".join(cells) + " |")
    # Summary: best CUTLASS int8 / fp8 per case (over cfgs and swizzles) against the cuBLASLt baselines.
    def best(dt, M, N, K):
        cand = [r for r in rows if r.get("lib") == "cutlass" and r.get("dtype") == dt and "tflops" in r and r.get("M") == str(M) and r.get("N") == str(N) and r.get("K") == str(K)]
        return max(cand, key=lambda r: float(r["tflops"])) if cand else None
    def lt(dt, M, N, K):
        return next((r for r in rows if r.get("lib") == "cublaslt" and r.get("dtype") == dt and "tflops" in r and r.get("M") == str(M) and r.get("N") == str(N) and r.get("K") == str(K)), None)
    def tf(r):
        return float(r["tflops"]) if r else None
    def ratio(a, b):
        return f"{a / b:.2f}" if (a and b) else ""
    lines += ["", "## Summary: best CUTLASS per dtype vs cuBLASLt (TFLOPS = 2MNK / median time; INT8 counted as TFLOPS)", "",
              "| N x K | M | cuBLASLt bf16 | cuBLASLt FP8 pt | cuBLASLt INT8 (s32 out) | cuBLASLt INT8+rescale | CUTLASS FP8 pt (cfg/swz) | CUTLASS INT8 pt (cfg/swz) | INT8/FP8 (CUTLASS) | INT8/cuBLASLt FP8 | INT8/bf16 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for (N, K) in shapes:
        for M in Ms:
            b8, i8 = best("fp8", M, N, K), best("int8", M, N, K)
            lbf, lf8, li8, li8r = lt("bf16", M, N, K), lt("fp8", M, N, K), lt("int8", M, N, K), lt("int8+rescale", M, N, K)
            fmt = lambda r: f"{float(r['tflops']):.0f} ({float(r['median_us']):.0f} us)" if r else ""
            fmtc = lambda r: f"{float(r['tflops']):.0f} ({float(r['median_us']):.0f} us; c{r['cfg']}/s{r.get('swizzle','')})" if r else ""
            lines.append(f"| {N}x{K} | {M} | {fmt(lbf)} | {fmt(lf8)} | {fmt(li8)} | {fmt(li8r)} | {fmtc(b8)} | {fmtc(i8)} | {ratio(tf(i8), tf(b8))} | {ratio(tf(i8), tf(lf8))} | {ratio(tf(i8), tf(lbf))} |")
    # best cfg per dtype summary
    for dt in ("fp8", "int8"):
        sub = [r for r in rows if r.get("lib") == "cutlass" and r.get("dtype") == dt and "tflops" in r]
        if not sub:
            continue
        lines += ["", f"## CUTLASS {dt}: best cfg per case", "", "| N x K | M | best cfg | swizzle | TFLOPS | desc |", "|---|---:|---:|---:|---:|---|"]
        for (N, K) in shapes:
            for M in Ms:
                cand = [r for r in sub if r.get("M") == str(M) and r.get("N") == str(N) and r.get("K") == str(K)]
                if cand:
                    b = max(cand, key=lambda r: float(r["tflops"]))
                    lines.append(f"| {N}x{K} | {M} | {b['cfg']} | {b.get('swizzle','')} | {float(b['tflops']):.0f} | {b.get('desc','')} |")
    with open(mdp, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
