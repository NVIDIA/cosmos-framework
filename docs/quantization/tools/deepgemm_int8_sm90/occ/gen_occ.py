"""Config table of the OCC (occupancy-2, approach D) extension (kernels/deepgemm_int8/occ, module deepgemm_int8_occ_ext). Isolated from
gen.py/configs.h and the pp/ extension; only `sm90_int8_gemm_1d2d_occ_impl` / `_occdb_impl` instantiations live here.
Fields: (block_m, block_n, stages, num_tma_multicast, multicast_on_a, compiled N, compiled K, variant '' | 'db', math_threads,
min_blocks_per_sm, math_regs). Register budget: launch regs = floor(65536 / (256 * mb) / 8) * 8; producer setmaxnreg.dec 40 frees
(launch - 40) * 128, so the math warpgroup can setmaxnreg.inc to 2 * launch - 40 (mb 2: 128 -> 216; mb 3: 80 -> 120).
smem per CTA must stay <= 233472 / mb - 1024 (mb 2: 115712 B; mb 3: 76800 B): the kernel keeps a separate bf16 D tile
(64x128: 16 KB), so 64x128 fits only 3 stages at 2 CTAs/SM (4 stages = 115904 B > 115712), 128x128 only 2, 64x256 only 2.
Re-run `python gen_occ.py` after editing CONFIGS."""
import os
HERE = os.path.dirname(os.path.abspath(__file__))

def ceil_div(a, b): return (a + b - 1) // b
def align(a, b): return ceil_div(a, b) * b
def smem(bm, bn, st, k=12288, hd=0):
    return align(bm * (64 if hd else bn) * 2, 1024) + st * (bm * 128 + bn * 128 + align(bm * 4, 128)) + align(ceil_div(k, 128) * 4 * (1 if 128 % bn == 0 else 2), 8) + st * 16
def launch_regs(mb): return (65536 // (256 * mb)) // 8 * 8
def math_regs(mb): return min(2 * launch_regs(mb) - 40, 248)

CONFIGS = [
    # (bm, bn, st, mc, on_a, sn, sk, variant, mt, mb, regs[, half_d])
    (64, 128, 3, 1, False, 0, 0, '', 128, 2, 216),            # 0: D1 dynamic shape (K = 128 / 256 edge cases)
    (64, 128, 3, 1, False, 4096, 4096, '', 128, 2, 216),      # 1: D1 64x128 s3 c1x1 N4096K4096, 2 CTAs/SM
    (64, 128, 3, 1, False, 4096, 4096, 'db', 128, 2, 216),    # 2: D4 = D1 + double-buffered accumulator (final 64 + 2x64 int32)
    (64, 128, 3, 2, False, 4096, 4096, '', 128, 2, 216),      # 3: D5 cluster 2x1 (B multicast)
    (64, 128, 3, 2, True, 4096, 4096, '', 128, 2, 216),       # 4: D5 cluster 1x2 (A multicast)
    (64, 64, 4, 1, False, 4096, 4096, '', 128, 3, 120),       # 5: 64x64 s4, 3 CTAs/SM (launch 80 regs -> math 120)
    (64, 64, 6, 1, False, 4096, 4096, '', 128, 2, 216),       # 6: 64x64 s6, 2 CTAs/SM
    (128, 128, 2, 1, False, 4096, 4096, '', 128, 2, 216),     # 7: D2 128x128 with ONE math warpgroup (2 sequential 64-row waves), 2 stages
    (64, 128, 3, 1, False, 12288, 4096, '', 128, 2, 216),     # 8: D1 N12288K4096
    (64, 128, 3, 1, False, 4096, 12288, '', 128, 2, 216),     # 9: D1 N4096K12288
    (64, 128, 3, 1, False, 1024, 4096, '', 128, 2, 216),      # 10: D1 N1024K4096
    (64, 128, 3, 1, False, 12288, 4096, 'db', 128, 2, 216),   # 11: D4 N12288K4096
    (64, 128, 3, 1, False, 4096, 12288, 'db', 128, 2, 216),   # 12: D4 N4096K12288
    (64, 128, 3, 1, False, 1024, 4096, 'db', 128, 2, 216),    # 13: D4 N1024K4096
    (64, 128, 3, 1, False, 0, 0, 'db', 128, 2, 216),          # 14: D4 dynamic shape (edge cases)
    (64, 128, 4, 1, False, 4096, 4096, '', 128, 2, 216, 1),    # 15: D1 + half-D epilogue -> 4 stages fit (107968 B)
    (64, 128, 4, 1, False, 4096, 4096, 'db', 128, 2, 216, 1),  # 16: D4 + half-D, 4 stages
    (64, 128, 4, 2, False, 4096, 4096, '', 128, 2, 216, 1),    # 17: D5 c2x1 + half-D, 4 stages
    (64, 128, 4, 1, False, 0, 0, '', 128, 2, 216, 1),          # 18: half-D dynamic shape (edge cases)
    # D3 (64x256): final 128 + int32 acc 128 = 256 regs > 216 AND smem 116000 B > 115712 B even with 2 stages -> infeasible at 2 CTAs/SM, not built
]

def rows():
    out = []
    for i, cfg in enumerate(CONFIGS):
        bm, bn, st, mc, on_a, sn, sk, variant, mt, mb, regs = cfg[:11]; hd = cfg[11] if len(cfg) > 11 else 0
        assert not hd or (bn * 2) % 128 == 0, 'half-D needs a 128B-swizzled D store'
        assert bm in (64, 128, 256) and bn % 16 == 0 and 16 <= bn <= 256 and mc in (1, 2) and variant in ('', 'db') and mt in (128, 256) and mb in (1, 2, 3)
        assert regs % 8 == 0 and regs <= math_regs(mb), f"cfg {i}: math regs {regs} exceed the budget {math_regs(mb)} for {mb} CTAs/SM"
        sm = smem(bm, bn, st, hd=hd)
        assert sm <= 232448, f"cfg {i}: smem {sm} exceeds 227 KB"
        fits = sm <= 233472 // mb - 1024
        cl = '1x2' if (mc == 2 and on_a) else '2x1' if mc == 2 else '1x1'
        name = f"{bm}x{bn} s{st} c{cl}" + (f" N{sn}K{sk}" if sn or sk else "") + (" DB" if variant == 'db' else "") + (" halfD" if hd else "") + f" mt{mt} occ{mb} r{regs}"
        if not fits: print(f"WARNING cfg {i} {name}: smem {sm} B > {233472 // mb - 1024} B limit for {mb} CTAs/SM")
        out.append(dict(i=i, bm=bm, bn=bn, st=st, mc=mc, on_a=on_a, sn=sn, sk=sk, variant=variant, mt=mt, mb=mb, regs=regs, hd=hd, name=name, smem=sm))
    return out

def main():
    rs = rows()
    for r in rs:
        tmpl = (f"{r['sn']}u, {r['sk']}u, {r['bm']}u, {r['bn']}u, {r['st']}u, {r['mc']}u, {'true' if r['on_a'] else 'false'}, "
                f"{r['mt']}u, {r['mb']}u, {r['regs']}u, {'true' if r['variant'] == 'db' else 'false'}, {'true' if r['hd'] else 'false'}")
        open(os.path.join(HERE, f"occcfg_{r['i']:02d}.cu"), "w").write(
            '// generated by occ/gen_occ.py -- one kernel instantiation per translation unit\n'
            '#include "launcher_occ.cuh"\n'
            + f"void dgint8_occ_run_{r['i']}(const dgint8::Args& args, dgint8::OccReport* rep) {{\n    dgint8::launch_occ<{tmpl}>(args, \"{r['name']}\", rep);\n}}\n")
    for f in os.listdir(HERE):
        if f.startswith("occcfg_") and f.endswith(".cu") and int(f[7:9]) >= len(rs):
            os.remove(os.path.join(HERE, f))
    with open(os.path.join(HERE, "occ_configs.h"), "w") as f:
        f.write('// generated by occ/gen_occ.py\n#pragma once\n#include "../dgint8_args.h"\n#include "occ_report.h"\n')
        f.write("struct DgInt8OccCfg { int block_m, block_n, stages, mcast, on_a, shape_n, shape_k, math_threads, min_blocks, math_regs, is_db, smem; const char* name; };\n")
        for r in rs:
            f.write(f"void dgint8_occ_run_{r['i']}(const dgint8::Args& args, dgint8::OccReport* rep);\n")
        f.write("static const DgInt8OccCfg kDgInt8OccCfgs[] = {\n")
        for r in rs:
            f.write(f"    {{{r['bm']}, {r['bn']}, {r['st']}, {r['mc']}, {int(r['on_a'])}, {r['sn']}, {r['sk']}, {r['mt']}, {r['mb']}, {r['regs']}, {int(r['variant'] == 'db')}, {r['smem']}, \"{r['name']}\"}},\n")
        f.write("};\ntypedef void (*dgint8_occ_run_fn)(const dgint8::Args&, dgint8::OccReport*);\nstatic const dgint8_occ_run_fn kDgInt8OccRuns[] = {\n")
        for r in rs:
            f.write(f"    dgint8_occ_run_{r['i']},\n")
        f.write("};\n")
        f.write(f"static const int kDgInt8OccNumCfgs = {len(rs)};\n")
    print("\n".join(f"occ cfg {r['i']}: {r['name']}  smem {r['smem']} B" for r in rs))

if __name__ == "__main__":
    main()
