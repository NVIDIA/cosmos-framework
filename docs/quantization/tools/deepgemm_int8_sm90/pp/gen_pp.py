"""Config table of the math-warpgroup ping-pong extension (kernels/deepgemm_int8/pp, module deepgemm_int8_pp_ext). Isolated from the
shared gen.py/configs.h so the default cfg ids and the DB variant are untouched; only `sm90_int8_gemm_1d2d_pp_impl` instantiations
live here (no duplicate kernel symbols -> can be loaded next to the main extension for in-process bit-identity checks).
Fields: (block_m, block_n, stages, num_tma_multicast, multicast_on_a, compiled N, compiled K, pp_mode) with pp_mode 1 = WG0 syncs
after its signal (uniform), 2 = both sync before issuing, WG0 skips the first wait (see ../gen_kernel_pp.py).
Re-run `python gen_pp.py` after editing CONFIGS."""
import os
HERE = os.path.dirname(os.path.abspath(__file__))

CONFIGS = [
    # 128x128 (one 64-row wave per math warpgroup): direct comparison with the DB kernel (main cfg 21/19 = 1140 / 1087 TFLOPS at 4096^3)
    (128, 128, 5, 1, False, 4096, 4096, 1),    # 0: 128x128 s5 c1x1 N4096K4096 pp1
    (128, 128, 5, 2, True, 4096, 4096, 1),     # 1: 128x128 s5 c1x2 N4096K4096 pp1
    (128, 128, 5, 1, False, 4096, 4096, 2),    # 2: 128x128 s5 c1x1 N4096K4096 pp2
    (128, 128, 5, 2, True, 4096, 4096, 2),     # 3: 128x128 s5 c1x2 N4096K4096 pp2
    (128, 128, 4, 2, True, 4096, 4096, 1),     # 4: 128x128 s4 c1x2 N4096K4096 pp1 (stage-count comparison)
    # 256x128 (two waves per warpgroup, handshake per wave): comparison with the default best (main cfg 8 = 1061 TFLOPS at 4096^3)
    (256, 128, 3, 2, True, 4096, 4096, 1),     # 5: 256x128 s3 c1x2 N4096K4096 pp1
    (256, 128, 3, 1, False, 4096, 4096, 1),    # 6: 256x128 s3 c1x1 N4096K4096 pp1
    (256, 128, 3, 2, True, 4096, 4096, 2),     # 7: 256x128 s3 c1x2 N4096K4096 pp2
    # other production (N, K)
    (128, 128, 5, 2, True, 4096, 12288, 1),    # 8: 128x128 s5 c1x2 N4096K12288 pp1
    (128, 128, 5, 2, True, 12288, 4096, 1),    # 9: 128x128 s5 c1x2 N12288K4096 pp1
    (128, 128, 5, 2, True, 1024, 4096, 1),     # 10: 128x128 s5 c1x2 N1024K4096 pp1
    (256, 128, 3, 2, True, 4096, 12288, 1),    # 11: 256x128 s3 c1x2 N4096K12288 pp1
    (256, 128, 3, 2, True, 12288, 4096, 1),    # 12: 256x128 s3 c1x2 N12288K4096 pp1
    (256, 128, 3, 2, True, 1024, 4096, 1),     # 13: 256x128 s3 c1x2 N1024K4096 pp1
    # dynamic shapes (K = 128 / 256 edge cases)
    (128, 128, 5, 1, False, 0, 0, 1),          # 14: 128x128 s5 c1x1 pp1 dynamic
    (256, 128, 3, 2, True, 0, 0, 1),           # 15: 256x128 s3 c1x2 pp1 dynamic
    # round 2 (dual wgmma chains per warpgroup, pp3/pp4) was exact but slower (930-945 / 891 TFLOPS at 4096^3): see logs/pp_time2.log
    # round 3: true tensor-core mutual exclusion -- signal after wait<0> (pp5) or after wait<1> of a split commit group (pp6)
    (128, 128, 5, 1, False, 4096, 4096, 5),    # 16: 128x128 s5 c1x1 N4096K4096 pp5
    (128, 128, 5, 2, True, 4096, 4096, 5),     # 17: 128x128 s5 c1x2 N4096K4096 pp5
    (256, 128, 3, 2, True, 4096, 4096, 5),     # 18: 256x128 s3 c1x2 N4096K4096 pp5
    (128, 128, 5, 1, False, 4096, 4096, 6),    # 19: 128x128 s5 c1x1 N4096K4096 pp6
    (256, 128, 3, 2, True, 4096, 4096, 6),     # 20: 256x128 s3 c1x2 N4096K4096 pp6
    (128, 128, 5, 2, True, 12288, 4096, 5),    # 21: 128x128 s5 c1x2 N12288K4096 pp5
    (256, 128, 3, 2, True, 12288, 4096, 5),    # 22: 256x128 s3 c1x2 N12288K4096 pp5
    (128, 128, 5, 1, False, 0, 0, 5),          # 23: 128x128 s5 c1x1 pp5 dynamic (K = 128 / 256 edge cases)
    (256, 128, 3, 2, True, 4096, 12288, 5),    # 24: 256x128 s3 c1x2 N4096K12288 pp5
    (256, 128, 3, 2, True, 1024, 4096, 5),     # 25: 256x128 s3 c1x2 N1024K4096 pp5
    (256, 128, 3, 2, True, 0, 0, 6),           # 26: 256x128 s3 c1x2 pp6 dynamic (K = 128 / 256 edge cases for the split-group path)
    # round 4: tensor-core mutual exclusion + dual chain per warpgroup
    (128, 128, 5, 1, False, 4096, 4096, 7),    # 27: 128x128 s5 c1x1 N4096K4096 pp7
    (128, 128, 5, 2, True, 4096, 4096, 7),     # 28: 128x128 s5 c1x2 N4096K4096 pp7
    (128, 128, 5, 2, True, 12288, 4096, 7),    # 29: 128x128 s5 c1x2 N12288K4096 pp7
    (128, 128, 5, 2, True, 4096, 12288, 7),    # 30: 128x128 s5 c1x2 N4096K12288 pp7
    (128, 128, 5, 1, False, 0, 0, 7),          # 31: 128x128 s5 c1x1 pp7 dynamic (K = 128 / 256 edge cases)
]

def rows():
    out = []
    for i, (bm, bn, st, mc, on_a, sn, sk, pm) in enumerate(CONFIGS):
        assert bm in (128, 256) and pm in (1, 2, 3, 4, 5, 6, 7), "ping-pong needs two math warpgroups (BLOCK_M in {128, 256})"
        assert pm not in (3, 4, 7) or bm == 128, "dual-chain modes need BLOCK_M = 128"
        name = (f"{bm}x{bn} s{st} c{'1x2' if (mc == 2 and on_a) else '2x1' if mc == 2 else '1x1'}" + (f" N{sn}K{sk}" if sn or sk else "")
                + f" pp{pm}")
        out.append(dict(i=i, bm=bm, bn=bn, st=st, mc=mc, on_a=on_a, sn=sn, sk=sk, pm=pm, mt=256, name=name))
    return out

def main():
    rs = rows()
    for r in rs:
        tmpl = f"{r['sn']}u, {r['sk']}u, {r['bm']}u, {r['bn']}u, {r['st']}u, {r['mc']}u, {'true' if r['on_a'] else 'false'}, {r['pm']}u"
        open(os.path.join(HERE, f"ppcfg_{r['i']:02d}.cu"), "w").write(
            '// generated by pp/gen_pp.py -- one kernel instantiation per translation unit\n'
            '#include "../launcher_pp.cuh"\n'
            + f"void dgint8_pp_run_{r['i']}(const dgint8::Args& args) {{\n    dgint8::launch_pp<{tmpl}>(args);\n}}\n")
    for f in os.listdir(HERE):
        if f.startswith("ppcfg_") and f.endswith(".cu") and int(f[6:8]) >= len(rs):
            os.remove(os.path.join(HERE, f))
    with open(os.path.join(HERE, "pp_configs.h"), "w") as f:
        f.write('// generated by pp/gen_pp.py\n#pragma once\n#include "../dgint8_args.h"\n')
        for r in rs:
            f.write(f"void dgint8_pp_run_{r['i']}(const dgint8::Args& args);\n")
        f.write("static const dgint8::CfgInfo kDgInt8PpCfgs[] = {\n")
        for r in rs:
            f.write(f"    {{{r['bm']}, {r['bn']}, {r['st']}, {r['mc']}, {int(r['on_a'])}, {r['sn']}, {r['sk']}, {r['mt']}, {r['pm']}, \"{r['name']}\"}},\n")
        f.write("};\ntypedef void (*dgint8_pp_run_fn)(const dgint8::Args&);\nstatic const dgint8_pp_run_fn kDgInt8PpRuns[] = {\n")
        for r in rs:
            f.write(f"    dgint8_pp_run_{r['i']},\n")
        f.write("};\n")
        f.write(f"static const int kDgInt8PpNumCfgs = {len(rs)};\n")
    print("\n".join(f"pp cfg {r['i']}: {r['name']}" for r in rs))

if __name__ == "__main__":
    main()
