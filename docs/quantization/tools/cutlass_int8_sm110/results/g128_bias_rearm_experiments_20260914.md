# TMEM pre-biased promotion: re-arm warpgroup implementation and ablations (2026-09-14 evening)

Single-config debug builds (`-DTHOR_BW_DEBUG_ONLY16`, cfg16 = 2SM 256x256x128 per-col g128, epi 128x32), INT8, 4096x4096 M=1520,
hot-A/cold-W, 1 s warm-up, 30 iterations. All `verify=PASS` unless the variant is an ablation (marked wrong).

| variant | flags | us | TFLOPS | note |
|---|---|---:|---:|---|
| control (I2F promotion) | default | 323.6 | 157.6 | |
| **pre-bias, re-arm warpgroup (warps 12-15), complete** | `-DG128_OPT_BIAS` | 479.1 | 106.5 | PASS; slower |
| protocol only (re-arm warps wait+release, no stores, I2F math; wrong) | `BIAS + DBG_REARM_NOSTORE + DBG_BIAS_OLDMATH` | -- | 155.7 | earlier body; ~= control |
| no per-column scale loads (wrong) | `-DG128_EXP_NO_SFB` | 276.2 | **184.7** | the LDS.128 broadcasts of s_w cost 25 % |
| no scale loads + protocol only (wrong) | `NO_SFB + BIAS + NOSTORE + OLDMATH` | 319.3 | 159.7 | protocol hop costs 14 % once promotion is faster |
| no scale loads + bias math only, no protocol (wrong) | `NO_SFB + BIAS + REARM_PASSIVE + FORCEMATH` | 299.0 | 170.6 | 2-FFMA promotion is *slower* than I2F+FMUL+FFMA here |
| no scale loads + bias math + protocol (wrong) | `NO_SFB + BIAS + NOSTORE` | 321.5 | 158.7 | |
| FP8 control / FP8 no scale loads | | 286.6 / 231.3 | 178.0 / 220.5 | FP8 has no I2F and is not faster than INT8 per-col |

Conclusions
- The pre-bias trick works numerically (bit-identical promotion, PASS on every config) and the re-arm warpgroup removes the register
  problem (32 B spills), but the per-col kernel is **not FP-issue bound**: replacing I2F+FMUL+FFMA by two FFMAs does not speed it up
  (170.6 vs 184.7 with scale loads removed; 158.7 vs 157.6 in the complete kernel before stores), and FP8 per-col (no I2F at all) is
  only 1.13x INT8 per-col. The microbenchmark issue rates (3.27 vs 2.06 clk/element) do not translate because the promotion warps are
  bound by memory-side latency/bandwidth, not by the FP pipe.
- What the per-col promotion is bound by: (1) the warp-broadcast `ld.shared.v4` of the per-column scales (8 per 32-column sub-tile
  per thread, 64 per stage per warp, 512 wavefronts per stage per CTA competing with the MMA's smem operand reads): removing them gives
  +25 % (184.7); (2) the TMEM stage turnaround: the 2-stage TMEM ring (256x256 tile) makes any extra hop on the release path visible
  (re-arm protocol without stores: -14 %); (3) the tcgen05.st re-arm itself (32 x8 stores per warp per stage): 158.7 -> 106.5.
- Why g256 gained 1.7x: it halves *all* per-K-group work at once -- scale loads, TMEM loads, promotion FFMAs, barriers -- not just I2F.
- Therefore: the TMEM pre-bias path stays in the tree behind `-DG128_OPT_BIAS` (off) as a verified negative result. The lever for per-col
  g128 is the scale distribution: a TMEM-load layout in which a warp's lanes span columns (so the s_w loads are not 32-way broadcasts),
  fp16-packed scales (half the wavefronts), or TMEM-resident scales via UTCCP broadcast if the TMEM budget allows (256x128 tiles).
