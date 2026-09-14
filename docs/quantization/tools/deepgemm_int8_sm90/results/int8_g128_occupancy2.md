# INT8 g128 (1,128,128) block-scaled GEMM -- implementer D occupancy-2 (OCC, approach D) variant, verifier run (idle H100, serial timing)

- NVIDIA H100 80GB HBM3, torch 2.13.0a0+9186a08b2c.nv26.07, 2026-09-14T03:19:02; triton do_bench(warmup=25, rep=200, median, L2 flushed), 3 interleaved repeats, median of medians; SM MHz = nvidia-smi samples ~0.1 s into each do_bench (clocks unlocked, power-throttled at M=42240). TFLOPS = 2MNK/t with the true M.
- `D occ` = implementer D's approach D (isolated extension `kernels/deepgemm_int8/occ`, kernels `sm90_int8_gemm_1d2d_occ_impl` / `_occdb_impl`): ONE math warpgroup per CTA (128 producer + 128 math threads, setmaxnreg 40 / 216, `__launch_bounds__(256, 2)`), persistent grid of 264 CTAs = 2 co-resident CTAs per SM (3 for the 64x64 r120 cfg), so the promotion of one CTA can overlap the WGMMAs of the other. Tiles: 64x128 s3 (D1), + DB accumulator (D4), c2x1 / c1x2 clusters (D5), 128x128 s2 one-warpgroup two-wave (D2), 64x64 s4 at 3 CTAs/SM and s6, and `halfD` = D tile stored in two 8 KB passes so 4 stages fit at 2 CTAs/SM. D3 (64x256) is infeasible at 2 CTAs/SM (regs 256 > 216, smem > 115712 B) and was not built. `B DB` = implementer B's double-buffered accumulator (main cfg 21/23/24/25/27), `C pp` = implementer C's ping-pong mainloop (pp cfg 5/7/11/12/13), `default` = the unchanged INT8 port (main cfg 8/9/10/11 = 256x128 s3 c1x2 shape-compiled).
- occupancy (runtime cudaOccupancyMaxActiveBlocksPerMultiprocessor in the launcher, `probes/verify_int8_g128_occ.py`): 19 / 19 occ cfgs reach their target (values [2, 3]); ptxas launch budget 128 regs (80 for the 3-CTA cfg), spills 0 B for the shape-compiled 64x128 s3 cfgs, 24-104 B for halfD, 472 B for 128x128 s2, 232-752 B for the dynamic-shape cfgs.
- correctness (`probes/verify_int8_g128_occ.py`, logs/occ_v_check.log): 196 / 196 occ checks bit-identical (torch.equal) to the same-tile default kernel (REF1: main cfg 6 = 64x128 s8 c1x1, cfg 3 = 128x128 s5 c1x1) AND to the 256x128 default (REF2) AND to kernels/cutlass_int8_bw (REF3), rel-L2 vs an independent fp64 reference <= 3e-3 (all 19 occ cfgs; (1024,4096,901), (4096,12288,1802), (12288,4096,4096), (4096,4096,4096) + K=128/256/384 edge cases; random / adversarial 1e-4-1.0 scales / +-127 / -128 mixes, NaN sfa padding); 44 / 44 default == cutlass_int8_bw checks. stress 4096x4096: 200 iterations, M 901..4194 step 37, 13 cfgs, 2600 launches, 0 mismatches, no hang (1 s).
- every timed INT8 g128 row was additionally asserted torch.equal to the default kernel on the timed tensors (incl. M=42240).

## Best exact INT8 g128 kernel per shape vs references (TFLOPS)

| N | K | M | default 256x128 s3 c1x2 | best B DB (cfg) | best C pp (cfg) | best D occ (cfg) | D occ vs best exact | best exact INT8 g128 | DeepGEMM FP8 g128 | ratio vs FP8 g128 | cutlass INT8 per-tensor | ratio vs per-tensor | cuBLASLt FP8 per-tensor | SM MHz (default / best D occ) |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| 4096 | 4096 | 901 | 799 | 848 (128x128 s5 c1x1 N4096K4096 DB) | 814 (256x128 s3 c1x2 N4096K4096 pp1) | 733 (64x128 s3 c1x1 N4096K4096 mt128 occ2 r216) | 0.86 | 848 B DB cfg 21 | 962 | 0.88 | 978 | 0.87 | 1077 | 1980 / 1980 |
| 4096 | 4096 | 4096 | 1048 | 1122 (128x128 s5 c1x1 N4096K4096 DB) | 1090 (256x128 s3 c1x2 N4096K4096 pp2) | 921 (64x128 s3 c1x1 N4096K4096 DB mt128 occ2 r216) | 0.82 | 1122 B DB cfg 21 | 1278 | 0.88 | 1371 | 0.82 | 1335 | 1965 / 1965 |
| 4096 | 4096 | 42240 | 1071 | 1088 (128x128 s5 c1x1 N4096K4096 DB) | 1101 (256x128 s3 c1x2 N4096K4096 pp1) | 911 (64x128 s3 c2x1 N4096K4096 mt128 occ2 r216) | 0.83 | 1101 C pp cfg 5 | 1246 | 0.88 | 1393 | 0.79 | 1269 | 1785 / 1515 |
| 1024 | 4096 | 901 | 223 (pick_config 64x128 s8 c1x1: 428) | 479 (64x128 s8 c1x1 DB) | 218 (256x128 s3 c1x2 N1024K4096 pp1) | 359 (64x128 s3 c1x1 N1024K4096 mt128 occ2 r216) | 0.75 | 479 B DB cfg 27 | 380 | 1.26 | 268 | 1.79 | 516 | 1980 / 1980 |
| 1024 | 4096 | 4096 | 904 | 930 (128x128 s5 c1x2 N1024K4096 DB) | 914 (256x128 s3 c1x2 N1024K4096 pp1) | 825 (64x128 s3 c1x1 N1024K4096 mt128 occ2 r216) | 0.89 | 930 B DB cfg 25 | 1047 | 0.89 | 1118 | 0.83 | 1143 | 1980 / 1980 |
| 1024 | 4096 | 42240 | 1065 | 1092 (128x128 s5 c1x2 N1024K4096 DB) | 1089 (256x128 s3 c1x2 N1024K4096 pp1) | 857 (64x128 s3 c1x1 N1024K4096 DB mt128 occ2 r216) | 0.79 | 1092 B DB cfg 25 | 1223 | 0.89 | 1427 | 0.77 | 1264 | 1830 / 1770 |
| 12288 | 4096 | 901 | 914 | 950 (128x128 s5 c1x2 N12288K4096 DB) | 945 (256x128 s3 c1x2 N12288K4096 pp1) | 811 (64x128 s3 c1x1 N12288K4096 DB mt128 occ2 r216) | 0.85 | 950 B DB cfg 23 | 1127 | 0.84 | 1176 | 0.81 | 1293 | 1980 / 1980 |
| 12288 | 4096 | 4096 | 1066 | 1093 (128x128 s5 c1x2 N12288K4096 DB) | 1083 (256x128 s3 c1x2 N12288K4096 pp1) | 870 (64x128 s3 c1x1 N12288K4096 DB mt128 occ2 r216) | 0.80 | 1093 B DB cfg 23 | 1233 | 0.89 | 1397 | 0.78 | 1237 | 1860 / 1590 |
| 12288 | 4096 | 42240 | 1054 | 1103 (128x128 s5 c1x2 N12288K4096 DB) | 1105 (256x128 s3 c1x2 N12288K4096 pp1) | 749 (64x128 s3 c1x1 N12288K4096 DB mt128 occ2 r216) | 0.68 | 1105 C pp cfg 12 | 1254 | 0.88 | 1287 | 0.86 | 1265 | 1545 / 1515 |
| 4096 | 12288 | 901 | 941 | 944 (128x128 s5 c1x2 N4096K12288 DB) | 977 (256x128 s3 c1x2 N4096K12288 pp1) | 783 (64x128 s3 c1x1 N4096K12288 DB mt128 occ2 r216) | 0.80 | 977 C pp cfg 11 | 1164 | 0.84 | 1276 | 0.77 | 1410 | 1980 / 1980 |
| 4096 | 12288 | 4096 | 1091 | 1091 (128x128 s5 c1x2 N4096K12288 DB) | 1132 (256x128 s3 c1x2 N4096K12288 pp1) | 850 (64x128 s3 c1x1 N4096K12288 DB mt128 occ2 r216) | 0.75 | 1132 C pp cfg 11 | 1326 | 0.85 | 1497 | 0.76 | 1303 | 1860 / 1635 |
| 4096 | 12288 | 42240 | 1067 | 1099 (128x128 s5 c1x2 N4096K12288 DB) | 1113 (256x128 s3 c1x2 N4096K12288 pp1) | 723 (64x128 s3 c1x1 N4096K12288 DB mt128 occ2 r216) | 0.65 | 1113 C pp cfg 11 | 1329 | 0.84 | 1467 | 0.76 | 1309 | 1365 / 1380 |

## All rows: microseconds (median of 3 medians), TFLOPS, per-repeat medians, SM MHz samples

| N | K | M | kernel | config | us | TFLOPS | repeats (us) | SM MHz samples |
|---|---|---|---|---|---:|---:|---|---|
| 4096 | 4096 | 901 | default 256x128 cfg 8 | 256x128 s3 c1x2 N4096K4096 | 37.9 | 799 | 37.9, 38.0, 37.8 | [1980, 1980, 1980] |
| 4096 | 4096 | 901 | B DB cfg 21 | 128x128 s5 c1x1 N4096K4096 DB | 35.6 | 848 | 35.7, 35.6, 35.4 | [1980, 1980, 1980] |
| 4096 | 4096 | 901 | C pp cfg 5 | 256x128 s3 c1x2 N4096K4096 pp1 | 37.2 | 814 | 37.2, 37.2, 37.1 | [1980, 1980, 1980] |
| 4096 | 4096 | 901 | C pp cfg 7 | 256x128 s3 c1x2 N4096K4096 pp2 | 37.5 | 806 | 37.5, 37.2, 37.5 | [1980, 1980, 1980] |
| 4096 | 4096 | 901 | D occ cfg 1 | 64x128 s3 c1x1 N4096K4096 mt128 occ2 r216 | 41.2 | 733 | 41.3, 41.1, 41.2 | [1980, 1980, 1980] |
| 4096 | 4096 | 901 | D occ cfg 2 | 64x128 s3 c1x1 N4096K4096 DB mt128 occ2 r216 | 42.9 | 704 | 42.9, 42.8, 43.0 | [1980, 1980, 1980] |
| 4096 | 4096 | 901 | D occ cfg 3 | 64x128 s3 c2x1 N4096K4096 mt128 occ2 r216 | 44.4 | 682 | 44.5, 44.3, 44.4 | [1980, 1980, 1980] |
| 4096 | 4096 | 901 | D occ cfg 4 | 64x128 s3 c1x2 N4096K4096 mt128 occ2 r216 | 45.7 | 662 | 45.8, 45.7, 45.6 | [1980, 1980, 1980] |
| 4096 | 4096 | 901 | D occ cfg 15 | 64x128 s4 c1x1 N4096K4096 halfD mt128 occ2 r216 | 42.6 | 710 | 42.6, 42.7, 42.4 | [1980, 1980, 1980] |
| 4096 | 4096 | 901 | D occ cfg 16 | 64x128 s4 c1x1 N4096K4096 DB halfD mt128 occ2 r216 | 91.1 | 332 | 91.1, 91.1, 91.0 | [1980, 1980, 1980] |
| 4096 | 4096 | 901 | D occ cfg 17 | 64x128 s4 c2x1 N4096K4096 halfD mt128 occ2 r216 | 49.3 | 613 | 49.3, 49.3, 49.0 | [1980, 1980, 1980] |
| 4096 | 4096 | 901 | D occ cfg 7 | 128x128 s2 c1x1 N4096K4096 mt128 occ2 r216 | 99.9 | 303 | 99.9, 99.8, 99.9 | [1980, 1980, 1980] |
| 4096 | 4096 | 901 | D occ cfg 5 | 64x64 s4 c1x1 N4096K4096 mt128 occ3 r120 | 71.3 | 424 | 71.4, 71.2, 71.3 | [1980, 1980, 1980] |
| 4096 | 4096 | 901 | D occ cfg 6 | 64x64 s6 c1x1 N4096K4096 mt128 occ2 r216 | 66.3 | 456 | 66.3, 66.3, 66.4 | [1980, 1980, 1980] |
| 4096 | 4096 | 901 | D occ cfg 0 | 64x128 s3 c1x1 mt128 occ2 r216 | 78.5 | 385 | 78.5, 78.5, 78.5 | [1980, 1980, 1980] |
| 4096 | 4096 | 901 | D occ cfg 14 | 64x128 s3 c1x1 DB mt128 occ2 r216 | 245.7 | 123 | 245.3, 245.7, 245.7 | [1980, 1980, 1980] |
| 4096 | 4096 | 901 | D occ cfg 18 | 64x128 s4 c1x1 halfD mt128 occ2 r216 | 78.4 | 386 | 78.4, 78.3, 78.4 | [1980, 1980, 1980] |
| 4096 | 4096 | 901 | DeepGEMM FP8 g128 | upstream 2.8.0 1D2D, unpadded M | 31.4 | 962 | 31.4, 31.5, 31.4 | [1980, 1980, 1980] |
| 4096 | 4096 | 901 | cutlass_int8 per-tensor | 128x256 c2x1 (int8_scaled_mm cfg 0) | 30.9 | 978 | 30.9, 30.8, 30.9 | [1980, 1980, 1980] |
| 4096 | 4096 | 901 | cuBLASLt FP8 per-tensor | torch._scaled_mm use_fast_accum=True | 28.1 | 1077 | 28.1, 27.8, 28.1 | [1980, 1980, 1980] |
| 4096 | 4096 | 4096 | default 256x128 cfg 8 | 256x128 s3 c1x2 N4096K4096 | 131.1 | 1048 | 131.0, 131.2, 131.1 | [1980, 1965, 1965] |
| 4096 | 4096 | 4096 | B DB cfg 21 | 128x128 s5 c1x1 N4096K4096 DB | 122.5 | 1122 | 122.6, 122.5, 122.2 | [1905, 1950, 1950] |
| 4096 | 4096 | 4096 | C pp cfg 5 | 256x128 s3 c1x2 N4096K4096 pp1 | 126.1 | 1090 | 126.1, 126.3, 125.8 | [1965, 1965, 1965] |
| 4096 | 4096 | 4096 | C pp cfg 7 | 256x128 s3 c1x2 N4096K4096 pp2 | 126.0 | 1090 | 126.0, 126.1, 125.8 | [1980, 1980, 1980] |
| 4096 | 4096 | 4096 | D occ cfg 1 | 64x128 s3 c1x1 N4096K4096 mt128 occ2 r216 | 150.0 | 916 | 150.0, 149.5, 150.1 | [1965, 1965, 1965] |
| 4096 | 4096 | 4096 | D occ cfg 2 | 64x128 s3 c1x1 N4096K4096 DB mt128 occ2 r216 | 149.2 | 921 | 149.2, 149.3, 148.9 | [1950, 1965, 1965] |
| 4096 | 4096 | 4096 | D occ cfg 3 | 64x128 s3 c2x1 N4096K4096 mt128 occ2 r216 | 150.0 | 916 | 150.2, 150.0, 150.0 | [1935, 1965, 1965] |
| 4096 | 4096 | 4096 | D occ cfg 4 | 64x128 s3 c1x2 N4096K4096 mt128 occ2 r216 | 158.2 | 869 | 157.9, 158.2, 158.3 | [1965, 1950, 1920] |
| 4096 | 4096 | 4096 | D occ cfg 15 | 64x128 s4 c1x1 N4096K4096 halfD mt128 occ2 r216 | 153.8 | 894 | 153.6, 154.0, 153.8 | [1965, 1965, 1965] |
| 4096 | 4096 | 4096 | D occ cfg 16 | 64x128 s4 c1x1 N4096K4096 DB halfD mt128 occ2 r216 | 339.7 | 405 | 339.5, 339.7, 339.8 | [1965, 1965, 1965] |
| 4096 | 4096 | 4096 | D occ cfg 17 | 64x128 s4 c2x1 N4096K4096 halfD mt128 occ2 r216 | 167.1 | 822 | 167.1, 166.9, 167.1 | [1950, 1965, 1935] |
| 4096 | 4096 | 4096 | D occ cfg 7 | 128x128 s2 c1x1 N4096K4096 mt128 occ2 r216 | 367.2 | 374 | 367.2, 367.2, 367.4 | [1965, 1965, 1965] |
| 4096 | 4096 | 4096 | D occ cfg 5 | 64x64 s4 c1x1 N4096K4096 mt128 occ3 r120 | 269.2 | 511 | 269.1, 269.2, 269.2 | [1980, 1980, 1980] |
| 4096 | 4096 | 4096 | D occ cfg 6 | 64x64 s6 c1x1 N4096K4096 mt128 occ2 r216 | 218.1 | 630 | 218.7, 218.1, 217.9 | [1980, 1980, 1980] |
| 4096 | 4096 | 4096 | D occ cfg 0 | 64x128 s3 c1x1 mt128 occ2 r216 | 291.1 | 472 | 291.1, 291.1, 291.0 | [1980, 1980, 1980] |
| 4096 | 4096 | 4096 | D occ cfg 14 | 64x128 s3 c1x1 DB mt128 occ2 r216 | 982.2 | 140 | 982.0, 982.2, 982.7 | [1980, 1980, 1980] |
| 4096 | 4096 | 4096 | D occ cfg 18 | 64x128 s4 c1x1 halfD mt128 occ2 r216 | 291.3 | 472 | 290.7, 291.3, 292.4 | [1980, 1980, 1965] |
| 4096 | 4096 | 4096 | DeepGEMM FP8 g128 | upstream 2.8.0 1D2D, unpadded M | 107.6 | 1278 | 107.1, 107.6, 108.1 | [1845, 1860, 1815] |
| 4096 | 4096 | 4096 | cutlass_int8 per-tensor | 128x256 c2x1 (int8_scaled_mm cfg 0) | 100.2 | 1371 | 100.2, 100.2, 100.6 | [1950, 1950, 1950] |
| 4096 | 4096 | 4096 | cuBLASLt FP8 per-tensor | torch._scaled_mm use_fast_accum=True | 102.9 | 1335 | 102.9, 102.9, 102.9 | [1860, 1920, 1920] |
| 4096 | 4096 | 42240 | default 256x128 cfg 8 | 256x128 s3 c1x2 N4096K4096 | 1323.2 | 1071 | 1323.2, 1308.2, 1323.2 | [1950, 1740, 1785] |
| 4096 | 4096 | 42240 | B DB cfg 21 | 128x128 s5 c1x1 N4096K4096 DB | 1303.3 | 1088 | 1303.3, 1299.9, 1326.7 | [1575, 1590, 1530] |
| 4096 | 4096 | 42240 | C pp cfg 5 | 256x128 s3 c1x2 N4096K4096 pp1 | 1287.0 | 1101 | 1323.4, 1285.3, 1287.0 | [1605, 1740, 1635] |
| 4096 | 4096 | 42240 | C pp cfg 7 | 256x128 s3 c1x2 N4096K4096 pp2 | 1310.8 | 1081 | 1310.8, 1307.2, 1319.4 | [1710, 1725, 1725] |
| 4096 | 4096 | 42240 | D occ cfg 1 | 64x128 s3 c1x1 N4096K4096 mt128 occ2 r216 | 1964.4 | 722 | 1973.2, 1960.8, 1964.4 | [1785, 1635, 1635] |
| 4096 | 4096 | 42240 | D occ cfg 2 | 64x128 s3 c1x1 N4096K4096 DB mt128 occ2 r216 | 1792.2 | 791 | 1789.2, 1792.2, 1803.4 | [1770, 1800, 1740] |
| 4096 | 4096 | 42240 | D occ cfg 3 | 64x128 s3 c2x1 N4096K4096 mt128 occ2 r216 | 1556.6 | 911 | 1556.6, 1561.4, 1541.5 | [1530, 1485, 1515] |
| 4096 | 4096 | 42240 | D occ cfg 4 | 64x128 s3 c1x2 N4096K4096 mt128 occ2 r216 | 1706.3 | 831 | 1723.4, 1705.5, 1706.3 | [1515, 1560, 1620] |
| 4096 | 4096 | 42240 | D occ cfg 15 | 64x128 s4 c1x1 N4096K4096 halfD mt128 occ2 r216 | 1992.3 | 711 | 1992.3, 1992.4, 1987.2 | [1695, 1650, 1680] |
| 4096 | 4096 | 42240 | D occ cfg 16 | 64x128 s4 c1x1 N4096K4096 DB halfD mt128 occ2 r216 | 4610.2 | 307 | 4610.2, 4605.5, 4625.4 | [1905, 1950, 1845] |
| 4096 | 4096 | 42240 | D occ cfg 17 | 64x128 s4 c2x1 N4096K4096 halfD mt128 occ2 r216 | 1801.5 | 787 | 1801.5, 1782.4, 1831.4 | [1440, 1515, 1365] |
| 4096 | 4096 | 42240 | D occ cfg 7 | 128x128 s2 c1x1 N4096K4096 mt128 occ2 r216 | 4533.5 | 313 | 4524.0, 4543.6, 4533.5 | [1830, 1815, 1860] |
| 4096 | 4096 | 42240 | D occ cfg 5 | 64x64 s4 c1x1 N4096K4096 mt128 occ3 r120 | 3123.7 | 454 | 3045.9, 3131.3, 3123.7 | [1500, 1455, 1485] |
| 4096 | 4096 | 42240 | D occ cfg 6 | 64x64 s6 c1x1 N4096K4096 mt128 occ2 r216 | 2737.6 | 518 | 2734.2, 2737.6, 2744.1 | [1785, 1875, 1860] |
| 4096 | 4096 | 42240 | D occ cfg 0 | 64x128 s3 c1x1 mt128 occ2 r216 | 4401.7 | 322 | 4408.9, 4401.7, 4375.7 | [1830, 1875, 1830] |
| 4096 | 4096 | 42240 | D occ cfg 14 | 64x128 s3 c1x1 DB mt128 occ2 r216 | 11909.0 | 119 | 11909.0, 11905.0, 11911.3 | [1965, 1950, 1965] |
| 4096 | 4096 | 42240 | D occ cfg 18 | 64x128 s4 c1x1 halfD mt128 occ2 r216 | 4610.6 | 307 | 4611.3, 4610.6, 4607.6 | [1965, 1980, 1980] |
| 4096 | 4096 | 42240 | DeepGEMM FP8 g128 | upstream 2.8.0 1D2D, unpadded M | 1137.9 | 1246 | 1137.9, 1155.9, 1137.1 | [1260, 1305, 1275] |
| 4096 | 4096 | 42240 | cutlass_int8 per-tensor | 128x256 c2x1 (int8_scaled_mm cfg 0) | 1017.2 | 1393 | 1015.8, 1017.2, 1037.0 | [1515, 1650, 1650] |
| 4096 | 4096 | 42240 | cuBLASLt FP8 per-tensor | torch._scaled_mm use_fast_accum=True | 1116.6 | 1269 | 1116.3, 1116.6, 1125.7 | [1395, 1380, 1395] |
| 1024 | 4096 | 901 | default 256x128 cfg 9 | 256x128 s3 c1x2 N1024K4096 | 33.9 | 223 | 34.2, 33.6, 33.9 | [1980, 1980, 1980] |
| 1024 | 4096 | 901 | default best cfg 6 | 64x128 s8 c1x1 | 17.7 | 428 | 17.7, 17.7, 17.7 | [1980, 1980, 1980] |
| 1024 | 4096 | 901 | B DB cfg 25 | 128x128 s5 c1x2 N1024K4096 DB | 21.1 | 358 | 21.1, 21.1, 21.2 | [1980, 1980, 1980] |
| 1024 | 4096 | 901 | B DB cfg 27 | 64x128 s8 c1x1 DB | 15.8 | 479 | 15.8, 15.8, 15.8 | [1980, 1980, 1980] |
| 1024 | 4096 | 901 | C pp cfg 13 | 256x128 s3 c1x2 N1024K4096 pp1 | 34.7 | 218 | 34.7, 35.0, 34.7 | [1980, 1980, 1980] |
| 1024 | 4096 | 901 | D occ cfg 10 | 64x128 s3 c1x1 N1024K4096 mt128 occ2 r216 | 21.1 | 359 | 21.1, 21.2, 21.0 | [1980, 1980, 1980] |
| 1024 | 4096 | 901 | D occ cfg 13 | 64x128 s3 c1x1 N1024K4096 DB mt128 occ2 r216 | 22.0 | 343 | 22.0, 22.0, 21.8 | [1980, 1980, 1980] |
| 1024 | 4096 | 901 | D occ cfg 0 | 64x128 s3 c1x1 mt128 occ2 r216 | 35.5 | 213 | 35.5, 35.6, 35.5 | [1980, 1980, 1980] |
| 1024 | 4096 | 901 | D occ cfg 14 | 64x128 s3 c1x1 DB mt128 occ2 r216 | 100.8 | 75 | 100.8, 100.8, 100.2 | [1980, 1980, 1980] |
| 1024 | 4096 | 901 | D occ cfg 18 | 64x128 s4 c1x1 halfD mt128 occ2 r216 | 35.9 | 211 | 35.9, 36.2, 35.9 | [1980, 1980, 1980] |
| 1024 | 4096 | 901 | DeepGEMM FP8 g128 | upstream 2.8.0 1D2D, unpadded M | 19.9 | 380 | 19.7, 19.9, 19.9 | [1980, 1980, 1980] |
| 1024 | 4096 | 901 | cutlass_int8 per-tensor | 128x256 c2x1 (int8_scaled_mm cfg 0) | 28.2 | 268 | 28.0, 28.3, 28.2 | [1980, 1980, 1980] |
| 1024 | 4096 | 901 | cuBLASLt FP8 per-tensor | torch._scaled_mm use_fast_accum=True | 14.7 | 516 | 14.6, 14.8, 14.7 | [1980, 1980, 1980] |
| 1024 | 4096 | 4096 | default 256x128 cfg 9 | 256x128 s3 c1x2 N1024K4096 | 38.0 | 904 | 38.4, 38.0, 38.0 | [1980, 1980, 1980] |
| 1024 | 4096 | 4096 | B DB cfg 25 | 128x128 s5 c1x2 N1024K4096 DB | 36.9 | 930 | 37.2, 36.9, 36.9 | [1980, 1980, 1980] |
| 1024 | 4096 | 4096 | B DB cfg 27 | 64x128 s8 c1x1 DB | 39.3 | 874 | 39.5, 39.3, 39.1 | [1980, 1980, 1980] |
| 1024 | 4096 | 4096 | C pp cfg 13 | 256x128 s3 c1x2 N1024K4096 pp1 | 37.6 | 914 | 37.6, 37.7, 37.5 | [1980, 1980, 1980] |
| 1024 | 4096 | 4096 | D occ cfg 10 | 64x128 s3 c1x1 N1024K4096 mt128 occ2 r216 | 41.7 | 825 | 41.7, 41.7, 41.7 | [1980, 1980, 1980] |
| 1024 | 4096 | 4096 | D occ cfg 13 | 64x128 s3 c1x1 N1024K4096 DB mt128 occ2 r216 | 43.4 | 792 | 43.4, 43.4, 43.4 | [1980, 1980, 1980] |
| 1024 | 4096 | 4096 | D occ cfg 0 | 64x128 s3 c1x1 mt128 occ2 r216 | 81.6 | 421 | 81.9, 81.6, 81.5 | [1980, 1980, 1980] |
| 1024 | 4096 | 4096 | D occ cfg 14 | 64x128 s3 c1x1 DB mt128 occ2 r216 | 256.1 | 134 | 256.1, 255.9, 256.1 | [1980, 1980, 1980] |
| 1024 | 4096 | 4096 | D occ cfg 18 | 64x128 s4 c1x1 halfD mt128 occ2 r216 | 80.1 | 429 | 80.3, 80.0, 80.1 | [1980, 1980, 1980] |
| 1024 | 4096 | 4096 | DeepGEMM FP8 g128 | upstream 2.8.0 1D2D, unpadded M | 32.8 | 1047 | 33.0, 32.8, 32.8 | [1980, 1980, 1980] |
| 1024 | 4096 | 4096 | cutlass_int8 per-tensor | 128x256 c2x1 (int8_scaled_mm cfg 0) | 30.7 | 1118 | 30.7, 30.7, 30.9 | [1980, 1980, 1980] |
| 1024 | 4096 | 4096 | cuBLASLt FP8 per-tensor | torch._scaled_mm use_fast_accum=True | 30.0 | 1143 | 30.0, 30.0, 30.2 | [1980, 1980, 1980] |
| 1024 | 4096 | 42240 | default 256x128 cfg 9 | 256x128 s3 c1x2 N1024K4096 | 332.8 | 1065 | 327.3, 334.0, 332.8 | [1845, 1830, 1740] |
| 1024 | 4096 | 42240 | B DB cfg 25 | 128x128 s5 c1x2 N1024K4096 DB | 324.6 | 1092 | 324.6, 330.5, 311.2 | [1695, 1620, 1725] |
| 1024 | 4096 | 42240 | B DB cfg 27 | 64x128 s8 c1x1 DB | 370.5 | 956 | 360.3, 370.5, 370.9 | [1755, 1605, 1665] |
| 1024 | 4096 | 42240 | C pp cfg 13 | 256x128 s3 c1x2 N1024K4096 pp1 | 325.3 | 1089 | 330.9, 324.5, 325.3 | [1695, 1770, 1755] |
| 1024 | 4096 | 42240 | D occ cfg 10 | 64x128 s3 c1x1 N1024K4096 mt128 occ2 r216 | 445.7 | 795 | 446.1, 445.6, 445.7 | [1725, 1695, 1695] |
| 1024 | 4096 | 42240 | D occ cfg 13 | 64x128 s3 c1x1 N1024K4096 DB mt128 occ2 r216 | 413.4 | 857 | 410.9, 416.0, 413.4 | [1785, 1725, 1770] |
| 1024 | 4096 | 42240 | D occ cfg 0 | 64x128 s3 c1x1 mt128 occ2 r216 | 797.6 | 444 | 798.2, 792.2, 797.6 | [1830, 1815, 1905] |
| 1024 | 4096 | 42240 | D occ cfg 14 | 64x128 s3 c1x1 DB mt128 occ2 r216 | 2572.5 | 138 | 2572.5, 2574.1, 2571.0 | [1965, 1965, 1965] |
| 1024 | 4096 | 42240 | D occ cfg 18 | 64x128 s4 c1x1 halfD mt128 occ2 r216 | 813.5 | 436 | 794.4, 824.2, 813.5 | [1800, 1875, 1905] |
| 1024 | 4096 | 42240 | DeepGEMM FP8 g128 | upstream 2.8.0 1D2D, unpadded M | 289.8 | 1223 | 289.2, 292.2, 289.8 | [1455, 1410, 1470] |
| 1024 | 4096 | 42240 | cutlass_int8 per-tensor | 128x256 c2x1 (int8_scaled_mm cfg 0) | 248.4 | 1427 | 247.8, 250.0, 248.4 | [1800, 1755, 1650] |
| 1024 | 4096 | 42240 | cuBLASLt FP8 per-tensor | torch._scaled_mm use_fast_accum=True | 280.4 | 1264 | 276.4, 283.6, 280.4 | [1530, 1455, 1470] |
| 12288 | 4096 | 901 | default 256x128 cfg 10 | 256x128 s3 c1x2 N12288K4096 | 99.2 | 914 | 98.6, 99.3, 99.2 | [1980, 1980, 1980] |
| 12288 | 4096 | 901 | B DB cfg 23 | 128x128 s5 c1x2 N12288K4096 DB | 95.5 | 950 | 95.5, 95.6, 95.5 | [1980, 1980, 1980] |
| 12288 | 4096 | 901 | C pp cfg 12 | 256x128 s3 c1x2 N12288K4096 pp1 | 96.0 | 945 | 96.0, 95.9, 96.1 | [1980, 1980, 1980] |
| 12288 | 4096 | 901 | D occ cfg 8 | 64x128 s3 c1x1 N12288K4096 mt128 occ2 r216 | 122.4 | 741 | 122.4, 122.6, 122.4 | [1980, 1980, 1980] |
| 12288 | 4096 | 901 | D occ cfg 11 | 64x128 s3 c1x1 N12288K4096 DB mt128 occ2 r216 | 111.8 | 811 | 111.7, 111.8, 111.8 | [1980, 1980, 1980] |
| 12288 | 4096 | 901 | D occ cfg 0 | 64x128 s3 c1x1 mt128 occ2 r216 | 223.4 | 406 | 224.1, 223.4, 223.2 | [1980, 1980, 1980] |
| 12288 | 4096 | 901 | D occ cfg 14 | 64x128 s3 c1x1 DB mt128 occ2 r216 | 707.0 | 128 | 707.0, 707.6, 706.9 | [1980, 1980, 1980] |
| 12288 | 4096 | 901 | D occ cfg 18 | 64x128 s4 c1x1 halfD mt128 occ2 r216 | 214.9 | 422 | 214.9, 214.7, 215.2 | [1980, 1980, 1980] |
| 12288 | 4096 | 901 | DeepGEMM FP8 g128 | upstream 2.8.0 1D2D, unpadded M | 80.5 | 1127 | 80.0, 80.5, 80.6 | [1980, 1905, 1950] |
| 12288 | 4096 | 901 | cutlass_int8 per-tensor | 128x256 c2x1 (int8_scaled_mm cfg 0) | 77.1 | 1176 | 77.1, 77.1, 77.4 | [1965, 1965, 1965] |
| 12288 | 4096 | 901 | cuBLASLt FP8 per-tensor | torch._scaled_mm use_fast_accum=True | 70.1 | 1293 | 70.1, 69.8, 70.5 | [1965, 1980, 1950] |
| 12288 | 4096 | 4096 | default 256x128 cfg 10 | 256x128 s3 c1x2 N12288K4096 | 386.6 | 1066 | 387.4, 385.7, 386.6 | [1860, 1740, 1875] |
| 12288 | 4096 | 4096 | B DB cfg 23 | 128x128 s5 c1x2 N12288K4096 DB | 377.3 | 1093 | 375.4, 377.3, 377.5 | [1785, 1620, 1890] |
| 12288 | 4096 | 4096 | C pp cfg 12 | 256x128 s3 c1x2 N12288K4096 pp1 | 380.8 | 1083 | 377.7, 382.1, 380.8 | [1860, 1800, 1800] |
| 12288 | 4096 | 4096 | D occ cfg 8 | 64x128 s3 c1x1 N12288K4096 mt128 occ2 r216 | 492.5 | 837 | 491.2, 493.0, 492.5 | [1620, 1725, 1740] |
| 12288 | 4096 | 4096 | D occ cfg 11 | 64x128 s3 c1x1 N12288K4096 DB mt128 occ2 r216 | 474.1 | 870 | 469.1, 476.4, 474.1 | [1665, 1545, 1590] |
| 12288 | 4096 | 4096 | D occ cfg 0 | 64x128 s3 c1x1 mt128 occ2 r216 | 1169.6 | 353 | 1160.5, 1171.3, 1169.6 | [1935, 1845, 1965] |
| 12288 | 4096 | 4096 | D occ cfg 14 | 64x128 s3 c1x1 DB mt128 occ2 r216 | 3348.6 | 123 | 3363.2, 3348.6, 3341.0 | [1965, 1965, 1965] |
| 12288 | 4096 | 4096 | D occ cfg 18 | 64x128 s4 c1x1 halfD mt128 occ2 r216 | 1060.1 | 389 | 1060.1, 1060.0, 1061.3 | [1935, 1905, 1980] |
| 12288 | 4096 | 4096 | DeepGEMM FP8 g128 | upstream 2.8.0 1D2D, unpadded M | 334.4 | 1233 | 333.2, 336.0, 334.4 | [1500, 1545, 1545] |
| 12288 | 4096 | 4096 | cutlass_int8 per-tensor | 128x256 c2x1 (int8_scaled_mm cfg 0) | 295.2 | 1397 | 295.2, 295.7, 291.6 | [1560, 1725, 1665] |
| 12288 | 4096 | 4096 | cuBLASLt FP8 per-tensor | torch._scaled_mm use_fast_accum=True | 333.2 | 1237 | 333.2, 327.6, 333.7 | [1470, 1440, 1635] |
| 12288 | 4096 | 42240 | default 256x128 cfg 10 | 256x128 s3 c1x2 N12288K4096 | 4032.5 | 1054 | 4014.5, 4056.3, 4032.5 | [1725, 1545, 1470] |
| 12288 | 4096 | 42240 | B DB cfg 23 | 128x128 s5 c1x2 N12288K4096 DB | 3855.8 | 1103 | 3878.9, 3847.8, 3855.8 | [1650, 1410, 1395] |
| 12288 | 4096 | 42240 | C pp cfg 12 | 256x128 s3 c1x2 N12288K4096 pp1 | 3846.8 | 1105 | 4038.8, 3811.6, 3846.8 | [1470, 1680, 1620] |
| 12288 | 4096 | 42240 | D occ cfg 8 | 64x128 s3 c1x1 N12288K4096 mt128 occ2 r216 | 6490.3 | 655 | 6269.3, 6568.1, 6490.3 | [1455, 1425, 1410] |
| 12288 | 4096 | 42240 | D occ cfg 11 | 64x128 s3 c1x1 N12288K4096 DB mt128 occ2 r216 | 5677.5 | 749 | 5612.4, 5743.4, 5677.5 | [1515, 1425, 1545] |
| 12288 | 4096 | 42240 | D occ cfg 0 | 64x128 s3 c1x1 mt128 occ2 r216 | 18743.8 | 227 | 18761.6, 18736.3, 18743.8 | [1500, 1965, 1965] |
| 12288 | 4096 | 42240 | D occ cfg 14 | 64x128 s3 c1x1 DB mt128 occ2 r216 | 39801.4 | 107 | 39777.8, 39801.4, 39807.9 | [1980, 1980, 1980] |
| 12288 | 4096 | 42240 | D occ cfg 18 | 64x128 s4 c1x1 halfD mt128 occ2 r216 | 18290.8 | 232 | 18290.8, 18296.8, 18270.2 | [1980, 1980, 1980] |
| 12288 | 4096 | 42240 | DeepGEMM FP8 g128 | upstream 2.8.0 1D2D, unpadded M | 3391.1 | 1254 | 3189.4, 3391.1, 3398.8 | [1410, 1410, 1410] |
| 12288 | 4096 | 42240 | cutlass_int8 per-tensor | 128x256 c2x1 (int8_scaled_mm cfg 0) | 3304.0 | 1287 | 3285.7, 3327.1, 3304.0 | [1425, 1395, 1425] |
| 12288 | 4096 | 42240 | cuBLASLt FP8 per-tensor | torch._scaled_mm use_fast_accum=True | 3360.6 | 1265 | 3360.0, 3398.9, 3360.6 | [1260, 1275, 1275] |
| 4096 | 12288 | 901 | default 256x128 cfg 11 | 256x128 s3 c1x2 N4096K12288 | 96.4 | 941 | 95.6, 96.8, 96.4 | [1980, 1965, 1980] |
| 4096 | 12288 | 901 | B DB cfg 24 | 128x128 s5 c1x2 N4096K12288 DB | 96.1 | 944 | 96.0, 96.4, 96.1 | [1980, 1935, 1950] |
| 4096 | 12288 | 901 | C pp cfg 11 | 256x128 s3 c1x2 N4096K12288 pp1 | 92.8 | 977 | 92.5, 92.8, 92.8 | [1980, 1965, 1965] |
| 4096 | 12288 | 901 | D occ cfg 9 | 64x128 s3 c1x1 N4096K12288 mt128 occ2 r216 | 127.1 | 714 | 127.6, 126.7, 127.1 | [1980, 1980, 1980] |
| 4096 | 12288 | 901 | D occ cfg 12 | 64x128 s3 c1x1 N4096K12288 DB mt128 occ2 r216 | 115.8 | 783 | 115.8, 115.6, 115.8 | [1980, 1980, 1980] |
| 4096 | 12288 | 901 | D occ cfg 0 | 64x128 s3 c1x1 mt128 occ2 r216 | 213.5 | 425 | 213.5, 214.1, 213.5 | [1980, 1980, 1980] |
| 4096 | 12288 | 901 | D occ cfg 14 | 64x128 s3 c1x1 DB mt128 occ2 r216 | 712.6 | 127 | 711.9, 712.6, 713.7 | [1980, 1980, 1980] |
| 4096 | 12288 | 901 | D occ cfg 18 | 64x128 s4 c1x1 halfD mt128 occ2 r216 | 212.8 | 426 | 212.7, 212.9, 212.8 | [1980, 1980, 1980] |
| 4096 | 12288 | 901 | DeepGEMM FP8 g128 | upstream 2.8.0 1D2D, unpadded M | 78.0 | 1164 | 78.6, 78.0, 77.8 | [1935, 1935, 1950] |
| 4096 | 12288 | 901 | cutlass_int8 per-tensor | 128x256 c2x1 (int8_scaled_mm cfg 0) | 71.1 | 1276 | 71.1, 71.3, 71.0 | [1965, 1965, 1965] |
| 4096 | 12288 | 901 | cuBLASLt FP8 per-tensor | torch._scaled_mm use_fast_accum=True | 64.3 | 1410 | 64.7, 64.1, 64.3 | [1965, 1980, 1935] |
| 4096 | 12288 | 4096 | default 256x128 cfg 11 | 256x128 s3 c1x2 N4096K12288 | 377.9 | 1091 | 376.0, 378.3, 377.9 | [1860, 1860, 1860] |
| 4096 | 12288 | 4096 | B DB cfg 24 | 128x128 s5 c1x2 N4096K12288 DB | 378.0 | 1091 | 373.2, 378.0, 378.8 | [1695, 1740, 1740] |
| 4096 | 12288 | 4096 | C pp cfg 11 | 256x128 s3 c1x2 N4096K12288 pp1 | 364.2 | 1132 | 367.0, 364.2, 361.6 | [1830, 1830, 1785] |
| 4096 | 12288 | 4096 | D occ cfg 9 | 64x128 s3 c1x1 N4096K12288 mt128 occ2 r216 | 539.3 | 765 | 538.0, 539.7, 539.3 | [1680, 1800, 1845] |
| 4096 | 12288 | 4096 | D occ cfg 12 | 64x128 s3 c1x1 N4096K12288 DB mt128 occ2 r216 | 485.0 | 850 | 478.0, 490.6, 485.0 | [1665, 1575, 1635] |
| 4096 | 12288 | 4096 | D occ cfg 0 | 64x128 s3 c1x1 mt128 occ2 r216 | 1191.7 | 346 | 1191.6, 1191.7, 1196.1 | [1935, 1965, 1950] |
| 4096 | 12288 | 4096 | D occ cfg 14 | 64x128 s3 c1x1 DB mt128 occ2 r216 | 3403.6 | 121 | 3403.6, 3400.6, 3409.3 | [1965, 1965, 1965] |
| 4096 | 12288 | 4096 | D occ cfg 18 | 64x128 s4 c1x1 halfD mt128 occ2 r216 | 1209.9 | 341 | 1208.7, 1209.9, 1210.6 | [1980, 1980, 1980] |
| 4096 | 12288 | 4096 | DeepGEMM FP8 g128 | upstream 2.8.0 1D2D, unpadded M | 311.0 | 1326 | 310.8, 322.0, 311.0 | [1545, 1530, 1590] |
| 4096 | 12288 | 4096 | cutlass_int8 per-tensor | 128x256 c2x1 (int8_scaled_mm cfg 0) | 275.4 | 1497 | 274.0, 275.6, 275.4 | [1755, 1770, 1470] |
| 4096 | 12288 | 4096 | cuBLASLt FP8 per-tensor | torch._scaled_mm use_fast_accum=True | 316.4 | 1303 | 316.4, 320.6, 300.7 | [1620, 1485, 1605] |
| 4096 | 12288 | 42240 | default 256x128 cfg 11 | 256x128 s3 c1x2 N4096K12288 | 3984.6 | 1067 | 3904.9, 3984.6, 3992.5 | [1725, 1260, 1365] |
| 4096 | 12288 | 42240 | B DB cfg 24 | 128x128 s5 c1x2 N4096K12288 DB | 3869.4 | 1099 | 3873.5, 3844.1, 3869.4 | [1455, 1425, 1575] |
| 4096 | 12288 | 42240 | C pp cfg 11 | 256x128 s3 c1x2 N4096K12288 pp1 | 3821.4 | 1113 | 3931.4, 3821.4, 3780.5 | [1470, 1515, 1575] |
| 4096 | 12288 | 42240 | D occ cfg 9 | 64x128 s3 c1x1 N4096K12288 mt128 occ2 r216 | 6960.8 | 611 | 6816.3, 7030.8, 6960.8 | [1350, 1395, 1425] |
| 4096 | 12288 | 42240 | D occ cfg 12 | 64x128 s3 c1x1 N4096K12288 DB mt128 occ2 r216 | 5883.5 | 723 | 5778.2, 5883.5, 5978.0 | [1470, 1380, 1380] |
| 4096 | 12288 | 42240 | D occ cfg 0 | 64x128 s3 c1x1 mt128 occ2 r216 | 18226.1 | 233 | 18233.3, 18135.2, 18226.1 | [1905, 1935, 1695] |
| 4096 | 12288 | 42240 | D occ cfg 14 | 64x128 s3 c1x1 DB mt128 occ2 r216 | 39833.2 | 107 | 39833.2, 39799.4, 39844.7 | [1965, 1965, 1980] |
| 4096 | 12288 | 42240 | D occ cfg 18 | 64x128 s4 c1x1 halfD mt128 occ2 r216 | 18335.3 | 232 | 18320.9, 18374.8, 18335.3 | [1980, 1980, 1980] |
| 4096 | 12288 | 42240 | DeepGEMM FP8 g128 | upstream 2.8.0 1D2D, unpadded M | 3198.4 | 1329 | 3300.7, 3198.4, 3141.6 | [1380, 1365, 1380] |
| 4096 | 12288 | 42240 | cutlass_int8 per-tensor | 128x256 c2x1 (int8_scaled_mm cfg 0) | 2899.2 | 1467 | 2899.7, 2866.6, 2899.2 | [1395, 1500, 1440] |
| 4096 | 12288 | 42240 | cuBLASLt FP8 per-tensor | torch._scaled_mm use_fast_accum=True | 3249.2 | 1309 | 3212.0, 3249.5, 3249.2 | [1260, 1350, 1245] |

## Occupancy report per occ cfg (runtime, cudaOccupancyMaxActiveBlocksPerMultiprocessor / cudaOccupancyMaxActiveClusters, cudaFuncGetAttributes)

| occ cfg | name | CTAs/SM | target | max active clusters | ptxas regs (launch budget) | spill B | dyn smem B | ok |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 0 | 64x128 s3 c1x1 mt128 occ2 r216 | 2 | 2 | -1 | 128 | 264 | 91056 | True |
| 1 | 64x128 s3 c1x1 N4096K4096 mt128 occ2 r216 | 2 | 2 | -1 | 128 | 0 | 91056 | True |
| 2 | 64x128 s3 c1x1 N4096K4096 DB mt128 occ2 r216 | 2 | 2 | -1 | 128 | 0 | 91056 | True |
| 3 | 64x128 s3 c2x1 N4096K4096 mt128 occ2 r216 | 2 | 2 | 132 | 128 | 0 | 91056 | True |
| 4 | 64x128 s3 c1x2 N4096K4096 mt128 occ2 r216 | 2 | 2 | 132 | 128 | 0 | 91056 | True |
| 5 | 64x64 s4 c1x1 N4096K4096 mt128 occ3 r120 | 3 | 3 | -1 | 80 | 64 | 74944 | True |
| 6 | 64x64 s6 c1x1 N4096K4096 mt128 occ2 r216 | 2 | 2 | -1 | 128 | 0 | 108256 | True |
| 7 | 128x128 s2 c1x1 N4096K4096 mt128 occ2 r216 | 2 | 2 | -1 | 128 | 472 | 99488 | True |
| 8 | 64x128 s3 c1x1 N12288K4096 mt128 occ2 r216 | 2 | 2 | -1 | 128 | 0 | 91056 | True |
| 9 | 64x128 s3 c1x1 N4096K12288 mt128 occ2 r216 | 2 | 2 | -1 | 128 | 0 | 91312 | True |
| 10 | 64x128 s3 c1x1 N1024K4096 mt128 occ2 r216 | 2 | 2 | -1 | 128 | 0 | 91056 | True |
| 11 | 64x128 s3 c1x1 N12288K4096 DB mt128 occ2 r216 | 2 | 2 | -1 | 128 | 0 | 91056 | True |
| 12 | 64x128 s3 c1x1 N4096K12288 DB mt128 occ2 r216 | 2 | 2 | -1 | 128 | 0 | 91312 | True |
| 13 | 64x128 s3 c1x1 N1024K4096 DB mt128 occ2 r216 | 2 | 2 | -1 | 128 | 0 | 91056 | True |
| 14 | 64x128 s3 c1x1 DB mt128 occ2 r216 | 2 | 2 | -1 | 128 | 752 | 91056 | True |
| 15 | 64x128 s4 c1x1 N4096K4096 halfD mt128 occ2 r216 | 2 | 2 | -1 | 128 | 24 | 107712 | True |
| 16 | 64x128 s4 c1x1 N4096K4096 DB halfD mt128 occ2 r216 | 2 | 2 | -1 | 128 | 264 | 107712 | True |
| 17 | 64x128 s4 c2x1 N4096K4096 halfD mt128 occ2 r216 | 2 | 2 | 132 | 128 | 104 | 107712 | True |
| 18 | 64x128 s4 c1x1 halfD mt128 occ2 r216 | 2 | 2 | -1 | 128 | 232 | 107712 | True |

## Verification detail (occ cfg vs same-tile default REF1 / 256x128 default REF2 / cutlass_int8_bw REF3; rel-L2 and max|d|/max|ref| vs fp64)

| occ cfg | name | N | K | M | case | ==REF1 (cfg) | ==REF2 | ==REF3 | #diff | rel-L2 | max rel | ok |
|---|---|---|---|---|---|---|---|---|---:|---:|---:|---|
| 0 | 64x128 s3 c1x1 mt128 occ2 r216 | 1024 | 4096 | 901 | random | True (6) | True | True | 0 | 1.66e-03 | 3.49e-03 | True |
| 10 | 64x128 s3 c1x1 N1024K4096 mt128 occ2 r216 | 1024 | 4096 | 901 | random | True (6) | True | True | 0 | 1.66e-03 | 3.49e-03 | True |
| 13 | 64x128 s3 c1x1 N1024K4096 DB mt128 occ2 r216 | 1024 | 4096 | 901 | random | True (6) | True | True | 0 | 1.66e-03 | 3.49e-03 | True |
| 14 | 64x128 s3 c1x1 DB mt128 occ2 r216 | 1024 | 4096 | 901 | random | True (6) | True | True | 0 | 1.66e-03 | 3.49e-03 | True |
| 18 | 64x128 s4 c1x1 halfD mt128 occ2 r216 | 1024 | 4096 | 901 | random | True (6) | True | True | 0 | 1.66e-03 | 3.49e-03 | True |
| 0 | 64x128 s3 c1x1 mt128 occ2 r216 | 1024 | 4096 | 901 | adversarial_scales | True (6) | True | True | 0 | 1.66e-03 | 2.32e-03 | True |
| 10 | 64x128 s3 c1x1 N1024K4096 mt128 occ2 r216 | 1024 | 4096 | 901 | adversarial_scales | True (6) | True | True | 0 | 1.66e-03 | 2.32e-03 | True |
| 13 | 64x128 s3 c1x1 N1024K4096 DB mt128 occ2 r216 | 1024 | 4096 | 901 | adversarial_scales | True (6) | True | True | 0 | 1.66e-03 | 2.32e-03 | True |
| 14 | 64x128 s3 c1x1 DB mt128 occ2 r216 | 1024 | 4096 | 901 | adversarial_scales | True (6) | True | True | 0 | 1.66e-03 | 2.32e-03 | True |
| 18 | 64x128 s4 c1x1 halfD mt128 occ2 r216 | 1024 | 4096 | 901 | adversarial_scales | True (6) | True | True | 0 | 1.66e-03 | 2.32e-03 | True |
| 0 | 64x128 s3 c1x1 mt128 occ2 r216 | 1024 | 4096 | 901 | extremes_pm127 | True (6) | True | True | 0 | 1.72e-03 | 2.32e-03 | True |
| 10 | 64x128 s3 c1x1 N1024K4096 mt128 occ2 r216 | 1024 | 4096 | 901 | extremes_pm127 | True (6) | True | True | 0 | 1.72e-03 | 2.32e-03 | True |
| 13 | 64x128 s3 c1x1 N1024K4096 DB mt128 occ2 r216 | 1024 | 4096 | 901 | extremes_pm127 | True (6) | True | True | 0 | 1.72e-03 | 2.32e-03 | True |
| 14 | 64x128 s3 c1x1 DB mt128 occ2 r216 | 1024 | 4096 | 901 | extremes_pm127 | True (6) | True | True | 0 | 1.72e-03 | 2.32e-03 | True |
| 18 | 64x128 s4 c1x1 halfD mt128 occ2 r216 | 1024 | 4096 | 901 | extremes_pm127 | True (6) | True | True | 0 | 1.72e-03 | 2.32e-03 | True |
| 0 | 64x128 s3 c1x1 mt128 occ2 r216 | 1024 | 4096 | 901 | extremes_m128 | True (6) | True | True | 0 | 1.62e-03 | 2.05e-03 | True |
| 10 | 64x128 s3 c1x1 N1024K4096 mt128 occ2 r216 | 1024 | 4096 | 901 | extremes_m128 | True (6) | True | True | 0 | 1.62e-03 | 2.05e-03 | True |
| 13 | 64x128 s3 c1x1 N1024K4096 DB mt128 occ2 r216 | 1024 | 4096 | 901 | extremes_m128 | True (6) | True | True | 0 | 1.62e-03 | 2.05e-03 | True |
| 14 | 64x128 s3 c1x1 DB mt128 occ2 r216 | 1024 | 4096 | 901 | extremes_m128 | True (6) | True | True | 0 | 1.62e-03 | 2.05e-03 | True |
| 18 | 64x128 s4 c1x1 halfD mt128 occ2 r216 | 1024 | 4096 | 901 | extremes_m128 | True (6) | True | True | 0 | 1.62e-03 | 2.05e-03 | True |
| 0 | 64x128 s3 c1x1 mt128 occ2 r216 | 4096 | 12288 | 1802 | random | True (6) | True | True | 0 | 1.66e-03 | 2.04e-03 | True |
| 9 | 64x128 s3 c1x1 N4096K12288 mt128 occ2 r216 | 4096 | 12288 | 1802 | random | True (6) | True | True | 0 | 1.66e-03 | 2.04e-03 | True |
| 12 | 64x128 s3 c1x1 N4096K12288 DB mt128 occ2 r216 | 4096 | 12288 | 1802 | random | True (6) | True | True | 0 | 1.66e-03 | 2.04e-03 | True |
| 14 | 64x128 s3 c1x1 DB mt128 occ2 r216 | 4096 | 12288 | 1802 | random | True (6) | True | True | 0 | 1.66e-03 | 2.04e-03 | True |
| 18 | 64x128 s4 c1x1 halfD mt128 occ2 r216 | 4096 | 12288 | 1802 | random | True (6) | True | True | 0 | 1.66e-03 | 2.04e-03 | True |
| 0 | 64x128 s3 c1x1 mt128 occ2 r216 | 4096 | 12288 | 1802 | adversarial_scales | True (6) | True | True | 0 | 1.66e-03 | 2.72e-03 | True |
| 9 | 64x128 s3 c1x1 N4096K12288 mt128 occ2 r216 | 4096 | 12288 | 1802 | adversarial_scales | True (6) | True | True | 0 | 1.66e-03 | 2.72e-03 | True |
| 12 | 64x128 s3 c1x1 N4096K12288 DB mt128 occ2 r216 | 4096 | 12288 | 1802 | adversarial_scales | True (6) | True | True | 0 | 1.66e-03 | 2.72e-03 | True |
| 14 | 64x128 s3 c1x1 DB mt128 occ2 r216 | 4096 | 12288 | 1802 | adversarial_scales | True (6) | True | True | 0 | 1.66e-03 | 2.72e-03 | True |
| 18 | 64x128 s4 c1x1 halfD mt128 occ2 r216 | 4096 | 12288 | 1802 | adversarial_scales | True (6) | True | True | 0 | 1.66e-03 | 2.72e-03 | True |
| 0 | 64x128 s3 c1x1 mt128 occ2 r216 | 4096 | 12288 | 1802 | extremes_pm127 | True (6) | True | True | 0 | 1.62e-03 | 1.62e-03 | True |
| 9 | 64x128 s3 c1x1 N4096K12288 mt128 occ2 r216 | 4096 | 12288 | 1802 | extremes_pm127 | True (6) | True | True | 0 | 1.62e-03 | 1.62e-03 | True |
| 12 | 64x128 s3 c1x1 N4096K12288 DB mt128 occ2 r216 | 4096 | 12288 | 1802 | extremes_pm127 | True (6) | True | True | 0 | 1.62e-03 | 1.62e-03 | True |
| 14 | 64x128 s3 c1x1 DB mt128 occ2 r216 | 4096 | 12288 | 1802 | extremes_pm127 | True (6) | True | True | 0 | 1.62e-03 | 1.62e-03 | True |
| 18 | 64x128 s4 c1x1 halfD mt128 occ2 r216 | 4096 | 12288 | 1802 | extremes_pm127 | True (6) | True | True | 0 | 1.62e-03 | 1.62e-03 | True |
| 0 | 64x128 s3 c1x1 mt128 occ2 r216 | 4096 | 12288 | 1802 | extremes_m128 | True (6) | True | True | 0 | 1.68e-03 | 3.03e-03 | True |
| 9 | 64x128 s3 c1x1 N4096K12288 mt128 occ2 r216 | 4096 | 12288 | 1802 | extremes_m128 | True (6) | True | True | 0 | 1.68e-03 | 3.03e-03 | True |
| 12 | 64x128 s3 c1x1 N4096K12288 DB mt128 occ2 r216 | 4096 | 12288 | 1802 | extremes_m128 | True (6) | True | True | 0 | 1.68e-03 | 3.03e-03 | True |
| 14 | 64x128 s3 c1x1 DB mt128 occ2 r216 | 4096 | 12288 | 1802 | extremes_m128 | True (6) | True | True | 0 | 1.68e-03 | 3.03e-03 | True |
| 18 | 64x128 s4 c1x1 halfD mt128 occ2 r216 | 4096 | 12288 | 1802 | extremes_m128 | True (6) | True | True | 0 | 1.68e-03 | 3.03e-03 | True |
| 0 | 64x128 s3 c1x1 mt128 occ2 r216 | 12288 | 4096 | 4096 | random | True (6) | True | True | 0 | 1.66e-03 | 3.19e-03 | True |
| 8 | 64x128 s3 c1x1 N12288K4096 mt128 occ2 r216 | 12288 | 4096 | 4096 | random | True (6) | True | True | 0 | 1.66e-03 | 3.19e-03 | True |
| 11 | 64x128 s3 c1x1 N12288K4096 DB mt128 occ2 r216 | 12288 | 4096 | 4096 | random | True (6) | True | True | 0 | 1.66e-03 | 3.19e-03 | True |
| 14 | 64x128 s3 c1x1 DB mt128 occ2 r216 | 12288 | 4096 | 4096 | random | True (6) | True | True | 0 | 1.66e-03 | 3.19e-03 | True |
| 18 | 64x128 s4 c1x1 halfD mt128 occ2 r216 | 12288 | 4096 | 4096 | random | True (6) | True | True | 0 | 1.66e-03 | 3.19e-03 | True |
| 0 | 64x128 s3 c1x1 mt128 occ2 r216 | 12288 | 4096 | 4096 | adversarial_scales | True (6) | True | True | 0 | 1.66e-03 | 2.16e-03 | True |
| 8 | 64x128 s3 c1x1 N12288K4096 mt128 occ2 r216 | 12288 | 4096 | 4096 | adversarial_scales | True (6) | True | True | 0 | 1.66e-03 | 2.16e-03 | True |
| 11 | 64x128 s3 c1x1 N12288K4096 DB mt128 occ2 r216 | 12288 | 4096 | 4096 | adversarial_scales | True (6) | True | True | 0 | 1.66e-03 | 2.16e-03 | True |
| 14 | 64x128 s3 c1x1 DB mt128 occ2 r216 | 12288 | 4096 | 4096 | adversarial_scales | True (6) | True | True | 0 | 1.66e-03 | 2.16e-03 | True |
| 18 | 64x128 s4 c1x1 halfD mt128 occ2 r216 | 12288 | 4096 | 4096 | adversarial_scales | True (6) | True | True | 0 | 1.66e-03 | 2.16e-03 | True |
| 0 | 64x128 s3 c1x1 mt128 occ2 r216 | 12288 | 4096 | 4096 | extremes_pm127 | True (6) | True | True | 0 | 1.66e-03 | 2.76e-03 | True |
| 8 | 64x128 s3 c1x1 N12288K4096 mt128 occ2 r216 | 12288 | 4096 | 4096 | extremes_pm127 | True (6) | True | True | 0 | 1.66e-03 | 2.76e-03 | True |
| 11 | 64x128 s3 c1x1 N12288K4096 DB mt128 occ2 r216 | 12288 | 4096 | 4096 | extremes_pm127 | True (6) | True | True | 0 | 1.66e-03 | 2.76e-03 | True |
| 14 | 64x128 s3 c1x1 DB mt128 occ2 r216 | 12288 | 4096 | 4096 | extremes_pm127 | True (6) | True | True | 0 | 1.66e-03 | 2.76e-03 | True |
| 18 | 64x128 s4 c1x1 halfD mt128 occ2 r216 | 12288 | 4096 | 4096 | extremes_pm127 | True (6) | True | True | 0 | 1.66e-03 | 2.76e-03 | True |
| 0 | 64x128 s3 c1x1 mt128 occ2 r216 | 12288 | 4096 | 4096 | extremes_m128 | True (6) | True | True | 0 | 1.67e-03 | 3.58e-03 | True |
| 8 | 64x128 s3 c1x1 N12288K4096 mt128 occ2 r216 | 12288 | 4096 | 4096 | extremes_m128 | True (6) | True | True | 0 | 1.67e-03 | 3.58e-03 | True |
| 11 | 64x128 s3 c1x1 N12288K4096 DB mt128 occ2 r216 | 12288 | 4096 | 4096 | extremes_m128 | True (6) | True | True | 0 | 1.67e-03 | 3.58e-03 | True |
| 14 | 64x128 s3 c1x1 DB mt128 occ2 r216 | 12288 | 4096 | 4096 | extremes_m128 | True (6) | True | True | 0 | 1.67e-03 | 3.58e-03 | True |
| 18 | 64x128 s4 c1x1 halfD mt128 occ2 r216 | 12288 | 4096 | 4096 | extremes_m128 | True (6) | True | True | 0 | 1.67e-03 | 3.58e-03 | True |
| 0 | 64x128 s3 c1x1 mt128 occ2 r216 | 4096 | 4096 | 4096 | random | True (6) | True | True | 0 | 1.66e-03 | 3.08e-03 | True |
| 1 | 64x128 s3 c1x1 N4096K4096 mt128 occ2 r216 | 4096 | 4096 | 4096 | random | True (6) | True | True | 0 | 1.66e-03 | 3.08e-03 | True |
| 2 | 64x128 s3 c1x1 N4096K4096 DB mt128 occ2 r216 | 4096 | 4096 | 4096 | random | True (6) | True | True | 0 | 1.66e-03 | 3.08e-03 | True |
| 3 | 64x128 s3 c2x1 N4096K4096 mt128 occ2 r216 | 4096 | 4096 | 4096 | random | True (None) | True | True | 0 | 1.66e-03 | 3.08e-03 | True |
| 4 | 64x128 s3 c1x2 N4096K4096 mt128 occ2 r216 | 4096 | 4096 | 4096 | random | True (None) | True | True | 0 | 1.66e-03 | 3.08e-03 | True |
| 5 | 64x64 s4 c1x1 N4096K4096 mt128 occ3 r120 | 4096 | 4096 | 4096 | random | True (None) | True | True | 0 | 1.66e-03 | 3.08e-03 | True |
| 6 | 64x64 s6 c1x1 N4096K4096 mt128 occ2 r216 | 4096 | 4096 | 4096 | random | True (None) | True | True | 0 | 1.66e-03 | 3.08e-03 | True |
| 7 | 128x128 s2 c1x1 N4096K4096 mt128 occ2 r216 | 4096 | 4096 | 4096 | random | True (3) | True | True | 0 | 1.66e-03 | 3.08e-03 | True |
| 14 | 64x128 s3 c1x1 DB mt128 occ2 r216 | 4096 | 4096 | 4096 | random | True (6) | True | True | 0 | 1.66e-03 | 3.08e-03 | True |
| 15 | 64x128 s4 c1x1 N4096K4096 halfD mt128 occ2 r216 | 4096 | 4096 | 4096 | random | True (6) | True | True | 0 | 1.66e-03 | 3.08e-03 | True |
| 16 | 64x128 s4 c1x1 N4096K4096 DB halfD mt128 occ2 r216 | 4096 | 4096 | 4096 | random | True (6) | True | True | 0 | 1.66e-03 | 3.08e-03 | True |
| 17 | 64x128 s4 c2x1 N4096K4096 halfD mt128 occ2 r216 | 4096 | 4096 | 4096 | random | True (None) | True | True | 0 | 1.66e-03 | 3.08e-03 | True |
| 18 | 64x128 s4 c1x1 halfD mt128 occ2 r216 | 4096 | 4096 | 4096 | random | True (6) | True | True | 0 | 1.66e-03 | 3.08e-03 | True |
| 0 | 64x128 s3 c1x1 mt128 occ2 r216 | 4096 | 4096 | 4096 | adversarial_scales | True (6) | True | True | 0 | 1.66e-03 | 2.40e-03 | True |
| 1 | 64x128 s3 c1x1 N4096K4096 mt128 occ2 r216 | 4096 | 4096 | 4096 | adversarial_scales | True (6) | True | True | 0 | 1.66e-03 | 2.40e-03 | True |
| 2 | 64x128 s3 c1x1 N4096K4096 DB mt128 occ2 r216 | 4096 | 4096 | 4096 | adversarial_scales | True (6) | True | True | 0 | 1.66e-03 | 2.40e-03 | True |
| 3 | 64x128 s3 c2x1 N4096K4096 mt128 occ2 r216 | 4096 | 4096 | 4096 | adversarial_scales | True (None) | True | True | 0 | 1.66e-03 | 2.40e-03 | True |
| 4 | 64x128 s3 c1x2 N4096K4096 mt128 occ2 r216 | 4096 | 4096 | 4096 | adversarial_scales | True (None) | True | True | 0 | 1.66e-03 | 2.40e-03 | True |
| 5 | 64x64 s4 c1x1 N4096K4096 mt128 occ3 r120 | 4096 | 4096 | 4096 | adversarial_scales | True (None) | True | True | 0 | 1.66e-03 | 2.40e-03 | True |
| 6 | 64x64 s6 c1x1 N4096K4096 mt128 occ2 r216 | 4096 | 4096 | 4096 | adversarial_scales | True (None) | True | True | 0 | 1.66e-03 | 2.40e-03 | True |
| 7 | 128x128 s2 c1x1 N4096K4096 mt128 occ2 r216 | 4096 | 4096 | 4096 | adversarial_scales | True (3) | True | True | 0 | 1.66e-03 | 2.40e-03 | True |
| 14 | 64x128 s3 c1x1 DB mt128 occ2 r216 | 4096 | 4096 | 4096 | adversarial_scales | True (6) | True | True | 0 | 1.66e-03 | 2.40e-03 | True |
| 15 | 64x128 s4 c1x1 N4096K4096 halfD mt128 occ2 r216 | 4096 | 4096 | 4096 | adversarial_scales | True (6) | True | True | 0 | 1.66e-03 | 2.40e-03 | True |
| 16 | 64x128 s4 c1x1 N4096K4096 DB halfD mt128 occ2 r216 | 4096 | 4096 | 4096 | adversarial_scales | True (6) | True | True | 0 | 1.66e-03 | 2.40e-03 | True |
| 17 | 64x128 s4 c2x1 N4096K4096 halfD mt128 occ2 r216 | 4096 | 4096 | 4096 | adversarial_scales | True (None) | True | True | 0 | 1.66e-03 | 2.40e-03 | True |
| 18 | 64x128 s4 c1x1 halfD mt128 occ2 r216 | 4096 | 4096 | 4096 | adversarial_scales | True (6) | True | True | 0 | 1.66e-03 | 2.40e-03 | True |
| 0 | 64x128 s3 c1x1 mt128 occ2 r216 | 4096 | 4096 | 4096 | extremes_pm127 | True (6) | True | True | 0 | 1.72e-03 | 1.77e-03 | True |
| 1 | 64x128 s3 c1x1 N4096K4096 mt128 occ2 r216 | 4096 | 4096 | 4096 | extremes_pm127 | True (6) | True | True | 0 | 1.72e-03 | 1.77e-03 | True |
| 2 | 64x128 s3 c1x1 N4096K4096 DB mt128 occ2 r216 | 4096 | 4096 | 4096 | extremes_pm127 | True (6) | True | True | 0 | 1.72e-03 | 1.77e-03 | True |
| 3 | 64x128 s3 c2x1 N4096K4096 mt128 occ2 r216 | 4096 | 4096 | 4096 | extremes_pm127 | True (None) | True | True | 0 | 1.72e-03 | 1.77e-03 | True |
| 4 | 64x128 s3 c1x2 N4096K4096 mt128 occ2 r216 | 4096 | 4096 | 4096 | extremes_pm127 | True (None) | True | True | 0 | 1.72e-03 | 1.77e-03 | True |
| 5 | 64x64 s4 c1x1 N4096K4096 mt128 occ3 r120 | 4096 | 4096 | 4096 | extremes_pm127 | True (None) | True | True | 0 | 1.72e-03 | 1.77e-03 | True |
| 6 | 64x64 s6 c1x1 N4096K4096 mt128 occ2 r216 | 4096 | 4096 | 4096 | extremes_pm127 | True (None) | True | True | 0 | 1.72e-03 | 1.77e-03 | True |
| 7 | 128x128 s2 c1x1 N4096K4096 mt128 occ2 r216 | 4096 | 4096 | 4096 | extremes_pm127 | True (3) | True | True | 0 | 1.72e-03 | 1.77e-03 | True |
| 14 | 64x128 s3 c1x1 DB mt128 occ2 r216 | 4096 | 4096 | 4096 | extremes_pm127 | True (6) | True | True | 0 | 1.72e-03 | 1.77e-03 | True |
| 15 | 64x128 s4 c1x1 N4096K4096 halfD mt128 occ2 r216 | 4096 | 4096 | 4096 | extremes_pm127 | True (6) | True | True | 0 | 1.72e-03 | 1.77e-03 | True |
| 16 | 64x128 s4 c1x1 N4096K4096 DB halfD mt128 occ2 r216 | 4096 | 4096 | 4096 | extremes_pm127 | True (6) | True | True | 0 | 1.72e-03 | 1.77e-03 | True |
| 17 | 64x128 s4 c2x1 N4096K4096 halfD mt128 occ2 r216 | 4096 | 4096 | 4096 | extremes_pm127 | True (None) | True | True | 0 | 1.72e-03 | 1.77e-03 | True |
| 18 | 64x128 s4 c1x1 halfD mt128 occ2 r216 | 4096 | 4096 | 4096 | extremes_pm127 | True (6) | True | True | 0 | 1.72e-03 | 1.77e-03 | True |
| 0 | 64x128 s3 c1x1 mt128 occ2 r216 | 4096 | 4096 | 4096 | extremes_m128 | True (6) | True | True | 0 | 1.68e-03 | 2.05e-03 | True |
| 1 | 64x128 s3 c1x1 N4096K4096 mt128 occ2 r216 | 4096 | 4096 | 4096 | extremes_m128 | True (6) | True | True | 0 | 1.68e-03 | 2.05e-03 | True |
| 2 | 64x128 s3 c1x1 N4096K4096 DB mt128 occ2 r216 | 4096 | 4096 | 4096 | extremes_m128 | True (6) | True | True | 0 | 1.68e-03 | 2.05e-03 | True |
| 3 | 64x128 s3 c2x1 N4096K4096 mt128 occ2 r216 | 4096 | 4096 | 4096 | extremes_m128 | True (None) | True | True | 0 | 1.68e-03 | 2.05e-03 | True |
| 4 | 64x128 s3 c1x2 N4096K4096 mt128 occ2 r216 | 4096 | 4096 | 4096 | extremes_m128 | True (None) | True | True | 0 | 1.68e-03 | 2.05e-03 | True |
| 5 | 64x64 s4 c1x1 N4096K4096 mt128 occ3 r120 | 4096 | 4096 | 4096 | extremes_m128 | True (None) | True | True | 0 | 1.68e-03 | 2.05e-03 | True |
| 6 | 64x64 s6 c1x1 N4096K4096 mt128 occ2 r216 | 4096 | 4096 | 4096 | extremes_m128 | True (None) | True | True | 0 | 1.68e-03 | 2.05e-03 | True |
| 7 | 128x128 s2 c1x1 N4096K4096 mt128 occ2 r216 | 4096 | 4096 | 4096 | extremes_m128 | True (3) | True | True | 0 | 1.68e-03 | 2.05e-03 | True |
| 14 | 64x128 s3 c1x1 DB mt128 occ2 r216 | 4096 | 4096 | 4096 | extremes_m128 | True (6) | True | True | 0 | 1.68e-03 | 2.05e-03 | True |
| 15 | 64x128 s4 c1x1 N4096K4096 halfD mt128 occ2 r216 | 4096 | 4096 | 4096 | extremes_m128 | True (6) | True | True | 0 | 1.68e-03 | 2.05e-03 | True |
| 16 | 64x128 s4 c1x1 N4096K4096 DB halfD mt128 occ2 r216 | 4096 | 4096 | 4096 | extremes_m128 | True (6) | True | True | 0 | 1.68e-03 | 2.05e-03 | True |
| 17 | 64x128 s4 c2x1 N4096K4096 halfD mt128 occ2 r216 | 4096 | 4096 | 4096 | extremes_m128 | True (None) | True | True | 0 | 1.68e-03 | 2.05e-03 | True |
| 18 | 64x128 s4 c1x1 halfD mt128 occ2 r216 | 4096 | 4096 | 4096 | extremes_m128 | True (6) | True | True | 0 | 1.68e-03 | 2.05e-03 | True |
| 0 | 64x128 s3 c1x1 mt128 occ2 r216 | 4096 | 128 | 901 | random | True (6) | True | True | 0 | 1.66e-03 | 2.62e-03 | True |
| 14 | 64x128 s3 c1x1 DB mt128 occ2 r216 | 4096 | 128 | 901 | random | True (6) | True | True | 0 | 1.66e-03 | 2.62e-03 | True |
| 18 | 64x128 s4 c1x1 halfD mt128 occ2 r216 | 4096 | 128 | 901 | random | True (6) | True | True | 0 | 1.66e-03 | 2.62e-03 | True |
| 0 | 64x128 s3 c1x1 mt128 occ2 r216 | 4096 | 128 | 901 | adversarial_scales | True (6) | True | True | 0 | 1.66e-03 | 2.95e-03 | True |
| 14 | 64x128 s3 c1x1 DB mt128 occ2 r216 | 4096 | 128 | 901 | adversarial_scales | True (6) | True | True | 0 | 1.66e-03 | 2.95e-03 | True |
| 18 | 64x128 s4 c1x1 halfD mt128 occ2 r216 | 4096 | 128 | 901 | adversarial_scales | True (6) | True | True | 0 | 1.66e-03 | 2.95e-03 | True |
| 0 | 64x128 s3 c1x1 mt128 occ2 r216 | 4096 | 128 | 901 | extremes_pm127 | True (6) | True | True | 0 | 9.20e-04 | 9.73e-04 | True |
| 14 | 64x128 s3 c1x1 DB mt128 occ2 r216 | 4096 | 128 | 901 | extremes_pm127 | True (6) | True | True | 0 | 9.20e-04 | 9.73e-04 | True |
| 18 | 64x128 s4 c1x1 halfD mt128 occ2 r216 | 4096 | 128 | 901 | extremes_pm127 | True (6) | True | True | 0 | 9.20e-04 | 9.73e-04 | True |
| 0 | 64x128 s3 c1x1 mt128 occ2 r216 | 4096 | 128 | 901 | extremes_m128 | True (6) | True | True | 0 | 1.63e-03 | 3.60e-03 | True |
| 14 | 64x128 s3 c1x1 DB mt128 occ2 r216 | 4096 | 128 | 901 | extremes_m128 | True (6) | True | True | 0 | 1.63e-03 | 3.60e-03 | True |
| 18 | 64x128 s4 c1x1 halfD mt128 occ2 r216 | 4096 | 128 | 901 | extremes_m128 | True (6) | True | True | 0 | 1.63e-03 | 3.60e-03 | True |
| 0 | 64x128 s3 c1x1 mt128 occ2 r216 | 4096 | 128 | 4096 | random | True (6) | True | True | 0 | 1.66e-03 | 2.34e-03 | True |
| 14 | 64x128 s3 c1x1 DB mt128 occ2 r216 | 4096 | 128 | 4096 | random | True (6) | True | True | 0 | 1.66e-03 | 2.34e-03 | True |
| 18 | 64x128 s4 c1x1 halfD mt128 occ2 r216 | 4096 | 128 | 4096 | random | True (6) | True | True | 0 | 1.66e-03 | 2.34e-03 | True |
| 0 | 64x128 s3 c1x1 mt128 occ2 r216 | 4096 | 128 | 4096 | adversarial_scales | True (6) | True | True | 0 | 1.66e-03 | 2.75e-03 | True |
| 14 | 64x128 s3 c1x1 DB mt128 occ2 r216 | 4096 | 128 | 4096 | adversarial_scales | True (6) | True | True | 0 | 1.66e-03 | 2.75e-03 | True |
| 18 | 64x128 s4 c1x1 halfD mt128 occ2 r216 | 4096 | 128 | 4096 | adversarial_scales | True (6) | True | True | 0 | 1.66e-03 | 2.75e-03 | True |
| 0 | 64x128 s3 c1x1 mt128 occ2 r216 | 4096 | 128 | 4096 | extremes_pm127 | True (6) | True | True | 0 | 1.51e-03 | 9.73e-04 | True |
| 14 | 64x128 s3 c1x1 DB mt128 occ2 r216 | 4096 | 128 | 4096 | extremes_pm127 | True (6) | True | True | 0 | 1.51e-03 | 9.73e-04 | True |
| 18 | 64x128 s4 c1x1 halfD mt128 occ2 r216 | 4096 | 128 | 4096 | extremes_pm127 | True (6) | True | True | 0 | 1.51e-03 | 9.73e-04 | True |
| 0 | 64x128 s3 c1x1 mt128 occ2 r216 | 4096 | 128 | 4096 | extremes_m128 | True (6) | True | True | 0 | 1.66e-03 | 3.75e-03 | True |
| 14 | 64x128 s3 c1x1 DB mt128 occ2 r216 | 4096 | 128 | 4096 | extremes_m128 | True (6) | True | True | 0 | 1.66e-03 | 3.75e-03 | True |
| 18 | 64x128 s4 c1x1 halfD mt128 occ2 r216 | 4096 | 128 | 4096 | extremes_m128 | True (6) | True | True | 0 | 1.66e-03 | 3.75e-03 | True |
| 0 | 64x128 s3 c1x1 mt128 occ2 r216 | 1024 | 128 | 1802 | random | True (6) | True | True | 0 | 1.66e-03 | 2.84e-03 | True |
| 14 | 64x128 s3 c1x1 DB mt128 occ2 r216 | 1024 | 128 | 1802 | random | True (6) | True | True | 0 | 1.66e-03 | 2.84e-03 | True |
| 18 | 64x128 s4 c1x1 halfD mt128 occ2 r216 | 1024 | 128 | 1802 | random | True (6) | True | True | 0 | 1.66e-03 | 2.84e-03 | True |
| 0 | 64x128 s3 c1x1 mt128 occ2 r216 | 1024 | 128 | 1802 | adversarial_scales | True (6) | True | True | 0 | 1.66e-03 | 3.21e-03 | True |
| 14 | 64x128 s3 c1x1 DB mt128 occ2 r216 | 1024 | 128 | 1802 | adversarial_scales | True (6) | True | True | 0 | 1.66e-03 | 3.21e-03 | True |
| 18 | 64x128 s4 c1x1 halfD mt128 occ2 r216 | 1024 | 128 | 1802 | adversarial_scales | True (6) | True | True | 0 | 1.66e-03 | 3.21e-03 | True |
| 0 | 64x128 s3 c1x1 mt128 occ2 r216 | 1024 | 128 | 1802 | extremes_pm127 | True (6) | True | True | 0 | 9.18e-04 | 9.73e-04 | True |
| 14 | 64x128 s3 c1x1 DB mt128 occ2 r216 | 1024 | 128 | 1802 | extremes_pm127 | True (6) | True | True | 0 | 9.18e-04 | 9.73e-04 | True |
| 18 | 64x128 s4 c1x1 halfD mt128 occ2 r216 | 1024 | 128 | 1802 | extremes_pm127 | True (6) | True | True | 0 | 9.18e-04 | 9.73e-04 | True |
| 0 | 64x128 s3 c1x1 mt128 occ2 r216 | 1024 | 128 | 1802 | extremes_m128 | True (6) | True | True | 0 | 1.65e-03 | 3.68e-03 | True |
| 14 | 64x128 s3 c1x1 DB mt128 occ2 r216 | 1024 | 128 | 1802 | extremes_m128 | True (6) | True | True | 0 | 1.65e-03 | 3.68e-03 | True |
| 18 | 64x128 s4 c1x1 halfD mt128 occ2 r216 | 1024 | 128 | 1802 | extremes_m128 | True (6) | True | True | 0 | 1.65e-03 | 3.68e-03 | True |
| 0 | 64x128 s3 c1x1 mt128 occ2 r216 | 4096 | 256 | 901 | random | True (6) | True | True | 0 | 1.66e-03 | 2.11e-03 | True |
| 14 | 64x128 s3 c1x1 DB mt128 occ2 r216 | 4096 | 256 | 901 | random | True (6) | True | True | 0 | 1.66e-03 | 2.11e-03 | True |
| 18 | 64x128 s4 c1x1 halfD mt128 occ2 r216 | 4096 | 256 | 901 | random | True (6) | True | True | 0 | 1.66e-03 | 2.11e-03 | True |
| 0 | 64x128 s3 c1x1 mt128 occ2 r216 | 4096 | 256 | 901 | adversarial_scales | True (6) | True | True | 0 | 1.66e-03 | 2.43e-03 | True |
| 14 | 64x128 s3 c1x1 DB mt128 occ2 r216 | 4096 | 256 | 901 | adversarial_scales | True (6) | True | True | 0 | 1.66e-03 | 2.43e-03 | True |
| 18 | 64x128 s4 c1x1 halfD mt128 occ2 r216 | 4096 | 256 | 901 | adversarial_scales | True (6) | True | True | 0 | 1.66e-03 | 2.43e-03 | True |
| 0 | 64x128 s3 c1x1 mt128 occ2 r216 | 4096 | 256 | 901 | extremes_pm127 | True (6) | True | True | 0 | 1.18e-03 | 9.91e-04 | True |
| 14 | 64x128 s3 c1x1 DB mt128 occ2 r216 | 4096 | 256 | 901 | extremes_pm127 | True (6) | True | True | 0 | 1.18e-03 | 9.91e-04 | True |
| 18 | 64x128 s4 c1x1 halfD mt128 occ2 r216 | 4096 | 256 | 901 | extremes_pm127 | True (6) | True | True | 0 | 1.18e-03 | 9.91e-04 | True |
| 0 | 64x128 s3 c1x1 mt128 occ2 r216 | 4096 | 256 | 901 | extremes_m128 | True (6) | True | True | 0 | 1.67e-03 | 1.96e-03 | True |
| 14 | 64x128 s3 c1x1 DB mt128 occ2 r216 | 4096 | 256 | 901 | extremes_m128 | True (6) | True | True | 0 | 1.67e-03 | 1.96e-03 | True |
| 18 | 64x128 s4 c1x1 halfD mt128 occ2 r216 | 4096 | 256 | 901 | extremes_m128 | True (6) | True | True | 0 | 1.67e-03 | 1.96e-03 | True |
| 0 | 64x128 s3 c1x1 mt128 occ2 r216 | 4096 | 256 | 4096 | random | True (6) | True | True | 0 | 1.66e-03 | 1.72e-03 | True |
| 14 | 64x128 s3 c1x1 DB mt128 occ2 r216 | 4096 | 256 | 4096 | random | True (6) | True | True | 0 | 1.66e-03 | 1.72e-03 | True |
| 18 | 64x128 s4 c1x1 halfD mt128 occ2 r216 | 4096 | 256 | 4096 | random | True (6) | True | True | 0 | 1.66e-03 | 1.72e-03 | True |
| 0 | 64x128 s3 c1x1 mt128 occ2 r216 | 4096 | 256 | 4096 | adversarial_scales | True (6) | True | True | 0 | 1.66e-03 | 1.96e-03 | True |
| 14 | 64x128 s3 c1x1 DB mt128 occ2 r216 | 4096 | 256 | 4096 | adversarial_scales | True (6) | True | True | 0 | 1.66e-03 | 1.96e-03 | True |
| 18 | 64x128 s4 c1x1 halfD mt128 occ2 r216 | 4096 | 256 | 4096 | adversarial_scales | True (6) | True | True | 0 | 1.66e-03 | 1.96e-03 | True |
| 0 | 64x128 s3 c1x1 mt128 occ2 r216 | 4096 | 256 | 4096 | extremes_pm127 | True (6) | True | True | 0 | 1.44e-03 | 8.49e-04 | True |
| 14 | 64x128 s3 c1x1 DB mt128 occ2 r216 | 4096 | 256 | 4096 | extremes_pm127 | True (6) | True | True | 0 | 1.44e-03 | 8.49e-04 | True |
| 18 | 64x128 s4 c1x1 halfD mt128 occ2 r216 | 4096 | 256 | 4096 | extremes_pm127 | True (6) | True | True | 0 | 1.44e-03 | 8.49e-04 | True |
| 0 | 64x128 s3 c1x1 mt128 occ2 r216 | 4096 | 256 | 4096 | extremes_m128 | True (6) | True | True | 0 | 1.67e-03 | 3.50e-03 | True |
| 14 | 64x128 s3 c1x1 DB mt128 occ2 r216 | 4096 | 256 | 4096 | extremes_m128 | True (6) | True | True | 0 | 1.67e-03 | 3.50e-03 | True |
| 18 | 64x128 s4 c1x1 halfD mt128 occ2 r216 | 4096 | 256 | 4096 | extremes_m128 | True (6) | True | True | 0 | 1.67e-03 | 3.50e-03 | True |
| 0 | 64x128 s3 c1x1 mt128 occ2 r216 | 1024 | 256 | 4096 | random | True (6) | True | True | 0 | 1.66e-03 | 2.41e-03 | True |
| 14 | 64x128 s3 c1x1 DB mt128 occ2 r216 | 1024 | 256 | 4096 | random | True (6) | True | True | 0 | 1.66e-03 | 2.41e-03 | True |
| 18 | 64x128 s4 c1x1 halfD mt128 occ2 r216 | 1024 | 256 | 4096 | random | True (6) | True | True | 0 | 1.66e-03 | 2.41e-03 | True |
| 0 | 64x128 s3 c1x1 mt128 occ2 r216 | 1024 | 256 | 4096 | adversarial_scales | True (6) | True | True | 0 | 1.66e-03 | 2.43e-03 | True |
| 14 | 64x128 s3 c1x1 DB mt128 occ2 r216 | 1024 | 256 | 4096 | adversarial_scales | True (6) | True | True | 0 | 1.66e-03 | 2.43e-03 | True |
| 18 | 64x128 s4 c1x1 halfD mt128 occ2 r216 | 1024 | 256 | 4096 | adversarial_scales | True (6) | True | True | 0 | 1.66e-03 | 2.43e-03 | True |
| 0 | 64x128 s3 c1x1 mt128 occ2 r216 | 1024 | 256 | 4096 | extremes_pm127 | True (6) | True | True | 0 | 1.02e-03 | 6.02e-04 | True |
| 14 | 64x128 s3 c1x1 DB mt128 occ2 r216 | 1024 | 256 | 4096 | extremes_pm127 | True (6) | True | True | 0 | 1.02e-03 | 6.02e-04 | True |
| 18 | 64x128 s4 c1x1 halfD mt128 occ2 r216 | 1024 | 256 | 4096 | extremes_pm127 | True (6) | True | True | 0 | 1.02e-03 | 6.02e-04 | True |
| 0 | 64x128 s3 c1x1 mt128 occ2 r216 | 1024 | 256 | 4096 | extremes_m128 | True (6) | True | True | 0 | 1.64e-03 | 3.61e-03 | True |
| 14 | 64x128 s3 c1x1 DB mt128 occ2 r216 | 1024 | 256 | 4096 | extremes_m128 | True (6) | True | True | 0 | 1.64e-03 | 3.61e-03 | True |
| 18 | 64x128 s4 c1x1 halfD mt128 occ2 r216 | 1024 | 256 | 4096 | extremes_m128 | True (6) | True | True | 0 | 1.64e-03 | 3.61e-03 | True |
| 0 | 64x128 s3 c1x1 mt128 occ2 r216 | 1024 | 384 | 901 | random | True (6) | True | True | 0 | 1.66e-03 | 2.58e-03 | True |
| 14 | 64x128 s3 c1x1 DB mt128 occ2 r216 | 1024 | 384 | 901 | random | True (6) | True | True | 0 | 1.66e-03 | 2.58e-03 | True |
| 18 | 64x128 s4 c1x1 halfD mt128 occ2 r216 | 1024 | 384 | 901 | random | True (6) | True | True | 0 | 1.66e-03 | 2.58e-03 | True |
| 0 | 64x128 s3 c1x1 mt128 occ2 r216 | 1024 | 384 | 901 | adversarial_scales | True (6) | True | True | 0 | 1.66e-03 | 2.10e-03 | True |
| 14 | 64x128 s3 c1x1 DB mt128 occ2 r216 | 1024 | 384 | 901 | adversarial_scales | True (6) | True | True | 0 | 1.66e-03 | 2.10e-03 | True |
| 18 | 64x128 s4 c1x1 halfD mt128 occ2 r216 | 1024 | 384 | 901 | adversarial_scales | True (6) | True | True | 0 | 1.66e-03 | 2.10e-03 | True |
| 0 | 64x128 s3 c1x1 mt128 occ2 r216 | 1024 | 384 | 901 | extremes_pm127 | True (6) | True | True | 0 | 4.71e-04 | 8.49e-04 | True |
| 14 | 64x128 s3 c1x1 DB mt128 occ2 r216 | 1024 | 384 | 901 | extremes_pm127 | True (6) | True | True | 0 | 4.71e-04 | 8.49e-04 | True |
| 18 | 64x128 s4 c1x1 halfD mt128 occ2 r216 | 1024 | 384 | 901 | extremes_pm127 | True (6) | True | True | 0 | 4.71e-04 | 8.49e-04 | True |
| 0 | 64x128 s3 c1x1 mt128 occ2 r216 | 1024 | 384 | 901 | extremes_m128 | True (6) | True | True | 0 | 1.64e-03 | 3.36e-03 | True |
| 14 | 64x128 s3 c1x1 DB mt128 occ2 r216 | 1024 | 384 | 901 | extremes_m128 | True (6) | True | True | 0 | 1.64e-03 | 3.36e-03 | True |
| 18 | 64x128 s4 c1x1 halfD mt128 occ2 r216 | 1024 | 384 | 901 | extremes_m128 | True (6) | True | True | 0 | 1.64e-03 | 3.36e-03 | True |

## Verdict (verifier)

- Exactness: confirmed. All 19 OCC cfgs are bit-identical (torch.equal) to the same-tile default kernel, to the 256x128 s3 c1x2 default and to kernels/cutlass_int8_bw in 196 / 196 checks (4 shapes + 7 K=128/256/384 edge cases x 4 input cases), rel-L2 vs fp64 <= 1.66e-3 (= bf16 output rounding, same as the reference kernels); 2600-launch stress loop over M = 901..4194 step 37 with 13 cfgs: 0 mismatches, no hang; every timed row asserted torch.equal on the timed tensors. Occupancy is real: cudaOccupancyMaxActiveBlocksPerMultiprocessor = 2 for every 2-CTA cfg (3 for the 64x64 r120 cfg), cudaOccupancyMaxActiveClusters = 132 for the cluster cfgs, ptxas launch budget 128 regs (setmaxnreg 40 / 216), 0 spill bytes for the shape-compiled 64x128 s3 cfgs (24-104 B halfD, 472 B 128x128 s2, 232-752 B dynamic-shape cfgs).
- Speed: approach D is exact but never faster. The best D cfg per shape reaches 0.65-0.89x of the best existing exact kernel (B DB cfg 21/23/25/27 or C pp1 cfg 5/11/12): 4096^3 921 vs 1122 (DB), 4096x4096x42240 911 (c2x1) vs 1101 (pp1), 12288x4096x4096 870 vs 1093, 4096x12288x4096 850 vs 1132, and it collapses at M=42240 with large N or K (611-749 TFLOPS). Co-residency itself works: the same 64x128 c1x1 tile goes from 775 TFLOPS at 1 CTA/SM (main cfg 6, s8) to 916-941 at 2 CTAs/SM (occ cfg 1, s3) at 4096^3, i.e. the second CTA does hide the promotion, but the DB accumulator at 1 CTA/SM already gets the same tile to 961 (main cfg 27) and D4 (occ + DB) does not compound (921-929). So once promotion is hidden the 64x128 tile is bound by operand traffic: 1.5-2x the smem/L2 bytes per MAC of the 128x128 / 256x128 tiles, B fetched by both co-resident CTAs, only 3 stages under the 115712 B per-CTA smem limit; at M=42240 the B-multicast cluster (c2x1) is the only D cfg above 900, which points at the TMA/L2 feed as the limiter. The register/smem budget of 2 CTAs/SM (216 math regs, 115 KB) forbids the larger tiles: D3 64x256 is infeasible (final 128 + int32 128 accum > 216 regs, smem > limit), D2 128x128 with one warpgroup (2 sequential waves, 2 stages) runs at 300-372 TFLOPS, 64x64 at 3 CTAs/SM at 454-511. halfD (4 stages) is 2-3% slower than s3 and halfD+DB is 3x slower (register pressure: 264 B spills inside the mainloop).
- Best exact INT8 g128 today (this run): 848 / 1122 / 1101 TFLOPS at 4096x4096 x M 901 / 4096 / 42240; 1092 at 1024x4096x42240; 1093-1105 at 12288x4096; 977-1132 at 4096x12288. Ratios vs DeepGEMM FP8 g128: 0.84-0.89 (1.26 at 1024x4096x901 where FP8 pads M); vs CUTLASS INT8 per-tensor: 0.76-0.87 (1.79 at 1024x4096x901); cuBLASLt FP8 per-tensor sits at 1077-1410.
- What remains: the 11-16% gap to FP8 g128 is still the exposed int32->fp32 promotion of the two lock-stepped math warpgroups in the 128x128 / 256x128 tiles. Approach D cannot close it because 2 CTAs/SM only fit tiles whose operand traffic costs more than the overlap gains; approach C (ping-pong barriers) recovers 3-4%; approach B (DB accumulator) 5-7%. Remaining exact options are (a) a 128x128 / 256x128 two-warpgroup tile whose warpgroups are deliberately phase-shifted by half a k-block at kernel start without per-iteration barriers (pp already showed barrier-enforced alternation costs about what it gains), (b) a 2-CTA cluster with 128x128 tiles that shares B via multicast to cut the L2 traffic of approach D (64x128 c2x1 was the best D cfg at M=42240), or (c) accepting DB cfg 21 / pp1 at 1085-1130 TFLOPS (0.84-0.89x FP8 g128) as the practical ceiling of the exact INT8 g128 1D2D structure on H100.
