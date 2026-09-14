# INT8 g128 (1,128,128) block-scaled GEMM -- implementer C math-warpgroup ping-pong (PP) variant, verifier run (idle H100, serial timing)

- NVIDIA H100 80GB HBM3, torch 2.13.0a0+9186a08b2c.nv26.07, 2026-09-14T02:47:22; triton do_bench(warmup=25, rep=200, median, L2 flushed), 3 interleaved repeats, median of medians; SM MHz = nvidia-smi samples ~0.1 s into each do_bench (clocks unlocked, power-throttled at M=42240). TFLOPS = 2MNK/t with the true M.
- `C pp` = implementer C's ping-pong mainloop (isolated extension `kernels/deepgemm_int8/pp`, kernel `sm90_int8_gemm_1d2d_pp_impl<..., kPPMode>` in `include/deep_gemm/impls/sm90_int8_gemm_1d2d_pp.cuh`; pp1/pp2 = issue-order alternation of the two math warpgroups via named barriers 10/11, pp5/pp6 = tensor-core mutual exclusion, pp3/4/7 = dual wgmma chains). `B DB` = implementer B's double-buffered accumulator (main extension cfg 19-29). `default` = the unchanged INT8 port (main cfg 8/9/10/11 = 256x128 s3 c1x2 shape-compiled).
- correctness (`probes/verify_int8_g128_pp.py`, logs/pp_verify_check.log): 328 / 328 pp checks bit-identical (torch.equal) to the same-tile default kernel AND to the 256x128 default AND to kernels/cutlass_int8_bw, rel-L2 vs an independent fp64 reference <= 3e-3 (all 32 pp cfgs; (1024,4096,901), (4096,12288,1802), (12288,4096,4096), (4096,4096,4096) + K=128/256/384 edge cases; random / adversarial 1e-4-1.0 scales / +-127 / -128 mixes, NaN sfa padding); 44 / 44 default == cutlass_int8_bw checks. stress 1024x4096: 60 iterations, M 901..4194 step 37, 8 cfgs, 480 launches, 0 mismatches, no hang (0 s); stress 4096x4096: 200 iterations, M 901..4194 step 37, 20 cfgs, 4000 launches, 0 mismatches, no hang (1 s); stress 4096x12288: 60 iterations, M 901..4194 step 37, 9 cfgs, 540 launches, 0 mismatches, no hang (0 s).
- every timed INT8 g128 row was additionally asserted torch.equal to the default kernel on the timed tensors (incl. M=42240).

## Best exact INT8 g128 kernel per shape vs references (TFLOPS)

| N | K | M | default 256x128 s3 c1x2 | best B DB (cfg) | best C pp (cfg) | best exact INT8 g128 | DeepGEMM FP8 g128 | ratio vs FP8 g128 | cutlass INT8 per-tensor | ratio vs per-tensor | cuBLASLt FP8 per-tensor | SM MHz (default / best C pp) |
|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| 4096 | 4096 | 901 | 799 | 853 (128x128 s5 c1x1 N4096K4096 DB) | 811 (256x128 s3 c1x2 N4096K4096 pp1) | 853 B DB cfg 21 | 963 | 0.89 | 983 | 0.87 | 1091 | 1980 / 1980 |
| 4096 | 4096 | 4096 | 1051 | 1128 (128x128 s5 c1x1 N4096K4096 DB) | 1091 (256x128 s3 c1x2 N4096K4096 pp6) | 1128 B DB cfg 21 | 1280 | 0.88 | 1373 | 0.82 | 1342 | 1965 / 1980 |
| 4096 | 4096 | 42240 | 1083 | 1078 (128x128 s5 c1x1 N4096K4096 DB) | 1113 (256x128 s3 c1x2 N4096K4096 pp2) | 1113 C pp cfg 7 | 1230 | 0.91 | 1390 | 0.80 | 1303 | 1740 / 1755 |
| 1024 | 4096 | 901 | 221 (pick_config 64x128 s8 c1x1: 425) | 474 (64x128 s8 c1x1 DB) | 338 (128x128 s5 c1x2 N1024K4096 pp1) | 474 B DB cfg 27 | 382 | 1.24 | 271 | 1.75 | 509 | 1980 / 1980 |
| 1024 | 4096 | 4096 | 897 | 922 (128x128 s5 c1x2 N1024K4096 DB) | 911 (256x128 s3 c1x2 N1024K4096 pp1) | 922 B DB cfg 25 | 1038 | 0.89 | 1113 | 0.83 | 1134 | 1980 / 1980 |
| 1024 | 4096 | 42240 | 1079 | 1088 (128x128 s5 c1x2 N1024K4096 DB) | 1082 (256x128 s3 c1x2 N1024K4096 pp1) | 1088 B DB cfg 25 | 1247 | 0.87 | 1427 | 0.76 | 1262 | 1875 / 1770 |
| 12288 | 4096 | 901 | 915 | 950 (128x128 s5 c1x2 N12288K4096 DB) | 947 (256x128 s3 c1x2 N12288K4096 pp1) | 950 B DB cfg 23 | 1127 | 0.84 | 1175 | 0.81 | 1294 | 1980 / 1980 |
| 12288 | 4096 | 4096 | 1050 | 1083 (128x128 s5 c1x2 N12288K4096 DB) | 1083 (256x128 s3 c1x2 N12288K4096 pp1) | 1083 B DB cfg 23 | 1227 | 0.88 | 1420 | 0.76 | 1275 | 1725 / 1815 |
| 12288 | 4096 | 42240 | 1078 | 1059 (128x128 s5 c1x2 N12288K4096 DB) | 1094 (256x128 s3 c1x2 N12288K4096 pp1) | 1094 C pp cfg 12 | 1237 | 0.88 | 1278 | 0.86 | 1294 | 1575 / 1590 |
| 4096 | 12288 | 901 | 940 | 943 (128x128 s5 c1x2 N4096K12288 DB) | 979 (256x128 s3 c1x2 N4096K12288 pp1) | 979 C pp cfg 11 | 1167 | 0.84 | 1269 | 0.77 | 1407 | 1965 / 1980 |
| 4096 | 12288 | 4096 | 1095 | 1088 (128x128 s5 c1x2 N4096K12288 DB) | 1120 (256x128 s3 c1x2 N4096K12288 pp1) | 1120 C pp cfg 11 | 1292 | 0.87 | 1501 | 0.75 | 1379 | 1845 / 1755 |
| 4096 | 12288 | 42240 | 1076 | 1093 (128x128 s5 c1x2 N4096K12288 DB) | 1116 (256x128 s3 c1x2 N4096K12288 pp1) | 1116 C pp cfg 11 | 1270 | 0.88 | 1488 | 0.75 | 1338 | 1530 / 1470 |

## All rows: microseconds (median of 3 medians), TFLOPS, per-repeat medians, SM MHz samples

| N | K | M | kernel | config | us | TFLOPS | repeats (us) | SM MHz samples |
|---|---|---|---|---|---:|---:|---|---|
| 4096 | 4096 | 901 | default 256x128 cfg 8 | 256x128 s3 c1x2 N4096K4096 | 37.8 | 799 | 37.4, 37.8, 37.8 | [1980, 1980, 1980] |
| 4096 | 4096 | 901 | B DB cfg 21 | 128x128 s5 c1x1 N4096K4096 DB | 35.5 | 853 | 35.4, 35.5, 35.5 | [1980, 1980, 1980] |
| 4096 | 4096 | 901 | B DB cfg 19 | 128x128 s5 c1x2 N4096K4096 DB | 38.2 | 791 | 38.1, 38.3, 38.2 | [1980, 1980, 1980] |
| 4096 | 4096 | 901 | C pp cfg 5 | 256x128 s3 c1x2 N4096K4096 pp1 | 37.3 | 811 | 37.2, 37.3, 37.3 | [1980, 1980, 1980] |
| 4096 | 4096 | 901 | C pp cfg 6 | 256x128 s3 c1x1 N4096K4096 pp1 | 37.5 | 807 | 37.4, 37.5, 37.6 | [1980, 1980, 1980] |
| 4096 | 4096 | 901 | C pp cfg 7 | 256x128 s3 c1x2 N4096K4096 pp2 | 37.5 | 806 | 37.5, 37.6, 37.5 | [1980, 1980, 1980] |
| 4096 | 4096 | 901 | C pp cfg 18 | 256x128 s3 c1x2 N4096K4096 pp5 | 42.3 | 715 | 42.4, 42.3, 42.2 | [1980, 1980, 1980] |
| 4096 | 4096 | 901 | C pp cfg 20 | 256x128 s3 c1x2 N4096K4096 pp6 | 37.7 | 802 | 37.7, 37.6, 37.7 | [1980, 1980, 1980] |
| 4096 | 4096 | 901 | C pp cfg 0 | 128x128 s5 c1x1 N4096K4096 pp1 | 37.8 | 800 | 37.8, 37.7, 37.8 | [1980, 1980, 1980] |
| 4096 | 4096 | 901 | C pp cfg 1 | 128x128 s5 c1x2 N4096K4096 pp1 | 38.3 | 789 | 38.3, 38.2, 38.3 | [1980, 1980, 1980] |
| 4096 | 4096 | 901 | C pp cfg 2 | 128x128 s5 c1x1 N4096K4096 pp2 | 37.9 | 797 | 38.0, 37.7, 37.9 | [1980, 1980, 1980] |
| 4096 | 4096 | 901 | C pp cfg 3 | 128x128 s5 c1x2 N4096K4096 pp2 | 38.5 | 785 | 38.5, 38.2, 38.5 | [1980, 1980, 1980] |
| 4096 | 4096 | 901 | DeepGEMM FP8 g128 | upstream 2.8.0 1D2D, unpadded M | 31.4 | 963 | 31.4, 31.2, 31.4 | [1980, 1980, 1980] |
| 4096 | 4096 | 901 | cutlass_int8 per-tensor | 128x256 c2x1 (int8_scaled_mm cfg 0) | 30.8 | 983 | 30.9, 30.8, 30.8 | [1980, 1980, 1980] |
| 4096 | 4096 | 901 | cuBLASLt FP8 per-tensor | torch._scaled_mm use_fast_accum=True | 27.7 | 1091 | 27.7, 27.7, 27.6 | [1980, 1980, 1980] |
| 4096 | 4096 | 4096 | default 256x128 cfg 8 | 256x128 s3 c1x2 N4096K4096 | 130.8 | 1051 | 130.2, 131.0, 130.8 | [1980, 1950, 1965] |
| 4096 | 4096 | 4096 | B DB cfg 21 | 128x128 s5 c1x1 N4096K4096 DB | 121.8 | 1128 | 121.6, 121.8, 121.9 | [1950, 1965, 1905] |
| 4096 | 4096 | 4096 | B DB cfg 19 | 128x128 s5 c1x2 N4096K4096 DB | 127.6 | 1078 | 127.4, 127.6, 127.8 | [1965, 1965, 1965] |
| 4096 | 4096 | 4096 | C pp cfg 5 | 256x128 s3 c1x2 N4096K4096 pp1 | 126.1 | 1090 | 125.7, 126.2, 126.1 | [1965, 1965, 1965] |
| 4096 | 4096 | 4096 | C pp cfg 6 | 256x128 s3 c1x1 N4096K4096 pp1 | 127.4 | 1079 | 127.2, 127.7, 127.4 | [1980, 1950, 1965] |
| 4096 | 4096 | 4096 | C pp cfg 7 | 256x128 s3 c1x2 N4096K4096 pp2 | 126.3 | 1088 | 126.3, 126.6, 125.8 | [1965, 1965, 1980] |
| 4096 | 4096 | 4096 | C pp cfg 18 | 256x128 s3 c1x2 N4096K4096 pp5 | 144.5 | 951 | 144.4, 144.7, 144.5 | [1965, 1980, 1980] |
| 4096 | 4096 | 4096 | C pp cfg 20 | 256x128 s3 c1x2 N4096K4096 pp6 | 126.0 | 1091 | 125.9, 126.0, 126.1 | [1980, 1980, 1965] |
| 4096 | 4096 | 4096 | C pp cfg 0 | 128x128 s5 c1x1 N4096K4096 pp1 | 131.2 | 1048 | 131.2, 130.9, 131.5 | [1965, 1935, 1950] |
| 4096 | 4096 | 4096 | C pp cfg 1 | 128x128 s5 c1x2 N4096K4096 pp1 | 132.4 | 1038 | 132.4, 131.9, 132.4 | [1950, 1965, 1950] |
| 4096 | 4096 | 4096 | C pp cfg 2 | 128x128 s5 c1x1 N4096K4096 pp2 | 131.9 | 1042 | 131.9, 131.8, 132.0 | [1920, 1965, 1965] |
| 4096 | 4096 | 4096 | C pp cfg 3 | 128x128 s5 c1x2 N4096K4096 pp2 | 133.5 | 1029 | 133.5, 132.9, 133.6 | [1905, 1965, 1965] |
| 4096 | 4096 | 4096 | DeepGEMM FP8 g128 | upstream 2.8.0 1D2D, unpadded M | 107.4 | 1280 | 107.4, 107.4, 108.1 | [1965, 1770, 1830] |
| 4096 | 4096 | 4096 | cutlass_int8 per-tensor | 128x256 c2x1 (int8_scaled_mm cfg 0) | 100.1 | 1373 | 100.1, 100.4, 100.0 | [1965, 1965, 1950] |
| 4096 | 4096 | 4096 | cuBLASLt FP8 per-tensor | torch._scaled_mm use_fast_accum=True | 102.4 | 1342 | 102.4, 103.0, 102.4 | [1845, 1890, 1965] |
| 4096 | 4096 | 42240 | default 256x128 cfg 8 | 256x128 s3 c1x2 N4096K4096 | 1308.9 | 1083 | 1308.9, 1304.0, 1309.8 | [1740, 1635, 1785] |
| 4096 | 4096 | 42240 | B DB cfg 21 | 128x128 s5 c1x1 N4096K4096 DB | 1315.2 | 1078 | 1283.0, 1342.8, 1315.2 | [1710, 1530, 1770] |
| 4096 | 4096 | 42240 | B DB cfg 19 | 128x128 s5 c1x2 N4096K4096 DB | 1338.4 | 1059 | 1338.4, 1346.4, 1309.2 | [1500, 1515, 1515] |
| 4096 | 4096 | 42240 | C pp cfg 5 | 256x128 s3 c1x2 N4096K4096 pp1 | 1312.4 | 1080 | 1337.7, 1261.4, 1312.4 | [1515, 1725, 1560] |
| 4096 | 4096 | 42240 | C pp cfg 6 | 256x128 s3 c1x1 N4096K4096 pp1 | 1324.0 | 1070 | 1320.0, 1324.0, 1337.6 | [1710, 1710, 1665] |
| 4096 | 4096 | 42240 | C pp cfg 7 | 256x128 s3 c1x2 N4096K4096 pp2 | 1273.2 | 1113 | 1273.2, 1287.6, 1272.8 | [1740, 1755, 1755] |
| 4096 | 4096 | 42240 | C pp cfg 18 | 256x128 s3 c1x2 N4096K4096 pp5 | 1416.4 | 1001 | 1413.4, 1426.8, 1416.4 | [1845, 1830, 1815] |
| 4096 | 4096 | 42240 | C pp cfg 20 | 256x128 s3 c1x2 N4096K4096 pp6 | 1311.0 | 1081 | 1324.0, 1273.2, 1311.0 | [1740, 1710, 1770] |
| 4096 | 4096 | 42240 | C pp cfg 0 | 128x128 s5 c1x1 N4096K4096 pp1 | 1421.7 | 997 | 1421.7, 1404.9, 1449.4 | [1470, 1500, 1515] |
| 4096 | 4096 | 42240 | C pp cfg 1 | 128x128 s5 c1x2 N4096K4096 pp1 | 1390.8 | 1019 | 1390.8, 1432.2, 1383.3 | [1560, 1515, 1620] |
| 4096 | 4096 | 42240 | C pp cfg 2 | 128x128 s5 c1x1 N4096K4096 pp2 | 1404.5 | 1009 | 1402.5, 1404.5, 1404.5 | [1665, 1635, 1545] |
| 4096 | 4096 | 42240 | C pp cfg 3 | 128x128 s5 c1x2 N4096K4096 pp2 | 1406.6 | 1008 | 1451.7, 1401.9, 1406.6 | [1620, 1545, 1560] |
| 4096 | 4096 | 42240 | DeepGEMM FP8 g128 | upstream 2.8.0 1D2D, unpadded M | 1152.3 | 1230 | 1162.5, 1152.3, 1150.8 | [1350, 1380, 1335] |
| 4096 | 4096 | 42240 | cutlass_int8 per-tensor | 128x256 c2x1 (int8_scaled_mm cfg 0) | 1019.6 | 1390 | 1001.6, 1028.8, 1019.6 | [1665, 1620, 1680] |
| 4096 | 4096 | 42240 | cuBLASLt FP8 per-tensor | torch._scaled_mm use_fast_accum=True | 1088.0 | 1303 | 1086.9, 1099.5, 1088.0 | [1440, 1425, 1440] |
| 1024 | 4096 | 901 | default 256x128 cfg 9 | 256x128 s3 c1x2 N1024K4096 | 34.1 | 221 | 34.0, 34.1, 34.1 | [1980, 1980, 1980] |
| 1024 | 4096 | 901 | default best cfg 6 | 64x128 s8 c1x1 | 17.8 | 425 | 17.8, 17.8, 17.8 | [1980, 1980, 1980] |
| 1024 | 4096 | 901 | B DB cfg 25 | 128x128 s5 c1x2 N1024K4096 DB | 21.3 | 355 | 21.3, 21.3, 21.3 | [1980, 1980, 1980] |
| 1024 | 4096 | 901 | B DB cfg 27 | 64x128 s8 c1x1 DB | 15.9 | 474 | 15.9, 15.9, 15.9 | [1980, 1980, 1980] |
| 1024 | 4096 | 901 | C pp cfg 13 | 256x128 s3 c1x2 N1024K4096 pp1 | 35.1 | 215 | 35.1, 35.1, 35.1 | [1980, 1980, 1980] |
| 1024 | 4096 | 901 | C pp cfg 10 | 128x128 s5 c1x2 N1024K4096 pp1 | 22.3 | 338 | 22.3, 22.3, 22.4 | [1980, 1980, 1980] |
| 1024 | 4096 | 901 | C pp cfg 25 | 256x128 s3 c1x2 N1024K4096 pp5 | 40.6 | 186 | 40.7, 40.6, 40.6 | [1980, 1980, 1980] |
| 1024 | 4096 | 901 | DeepGEMM FP8 g128 | upstream 2.8.0 1D2D, unpadded M | 19.8 | 382 | 19.9, 19.8, 19.7 | [1980, 1980, 1980] |
| 1024 | 4096 | 901 | cutlass_int8 per-tensor | 128x256 c2x1 (int8_scaled_mm cfg 0) | 27.8 | 271 | 27.9, 27.8, 27.7 | [1980, 1980, 1980] |
| 1024 | 4096 | 901 | cuBLASLt FP8 per-tensor | torch._scaled_mm use_fast_accum=True | 14.8 | 509 | 14.8, 14.8, 15.1 | [1980, 1980, 1980] |
| 1024 | 4096 | 4096 | default 256x128 cfg 9 | 256x128 s3 c1x2 N1024K4096 | 38.3 | 897 | 38.3, 38.5, 38.3 | [1980, 1980, 1980] |
| 1024 | 4096 | 4096 | B DB cfg 25 | 128x128 s5 c1x2 N1024K4096 DB | 37.3 | 922 | 37.3, 37.3, 37.1 | [1980, 1980, 1980] |
| 1024 | 4096 | 4096 | B DB cfg 27 | 64x128 s8 c1x1 DB | 39.5 | 870 | 39.5, 39.6, 39.3 | [1980, 1980, 1980] |
| 1024 | 4096 | 4096 | C pp cfg 13 | 256x128 s3 c1x2 N1024K4096 pp1 | 37.7 | 911 | 37.7, 37.7, 37.7 | [1980, 1980, 1980] |
| 1024 | 4096 | 4096 | C pp cfg 10 | 128x128 s5 c1x2 N1024K4096 pp1 | 38.3 | 896 | 38.4, 38.3, 38.3 | [1980, 1980, 1980] |
| 1024 | 4096 | 4096 | C pp cfg 25 | 256x128 s3 c1x2 N1024K4096 pp5 | 42.7 | 804 | 42.7, 42.8, 42.7 | [1980, 1980, 1980] |
| 1024 | 4096 | 4096 | DeepGEMM FP8 g128 | upstream 2.8.0 1D2D, unpadded M | 33.1 | 1038 | 33.1, 33.1, 33.1 | [1980, 1980, 1980] |
| 1024 | 4096 | 4096 | cutlass_int8 per-tensor | 128x256 c2x1 (int8_scaled_mm cfg 0) | 30.9 | 1113 | 30.9, 30.8, 30.9 | [1980, 1980, 1980] |
| 1024 | 4096 | 4096 | cuBLASLt FP8 per-tensor | torch._scaled_mm use_fast_accum=True | 30.3 | 1134 | 30.3, 30.1, 30.3 | [1980, 1980, 1980] |
| 1024 | 4096 | 42240 | default 256x128 cfg 9 | 256x128 s3 c1x2 N1024K4096 | 328.2 | 1079 | 328.2, 329.9, 319.3 | [1800, 1875, 1905] |
| 1024 | 4096 | 42240 | B DB cfg 25 | 128x128 s5 c1x2 N1024K4096 DB | 325.8 | 1088 | 328.2, 325.8, 321.3 | [1650, 1680, 1650] |
| 1024 | 4096 | 42240 | B DB cfg 27 | 64x128 s8 c1x1 DB | 362.1 | 979 | 361.4, 365.4, 362.1 | [1755, 1725, 1755] |
| 1024 | 4096 | 42240 | C pp cfg 13 | 256x128 s3 c1x2 N1024K4096 pp1 | 327.6 | 1082 | 332.5, 327.6, 324.7 | [1770, 1785, 1755] |
| 1024 | 4096 | 42240 | C pp cfg 10 | 128x128 s5 c1x2 N1024K4096 pp1 | 352.8 | 1004 | 346.2, 352.8, 359.4 | [1740, 1605, 1620] |
| 1024 | 4096 | 42240 | C pp cfg 25 | 256x128 s3 c1x2 N1024K4096 pp5 | 353.2 | 1003 | 351.6, 353.2, 357.6 | [1950, 1935, 1860] |
| 1024 | 4096 | 42240 | DeepGEMM FP8 g128 | upstream 2.8.0 1D2D, unpadded M | 284.1 | 1247 | 283.2, 284.1, 287.9 | [1530, 1530, 1470] |
| 1024 | 4096 | 42240 | cutlass_int8 per-tensor | 128x256 c2x1 (int8_scaled_mm cfg 0) | 248.2 | 1427 | 248.2, 252.0, 248.2 | [1770, 1650, 1770] |
| 1024 | 4096 | 42240 | cuBLASLt FP8 per-tensor | torch._scaled_mm use_fast_accum=True | 280.8 | 1262 | 280.8, 281.5, 274.3 | [1440, 1500, 1560] |
| 12288 | 4096 | 901 | default 256x128 cfg 10 | 256x128 s3 c1x2 N12288K4096 | 99.1 | 915 | 98.4, 99.1, 99.3 | [1980, 1980, 1965] |
| 12288 | 4096 | 901 | B DB cfg 23 | 128x128 s5 c1x2 N12288K4096 DB | 95.5 | 950 | 95.5, 95.6, 95.5 | [1980, 1965, 1980] |
| 12288 | 4096 | 901 | C pp cfg 12 | 256x128 s3 c1x2 N12288K4096 pp1 | 95.7 | 947 | 95.7, 95.7, 95.8 | [1980, 1980, 1980] |
| 12288 | 4096 | 901 | C pp cfg 9 | 128x128 s5 c1x2 N12288K4096 pp1 | 101.1 | 897 | 101.1, 101.0, 101.2 | [1980, 1980, 1980] |
| 12288 | 4096 | 901 | C pp cfg 22 | 256x128 s3 c1x2 N12288K4096 pp5 | 111.1 | 816 | 111.1, 111.2, 111.1 | [1980, 1980, 1980] |
| 12288 | 4096 | 901 | DeepGEMM FP8 g128 | upstream 2.8.0 1D2D, unpadded M | 80.4 | 1127 | 80.5, 80.1, 80.4 | [1950, 1965, 1965] |
| 12288 | 4096 | 901 | cutlass_int8 per-tensor | 128x256 c2x1 (int8_scaled_mm cfg 0) | 77.2 | 1175 | 77.2, 77.2, 77.2 | [1965, 1965, 1965] |
| 12288 | 4096 | 901 | cuBLASLt FP8 per-tensor | torch._scaled_mm use_fast_accum=True | 70.1 | 1294 | 70.3, 70.1, 69.9 | [1965, 1965, 1980] |
| 12288 | 4096 | 4096 | default 256x128 cfg 10 | 256x128 s3 c1x2 N12288K4096 | 392.6 | 1050 | 386.7, 392.6, 394.0 | [1860, 1680, 1725] |
| 12288 | 4096 | 4096 | B DB cfg 23 | 128x128 s5 c1x2 N12288K4096 DB | 380.6 | 1083 | 378.7, 380.6, 386.5 | [1725, 1710, 1635] |
| 12288 | 4096 | 4096 | C pp cfg 12 | 256x128 s3 c1x2 N12288K4096 pp1 | 380.6 | 1083 | 379.6, 380.6, 381.5 | [1845, 1815, 1815] |
| 12288 | 4096 | 4096 | C pp cfg 9 | 128x128 s5 c1x2 N12288K4096 pp1 | 410.6 | 1004 | 430.2, 410.6, 408.3 | [1665, 1755, 1755] |
| 12288 | 4096 | 4096 | C pp cfg 22 | 256x128 s3 c1x2 N12288K4096 pp5 | 429.9 | 959 | 421.5, 433.7, 429.9 | [1845, 1800, 1875] |
| 12288 | 4096 | 4096 | DeepGEMM FP8 g128 | upstream 2.8.0 1D2D, unpadded M | 336.0 | 1227 | 330.8, 336.0, 338.9 | [1515, 1500, 1590] |
| 12288 | 4096 | 4096 | cutlass_int8 per-tensor | 128x256 c2x1 (int8_scaled_mm cfg 0) | 290.3 | 1420 | 290.3, 288.9, 292.8 | [1800, 1815, 1785] |
| 12288 | 4096 | 4096 | cuBLASLt FP8 per-tensor | torch._scaled_mm use_fast_accum=True | 323.3 | 1275 | 323.3, 304.4, 327.5 | [1485, 1635, 1515] |
| 12288 | 4096 | 42240 | default 256x128 cfg 10 | 256x128 s3 c1x2 N12288K4096 | 3943.2 | 1078 | 3988.6, 3943.2, 3939.9 | [1695, 1410, 1575] |
| 12288 | 4096 | 42240 | B DB cfg 23 | 128x128 s5 c1x2 N12288K4096 DB | 4015.2 | 1059 | 4015.2, 4024.9, 3887.3 | [1335, 1485, 1455] |
| 12288 | 4096 | 42240 | C pp cfg 12 | 256x128 s3 c1x2 N12288K4096 pp1 | 3886.6 | 1094 | 3923.3, 3848.7, 3886.6 | [1635, 1485, 1590] |
| 12288 | 4096 | 42240 | C pp cfg 9 | 128x128 s5 c1x2 N12288K4096 pp1 | 4293.7 | 990 | 4293.7, 4312.0, 4222.8 | [1455, 1590, 1470] |
| 12288 | 4096 | 42240 | C pp cfg 22 | 256x128 s3 c1x2 N12288K4096 pp5 | 4227.7 | 1006 | 4223.7, 4227.7, 4261.5 | [1485, 1665, 1740] |
| 12288 | 4096 | 42240 | DeepGEMM FP8 g128 | upstream 2.8.0 1D2D, unpadded M | 3436.8 | 1237 | 3508.4, 3436.8, 3315.1 | [1320, 1275, 1365] |
| 12288 | 4096 | 42240 | cutlass_int8 per-tensor | 128x256 c2x1 (int8_scaled_mm cfg 0) | 3326.8 | 1278 | 3328.4, 3326.8, 3322.7 | [1245, 1335, 1425] |
| 12288 | 4096 | 42240 | cuBLASLt FP8 per-tensor | torch._scaled_mm use_fast_accum=True | 3285.7 | 1294 | 3285.7, 3287.5, 3282.7 | [1320, 1425, 1320] |
| 4096 | 12288 | 901 | default 256x128 cfg 11 | 256x128 s3 c1x2 N4096K12288 | 96.5 | 940 | 95.4, 96.8, 96.5 | [1980, 1965, 1965] |
| 4096 | 12288 | 901 | B DB cfg 24 | 128x128 s5 c1x2 N4096K12288 DB | 96.2 | 943 | 96.1, 96.2, 96.4 | [1950, 1980, 1965] |
| 4096 | 12288 | 901 | C pp cfg 11 | 256x128 s3 c1x2 N4096K12288 pp1 | 92.6 | 979 | 92.8, 92.5, 92.6 | [1980, 1980, 1965] |
| 4096 | 12288 | 901 | C pp cfg 8 | 128x128 s5 c1x2 N4096K12288 pp1 | 98.1 | 924 | 97.8, 98.3, 98.1 | [1980, 1980, 1980] |
| 4096 | 12288 | 901 | C pp cfg 24 | 256x128 s3 c1x2 N4096K12288 pp5 | 107.9 | 841 | 107.9, 108.2, 107.5 | [1980, 1980, 1980] |
| 4096 | 12288 | 901 | DeepGEMM FP8 g128 | upstream 2.8.0 1D2D, unpadded M | 77.7 | 1167 | 77.6, 77.9, 77.7 | [1980, 1935, 1935] |
| 4096 | 12288 | 901 | cutlass_int8 per-tensor | 128x256 c2x1 (int8_scaled_mm cfg 0) | 71.5 | 1269 | 71.5, 71.6, 71.5 | [1965, 1965, 1965] |
| 4096 | 12288 | 901 | cuBLASLt FP8 per-tensor | torch._scaled_mm use_fast_accum=True | 64.5 | 1407 | 64.4, 64.5, 64.6 | [1965, 1965, 1890] |
| 4096 | 12288 | 4096 | default 256x128 cfg 11 | 256x128 s3 c1x2 N4096K12288 | 376.7 | 1095 | 375.1, 384.8, 376.7 | [1875, 1755, 1845] |
| 4096 | 12288 | 4096 | B DB cfg 24 | 128x128 s5 c1x2 N4096K12288 DB | 378.9 | 1088 | 372.8, 378.9, 384.9 | [1710, 1710, 1695] |
| 4096 | 12288 | 4096 | C pp cfg 11 | 256x128 s3 c1x2 N4096K12288 pp1 | 368.2 | 1120 | 366.2, 368.2, 370.3 | [1785, 1755, 1680] |
| 4096 | 12288 | 4096 | C pp cfg 8 | 128x128 s5 c1x2 N4096K12288 pp1 | 394.9 | 1044 | 417.9, 394.9, 393.6 | [1665, 1860, 1905] |
| 4096 | 12288 | 4096 | C pp cfg 24 | 256x128 s3 c1x2 N4096K12288 pp5 | 409.8 | 1006 | 403.1, 418.5, 409.8 | [1920, 1875, 1905] |
| 4096 | 12288 | 4096 | DeepGEMM FP8 g128 | upstream 2.8.0 1D2D, unpadded M | 319.0 | 1292 | 313.0, 319.0, 331.7 | [1590, 1500, 1440] |
| 4096 | 12288 | 4096 | cutlass_int8 per-tensor | 128x256 c2x1 (int8_scaled_mm cfg 0) | 274.7 | 1501 | 274.7, 274.1, 282.2 | [1770, 1650, 1605] |
| 4096 | 12288 | 4096 | cuBLASLt FP8 per-tensor | torch._scaled_mm use_fast_accum=True | 298.9 | 1379 | 300.0, 298.4, 298.9 | [1575, 1605, 1575] |
| 4096 | 12288 | 42240 | default 256x128 cfg 11 | 256x128 s3 c1x2 N4096K12288 | 3951.1 | 1076 | 4016.4, 3951.1, 3915.4 | [1710, 1530, 1260] |
| 4096 | 12288 | 42240 | B DB cfg 24 | 128x128 s5 c1x2 N4096K12288 DB | 3888.5 | 1093 | 3982.6, 3866.5, 3888.5 | [1485, 1455, 1380] |
| 4096 | 12288 | 42240 | C pp cfg 11 | 256x128 s3 c1x2 N4096K12288 pp1 | 3810.4 | 1116 | 3810.4, 3746.1, 3828.6 | [1395, 1545, 1470] |
| 4096 | 12288 | 42240 | C pp cfg 8 | 128x128 s5 c1x2 N4096K12288 pp1 | 4134.5 | 1028 | 4169.7, 4126.1, 4134.5 | [1425, 1470, 1455] |
| 4096 | 12288 | 42240 | C pp cfg 24 | 256x128 s3 c1x2 N4096K12288 pp5 | 4171.4 | 1019 | 4171.4, 4110.8, 4298.2 | [1590, 1410, 1620] |
| 4096 | 12288 | 42240 | DeepGEMM FP8 g128 | upstream 2.8.0 1D2D, unpadded M | 3347.4 | 1270 | 3374.2, 3347.4, 3344.4 | [1350, 1200, 1410] |
| 4096 | 12288 | 42240 | cutlass_int8 per-tensor | 128x256 c2x1 (int8_scaled_mm cfg 0) | 2858.1 | 1488 | 2866.6, 2838.5, 2858.1 | [1335, 1470, 1440] |
| 4096 | 12288 | 42240 | cuBLASLt FP8 per-tensor | torch._scaled_mm use_fast_accum=True | 3177.4 | 1338 | 3209.7, 3177.4, 3099.6 | [1410, 1200, 1305] |

## Verification detail (pp cfg vs same-tile default REF1 / 256x128 default REF2 / cutlass_int8_bw REF3; rel-L2 and max|d|/max|ref| vs fp64)

| pp cfg | name | N | K | M | case | ==REF1 (cfg) | ==REF2 | ==REF3 | #diff | rel-L2 | max rel | ok |
|---|---|---|---|---|---|---|---|---|---:|---:|---:|---|
| 10 | 128x128 s5 c1x2 N1024K4096 pp1 | 1024 | 4096 | 901 | random | True (None) | True | True | 0 | 1.66e-03 | 3.42e-03 | True |
| 13 | 256x128 s3 c1x2 N1024K4096 pp1 | 1024 | 4096 | 901 | random | True (9) | True | True | 0 | 1.66e-03 | 3.42e-03 | True |
| 14 | 128x128 s5 c1x1 pp1 | 1024 | 4096 | 901 | random | True (3) | True | True | 0 | 1.66e-03 | 3.42e-03 | True |
| 15 | 256x128 s3 c1x2 pp1 | 1024 | 4096 | 901 | random | True (9) | True | True | 0 | 1.66e-03 | 3.42e-03 | True |
| 23 | 128x128 s5 c1x1 pp5 | 1024 | 4096 | 901 | random | True (3) | True | True | 0 | 1.66e-03 | 3.42e-03 | True |
| 25 | 256x128 s3 c1x2 N1024K4096 pp5 | 1024 | 4096 | 901 | random | True (9) | True | True | 0 | 1.66e-03 | 3.42e-03 | True |
| 26 | 256x128 s3 c1x2 pp6 | 1024 | 4096 | 901 | random | True (9) | True | True | 0 | 1.66e-03 | 3.42e-03 | True |
| 31 | 128x128 s5 c1x1 pp7 | 1024 | 4096 | 901 | random | True (3) | True | True | 0 | 1.66e-03 | 3.42e-03 | True |
| 10 | 128x128 s5 c1x2 N1024K4096 pp1 | 1024 | 4096 | 901 | adversarial_scales | True (None) | True | True | 0 | 1.66e-03 | 2.82e-03 | True |
| 13 | 256x128 s3 c1x2 N1024K4096 pp1 | 1024 | 4096 | 901 | adversarial_scales | True (9) | True | True | 0 | 1.66e-03 | 2.82e-03 | True |
| 14 | 128x128 s5 c1x1 pp1 | 1024 | 4096 | 901 | adversarial_scales | True (3) | True | True | 0 | 1.66e-03 | 2.82e-03 | True |
| 15 | 256x128 s3 c1x2 pp1 | 1024 | 4096 | 901 | adversarial_scales | True (9) | True | True | 0 | 1.66e-03 | 2.82e-03 | True |
| 23 | 128x128 s5 c1x1 pp5 | 1024 | 4096 | 901 | adversarial_scales | True (3) | True | True | 0 | 1.66e-03 | 2.82e-03 | True |
| 25 | 256x128 s3 c1x2 N1024K4096 pp5 | 1024 | 4096 | 901 | adversarial_scales | True (9) | True | True | 0 | 1.66e-03 | 2.82e-03 | True |
| 26 | 256x128 s3 c1x2 pp6 | 1024 | 4096 | 901 | adversarial_scales | True (9) | True | True | 0 | 1.66e-03 | 2.82e-03 | True |
| 31 | 128x128 s5 c1x1 pp7 | 1024 | 4096 | 901 | adversarial_scales | True (3) | True | True | 0 | 1.66e-03 | 2.82e-03 | True |
| 10 | 128x128 s5 c1x2 N1024K4096 pp1 | 1024 | 4096 | 901 | extremes_pm127 | True (None) | True | True | 0 | 1.82e-03 | 2.32e-03 | True |
| 13 | 256x128 s3 c1x2 N1024K4096 pp1 | 1024 | 4096 | 901 | extremes_pm127 | True (9) | True | True | 0 | 1.82e-03 | 2.32e-03 | True |
| 14 | 128x128 s5 c1x1 pp1 | 1024 | 4096 | 901 | extremes_pm127 | True (3) | True | True | 0 | 1.82e-03 | 2.32e-03 | True |
| 15 | 256x128 s3 c1x2 pp1 | 1024 | 4096 | 901 | extremes_pm127 | True (9) | True | True | 0 | 1.82e-03 | 2.32e-03 | True |
| 23 | 128x128 s5 c1x1 pp5 | 1024 | 4096 | 901 | extremes_pm127 | True (3) | True | True | 0 | 1.82e-03 | 2.32e-03 | True |
| 25 | 256x128 s3 c1x2 N1024K4096 pp5 | 1024 | 4096 | 901 | extremes_pm127 | True (9) | True | True | 0 | 1.82e-03 | 2.32e-03 | True |
| 26 | 256x128 s3 c1x2 pp6 | 1024 | 4096 | 901 | extremes_pm127 | True (9) | True | True | 0 | 1.82e-03 | 2.32e-03 | True |
| 31 | 128x128 s5 c1x1 pp7 | 1024 | 4096 | 901 | extremes_pm127 | True (3) | True | True | 0 | 1.82e-03 | 2.32e-03 | True |
| 10 | 128x128 s5 c1x2 N1024K4096 pp1 | 1024 | 4096 | 901 | extremes_m128 | True (None) | True | True | 0 | 1.73e-03 | 2.11e-03 | True |
| 13 | 256x128 s3 c1x2 N1024K4096 pp1 | 1024 | 4096 | 901 | extremes_m128 | True (9) | True | True | 0 | 1.73e-03 | 2.11e-03 | True |
| 14 | 128x128 s5 c1x1 pp1 | 1024 | 4096 | 901 | extremes_m128 | True (3) | True | True | 0 | 1.73e-03 | 2.11e-03 | True |
| 15 | 256x128 s3 c1x2 pp1 | 1024 | 4096 | 901 | extremes_m128 | True (9) | True | True | 0 | 1.73e-03 | 2.11e-03 | True |
| 23 | 128x128 s5 c1x1 pp5 | 1024 | 4096 | 901 | extremes_m128 | True (3) | True | True | 0 | 1.73e-03 | 2.11e-03 | True |
| 25 | 256x128 s3 c1x2 N1024K4096 pp5 | 1024 | 4096 | 901 | extremes_m128 | True (9) | True | True | 0 | 1.73e-03 | 2.11e-03 | True |
| 26 | 256x128 s3 c1x2 pp6 | 1024 | 4096 | 901 | extremes_m128 | True (9) | True | True | 0 | 1.73e-03 | 2.11e-03 | True |
| 31 | 128x128 s5 c1x1 pp7 | 1024 | 4096 | 901 | extremes_m128 | True (3) | True | True | 0 | 1.73e-03 | 2.11e-03 | True |
| 8 | 128x128 s5 c1x2 N4096K12288 pp1 | 4096 | 12288 | 1802 | random | True (None) | True | True | 0 | 1.66e-03 | 2.61e-03 | True |
| 11 | 256x128 s3 c1x2 N4096K12288 pp1 | 4096 | 12288 | 1802 | random | True (11) | True | True | 0 | 1.66e-03 | 2.61e-03 | True |
| 14 | 128x128 s5 c1x1 pp1 | 4096 | 12288 | 1802 | random | True (3) | True | True | 0 | 1.66e-03 | 2.61e-03 | True |
| 15 | 256x128 s3 c1x2 pp1 | 4096 | 12288 | 1802 | random | True (11) | True | True | 0 | 1.66e-03 | 2.61e-03 | True |
| 23 | 128x128 s5 c1x1 pp5 | 4096 | 12288 | 1802 | random | True (3) | True | True | 0 | 1.66e-03 | 2.61e-03 | True |
| 24 | 256x128 s3 c1x2 N4096K12288 pp5 | 4096 | 12288 | 1802 | random | True (11) | True | True | 0 | 1.66e-03 | 2.61e-03 | True |
| 26 | 256x128 s3 c1x2 pp6 | 4096 | 12288 | 1802 | random | True (11) | True | True | 0 | 1.66e-03 | 2.61e-03 | True |
| 30 | 128x128 s5 c1x2 N4096K12288 pp7 | 4096 | 12288 | 1802 | random | True (None) | True | True | 0 | 1.66e-03 | 2.61e-03 | True |
| 31 | 128x128 s5 c1x1 pp7 | 4096 | 12288 | 1802 | random | True (3) | True | True | 0 | 1.66e-03 | 2.61e-03 | True |
| 8 | 128x128 s5 c1x2 N4096K12288 pp1 | 4096 | 12288 | 1802 | adversarial_scales | True (None) | True | True | 0 | 1.66e-03 | 3.02e-03 | True |
| 11 | 256x128 s3 c1x2 N4096K12288 pp1 | 4096 | 12288 | 1802 | adversarial_scales | True (11) | True | True | 0 | 1.66e-03 | 3.02e-03 | True |
| 14 | 128x128 s5 c1x1 pp1 | 4096 | 12288 | 1802 | adversarial_scales | True (3) | True | True | 0 | 1.66e-03 | 3.02e-03 | True |
| 15 | 256x128 s3 c1x2 pp1 | 4096 | 12288 | 1802 | adversarial_scales | True (11) | True | True | 0 | 1.66e-03 | 3.02e-03 | True |
| 23 | 128x128 s5 c1x1 pp5 | 4096 | 12288 | 1802 | adversarial_scales | True (3) | True | True | 0 | 1.66e-03 | 3.02e-03 | True |
| 24 | 256x128 s3 c1x2 N4096K12288 pp5 | 4096 | 12288 | 1802 | adversarial_scales | True (11) | True | True | 0 | 1.66e-03 | 3.02e-03 | True |
| 26 | 256x128 s3 c1x2 pp6 | 4096 | 12288 | 1802 | adversarial_scales | True (11) | True | True | 0 | 1.66e-03 | 3.02e-03 | True |
| 30 | 128x128 s5 c1x2 N4096K12288 pp7 | 4096 | 12288 | 1802 | adversarial_scales | True (None) | True | True | 0 | 1.66e-03 | 3.02e-03 | True |
| 31 | 128x128 s5 c1x1 pp7 | 4096 | 12288 | 1802 | adversarial_scales | True (3) | True | True | 0 | 1.66e-03 | 3.02e-03 | True |
| 8 | 128x128 s5 c1x2 N4096K12288 pp1 | 4096 | 12288 | 1802 | extremes_pm127 | True (None) | True | True | 0 | 1.48e-03 | 1.98e-03 | True |
| 11 | 256x128 s3 c1x2 N4096K12288 pp1 | 4096 | 12288 | 1802 | extremes_pm127 | True (11) | True | True | 0 | 1.48e-03 | 1.98e-03 | True |
| 14 | 128x128 s5 c1x1 pp1 | 4096 | 12288 | 1802 | extremes_pm127 | True (3) | True | True | 0 | 1.48e-03 | 1.98e-03 | True |
| 15 | 256x128 s3 c1x2 pp1 | 4096 | 12288 | 1802 | extremes_pm127 | True (11) | True | True | 0 | 1.48e-03 | 1.98e-03 | True |
| 23 | 128x128 s5 c1x1 pp5 | 4096 | 12288 | 1802 | extremes_pm127 | True (3) | True | True | 0 | 1.48e-03 | 1.98e-03 | True |
| 24 | 256x128 s3 c1x2 N4096K12288 pp5 | 4096 | 12288 | 1802 | extremes_pm127 | True (11) | True | True | 0 | 1.48e-03 | 1.98e-03 | True |
| 26 | 256x128 s3 c1x2 pp6 | 4096 | 12288 | 1802 | extremes_pm127 | True (11) | True | True | 0 | 1.48e-03 | 1.98e-03 | True |
| 30 | 128x128 s5 c1x2 N4096K12288 pp7 | 4096 | 12288 | 1802 | extremes_pm127 | True (None) | True | True | 0 | 1.48e-03 | 1.98e-03 | True |
| 31 | 128x128 s5 c1x1 pp7 | 4096 | 12288 | 1802 | extremes_pm127 | True (3) | True | True | 0 | 1.48e-03 | 1.98e-03 | True |
| 8 | 128x128 s5 c1x2 N4096K12288 pp1 | 4096 | 12288 | 1802 | extremes_m128 | True (None) | True | True | 0 | 1.74e-03 | 3.02e-03 | True |
| 11 | 256x128 s3 c1x2 N4096K12288 pp1 | 4096 | 12288 | 1802 | extremes_m128 | True (11) | True | True | 0 | 1.74e-03 | 3.02e-03 | True |
| 14 | 128x128 s5 c1x1 pp1 | 4096 | 12288 | 1802 | extremes_m128 | True (3) | True | True | 0 | 1.74e-03 | 3.02e-03 | True |
| 15 | 256x128 s3 c1x2 pp1 | 4096 | 12288 | 1802 | extremes_m128 | True (11) | True | True | 0 | 1.74e-03 | 3.02e-03 | True |
| 23 | 128x128 s5 c1x1 pp5 | 4096 | 12288 | 1802 | extremes_m128 | True (3) | True | True | 0 | 1.74e-03 | 3.02e-03 | True |
| 24 | 256x128 s3 c1x2 N4096K12288 pp5 | 4096 | 12288 | 1802 | extremes_m128 | True (11) | True | True | 0 | 1.74e-03 | 3.02e-03 | True |
| 26 | 256x128 s3 c1x2 pp6 | 4096 | 12288 | 1802 | extremes_m128 | True (11) | True | True | 0 | 1.74e-03 | 3.02e-03 | True |
| 30 | 128x128 s5 c1x2 N4096K12288 pp7 | 4096 | 12288 | 1802 | extremes_m128 | True (None) | True | True | 0 | 1.74e-03 | 3.02e-03 | True |
| 31 | 128x128 s5 c1x1 pp7 | 4096 | 12288 | 1802 | extremes_m128 | True (3) | True | True | 0 | 1.74e-03 | 3.02e-03 | True |
| 9 | 128x128 s5 c1x2 N12288K4096 pp1 | 12288 | 4096 | 4096 | random | True (None) | True | True | 0 | 1.66e-03 | 2.78e-03 | True |
| 12 | 256x128 s3 c1x2 N12288K4096 pp1 | 12288 | 4096 | 4096 | random | True (10) | True | True | 0 | 1.66e-03 | 2.78e-03 | True |
| 14 | 128x128 s5 c1x1 pp1 | 12288 | 4096 | 4096 | random | True (3) | True | True | 0 | 1.66e-03 | 2.78e-03 | True |
| 15 | 256x128 s3 c1x2 pp1 | 12288 | 4096 | 4096 | random | True (10) | True | True | 0 | 1.66e-03 | 2.78e-03 | True |
| 21 | 128x128 s5 c1x2 N12288K4096 pp5 | 12288 | 4096 | 4096 | random | True (None) | True | True | 0 | 1.66e-03 | 2.78e-03 | True |
| 22 | 256x128 s3 c1x2 N12288K4096 pp5 | 12288 | 4096 | 4096 | random | True (10) | True | True | 0 | 1.66e-03 | 2.78e-03 | True |
| 23 | 128x128 s5 c1x1 pp5 | 12288 | 4096 | 4096 | random | True (3) | True | True | 0 | 1.66e-03 | 2.78e-03 | True |
| 26 | 256x128 s3 c1x2 pp6 | 12288 | 4096 | 4096 | random | True (10) | True | True | 0 | 1.66e-03 | 2.78e-03 | True |
| 29 | 128x128 s5 c1x2 N12288K4096 pp7 | 12288 | 4096 | 4096 | random | True (None) | True | True | 0 | 1.66e-03 | 2.78e-03 | True |
| 31 | 128x128 s5 c1x1 pp7 | 12288 | 4096 | 4096 | random | True (3) | True | True | 0 | 1.66e-03 | 2.78e-03 | True |
| 9 | 128x128 s5 c1x2 N12288K4096 pp1 | 12288 | 4096 | 4096 | adversarial_scales | True (None) | True | True | 0 | 1.66e-03 | 2.17e-03 | True |
| 12 | 256x128 s3 c1x2 N12288K4096 pp1 | 12288 | 4096 | 4096 | adversarial_scales | True (10) | True | True | 0 | 1.66e-03 | 2.17e-03 | True |
| 14 | 128x128 s5 c1x1 pp1 | 12288 | 4096 | 4096 | adversarial_scales | True (3) | True | True | 0 | 1.66e-03 | 2.17e-03 | True |
| 15 | 256x128 s3 c1x2 pp1 | 12288 | 4096 | 4096 | adversarial_scales | True (10) | True | True | 0 | 1.66e-03 | 2.17e-03 | True |
| 21 | 128x128 s5 c1x2 N12288K4096 pp5 | 12288 | 4096 | 4096 | adversarial_scales | True (None) | True | True | 0 | 1.66e-03 | 2.17e-03 | True |
| 22 | 256x128 s3 c1x2 N12288K4096 pp5 | 12288 | 4096 | 4096 | adversarial_scales | True (10) | True | True | 0 | 1.66e-03 | 2.17e-03 | True |
| 23 | 128x128 s5 c1x1 pp5 | 12288 | 4096 | 4096 | adversarial_scales | True (3) | True | True | 0 | 1.66e-03 | 2.17e-03 | True |
| 26 | 256x128 s3 c1x2 pp6 | 12288 | 4096 | 4096 | adversarial_scales | True (10) | True | True | 0 | 1.66e-03 | 2.17e-03 | True |
| 29 | 128x128 s5 c1x2 N12288K4096 pp7 | 12288 | 4096 | 4096 | adversarial_scales | True (None) | True | True | 0 | 1.66e-03 | 2.17e-03 | True |
| 31 | 128x128 s5 c1x1 pp7 | 12288 | 4096 | 4096 | adversarial_scales | True (3) | True | True | 0 | 1.66e-03 | 2.17e-03 | True |
| 9 | 128x128 s5 c1x2 N12288K4096 pp1 | 12288 | 4096 | 4096 | extremes_pm127 | True (None) | True | True | 0 | 1.66e-03 | 1.67e-03 | True |
| 12 | 256x128 s3 c1x2 N12288K4096 pp1 | 12288 | 4096 | 4096 | extremes_pm127 | True (10) | True | True | 0 | 1.66e-03 | 1.67e-03 | True |
| 14 | 128x128 s5 c1x1 pp1 | 12288 | 4096 | 4096 | extremes_pm127 | True (3) | True | True | 0 | 1.66e-03 | 1.67e-03 | True |
| 15 | 256x128 s3 c1x2 pp1 | 12288 | 4096 | 4096 | extremes_pm127 | True (10) | True | True | 0 | 1.66e-03 | 1.67e-03 | True |
| 21 | 128x128 s5 c1x2 N12288K4096 pp5 | 12288 | 4096 | 4096 | extremes_pm127 | True (None) | True | True | 0 | 1.66e-03 | 1.67e-03 | True |
| 22 | 256x128 s3 c1x2 N12288K4096 pp5 | 12288 | 4096 | 4096 | extremes_pm127 | True (10) | True | True | 0 | 1.66e-03 | 1.67e-03 | True |
| 23 | 128x128 s5 c1x1 pp5 | 12288 | 4096 | 4096 | extremes_pm127 | True (3) | True | True | 0 | 1.66e-03 | 1.67e-03 | True |
| 26 | 256x128 s3 c1x2 pp6 | 12288 | 4096 | 4096 | extremes_pm127 | True (10) | True | True | 0 | 1.66e-03 | 1.67e-03 | True |
| 29 | 128x128 s5 c1x2 N12288K4096 pp7 | 12288 | 4096 | 4096 | extremes_pm127 | True (None) | True | True | 0 | 1.66e-03 | 1.67e-03 | True |
| 31 | 128x128 s5 c1x1 pp7 | 12288 | 4096 | 4096 | extremes_pm127 | True (3) | True | True | 0 | 1.66e-03 | 1.67e-03 | True |
| 9 | 128x128 s5 c1x2 N12288K4096 pp1 | 12288 | 4096 | 4096 | extremes_m128 | True (None) | True | True | 0 | 1.69e-03 | 2.54e-03 | True |
| 12 | 256x128 s3 c1x2 N12288K4096 pp1 | 12288 | 4096 | 4096 | extremes_m128 | True (10) | True | True | 0 | 1.69e-03 | 2.54e-03 | True |
| 14 | 128x128 s5 c1x1 pp1 | 12288 | 4096 | 4096 | extremes_m128 | True (3) | True | True | 0 | 1.69e-03 | 2.54e-03 | True |
| 15 | 256x128 s3 c1x2 pp1 | 12288 | 4096 | 4096 | extremes_m128 | True (10) | True | True | 0 | 1.69e-03 | 2.54e-03 | True |
| 21 | 128x128 s5 c1x2 N12288K4096 pp5 | 12288 | 4096 | 4096 | extremes_m128 | True (None) | True | True | 0 | 1.69e-03 | 2.54e-03 | True |
| 22 | 256x128 s3 c1x2 N12288K4096 pp5 | 12288 | 4096 | 4096 | extremes_m128 | True (10) | True | True | 0 | 1.69e-03 | 2.54e-03 | True |
| 23 | 128x128 s5 c1x1 pp5 | 12288 | 4096 | 4096 | extremes_m128 | True (3) | True | True | 0 | 1.69e-03 | 2.54e-03 | True |
| 26 | 256x128 s3 c1x2 pp6 | 12288 | 4096 | 4096 | extremes_m128 | True (10) | True | True | 0 | 1.69e-03 | 2.54e-03 | True |
| 29 | 128x128 s5 c1x2 N12288K4096 pp7 | 12288 | 4096 | 4096 | extremes_m128 | True (None) | True | True | 0 | 1.69e-03 | 2.54e-03 | True |
| 31 | 128x128 s5 c1x1 pp7 | 12288 | 4096 | 4096 | extremes_m128 | True (3) | True | True | 0 | 1.69e-03 | 2.54e-03 | True |
| 0 | 128x128 s5 c1x1 N4096K4096 pp1 | 4096 | 4096 | 4096 | random | True (3) | True | True | 0 | 1.66e-03 | 3.23e-03 | True |
| 1 | 128x128 s5 c1x2 N4096K4096 pp1 | 4096 | 4096 | 4096 | random | True (17) | True | True | 0 | 1.66e-03 | 3.23e-03 | True |
| 2 | 128x128 s5 c1x1 N4096K4096 pp2 | 4096 | 4096 | 4096 | random | True (3) | True | True | 0 | 1.66e-03 | 3.23e-03 | True |
| 3 | 128x128 s5 c1x2 N4096K4096 pp2 | 4096 | 4096 | 4096 | random | True (17) | True | True | 0 | 1.66e-03 | 3.23e-03 | True |
| 4 | 128x128 s4 c1x2 N4096K4096 pp1 | 4096 | 4096 | 4096 | random | True (17) | True | True | 0 | 1.66e-03 | 3.23e-03 | True |
| 5 | 256x128 s3 c1x2 N4096K4096 pp1 | 4096 | 4096 | 4096 | random | True (8) | True | True | 0 | 1.66e-03 | 3.23e-03 | True |
| 6 | 256x128 s3 c1x1 N4096K4096 pp1 | 4096 | 4096 | 4096 | random | True (12) | True | True | 0 | 1.66e-03 | 3.23e-03 | True |
| 7 | 256x128 s3 c1x2 N4096K4096 pp2 | 4096 | 4096 | 4096 | random | True (8) | True | True | 0 | 1.66e-03 | 3.23e-03 | True |
| 14 | 128x128 s5 c1x1 pp1 | 4096 | 4096 | 4096 | random | True (3) | True | True | 0 | 1.66e-03 | 3.23e-03 | True |
| 15 | 256x128 s3 c1x2 pp1 | 4096 | 4096 | 4096 | random | True (8) | True | True | 0 | 1.66e-03 | 3.23e-03 | True |
| 16 | 128x128 s5 c1x1 N4096K4096 pp5 | 4096 | 4096 | 4096 | random | True (3) | True | True | 0 | 1.66e-03 | 3.23e-03 | True |
| 17 | 128x128 s5 c1x2 N4096K4096 pp5 | 4096 | 4096 | 4096 | random | True (17) | True | True | 0 | 1.66e-03 | 3.23e-03 | True |
| 18 | 256x128 s3 c1x2 N4096K4096 pp5 | 4096 | 4096 | 4096 | random | True (8) | True | True | 0 | 1.66e-03 | 3.23e-03 | True |
| 19 | 128x128 s5 c1x1 N4096K4096 pp6 | 4096 | 4096 | 4096 | random | True (3) | True | True | 0 | 1.66e-03 | 3.23e-03 | True |
| 20 | 256x128 s3 c1x2 N4096K4096 pp6 | 4096 | 4096 | 4096 | random | True (8) | True | True | 0 | 1.66e-03 | 3.23e-03 | True |
| 23 | 128x128 s5 c1x1 pp5 | 4096 | 4096 | 4096 | random | True (3) | True | True | 0 | 1.66e-03 | 3.23e-03 | True |
| 26 | 256x128 s3 c1x2 pp6 | 4096 | 4096 | 4096 | random | True (8) | True | True | 0 | 1.66e-03 | 3.23e-03 | True |
| 27 | 128x128 s5 c1x1 N4096K4096 pp7 | 4096 | 4096 | 4096 | random | True (3) | True | True | 0 | 1.66e-03 | 3.23e-03 | True |
| 28 | 128x128 s5 c1x2 N4096K4096 pp7 | 4096 | 4096 | 4096 | random | True (17) | True | True | 0 | 1.66e-03 | 3.23e-03 | True |
| 31 | 128x128 s5 c1x1 pp7 | 4096 | 4096 | 4096 | random | True (3) | True | True | 0 | 1.66e-03 | 3.23e-03 | True |
| 0 | 128x128 s5 c1x1 N4096K4096 pp1 | 4096 | 4096 | 4096 | adversarial_scales | True (3) | True | True | 0 | 1.66e-03 | 2.63e-03 | True |
| 1 | 128x128 s5 c1x2 N4096K4096 pp1 | 4096 | 4096 | 4096 | adversarial_scales | True (17) | True | True | 0 | 1.66e-03 | 2.63e-03 | True |
| 2 | 128x128 s5 c1x1 N4096K4096 pp2 | 4096 | 4096 | 4096 | adversarial_scales | True (3) | True | True | 0 | 1.66e-03 | 2.63e-03 | True |
| 3 | 128x128 s5 c1x2 N4096K4096 pp2 | 4096 | 4096 | 4096 | adversarial_scales | True (17) | True | True | 0 | 1.66e-03 | 2.63e-03 | True |
| 4 | 128x128 s4 c1x2 N4096K4096 pp1 | 4096 | 4096 | 4096 | adversarial_scales | True (17) | True | True | 0 | 1.66e-03 | 2.63e-03 | True |
| 5 | 256x128 s3 c1x2 N4096K4096 pp1 | 4096 | 4096 | 4096 | adversarial_scales | True (8) | True | True | 0 | 1.66e-03 | 2.63e-03 | True |
| 6 | 256x128 s3 c1x1 N4096K4096 pp1 | 4096 | 4096 | 4096 | adversarial_scales | True (12) | True | True | 0 | 1.66e-03 | 2.63e-03 | True |
| 7 | 256x128 s3 c1x2 N4096K4096 pp2 | 4096 | 4096 | 4096 | adversarial_scales | True (8) | True | True | 0 | 1.66e-03 | 2.63e-03 | True |
| 14 | 128x128 s5 c1x1 pp1 | 4096 | 4096 | 4096 | adversarial_scales | True (3) | True | True | 0 | 1.66e-03 | 2.63e-03 | True |
| 15 | 256x128 s3 c1x2 pp1 | 4096 | 4096 | 4096 | adversarial_scales | True (8) | True | True | 0 | 1.66e-03 | 2.63e-03 | True |
| 16 | 128x128 s5 c1x1 N4096K4096 pp5 | 4096 | 4096 | 4096 | adversarial_scales | True (3) | True | True | 0 | 1.66e-03 | 2.63e-03 | True |
| 17 | 128x128 s5 c1x2 N4096K4096 pp5 | 4096 | 4096 | 4096 | adversarial_scales | True (17) | True | True | 0 | 1.66e-03 | 2.63e-03 | True |
| 18 | 256x128 s3 c1x2 N4096K4096 pp5 | 4096 | 4096 | 4096 | adversarial_scales | True (8) | True | True | 0 | 1.66e-03 | 2.63e-03 | True |
| 19 | 128x128 s5 c1x1 N4096K4096 pp6 | 4096 | 4096 | 4096 | adversarial_scales | True (3) | True | True | 0 | 1.66e-03 | 2.63e-03 | True |
| 20 | 256x128 s3 c1x2 N4096K4096 pp6 | 4096 | 4096 | 4096 | adversarial_scales | True (8) | True | True | 0 | 1.66e-03 | 2.63e-03 | True |
| 23 | 128x128 s5 c1x1 pp5 | 4096 | 4096 | 4096 | adversarial_scales | True (3) | True | True | 0 | 1.66e-03 | 2.63e-03 | True |
| 26 | 256x128 s3 c1x2 pp6 | 4096 | 4096 | 4096 | adversarial_scales | True (8) | True | True | 0 | 1.66e-03 | 2.63e-03 | True |
| 27 | 128x128 s5 c1x1 N4096K4096 pp7 | 4096 | 4096 | 4096 | adversarial_scales | True (3) | True | True | 0 | 1.66e-03 | 2.63e-03 | True |
| 28 | 128x128 s5 c1x2 N4096K4096 pp7 | 4096 | 4096 | 4096 | adversarial_scales | True (17) | True | True | 0 | 1.66e-03 | 2.63e-03 | True |
| 31 | 128x128 s5 c1x1 pp7 | 4096 | 4096 | 4096 | adversarial_scales | True (3) | True | True | 0 | 1.66e-03 | 2.63e-03 | True |
| 0 | 128x128 s5 c1x1 N4096K4096 pp1 | 4096 | 4096 | 4096 | extremes_pm127 | True (3) | True | True | 0 | 1.71e-03 | 2.52e-03 | True |
| 1 | 128x128 s5 c1x2 N4096K4096 pp1 | 4096 | 4096 | 4096 | extremes_pm127 | True (17) | True | True | 0 | 1.71e-03 | 2.52e-03 | True |
| 2 | 128x128 s5 c1x1 N4096K4096 pp2 | 4096 | 4096 | 4096 | extremes_pm127 | True (3) | True | True | 0 | 1.71e-03 | 2.52e-03 | True |
| 3 | 128x128 s5 c1x2 N4096K4096 pp2 | 4096 | 4096 | 4096 | extremes_pm127 | True (17) | True | True | 0 | 1.71e-03 | 2.52e-03 | True |
| 4 | 128x128 s4 c1x2 N4096K4096 pp1 | 4096 | 4096 | 4096 | extremes_pm127 | True (17) | True | True | 0 | 1.71e-03 | 2.52e-03 | True |
| 5 | 256x128 s3 c1x2 N4096K4096 pp1 | 4096 | 4096 | 4096 | extremes_pm127 | True (8) | True | True | 0 | 1.71e-03 | 2.52e-03 | True |
| 6 | 256x128 s3 c1x1 N4096K4096 pp1 | 4096 | 4096 | 4096 | extremes_pm127 | True (12) | True | True | 0 | 1.71e-03 | 2.52e-03 | True |
| 7 | 256x128 s3 c1x2 N4096K4096 pp2 | 4096 | 4096 | 4096 | extremes_pm127 | True (8) | True | True | 0 | 1.71e-03 | 2.52e-03 | True |
| 14 | 128x128 s5 c1x1 pp1 | 4096 | 4096 | 4096 | extremes_pm127 | True (3) | True | True | 0 | 1.71e-03 | 2.52e-03 | True |
| 15 | 256x128 s3 c1x2 pp1 | 4096 | 4096 | 4096 | extremes_pm127 | True (8) | True | True | 0 | 1.71e-03 | 2.52e-03 | True |
| 16 | 128x128 s5 c1x1 N4096K4096 pp5 | 4096 | 4096 | 4096 | extremes_pm127 | True (3) | True | True | 0 | 1.71e-03 | 2.52e-03 | True |
| 17 | 128x128 s5 c1x2 N4096K4096 pp5 | 4096 | 4096 | 4096 | extremes_pm127 | True (17) | True | True | 0 | 1.71e-03 | 2.52e-03 | True |
| 18 | 256x128 s3 c1x2 N4096K4096 pp5 | 4096 | 4096 | 4096 | extremes_pm127 | True (8) | True | True | 0 | 1.71e-03 | 2.52e-03 | True |
| 19 | 128x128 s5 c1x1 N4096K4096 pp6 | 4096 | 4096 | 4096 | extremes_pm127 | True (3) | True | True | 0 | 1.71e-03 | 2.52e-03 | True |
| 20 | 256x128 s3 c1x2 N4096K4096 pp6 | 4096 | 4096 | 4096 | extremes_pm127 | True (8) | True | True | 0 | 1.71e-03 | 2.52e-03 | True |
| 23 | 128x128 s5 c1x1 pp5 | 4096 | 4096 | 4096 | extremes_pm127 | True (3) | True | True | 0 | 1.71e-03 | 2.52e-03 | True |
| 26 | 256x128 s3 c1x2 pp6 | 4096 | 4096 | 4096 | extremes_pm127 | True (8) | True | True | 0 | 1.71e-03 | 2.52e-03 | True |
| 27 | 128x128 s5 c1x1 N4096K4096 pp7 | 4096 | 4096 | 4096 | extremes_pm127 | True (3) | True | True | 0 | 1.71e-03 | 2.52e-03 | True |
| 28 | 128x128 s5 c1x2 N4096K4096 pp7 | 4096 | 4096 | 4096 | extremes_pm127 | True (17) | True | True | 0 | 1.71e-03 | 2.52e-03 | True |
| 31 | 128x128 s5 c1x1 pp7 | 4096 | 4096 | 4096 | extremes_pm127 | True (3) | True | True | 0 | 1.71e-03 | 2.52e-03 | True |
| 0 | 128x128 s5 c1x1 N4096K4096 pp1 | 4096 | 4096 | 4096 | extremes_m128 | True (3) | True | True | 0 | 1.65e-03 | 3.53e-03 | True |
| 1 | 128x128 s5 c1x2 N4096K4096 pp1 | 4096 | 4096 | 4096 | extremes_m128 | True (17) | True | True | 0 | 1.65e-03 | 3.53e-03 | True |
| 2 | 128x128 s5 c1x1 N4096K4096 pp2 | 4096 | 4096 | 4096 | extremes_m128 | True (3) | True | True | 0 | 1.65e-03 | 3.53e-03 | True |
| 3 | 128x128 s5 c1x2 N4096K4096 pp2 | 4096 | 4096 | 4096 | extremes_m128 | True (17) | True | True | 0 | 1.65e-03 | 3.53e-03 | True |
| 4 | 128x128 s4 c1x2 N4096K4096 pp1 | 4096 | 4096 | 4096 | extremes_m128 | True (17) | True | True | 0 | 1.65e-03 | 3.53e-03 | True |
| 5 | 256x128 s3 c1x2 N4096K4096 pp1 | 4096 | 4096 | 4096 | extremes_m128 | True (8) | True | True | 0 | 1.65e-03 | 3.53e-03 | True |
| 6 | 256x128 s3 c1x1 N4096K4096 pp1 | 4096 | 4096 | 4096 | extremes_m128 | True (12) | True | True | 0 | 1.65e-03 | 3.53e-03 | True |
| 7 | 256x128 s3 c1x2 N4096K4096 pp2 | 4096 | 4096 | 4096 | extremes_m128 | True (8) | True | True | 0 | 1.65e-03 | 3.53e-03 | True |
| 14 | 128x128 s5 c1x1 pp1 | 4096 | 4096 | 4096 | extremes_m128 | True (3) | True | True | 0 | 1.65e-03 | 3.53e-03 | True |
| 15 | 256x128 s3 c1x2 pp1 | 4096 | 4096 | 4096 | extremes_m128 | True (8) | True | True | 0 | 1.65e-03 | 3.53e-03 | True |
| 16 | 128x128 s5 c1x1 N4096K4096 pp5 | 4096 | 4096 | 4096 | extremes_m128 | True (3) | True | True | 0 | 1.65e-03 | 3.53e-03 | True |
| 17 | 128x128 s5 c1x2 N4096K4096 pp5 | 4096 | 4096 | 4096 | extremes_m128 | True (17) | True | True | 0 | 1.65e-03 | 3.53e-03 | True |
| 18 | 256x128 s3 c1x2 N4096K4096 pp5 | 4096 | 4096 | 4096 | extremes_m128 | True (8) | True | True | 0 | 1.65e-03 | 3.53e-03 | True |
| 19 | 128x128 s5 c1x1 N4096K4096 pp6 | 4096 | 4096 | 4096 | extremes_m128 | True (3) | True | True | 0 | 1.65e-03 | 3.53e-03 | True |
| 20 | 256x128 s3 c1x2 N4096K4096 pp6 | 4096 | 4096 | 4096 | extremes_m128 | True (8) | True | True | 0 | 1.65e-03 | 3.53e-03 | True |
| 23 | 128x128 s5 c1x1 pp5 | 4096 | 4096 | 4096 | extremes_m128 | True (3) | True | True | 0 | 1.65e-03 | 3.53e-03 | True |
| 26 | 256x128 s3 c1x2 pp6 | 4096 | 4096 | 4096 | extremes_m128 | True (8) | True | True | 0 | 1.65e-03 | 3.53e-03 | True |
| 27 | 128x128 s5 c1x1 N4096K4096 pp7 | 4096 | 4096 | 4096 | extremes_m128 | True (3) | True | True | 0 | 1.65e-03 | 3.53e-03 | True |
| 28 | 128x128 s5 c1x2 N4096K4096 pp7 | 4096 | 4096 | 4096 | extremes_m128 | True (17) | True | True | 0 | 1.65e-03 | 3.53e-03 | True |
| 31 | 128x128 s5 c1x1 pp7 | 4096 | 4096 | 4096 | extremes_m128 | True (3) | True | True | 0 | 1.65e-03 | 3.53e-03 | True |
| 14 | 128x128 s5 c1x1 pp1 | 4096 | 128 | 901 | random | True (3) | True | True | 0 | 1.66e-03 | 3.00e-03 | True |
| 15 | 256x128 s3 c1x2 pp1 | 4096 | 128 | 901 | random | True (0) | True | True | 0 | 1.66e-03 | 3.00e-03 | True |
| 23 | 128x128 s5 c1x1 pp5 | 4096 | 128 | 901 | random | True (3) | True | True | 0 | 1.66e-03 | 3.00e-03 | True |
| 26 | 256x128 s3 c1x2 pp6 | 4096 | 128 | 901 | random | True (0) | True | True | 0 | 1.66e-03 | 3.00e-03 | True |
| 31 | 128x128 s5 c1x1 pp7 | 4096 | 128 | 901 | random | True (3) | True | True | 0 | 1.66e-03 | 3.00e-03 | True |
| 14 | 128x128 s5 c1x1 pp1 | 4096 | 128 | 901 | adversarial_scales | True (3) | True | True | 0 | 1.66e-03 | 2.82e-03 | True |
| 15 | 256x128 s3 c1x2 pp1 | 4096 | 128 | 901 | adversarial_scales | True (0) | True | True | 0 | 1.66e-03 | 2.82e-03 | True |
| 23 | 128x128 s5 c1x1 pp5 | 4096 | 128 | 901 | adversarial_scales | True (3) | True | True | 0 | 1.66e-03 | 2.82e-03 | True |
| 26 | 256x128 s3 c1x2 pp6 | 4096 | 128 | 901 | adversarial_scales | True (0) | True | True | 0 | 1.66e-03 | 2.82e-03 | True |
| 31 | 128x128 s5 c1x1 pp7 | 4096 | 128 | 901 | adversarial_scales | True (3) | True | True | 0 | 1.66e-03 | 2.82e-03 | True |
| 14 | 128x128 s5 c1x1 pp1 | 4096 | 128 | 901 | extremes_pm127 | True (3) | True | True | 0 | 8.71e-04 | 9.73e-04 | True |
| 15 | 256x128 s3 c1x2 pp1 | 4096 | 128 | 901 | extremes_pm127 | True (0) | True | True | 0 | 8.71e-04 | 9.73e-04 | True |
| 23 | 128x128 s5 c1x1 pp5 | 4096 | 128 | 901 | extremes_pm127 | True (3) | True | True | 0 | 8.71e-04 | 9.73e-04 | True |
| 26 | 256x128 s3 c1x2 pp6 | 4096 | 128 | 901 | extremes_pm127 | True (0) | True | True | 0 | 8.71e-04 | 9.73e-04 | True |
| 31 | 128x128 s5 c1x1 pp7 | 4096 | 128 | 901 | extremes_pm127 | True (3) | True | True | 0 | 8.71e-04 | 9.73e-04 | True |
| 14 | 128x128 s5 c1x1 pp1 | 4096 | 128 | 901 | extremes_m128 | True (3) | True | True | 0 | 1.66e-03 | 3.55e-03 | True |
| 15 | 256x128 s3 c1x2 pp1 | 4096 | 128 | 901 | extremes_m128 | True (0) | True | True | 0 | 1.66e-03 | 3.55e-03 | True |
| 23 | 128x128 s5 c1x1 pp5 | 4096 | 128 | 901 | extremes_m128 | True (3) | True | True | 0 | 1.66e-03 | 3.55e-03 | True |
| 26 | 256x128 s3 c1x2 pp6 | 4096 | 128 | 901 | extremes_m128 | True (0) | True | True | 0 | 1.66e-03 | 3.55e-03 | True |
| 31 | 128x128 s5 c1x1 pp7 | 4096 | 128 | 901 | extremes_m128 | True (3) | True | True | 0 | 1.66e-03 | 3.55e-03 | True |
| 14 | 128x128 s5 c1x1 pp1 | 4096 | 128 | 4096 | random | True (3) | True | True | 0 | 1.66e-03 | 2.44e-03 | True |
| 15 | 256x128 s3 c1x2 pp1 | 4096 | 128 | 4096 | random | True (0) | True | True | 0 | 1.66e-03 | 2.44e-03 | True |
| 23 | 128x128 s5 c1x1 pp5 | 4096 | 128 | 4096 | random | True (3) | True | True | 0 | 1.66e-03 | 2.44e-03 | True |
| 26 | 256x128 s3 c1x2 pp6 | 4096 | 128 | 4096 | random | True (0) | True | True | 0 | 1.66e-03 | 2.44e-03 | True |
| 31 | 128x128 s5 c1x1 pp7 | 4096 | 128 | 4096 | random | True (3) | True | True | 0 | 1.66e-03 | 2.44e-03 | True |
| 14 | 128x128 s5 c1x1 pp1 | 4096 | 128 | 4096 | adversarial_scales | True (3) | True | True | 0 | 1.66e-03 | 3.01e-03 | True |
| 15 | 256x128 s3 c1x2 pp1 | 4096 | 128 | 4096 | adversarial_scales | True (0) | True | True | 0 | 1.66e-03 | 3.01e-03 | True |
| 23 | 128x128 s5 c1x1 pp5 | 4096 | 128 | 4096 | adversarial_scales | True (3) | True | True | 0 | 1.66e-03 | 3.01e-03 | True |
| 26 | 256x128 s3 c1x2 pp6 | 4096 | 128 | 4096 | adversarial_scales | True (0) | True | True | 0 | 1.66e-03 | 3.01e-03 | True |
| 31 | 128x128 s5 c1x1 pp7 | 4096 | 128 | 4096 | adversarial_scales | True (3) | True | True | 0 | 1.66e-03 | 3.01e-03 | True |
| 14 | 128x128 s5 c1x1 pp1 | 4096 | 128 | 4096 | extremes_pm127 | True (3) | True | True | 0 | 1.28e-03 | 9.73e-04 | True |
| 15 | 256x128 s3 c1x2 pp1 | 4096 | 128 | 4096 | extremes_pm127 | True (0) | True | True | 0 | 1.28e-03 | 9.73e-04 | True |
| 23 | 128x128 s5 c1x1 pp5 | 4096 | 128 | 4096 | extremes_pm127 | True (3) | True | True | 0 | 1.28e-03 | 9.73e-04 | True |
| 26 | 256x128 s3 c1x2 pp6 | 4096 | 128 | 4096 | extremes_pm127 | True (0) | True | True | 0 | 1.28e-03 | 9.73e-04 | True |
| 31 | 128x128 s5 c1x1 pp7 | 4096 | 128 | 4096 | extremes_pm127 | True (3) | True | True | 0 | 1.28e-03 | 9.73e-04 | True |
| 14 | 128x128 s5 c1x1 pp1 | 4096 | 128 | 4096 | extremes_m128 | True (3) | True | True | 0 | 1.62e-03 | 3.61e-03 | True |
| 15 | 256x128 s3 c1x2 pp1 | 4096 | 128 | 4096 | extremes_m128 | True (0) | True | True | 0 | 1.62e-03 | 3.61e-03 | True |
| 23 | 128x128 s5 c1x1 pp5 | 4096 | 128 | 4096 | extremes_m128 | True (3) | True | True | 0 | 1.62e-03 | 3.61e-03 | True |
| 26 | 256x128 s3 c1x2 pp6 | 4096 | 128 | 4096 | extremes_m128 | True (0) | True | True | 0 | 1.62e-03 | 3.61e-03 | True |
| 31 | 128x128 s5 c1x1 pp7 | 4096 | 128 | 4096 | extremes_m128 | True (3) | True | True | 0 | 1.62e-03 | 3.61e-03 | True |
| 14 | 128x128 s5 c1x1 pp1 | 1024 | 128 | 1802 | random | True (3) | True | True | 0 | 1.66e-03 | 2.67e-03 | True |
| 15 | 256x128 s3 c1x2 pp1 | 1024 | 128 | 1802 | random | True (0) | True | True | 0 | 1.66e-03 | 2.67e-03 | True |
| 23 | 128x128 s5 c1x1 pp5 | 1024 | 128 | 1802 | random | True (3) | True | True | 0 | 1.66e-03 | 2.67e-03 | True |
| 26 | 256x128 s3 c1x2 pp6 | 1024 | 128 | 1802 | random | True (0) | True | True | 0 | 1.66e-03 | 2.67e-03 | True |
| 31 | 128x128 s5 c1x1 pp7 | 1024 | 128 | 1802 | random | True (3) | True | True | 0 | 1.66e-03 | 2.67e-03 | True |
| 14 | 128x128 s5 c1x1 pp1 | 1024 | 128 | 1802 | adversarial_scales | True (3) | True | True | 0 | 1.66e-03 | 3.11e-03 | True |
| 15 | 256x128 s3 c1x2 pp1 | 1024 | 128 | 1802 | adversarial_scales | True (0) | True | True | 0 | 1.66e-03 | 3.11e-03 | True |
| 23 | 128x128 s5 c1x1 pp5 | 1024 | 128 | 1802 | adversarial_scales | True (3) | True | True | 0 | 1.66e-03 | 3.11e-03 | True |
| 26 | 256x128 s3 c1x2 pp6 | 1024 | 128 | 1802 | adversarial_scales | True (0) | True | True | 0 | 1.66e-03 | 3.11e-03 | True |
| 31 | 128x128 s5 c1x1 pp7 | 1024 | 128 | 1802 | adversarial_scales | True (3) | True | True | 0 | 1.66e-03 | 3.11e-03 | True |
| 14 | 128x128 s5 c1x1 pp1 | 1024 | 128 | 1802 | extremes_pm127 | True (3) | True | True | 0 | 8.11e-04 | 9.73e-04 | True |
| 15 | 256x128 s3 c1x2 pp1 | 1024 | 128 | 1802 | extremes_pm127 | True (0) | True | True | 0 | 8.11e-04 | 9.73e-04 | True |
| 23 | 128x128 s5 c1x1 pp5 | 1024 | 128 | 1802 | extremes_pm127 | True (3) | True | True | 0 | 8.11e-04 | 9.73e-04 | True |
| 26 | 256x128 s3 c1x2 pp6 | 1024 | 128 | 1802 | extremes_pm127 | True (0) | True | True | 0 | 8.11e-04 | 9.73e-04 | True |
| 31 | 128x128 s5 c1x1 pp7 | 1024 | 128 | 1802 | extremes_pm127 | True (3) | True | True | 0 | 8.11e-04 | 9.73e-04 | True |
| 14 | 128x128 s5 c1x1 pp1 | 1024 | 128 | 1802 | extremes_m128 | True (3) | True | True | 0 | 1.64e-03 | 3.64e-03 | True |
| 15 | 256x128 s3 c1x2 pp1 | 1024 | 128 | 1802 | extremes_m128 | True (0) | True | True | 0 | 1.64e-03 | 3.64e-03 | True |
| 23 | 128x128 s5 c1x1 pp5 | 1024 | 128 | 1802 | extremes_m128 | True (3) | True | True | 0 | 1.64e-03 | 3.64e-03 | True |
| 26 | 256x128 s3 c1x2 pp6 | 1024 | 128 | 1802 | extremes_m128 | True (0) | True | True | 0 | 1.64e-03 | 3.64e-03 | True |
| 31 | 128x128 s5 c1x1 pp7 | 1024 | 128 | 1802 | extremes_m128 | True (3) | True | True | 0 | 1.64e-03 | 3.64e-03 | True |
| 14 | 128x128 s5 c1x1 pp1 | 4096 | 256 | 901 | random | True (3) | True | True | 0 | 1.66e-03 | 2.07e-03 | True |
| 15 | 256x128 s3 c1x2 pp1 | 4096 | 256 | 901 | random | True (0) | True | True | 0 | 1.66e-03 | 2.07e-03 | True |
| 23 | 128x128 s5 c1x1 pp5 | 4096 | 256 | 901 | random | True (3) | True | True | 0 | 1.66e-03 | 2.07e-03 | True |
| 26 | 256x128 s3 c1x2 pp6 | 4096 | 256 | 901 | random | True (0) | True | True | 0 | 1.66e-03 | 2.07e-03 | True |
| 31 | 128x128 s5 c1x1 pp7 | 4096 | 256 | 901 | random | True (3) | True | True | 0 | 1.66e-03 | 2.07e-03 | True |
| 14 | 128x128 s5 c1x1 pp1 | 4096 | 256 | 901 | adversarial_scales | True (3) | True | True | 0 | 1.66e-03 | 2.29e-03 | True |
| 15 | 256x128 s3 c1x2 pp1 | 4096 | 256 | 901 | adversarial_scales | True (0) | True | True | 0 | 1.66e-03 | 2.29e-03 | True |
| 23 | 128x128 s5 c1x1 pp5 | 4096 | 256 | 901 | adversarial_scales | True (3) | True | True | 0 | 1.66e-03 | 2.29e-03 | True |
| 26 | 256x128 s3 c1x2 pp6 | 4096 | 256 | 901 | adversarial_scales | True (0) | True | True | 0 | 1.66e-03 | 2.29e-03 | True |
| 31 | 128x128 s5 c1x1 pp7 | 4096 | 256 | 901 | adversarial_scales | True (3) | True | True | 0 | 1.66e-03 | 2.29e-03 | True |
| 14 | 128x128 s5 c1x1 pp1 | 4096 | 256 | 901 | extremes_pm127 | True (3) | True | True | 0 | 1.17e-03 | 9.91e-04 | True |
| 15 | 256x128 s3 c1x2 pp1 | 4096 | 256 | 901 | extremes_pm127 | True (0) | True | True | 0 | 1.17e-03 | 9.91e-04 | True |
| 23 | 128x128 s5 c1x1 pp5 | 4096 | 256 | 901 | extremes_pm127 | True (3) | True | True | 0 | 1.17e-03 | 9.91e-04 | True |
| 26 | 256x128 s3 c1x2 pp6 | 4096 | 256 | 901 | extremes_pm127 | True (0) | True | True | 0 | 1.17e-03 | 9.91e-04 | True |
| 31 | 128x128 s5 c1x1 pp7 | 4096 | 256 | 901 | extremes_pm127 | True (3) | True | True | 0 | 1.17e-03 | 9.91e-04 | True |
| 14 | 128x128 s5 c1x1 pp1 | 4096 | 256 | 901 | extremes_m128 | True (3) | True | True | 0 | 1.66e-03 | 3.44e-03 | True |
| 15 | 256x128 s3 c1x2 pp1 | 4096 | 256 | 901 | extremes_m128 | True (0) | True | True | 0 | 1.66e-03 | 3.44e-03 | True |
| 23 | 128x128 s5 c1x1 pp5 | 4096 | 256 | 901 | extremes_m128 | True (3) | True | True | 0 | 1.66e-03 | 3.44e-03 | True |
| 26 | 256x128 s3 c1x2 pp6 | 4096 | 256 | 901 | extremes_m128 | True (0) | True | True | 0 | 1.66e-03 | 3.44e-03 | True |
| 31 | 128x128 s5 c1x1 pp7 | 4096 | 256 | 901 | extremes_m128 | True (3) | True | True | 0 | 1.66e-03 | 3.44e-03 | True |
| 14 | 128x128 s5 c1x1 pp1 | 4096 | 256 | 4096 | random | True (3) | True | True | 0 | 1.66e-03 | 1.85e-03 | True |
| 15 | 256x128 s3 c1x2 pp1 | 4096 | 256 | 4096 | random | True (0) | True | True | 0 | 1.66e-03 | 1.85e-03 | True |
| 23 | 128x128 s5 c1x1 pp5 | 4096 | 256 | 4096 | random | True (3) | True | True | 0 | 1.66e-03 | 1.85e-03 | True |
| 26 | 256x128 s3 c1x2 pp6 | 4096 | 256 | 4096 | random | True (0) | True | True | 0 | 1.66e-03 | 1.85e-03 | True |
| 31 | 128x128 s5 c1x1 pp7 | 4096 | 256 | 4096 | random | True (3) | True | True | 0 | 1.66e-03 | 1.85e-03 | True |
| 14 | 128x128 s5 c1x1 pp1 | 4096 | 256 | 4096 | adversarial_scales | True (3) | True | True | 0 | 1.66e-03 | 2.28e-03 | True |
| 15 | 256x128 s3 c1x2 pp1 | 4096 | 256 | 4096 | adversarial_scales | True (0) | True | True | 0 | 1.66e-03 | 2.28e-03 | True |
| 23 | 128x128 s5 c1x1 pp5 | 4096 | 256 | 4096 | adversarial_scales | True (3) | True | True | 0 | 1.66e-03 | 2.28e-03 | True |
| 26 | 256x128 s3 c1x2 pp6 | 4096 | 256 | 4096 | adversarial_scales | True (0) | True | True | 0 | 1.66e-03 | 2.28e-03 | True |
| 31 | 128x128 s5 c1x1 pp7 | 4096 | 256 | 4096 | adversarial_scales | True (3) | True | True | 0 | 1.66e-03 | 2.28e-03 | True |
| 14 | 128x128 s5 c1x1 pp1 | 4096 | 256 | 4096 | extremes_pm127 | True (3) | True | True | 0 | 1.43e-03 | 6.02e-04 | True |
| 15 | 256x128 s3 c1x2 pp1 | 4096 | 256 | 4096 | extremes_pm127 | True (0) | True | True | 0 | 1.43e-03 | 6.02e-04 | True |
| 23 | 128x128 s5 c1x1 pp5 | 4096 | 256 | 4096 | extremes_pm127 | True (3) | True | True | 0 | 1.43e-03 | 6.02e-04 | True |
| 26 | 256x128 s3 c1x2 pp6 | 4096 | 256 | 4096 | extremes_pm127 | True (0) | True | True | 0 | 1.43e-03 | 6.02e-04 | True |
| 31 | 128x128 s5 c1x1 pp7 | 4096 | 256 | 4096 | extremes_pm127 | True (3) | True | True | 0 | 1.43e-03 | 6.02e-04 | True |
| 14 | 128x128 s5 c1x1 pp1 | 4096 | 256 | 4096 | extremes_m128 | True (3) | True | True | 0 | 1.67e-03 | 2.76e-03 | True |
| 15 | 256x128 s3 c1x2 pp1 | 4096 | 256 | 4096 | extremes_m128 | True (0) | True | True | 0 | 1.67e-03 | 2.76e-03 | True |
| 23 | 128x128 s5 c1x1 pp5 | 4096 | 256 | 4096 | extremes_m128 | True (3) | True | True | 0 | 1.67e-03 | 2.76e-03 | True |
| 26 | 256x128 s3 c1x2 pp6 | 4096 | 256 | 4096 | extremes_m128 | True (0) | True | True | 0 | 1.67e-03 | 2.76e-03 | True |
| 31 | 128x128 s5 c1x1 pp7 | 4096 | 256 | 4096 | extremes_m128 | True (3) | True | True | 0 | 1.67e-03 | 2.76e-03 | True |
| 14 | 128x128 s5 c1x1 pp1 | 1024 | 256 | 4096 | random | True (3) | True | True | 0 | 1.66e-03 | 1.96e-03 | True |
| 15 | 256x128 s3 c1x2 pp1 | 1024 | 256 | 4096 | random | True (0) | True | True | 0 | 1.66e-03 | 1.96e-03 | True |
| 23 | 128x128 s5 c1x1 pp5 | 1024 | 256 | 4096 | random | True (3) | True | True | 0 | 1.66e-03 | 1.96e-03 | True |
| 26 | 256x128 s3 c1x2 pp6 | 1024 | 256 | 4096 | random | True (0) | True | True | 0 | 1.66e-03 | 1.96e-03 | True |
| 31 | 128x128 s5 c1x1 pp7 | 1024 | 256 | 4096 | random | True (3) | True | True | 0 | 1.66e-03 | 1.96e-03 | True |
| 14 | 128x128 s5 c1x1 pp1 | 1024 | 256 | 4096 | adversarial_scales | True (3) | True | True | 0 | 1.66e-03 | 2.45e-03 | True |
| 15 | 256x128 s3 c1x2 pp1 | 1024 | 256 | 4096 | adversarial_scales | True (0) | True | True | 0 | 1.66e-03 | 2.45e-03 | True |
| 23 | 128x128 s5 c1x1 pp5 | 1024 | 256 | 4096 | adversarial_scales | True (3) | True | True | 0 | 1.66e-03 | 2.45e-03 | True |
| 26 | 256x128 s3 c1x2 pp6 | 1024 | 256 | 4096 | adversarial_scales | True (0) | True | True | 0 | 1.66e-03 | 2.45e-03 | True |
| 31 | 128x128 s5 c1x1 pp7 | 1024 | 256 | 4096 | adversarial_scales | True (3) | True | True | 0 | 1.66e-03 | 2.45e-03 | True |
| 14 | 128x128 s5 c1x1 pp1 | 1024 | 256 | 4096 | extremes_pm127 | True (3) | True | True | 0 | 1.40e-03 | 1.78e-03 | True |
| 15 | 256x128 s3 c1x2 pp1 | 1024 | 256 | 4096 | extremes_pm127 | True (0) | True | True | 0 | 1.40e-03 | 1.78e-03 | True |
| 23 | 128x128 s5 c1x1 pp5 | 1024 | 256 | 4096 | extremes_pm127 | True (3) | True | True | 0 | 1.40e-03 | 1.78e-03 | True |
| 26 | 256x128 s3 c1x2 pp6 | 1024 | 256 | 4096 | extremes_pm127 | True (0) | True | True | 0 | 1.40e-03 | 1.78e-03 | True |
| 31 | 128x128 s5 c1x1 pp7 | 1024 | 256 | 4096 | extremes_pm127 | True (3) | True | True | 0 | 1.40e-03 | 1.78e-03 | True |
| 14 | 128x128 s5 c1x1 pp1 | 1024 | 256 | 4096 | extremes_m128 | True (3) | True | True | 0 | 1.67e-03 | 1.96e-03 | True |
| 15 | 256x128 s3 c1x2 pp1 | 1024 | 256 | 4096 | extremes_m128 | True (0) | True | True | 0 | 1.67e-03 | 1.96e-03 | True |
| 23 | 128x128 s5 c1x1 pp5 | 1024 | 256 | 4096 | extremes_m128 | True (3) | True | True | 0 | 1.67e-03 | 1.96e-03 | True |
| 26 | 256x128 s3 c1x2 pp6 | 1024 | 256 | 4096 | extremes_m128 | True (0) | True | True | 0 | 1.67e-03 | 1.96e-03 | True |
| 31 | 128x128 s5 c1x1 pp7 | 1024 | 256 | 4096 | extremes_m128 | True (3) | True | True | 0 | 1.67e-03 | 1.96e-03 | True |
| 14 | 128x128 s5 c1x1 pp1 | 1024 | 384 | 901 | random | True (3) | True | True | 0 | 1.66e-03 | 1.99e-03 | True |
| 15 | 256x128 s3 c1x2 pp1 | 1024 | 384 | 901 | random | True (0) | True | True | 0 | 1.66e-03 | 1.99e-03 | True |
| 23 | 128x128 s5 c1x1 pp5 | 1024 | 384 | 901 | random | True (3) | True | True | 0 | 1.66e-03 | 1.99e-03 | True |
| 26 | 256x128 s3 c1x2 pp6 | 1024 | 384 | 901 | random | True (0) | True | True | 0 | 1.66e-03 | 1.99e-03 | True |
| 31 | 128x128 s5 c1x1 pp7 | 1024 | 384 | 901 | random | True (3) | True | True | 0 | 1.66e-03 | 1.99e-03 | True |
| 14 | 128x128 s5 c1x1 pp1 | 1024 | 384 | 901 | adversarial_scales | True (3) | True | True | 0 | 1.66e-03 | 2.21e-03 | True |
| 15 | 256x128 s3 c1x2 pp1 | 1024 | 384 | 901 | adversarial_scales | True (0) | True | True | 0 | 1.66e-03 | 2.21e-03 | True |
| 23 | 128x128 s5 c1x1 pp5 | 1024 | 384 | 901 | adversarial_scales | True (3) | True | True | 0 | 1.66e-03 | 2.21e-03 | True |
| 26 | 256x128 s3 c1x2 pp6 | 1024 | 384 | 901 | adversarial_scales | True (0) | True | True | 0 | 1.66e-03 | 2.21e-03 | True |
| 31 | 128x128 s5 c1x1 pp7 | 1024 | 384 | 901 | adversarial_scales | True (3) | True | True | 0 | 1.66e-03 | 2.21e-03 | True |
| 14 | 128x128 s5 c1x1 pp1 | 1024 | 384 | 901 | extremes_pm127 | True (3) | True | True | 0 | 5.19e-04 | 4.96e-04 | True |
| 15 | 256x128 s3 c1x2 pp1 | 1024 | 384 | 901 | extremes_pm127 | True (0) | True | True | 0 | 5.19e-04 | 4.96e-04 | True |
| 23 | 128x128 s5 c1x1 pp5 | 1024 | 384 | 901 | extremes_pm127 | True (3) | True | True | 0 | 5.19e-04 | 4.96e-04 | True |
| 26 | 256x128 s3 c1x2 pp6 | 1024 | 384 | 901 | extremes_pm127 | True (0) | True | True | 0 | 5.19e-04 | 4.96e-04 | True |
| 31 | 128x128 s5 c1x1 pp7 | 1024 | 384 | 901 | extremes_pm127 | True (3) | True | True | 0 | 5.19e-04 | 4.96e-04 | True |
| 14 | 128x128 s5 c1x1 pp1 | 1024 | 384 | 901 | extremes_m128 | True (3) | True | True | 0 | 1.67e-03 | 3.07e-03 | True |
| 15 | 256x128 s3 c1x2 pp1 | 1024 | 384 | 901 | extremes_m128 | True (0) | True | True | 0 | 1.67e-03 | 3.07e-03 | True |
| 23 | 128x128 s5 c1x1 pp5 | 1024 | 384 | 901 | extremes_m128 | True (3) | True | True | 0 | 1.67e-03 | 3.07e-03 | True |
| 26 | 256x128 s3 c1x2 pp6 | 1024 | 384 | 901 | extremes_m128 | True (0) | True | True | 0 | 1.67e-03 | 3.07e-03 | True |
| 31 | 128x128 s5 c1x1 pp7 | 1024 | 384 | 901 | extremes_m128 | True (3) | True | True | 0 | 1.67e-03 | 3.07e-03 | True |

## Verdict (verifier)

* Exactness: confirmed independently. All 32 pp instantiations are bit-identical (torch.equal) to the default kernel of the same tile, to the 256x128 default and to the independent CUTLASS `cutlass_int8_bw` kernel in 328 / 328 checks (4 production shapes incl. (12288,4096,4096) and 4096^3, K=128/256/384 edge cases, random / adversarial / +-127 / -128 scale-and-value patterns, NaN sfa padding), rel-L2 vs fp64 equals the default kernel's (1.5e-3 - 1.8e-3). 5020 stress launches over M = 901..4194 step 37 (2..33 M-tiles, M % 4 != 0) with all handshake modes in random order: 0 mismatches, no hang. Code review of the handshake: per batch WG0 `bar.arrive 10` / `bar.sync 11` and WG1 `bar.sync 10` / `bar.arrive 11` (mode 1) or the sync-before-issue form with the first-wait skip and the kernel-end consumption (modes 2/5/6/7) are generation-safe (a warpgroup cannot re-arrive on a barrier before the other one has consumed the previous generation), both warpgroups execute the same batch count (GemmType::Normal -> is_computation_valid always true, identical scheduler state), the empty-barrier arrival count (8 warps) and the smem scale reads before `warpgroup_arrive` are unchanged, and the dual-chain modes add an exact int32 sum. No C7514 in any build log, 168 registers like the default/DB kernels.
* Speed: the best C variant is 256x128 s3 c1x2 pp1 (cfg 5/11/12/13; pp2 and pp6 equal within noise). It is +1..+4 % over the default 256x128 kernel at 11 / 12 shapes (+53 % at (1024,4096,901) where the 256x128 tile is a poor fit anyway: pick_config's 64x128 default reaches 425 and B's 64x128 DB 474 vs 338 for the best pp) and the best exact INT8 g128 kernel at 6 / 12 shapes (4096x12288 at every M: 979 / 1120 / 1116 TFLOPS; 12288x4096 M=42240: 1094; 4096x4096 M=42240: 1113 (pp2)), but at 4096^3 it stays 3 % below implementer B's DB kernel (1091 vs 1128; DB's earlier run 1140) and B DB also wins at M=901 for N >= 4096 and at N=1024. So "exact but not faster" holds for the 4096^3 high-water mark; across the 12 shapes the best exact kernel now sits at 0.84-0.91x DeepGEMM FP8 g128 (1.24x at N=1024, M=901) and 0.75-0.87x CUTLASS INT8 per-tensor. The 128x128 pp variants are never better than the 128x128 DB kernel (they equal the lockstep default: with issue-order alternation the tensor cores arbitrate both in-flight wgmma chains fairly, so both warpgroups still finish and promote together; true mutual exclusion (pp5) loses 8-15 % because one warpgroup alone drives a single dependent 64xNx32 chain at ~57 % of the tensor-core peak).
* What remains: the per-k-block promotion (64 I2F + 64 FFMA per thread per 128-K block) is still serialized with the tensor cores; the no-convert experiment (1405-1525 TFLOPS) bounds the headroom at ~25 %. Per-batch alternation on a shared smem stage cannot desynchronize the two warpgroups; the remaining exact options are tile-level ping-pong (each math warpgroup on its own tile / own smem stages, out of phase by half a mainloop, CUTLASS KernelTmaWarpSpecializedPingpong style, at the cost of half the A/B reuse per stage), or moving the promotion off the math warpgroups (a third consumer warpgroup fed the int32 accumulators through shared memory: 64x128x4 B = 32 KB per hand-off, which conflicts with the 3-5 stage budget at 256x128 / 128x128).
