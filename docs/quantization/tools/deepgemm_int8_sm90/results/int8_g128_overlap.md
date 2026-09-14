# INT8 g128 (1,128,128) block-scaled GEMM -- promotion-overlap variants, serial timing (idle H100, verifier run)

- NVIDIA H100 80GB HBM3, torch 2.13.0a0+9186a08b2c.nv26.07, 2026-09-14T01:57:12; triton do_bench(warmup=25, rep=200, median, L2 flushed), 3 interleaved repeats, median of medians; SM MHz = median of nvidia-smi samples ~0.1 s into each do_bench (clocks unlocked, power-throttled at M=42240).
- TFLOPS = 2MNK/t with the true M. All INT8 g128 kernels run on the same int8/scale tensors; the cutlass tuner gets A/SFA zero-padded to 904 rows at M=901 outside the timed region.
- `B DB` = implementer B's double-buffered-accumulator mainloop (main extension cfg ids 19-29, `kernels/deepgemm_int8/include/deep_gemm/impls/sm90_int8_gemm_1d2d_db.cuh`); `A wi` = implementer A's wave-interleave mainloop (isolated extension `kernels/deepgemm_int8/wi`, cfg 6-12; its `ref` cfgs are the unchanged base kernel, SASS-identical to the default build). The wi extension cannot be loaded into the same process as the main one (identically mangled kernel symbols -> second launch fails), so it was timed in separate runs against its own reference instantiation.
- correctness: `probes/verify_int8_g128_overlap.py` (independent fp64 reference; random / adversarial-scale / +-127 / -128 cases, NaN sfa padding, K=128/256/384 edge cases): 232 / 232 variant checks bit-identical to the default kernel, 88 / 88 default-kernel == cutlass_int8_bw checks. Logs: logs/verif_overlap_*.log.

## Best exact INT8 g128 kernel per shape vs references (TFLOPS)

| N | K | M | old best (256x128 s3 c1x2) | best B DB (cfg) | best A wi (cfg) | cutlass tuner coop128x128 c2x1 DB | best exact INT8 g128 | DeepGEMM FP8 g128 | ratio vs FP8 g128 | cutlass INT8 per-tensor | ratio vs per-tensor | cuBLASLt FP8 per-tensor | SM MHz (old / best DB) |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| 4096 | 4096 | 901 | 798 | 853 (128x128 s5 c1x1 N4096K4096 DB) | 665 (256x128 s3 c1x1 N4096K4096 wi2; its ref run 798) | 788 | 853 B DB cfg 21 | 975 | 0.88 | 985 | 0.87 | 1093 | 1980 / 1980 |
| 4096 | 4096 | 4096 | 1061 | 1140 (128x128 s5 c1x1 N4096K4096 DB) | 885 (256x128 s3 c1x1 N4096K4096 wi2; its ref run 1063) | 1104 | 1140 B DB cfg 21 | 1303 | 0.88 | 1398 | 0.82 | 1370 | 1965 / 1965 |
| 4096 | 4096 | 42240 | 1101 | 1094 (128x128 s5 c1x1 N4096K4096 DB) | 936 (256x128 s3 c1x2 N4096K4096 wi2; its ref run 1080) | 1089 | 1101 deepgemm_int8 old best | 1225 | 0.90 | 1405 | 0.78 | 1327 | 1800 / 1545 |
| 1024 | 4096 | 901 | 426 | 477 (64x128 s8 c1x1 DB) | 174 (256x128 s3 c1x2 N1024K4096 wi2; its ref run 224) | 359 | 477 B DB cfg 27 | 380 | 1.25 | 272 | 1.75 | 526 | 1980 / 1980 |
| 1024 | 4096 | 4096 | 896 | 922 (128x128 s5 c1x2 N1024K4096 DB) | 768 (256x128 s3 c1x2 N1024K4096 wi2; its ref run 919) | 939 | 939 cutlass tuner coop128x128 c2x1 DB | 1055 | 0.89 | 1124 | 0.83 | 1146 | 1980 / 1980 |
| 1024 | 4096 | 42240 | 1082 | 1106 (128x128 s5 c1x2 N1024K4096 DB) | 946 (256x128 s3 c1x2 N1024K4096 wi2; its ref run 1092) | 1042 | 1106 B DB cfg 25 | 1243 | 0.89 | 1451 | 0.76 | 1354 | 1920 / 1785 |
| 12288 | 4096 | 901 | 921 | 958 (128x128 s5 c1x2 N12288K4096 DB) | 772 (256x128 s3 c1x2 N12288K4096 wi2; its ref run 921) | 850 | 958 B DB cfg 23 | 1136 | 0.84 | 1179 | 0.81 | 1307 | 1980 / 1980 |
| 12288 | 4096 | 4096 | 1079 | 1099 (128x128 s5 c1x2 N12288K4096 DB) | 935 (256x128 s3 c1x2 N12288K4096 wi2; its ref run 1081) | 1066 | 1099 B DB cfg 23 | 1222 | 0.90 | 1438 | 0.76 | 1352 | 1890 / 1725 |
| 12288 | 4096 | 42240 | 1080 | 1092 (128x128 s5 c1x2 N12288K4096 DB) | 946 (256x128 s3 c1x2 N12288K4096 wi2; its ref run 1055) | 977 | 1092 B DB cfg 23 | 1285 | 0.85 | 1283 | 0.85 | 1338 | 1695 / 1395 |
| 4096 | 12288 | 901 | 946 | 950 (128x128 s5 c1x2 N4096K12288 DB) | 773 (256x128 s3 c1x2 N4096K12288 wi2; its ref run 946) | 897 | 950 B DB cfg 24 | 1174 | 0.81 | 1272 | 0.75 | 1417 | 1980 / 1980 |
| 4096 | 12288 | 4096 | 1117 | 1103 (128x128 s5 c1x2 N4096K12288 DB) | 931 (256x128 s3 c1x2 N4096K12288 wi2; its ref run 1104) | 1055 | 1117 deepgemm_int8 old best | 1252 | 0.89 | 1520 | 0.74 | 1386 | 1875 / 1740 |
| 4096 | 12288 | 42240 | 1100 | 1102 (128x128 s5 c1x2 N4096K12288 DB) | 939 (256x128 s3 c1x2 N4096K12288 wi2; its ref run 1063) | 1047 | 1102 B DB cfg 24 | 1325 | 0.83 | 1481 | 0.74 | 1386 | 1710 / 1395 |

## All rows: microseconds (median of 3 medians), TFLOPS, per-repeat medians, SM MHz samples

| N | K | M | kernel | config | us | TFLOPS | repeats (us) | SM MHz samples |
|---|---|---|---|---|---:|---:|---|---|
| 4096 | 4096 | 901 | deepgemm_int8 old best | 256x128 s3 c1x2 N4096K4096 | 37.9 | 798 | 37.6, 37.9, 37.9 | [1980, 1980, 1980] |
| 4096 | 4096 | 901 | B DB cfg 19 | 128x128 s5 c1x2 N4096K4096 DB | 38.0 | 795 | 38.0, 38.0, 38.1 | [1980, 1980, 1980] |
| 4096 | 4096 | 901 | B DB cfg 20 | 128x128 s5 c2x1 N4096K4096 DB | 39.0 | 775 | 39.0, 39.0, 39.1 | [1980, 1980, 1980] |
| 4096 | 4096 | 901 | B DB cfg 21 | 128x128 s5 c1x1 N4096K4096 DB | 35.4 | 853 | 35.4, 35.4, 35.4 | [1980, 1980, 1980] |
| 4096 | 4096 | 901 | B DB cfg 27 | 64x128 s8 c1x1 DB | 41.6 | 726 | 41.7, 41.6, 41.4 | [1980, 1980, 1980] |
| 4096 | 4096 | 901 | cutlass tuner coop128x128 c2x1 DB | spec 128,128,2,1,0,0,0,0,0,1 (v0) | 38.4 | 788 | 38.6, 38.4, 38.4 | [1980, 1980, 1980] |
| 4096 | 4096 | 901 | DeepGEMM FP8 g128 | upstream 2.8.0 1D2D, unpadded M | 31.0 | 975 | 31.2, 31.0, 31.0 | [1980, 1980, 1980] |
| 4096 | 4096 | 901 | cutlass_int8 per-tensor | 128x256 c2x1 (int8_scaled_mm cfg 0) | 30.7 | 985 | 30.7, 30.7, 30.7 | [1980, 1980, 1980] |
| 4096 | 4096 | 901 | cuBLASLt FP8 per-tensor | torch._scaled_mm use_fast_accum=True | 27.6 | 1093 | 27.6, 27.7, 27.6 | [1980, 1980, 1980] |
| 4096 | 4096 | 901 | A wi ref cfg 0 | 256x128 s3 c1x2 N4096K4096 ref | 37.9 | 798 | 37.9, 38.0, 37.9 | [1980, 1980, 1980] |
| 4096 | 4096 | 901 | A wi cfg 6 | 256x128 s3 c1x2 N4096K4096 wi2 | 45.9 | 659 | 46.0, 45.8, 45.9 | [1980, 1980, 1980] |
| 4096 | 4096 | 901 | A wi cfg 10 | 256x128 s3 c1x1 N4096K4096 wi2 | 45.4 | 665 | 45.4, 45.5, 45.2 | [1980, 1980, 1980] |
| 4096 | 4096 | 901 | A wi cfg 11 | 256x128 s3 c1x2 N4096K4096 wi1 | 62.5 | 484 | 62.8, 62.5, 62.5 | [1980, 1980, 1980] |
| 4096 | 4096 | 4096 | deepgemm_int8 old best | 256x128 s3 c1x2 N4096K4096 | 129.5 | 1061 | 129.6, 129.3, 129.5 | [1980, 1965, 1965] |
| 4096 | 4096 | 4096 | B DB cfg 19 | 128x128 s5 c1x2 N4096K4096 DB | 126.5 | 1087 | 126.5, 126.6, 126.3 | [1920, 1965, 1935] |
| 4096 | 4096 | 4096 | B DB cfg 20 | 128x128 s5 c2x1 N4096K4096 DB | 127.2 | 1081 | 127.2, 127.1, 127.4 | [1935, 1905, 1965] |
| 4096 | 4096 | 4096 | B DB cfg 21 | 128x128 s5 c1x1 N4096K4096 DB | 120.5 | 1140 | 120.1, 120.5, 120.7 | [1965, 1965, 1965] |
| 4096 | 4096 | 4096 | B DB cfg 27 | 64x128 s8 c1x1 DB | 142.0 | 968 | 142.0, 141.9, 142.2 | [1965, 1965, 1965] |
| 4096 | 4096 | 4096 | cutlass tuner coop128x128 c2x1 DB | spec 128,128,2,1,0,0,0,0,0,1 (v0) | 124.4 | 1104 | 124.4, 124.4, 125.0 | [1965, 1980, 1965] |
| 4096 | 4096 | 4096 | DeepGEMM FP8 g128 | upstream 2.8.0 1D2D, unpadded M | 105.5 | 1303 | 105.8, 105.5, 105.0 | [1755, 1830, 1965] |
| 4096 | 4096 | 4096 | cutlass_int8 per-tensor | 128x256 c2x1 (int8_scaled_mm cfg 0) | 98.3 | 1398 | 98.3, 98.3, 98.2 | [1965, 1965, 1965] |
| 4096 | 4096 | 4096 | cuBLASLt FP8 per-tensor | torch._scaled_mm use_fast_accum=True | 100.4 | 1370 | 100.1, 100.4, 100.5 | [1965, 1935, 1905] |
| 4096 | 4096 | 4096 | A wi ref cfg 0 | 256x128 s3 c1x2 N4096K4096 ref | 129.3 | 1063 | 129.0, 129.3, 129.4 | [1980, 1980, 1980] |
| 4096 | 4096 | 4096 | A wi cfg 6 | 256x128 s3 c1x2 N4096K4096 wi2 | 155.8 | 882 | 155.7, 155.8, 155.8 | [1980, 1980, 1980] |
| 4096 | 4096 | 4096 | A wi cfg 10 | 256x128 s3 c1x1 N4096K4096 wi2 | 155.3 | 885 | 155.3, 155.3, 155.2 | [1980, 1980, 1980] |
| 4096 | 4096 | 4096 | A wi cfg 11 | 256x128 s3 c1x2 N4096K4096 wi1 | 223.6 | 615 | 223.5, 223.6, 223.6 | [1980, 1980, 1980] |
| 4096 | 4096 | 42240 | deepgemm_int8 old best | 256x128 s3 c1x2 N4096K4096 | 1287.2 | 1101 | 1295.9, 1287.2, 1279.5 | [1770, 1800, 1800] |
| 4096 | 4096 | 42240 | B DB cfg 19 | 128x128 s5 c1x2 N4096K4096 DB | 1319.3 | 1074 | 1301.5, 1319.3, 1331.5 | [1635, 1620, 1620] |
| 4096 | 4096 | 42240 | B DB cfg 20 | 128x128 s5 c2x1 N4096K4096 DB | 1358.5 | 1043 | 1358.5, 1372.6, 1329.4 | [1515, 1545, 1515] |
| 4096 | 4096 | 42240 | B DB cfg 21 | 128x128 s5 c1x1 N4096K4096 DB | 1296.1 | 1094 | 1306.0, 1296.1, 1262.9 | [1470, 1545, 1560] |
| 4096 | 4096 | 42240 | B DB cfg 27 | 64x128 s8 c1x1 DB | 1454.1 | 975 | 1446.0, 1454.1, 1468.8 | [1635, 1635, 1545] |
| 4096 | 4096 | 42240 | cutlass tuner coop128x128 c2x1 DB | spec 128,128,2,1,0,0,0,0,0,1 (v0) | 1301.2 | 1089 | 1291.6, 1301.2, 1312.0 | [1680, 1680, 1665] |
| 4096 | 4096 | 42240 | DeepGEMM FP8 g128 | upstream 2.8.0 1D2D, unpadded M | 1157.2 | 1225 | 1157.9, 1157.2, 1151.2 | [1695, 1515, 1350] |
| 4096 | 4096 | 42240 | cutlass_int8 per-tensor | 128x256 c2x1 (int8_scaled_mm cfg 0) | 1008.5 | 1405 | 1029.9, 1006.9, 1008.5 | [1650, 1635, 1725] |
| 4096 | 4096 | 42240 | cuBLASLt FP8 per-tensor | torch._scaled_mm use_fast_accum=True | 1068.3 | 1327 | 1065.6, 1068.3, 1074.0 | [1500, 1410, 1455] |
| 4096 | 4096 | 42240 | A wi ref cfg 0 | 256x128 s3 c1x2 N4096K4096 ref | 1312.7 | 1080 | 1285.1, 1312.7, 1368.0 | [1815, 1605, 1530] |
| 4096 | 4096 | 42240 | A wi cfg 6 | 256x128 s3 c1x2 N4096K4096 wi2 | 1514.9 | 936 | 1478.8, 1514.9, 1535.4 | [1920, 1815, 1845] |
| 4096 | 4096 | 42240 | A wi cfg 10 | 256x128 s3 c1x1 N4096K4096 wi2 | 1542.3 | 919 | 1542.3, 1497.3, 1543.8 | [1875, 1890, 1890] |
| 4096 | 4096 | 42240 | A wi cfg 11 | 256x128 s3 c1x2 N4096K4096 wi1 | 2242.3 | 632 | 2300.9, 2242.3, 2213.0 | [1605, 1710, 1920] |
| 1024 | 4096 | 901 | deepgemm_int8 old best | 64x128 s8 c1x1 | 17.8 | 426 | 17.8, 17.8, 17.8 | [1980, 1980, 1980] |
| 1024 | 4096 | 901 | B DB cfg 25 | 128x128 s5 c1x2 N1024K4096 DB | 21.3 | 354 | 21.3, 21.3, 21.3 | [1980, 1980, 1980] |
| 1024 | 4096 | 901 | B DB cfg 26 | 128x64 s8 c1x1 N1024K4096 DB | 19.8 | 382 | 19.7, 19.8, 19.8 | [1980, 1980, 1980] |
| 1024 | 4096 | 901 | B DB cfg 27 | 64x128 s8 c1x1 DB | 15.8 | 477 | 15.9, 15.8, 15.8 | [1980, 1980, 1980] |
| 1024 | 4096 | 901 | cutlass tuner coop128x128 c2x1 DB | spec 128,128,2,1,0,0,0,0,0,1 (v0) | 21.1 | 359 | 21.1, 21.2, 21.1 | [1980, 1980, 1980] |
| 1024 | 4096 | 901 | DeepGEMM FP8 g128 | upstream 2.8.0 1D2D, unpadded M | 19.9 | 380 | 19.9, 20.0, 19.9 | [1980, 1980, 1980] |
| 1024 | 4096 | 901 | cutlass_int8 per-tensor | 128x256 c2x1 (int8_scaled_mm cfg 0) | 27.7 | 272 | 27.7, 27.7, 27.7 | [1980, 1980, 1980] |
| 1024 | 4096 | 901 | cuBLASLt FP8 per-tensor | torch._scaled_mm use_fast_accum=True | 14.4 | 526 | 14.4, 14.4, 14.4 | [1980, 1980, 1980] |
| 1024 | 4096 | 901 | A wi ref cfg 3 | 256x128 s3 c1x2 N1024K4096 ref | 33.8 | 224 | 33.9, 33.6, 33.8 | [1980, 1980, 1980] |
| 1024 | 4096 | 901 | A wi cfg 9 | 256x128 s3 c1x2 N1024K4096 wi2 | 43.4 | 174 | 43.4, 43.4, 43.4 | [1980, 1980, 1980] |
| 1024 | 4096 | 4096 | deepgemm_int8 old best | 256x128 s3 c1x2 N1024K4096 | 38.4 | 896 | 37.9, 38.4, 38.4 | [1980, 1980, 1980] |
| 1024 | 4096 | 4096 | B DB cfg 25 | 128x128 s5 c1x2 N1024K4096 DB | 37.3 | 922 | 37.3, 37.3, 37.3 | [1980, 1980, 1980] |
| 1024 | 4096 | 4096 | B DB cfg 26 | 128x64 s8 c1x1 N1024K4096 DB | 47.0 | 731 | 47.1, 47.0, 47.0 | [1980, 1980, 1980] |
| 1024 | 4096 | 4096 | B DB cfg 27 | 64x128 s8 c1x1 DB | 39.0 | 881 | 39.3, 39.0, 39.0 | [1980, 1980, 1980] |
| 1024 | 4096 | 4096 | cutlass tuner coop128x128 c2x1 DB | spec 128,128,2,1,0,0,0,0,0,1 (v0) | 36.6 | 939 | 36.6, 36.6, 36.5 | [1980, 1980, 1980] |
| 1024 | 4096 | 4096 | DeepGEMM FP8 g128 | upstream 2.8.0 1D2D, unpadded M | 32.6 | 1055 | 32.6, 32.6, 32.6 | [1980, 1980, 1980] |
| 1024 | 4096 | 4096 | cutlass_int8 per-tensor | 128x256 c2x1 (int8_scaled_mm cfg 0) | 30.6 | 1124 | 30.5, 30.6, 30.6 | [1980, 1980, 1980] |
| 1024 | 4096 | 4096 | cuBLASLt FP8 per-tensor | torch._scaled_mm use_fast_accum=True | 30.0 | 1146 | 29.8, 30.0, 30.0 | [1980, 1980, 1980] |
| 1024 | 4096 | 4096 | A wi ref cfg 3 | 256x128 s3 c1x2 N1024K4096 ref | 37.4 | 919 | 37.4, 37.3, 37.5 | [1980, 1980, 1980] |
| 1024 | 4096 | 4096 | A wi cfg 9 | 256x128 s3 c1x2 N1024K4096 wi2 | 44.8 | 768 | 44.7, 44.8, 44.8 | [1980, 1980, 1980] |
| 1024 | 4096 | 42240 | deepgemm_int8 old best | 256x128 s3 c1x2 N1024K4096 | 327.6 | 1082 | 314.6, 336.2, 327.6 | [1920, 1680, 1920] |
| 1024 | 4096 | 42240 | B DB cfg 25 | 128x128 s5 c1x2 N1024K4096 DB | 320.2 | 1106 | 313.6, 320.2, 326.7 | [1785, 1875, 1680] |
| 1024 | 4096 | 42240 | B DB cfg 26 | 128x64 s8 c1x1 N1024K4096 DB | 499.6 | 709 | 500.8, 499.6, 496.8 | [1905, 1860, 1725] |
| 1024 | 4096 | 42240 | B DB cfg 27 | 64x128 s8 c1x1 DB | 363.0 | 976 | 366.2, 359.6, 363.0 | [1770, 1680, 1740] |
| 1024 | 4096 | 42240 | cutlass tuner coop128x128 c2x1 DB | spec 128,128,2,1,0,0,0,0,0,1 (v0) | 340.0 | 1042 | 344.3, 340.0, 331.4 | [1515, 1650, 1695] |
| 1024 | 4096 | 42240 | DeepGEMM FP8 g128 | upstream 2.8.0 1D2D, unpadded M | 285.0 | 1243 | 283.1, 295.6, 285.0 | [1560, 1470, 1515] |
| 1024 | 4096 | 42240 | cutlass_int8 per-tensor | 128x256 c2x1 (int8_scaled_mm cfg 0) | 244.1 | 1451 | 244.1, 244.0, 247.0 | [1800, 1680, 1800] |
| 1024 | 4096 | 42240 | cuBLASLt FP8 per-tensor | torch._scaled_mm use_fast_accum=True | 261.7 | 1354 | 261.7, 261.2, 271.6 | [1605, 1560, 1560] |
| 1024 | 4096 | 42240 | A wi ref cfg 3 | 256x128 s3 c1x2 N1024K4096 ref | 324.4 | 1092 | 320.1, 324.4, 336.4 | [1725, 1920, 1785] |
| 1024 | 4096 | 42240 | A wi cfg 9 | 256x128 s3 c1x2 N1024K4096 wi2 | 374.6 | 946 | 374.0, 376.0, 374.6 | [1965, 1965, 1935] |
| 12288 | 4096 | 901 | deepgemm_int8 old best | 256x128 s3 c1x2 N12288K4096 | 98.5 | 921 | 97.0, 98.5, 98.6 | [1980, 1980, 1980] |
| 12288 | 4096 | 901 | B DB cfg 23 | 128x128 s5 c1x2 N12288K4096 DB | 94.7 | 958 | 94.6, 94.7, 94.7 | [1980, 1980, 1980] |
| 12288 | 4096 | 901 | B DB cfg 27 | 64x128 s8 c1x1 DB | 111.6 | 813 | 112.1, 110.9, 111.6 | [1980, 1980, 1980] |
| 12288 | 4096 | 901 | cutlass tuner coop128x128 c2x1 DB | spec 128,128,2,1,0,0,0,0,0,1 (v0) | 106.7 | 850 | 106.8, 106.7, 106.7 | [1980, 1980, 1980] |
| 12288 | 4096 | 901 | DeepGEMM FP8 g128 | upstream 2.8.0 1D2D, unpadded M | 79.9 | 1136 | 79.9, 79.5, 80.0 | [1860, 1905, 1950] |
| 12288 | 4096 | 901 | cutlass_int8 per-tensor | 128x256 c2x1 (int8_scaled_mm cfg 0) | 76.9 | 1179 | 76.9, 76.8, 77.2 | [1965, 1965, 1965] |
| 12288 | 4096 | 901 | cuBLASLt FP8 per-tensor | torch._scaled_mm use_fast_accum=True | 69.4 | 1307 | 69.3, 69.5, 69.4 | [1980, 1980, 1980] |
| 12288 | 4096 | 901 | A wi ref cfg 2 | 256x128 s3 c1x2 N12288K4096 ref | 98.5 | 921 | 97.8, 98.5, 98.6 | [1980, 1980, 1980] |
| 12288 | 4096 | 901 | A wi cfg 8 | 256x128 s3 c1x2 N12288K4096 wi2 | 117.5 | 772 | 117.1, 117.5, 117.5 | [1980, 1980, 1980] |
| 12288 | 4096 | 4096 | deepgemm_int8 old best | 256x128 s3 c1x2 N12288K4096 | 382.1 | 1079 | 382.1, 383.8, 379.3 | [1890, 1845, 1890] |
| 12288 | 4096 | 4096 | B DB cfg 23 | 128x128 s5 c1x2 N12288K4096 DB | 375.0 | 1099 | 370.2, 390.7, 375.0 | [1725, 1650, 1755] |
| 12288 | 4096 | 4096 | B DB cfg 27 | 64x128 s8 c1x1 DB | 430.9 | 957 | 427.3, 438.8, 430.9 | [1830, 1725, 1815] |
| 12288 | 4096 | 4096 | cutlass tuner coop128x128 c2x1 DB | spec 128,128,2,1,0,0,0,0,0,1 (v0) | 386.9 | 1066 | 401.0, 377.8, 386.9 | [1710, 1785, 1755] |
| 12288 | 4096 | 4096 | DeepGEMM FP8 g128 | upstream 2.8.0 1D2D, unpadded M | 337.4 | 1222 | 337.4, 337.2, 339.0 | [1425, 1485, 1440] |
| 12288 | 4096 | 4096 | cutlass_int8 per-tensor | 128x256 c2x1 (int8_scaled_mm cfg 0) | 286.8 | 1438 | 287.4, 286.8, 285.1 | [1845, 1665, 1725] |
| 12288 | 4096 | 4096 | cuBLASLt FP8 per-tensor | torch._scaled_mm use_fast_accum=True | 304.9 | 1352 | 304.9, 313.1, 300.7 | [1650, 1725, 1590] |
| 12288 | 4096 | 4096 | A wi ref cfg 2 | 256x128 s3 c1x2 N12288K4096 ref | 381.3 | 1081 | 381.3, 380.8, 388.1 | [1860, 1905, 1800] |
| 12288 | 4096 | 4096 | A wi cfg 8 | 256x128 s3 c1x2 N12288K4096 wi2 | 441.2 | 935 | 439.8, 446.4, 441.2 | [1935, 1965, 1965] |
| 12288 | 4096 | 42240 | deepgemm_int8 old best | 256x128 s3 c1x2 N12288K4096 | 3938.3 | 1080 | 3905.3, 3944.4, 3938.3 | [1770, 1695, 1365] |
| 12288 | 4096 | 42240 | B DB cfg 23 | 128x128 s5 c1x2 N12288K4096 DB | 3892.3 | 1092 | 3984.5, 3810.0, 3892.3 | [1320, 1425, 1395] |
| 12288 | 4096 | 42240 | B DB cfg 27 | 64x128 s8 c1x1 DB | 4485.6 | 948 | 4374.8, 4504.8, 4485.6 | [1485, 1455, 1455] |
| 12288 | 4096 | 42240 | cutlass tuner coop128x128 c2x1 DB | spec 128,128,2,1,0,0,0,0,0,1 (v0) | 4353.8 | 977 | 4354.9, 4353.8, 4349.8 | [1305, 1350, 1335] |
| 12288 | 4096 | 42240 | DeepGEMM FP8 g128 | upstream 2.8.0 1D2D, unpadded M | 3309.5 | 1285 | 3254.3, 3309.5, 3332.3 | [1410, 1305, 1320] |
| 12288 | 4096 | 42240 | cutlass_int8 per-tensor | 128x256 c2x1 (int8_scaled_mm cfg 0) | 3313.9 | 1283 | 3396.9, 3270.5, 3313.9 | [1365, 1425, 1395] |
| 12288 | 4096 | 42240 | cuBLASLt FP8 per-tensor | torch._scaled_mm use_fast_accum=True | 3177.3 | 1338 | 3176.7, 3184.5, 3177.3 | [1365, 1350, 1350] |
| 12288 | 4096 | 42240 | A wi ref cfg 2 | 256x128 s3 c1x2 N12288K4096 ref | 4028.9 | 1055 | 3981.2, 4136.8, 4028.9 | [1725, 1440, 1635] |
| 12288 | 4096 | 42240 | A wi cfg 8 | 256x128 s3 c1x2 N12288K4096 wi2 | 4494.5 | 946 | 4519.7, 4494.5, 4383.7 | [1620, 1695, 1725] |
| 4096 | 12288 | 901 | deepgemm_int8 old best | 256x128 s3 c1x2 N4096K12288 | 95.9 | 946 | 95.6, 96.1, 95.9 | [1980, 1980, 1965] |
| 4096 | 12288 | 901 | B DB cfg 24 | 128x128 s5 c1x2 N4096K12288 DB | 95.4 | 950 | 95.5, 95.4, 95.3 | [1980, 1980, 1980] |
| 4096 | 12288 | 901 | B DB cfg 27 | 64x128 s8 c1x1 DB | 116.3 | 780 | 116.2, 116.6, 116.3 | [1980, 1980, 1980] |
| 4096 | 12288 | 901 | cutlass tuner coop128x128 c2x1 DB | spec 128,128,2,1,0,0,0,0,0,1 (v0) | 101.1 | 897 | 100.8, 101.1, 101.1 | [1980, 1980, 1980] |
| 4096 | 12288 | 901 | DeepGEMM FP8 g128 | upstream 2.8.0 1D2D, unpadded M | 77.2 | 1174 | 77.1, 77.2, 77.4 | [1935, 1965, 1905] |
| 4096 | 12288 | 901 | cutlass_int8 per-tensor | 128x256 c2x1 (int8_scaled_mm cfg 0) | 71.3 | 1272 | 71.0, 71.4, 71.3 | [1965, 1965, 1965] |
| 4096 | 12288 | 901 | cuBLASLt FP8 per-tensor | torch._scaled_mm use_fast_accum=True | 64.0 | 1417 | 64.0, 64.2, 63.8 | [1980, 1965, 1980] |
| 4096 | 12288 | 901 | A wi ref cfg 1 | 256x128 s3 c1x2 N4096K12288 ref | 95.9 | 946 | 94.6, 95.9, 96.0 | [1980, 1980, 1980] |
| 4096 | 12288 | 901 | A wi cfg 7 | 256x128 s3 c1x2 N4096K12288 wi2 | 117.3 | 773 | 117.5, 116.9, 117.3 | [1980, 1980, 1980] |
| 4096 | 12288 | 4096 | deepgemm_int8 old best | 256x128 s3 c1x2 N4096K12288 | 369.0 | 1117 | 370.1, 367.5, 369.0 | [1890, 1830, 1875] |
| 4096 | 12288 | 4096 | B DB cfg 24 | 128x128 s5 c1x2 N4096K12288 DB | 373.7 | 1103 | 369.6, 388.2, 373.7 | [1770, 1620, 1740] |
| 4096 | 12288 | 4096 | B DB cfg 27 | 64x128 s8 c1x1 DB | 420.8 | 980 | 414.1, 422.2, 420.8 | [1710, 1695, 1800] |
| 4096 | 12288 | 4096 | cutlass tuner coop128x128 c2x1 DB | spec 128,128,2,1,0,0,0,0,0,1 (v0) | 390.8 | 1055 | 424.7, 383.4, 390.8 | [1530, 1680, 1650] |
| 4096 | 12288 | 4096 | DeepGEMM FP8 g128 | upstream 2.8.0 1D2D, unpadded M | 329.4 | 1252 | 329.4, 316.2, 330.3 | [1515, 1440, 1410] |
| 4096 | 12288 | 4096 | cutlass_int8 per-tensor | 128x256 c2x1 (int8_scaled_mm cfg 0) | 271.3 | 1520 | 271.3, 272.5, 270.0 | [1770, 1800, 1695] |
| 4096 | 12288 | 4096 | cuBLASLt FP8 per-tensor | torch._scaled_mm use_fast_accum=True | 297.5 | 1386 | 290.8, 300.4, 297.5 | [1605, 1590, 1635] |
| 4096 | 12288 | 4096 | A wi ref cfg 1 | 256x128 s3 c1x2 N4096K12288 ref | 373.6 | 1104 | 368.9, 373.6, 377.7 | [1920, 1875, 1845] |
| 4096 | 12288 | 4096 | A wi cfg 7 | 256x128 s3 c1x2 N4096K12288 wi2 | 442.8 | 931 | 442.8, 441.6, 443.1 | [1965, 1950, 1965] |
| 4096 | 12288 | 42240 | deepgemm_int8 old best | 256x128 s3 c1x2 N4096K12288 | 3867.1 | 1100 | 3867.1, 3854.3, 3914.3 | [1710, 1620, 1710] |
| 4096 | 12288 | 42240 | B DB cfg 24 | 128x128 s5 c1x2 N4096K12288 DB | 3856.9 | 1102 | 4015.7, 3813.8, 3856.9 | [1320, 1410, 1395] |
| 4096 | 12288 | 42240 | B DB cfg 27 | 64x128 s8 c1x1 DB | 4650.2 | 914 | 4650.2, 4643.2, 4671.3 | [1455, 1410, 1440] |
| 4096 | 12288 | 42240 | cutlass tuner coop128x128 c2x1 DB | spec 128,128,2,1,0,0,0,0,0,1 (v0) | 4060.6 | 1047 | 4060.1, 4060.6, 4090.2 | [1410, 1395, 1365] |
| 4096 | 12288 | 42240 | DeepGEMM FP8 g128 | upstream 2.8.0 1D2D, unpadded M | 3209.4 | 1325 | 3209.4, 3206.0, 3238.7 | [1335, 1275, 1275] |
| 4096 | 12288 | 42240 | cutlass_int8 per-tensor | 128x256 c2x1 (int8_scaled_mm cfg 0) | 2871.9 | 1481 | 2929.3, 2853.2, 2871.9 | [1440, 1455, 1425] |
| 4096 | 12288 | 42240 | cuBLASLt FP8 per-tensor | torch._scaled_mm use_fast_accum=True | 3068.2 | 1386 | 3068.2, 3111.0, 3066.8 | [1305, 1290, 1305] |
| 4096 | 12288 | 42240 | A wi ref cfg 1 | 256x128 s3 c1x2 N4096K12288 ref | 3998.8 | 1063 | 3899.9, 3998.8, 4028.7 | [1710, 1410, 1530] |
| 4096 | 12288 | 42240 | A wi cfg 7 | 256x128 s3 c1x2 N4096K12288 wi2 | 4527.3 | 939 | 4533.3, 4527.3, 4449.6 | [1785, 1725, 1620] |

## Verification detail (bit-identity vs the default kernel, rel-L2 vs fp64)

| ext | cfg | name | N | K | M | case | bit-identical | #diff | rel-L2 | max rel |
|---|---|---|---|---|---|---|---|---:|---:|---:|
| main | 25 | 128x128 s5 c1x2 N1024K4096 DB | 1024 | 4096 | 901 | random | True | 0 | 1.66e-03 | 3.42e-03 |
| main | 26 | 128x64 s8 c1x1 N1024K4096 DB | 1024 | 4096 | 901 | random | True | 0 | 1.66e-03 | 3.42e-03 |
| main | 27 | 64x128 s8 c1x1 DB | 1024 | 4096 | 901 | random | True | 0 | 1.66e-03 | 3.42e-03 |
| main | 28 | 128x128 s5 c1x1 DB | 1024 | 4096 | 901 | random | True | 0 | 1.66e-03 | 3.42e-03 |
| main | 29 | 128x128 s5 c1x2 DB | 1024 | 4096 | 901 | random | True | 0 | 1.66e-03 | 3.42e-03 |
| main | 25 | 128x128 s5 c1x2 N1024K4096 DB | 1024 | 4096 | 901 | adversarial_scales | True | 0 | 1.66e-03 | 2.82e-03 |
| main | 26 | 128x64 s8 c1x1 N1024K4096 DB | 1024 | 4096 | 901 | adversarial_scales | True | 0 | 1.66e-03 | 2.82e-03 |
| main | 27 | 64x128 s8 c1x1 DB | 1024 | 4096 | 901 | adversarial_scales | True | 0 | 1.66e-03 | 2.82e-03 |
| main | 28 | 128x128 s5 c1x1 DB | 1024 | 4096 | 901 | adversarial_scales | True | 0 | 1.66e-03 | 2.82e-03 |
| main | 29 | 128x128 s5 c1x2 DB | 1024 | 4096 | 901 | adversarial_scales | True | 0 | 1.66e-03 | 2.82e-03 |
| main | 25 | 128x128 s5 c1x2 N1024K4096 DB | 1024 | 4096 | 901 | extremes_pm127 | True | 0 | 1.82e-03 | 2.32e-03 |
| main | 26 | 128x64 s8 c1x1 N1024K4096 DB | 1024 | 4096 | 901 | extremes_pm127 | True | 0 | 1.82e-03 | 2.32e-03 |
| main | 27 | 64x128 s8 c1x1 DB | 1024 | 4096 | 901 | extremes_pm127 | True | 0 | 1.82e-03 | 2.32e-03 |
| main | 28 | 128x128 s5 c1x1 DB | 1024 | 4096 | 901 | extremes_pm127 | True | 0 | 1.82e-03 | 2.32e-03 |
| main | 29 | 128x128 s5 c1x2 DB | 1024 | 4096 | 901 | extremes_pm127 | True | 0 | 1.82e-03 | 2.32e-03 |
| main | 25 | 128x128 s5 c1x2 N1024K4096 DB | 1024 | 4096 | 901 | extremes_m128 | True | 0 | 1.73e-03 | 2.11e-03 |
| main | 26 | 128x64 s8 c1x1 N1024K4096 DB | 1024 | 4096 | 901 | extremes_m128 | True | 0 | 1.73e-03 | 2.11e-03 |
| main | 27 | 64x128 s8 c1x1 DB | 1024 | 4096 | 901 | extremes_m128 | True | 0 | 1.73e-03 | 2.11e-03 |
| main | 28 | 128x128 s5 c1x1 DB | 1024 | 4096 | 901 | extremes_m128 | True | 0 | 1.73e-03 | 2.11e-03 |
| main | 29 | 128x128 s5 c1x2 DB | 1024 | 4096 | 901 | extremes_m128 | True | 0 | 1.73e-03 | 2.11e-03 |
| main | 24 | 128x128 s5 c1x2 N4096K12288 DB | 4096 | 12288 | 1802 | random | True | 0 | 1.66e-03 | 2.61e-03 |
| main | 27 | 64x128 s8 c1x1 DB | 4096 | 12288 | 1802 | random | True | 0 | 1.66e-03 | 2.61e-03 |
| main | 28 | 128x128 s5 c1x1 DB | 4096 | 12288 | 1802 | random | True | 0 | 1.66e-03 | 2.61e-03 |
| main | 29 | 128x128 s5 c1x2 DB | 4096 | 12288 | 1802 | random | True | 0 | 1.66e-03 | 2.61e-03 |
| main | 24 | 128x128 s5 c1x2 N4096K12288 DB | 4096 | 12288 | 1802 | adversarial_scales | True | 0 | 1.66e-03 | 3.02e-03 |
| main | 27 | 64x128 s8 c1x1 DB | 4096 | 12288 | 1802 | adversarial_scales | True | 0 | 1.66e-03 | 3.02e-03 |
| main | 28 | 128x128 s5 c1x1 DB | 4096 | 12288 | 1802 | adversarial_scales | True | 0 | 1.66e-03 | 3.02e-03 |
| main | 29 | 128x128 s5 c1x2 DB | 4096 | 12288 | 1802 | adversarial_scales | True | 0 | 1.66e-03 | 3.02e-03 |
| main | 24 | 128x128 s5 c1x2 N4096K12288 DB | 4096 | 12288 | 1802 | extremes_pm127 | True | 0 | 1.48e-03 | 1.98e-03 |
| main | 27 | 64x128 s8 c1x1 DB | 4096 | 12288 | 1802 | extremes_pm127 | True | 0 | 1.48e-03 | 1.98e-03 |
| main | 28 | 128x128 s5 c1x1 DB | 4096 | 12288 | 1802 | extremes_pm127 | True | 0 | 1.48e-03 | 1.98e-03 |
| main | 29 | 128x128 s5 c1x2 DB | 4096 | 12288 | 1802 | extremes_pm127 | True | 0 | 1.48e-03 | 1.98e-03 |
| main | 24 | 128x128 s5 c1x2 N4096K12288 DB | 4096 | 12288 | 1802 | extremes_m128 | True | 0 | 1.74e-03 | 3.02e-03 |
| main | 27 | 64x128 s8 c1x1 DB | 4096 | 12288 | 1802 | extremes_m128 | True | 0 | 1.74e-03 | 3.02e-03 |
| main | 28 | 128x128 s5 c1x1 DB | 4096 | 12288 | 1802 | extremes_m128 | True | 0 | 1.74e-03 | 3.02e-03 |
| main | 29 | 128x128 s5 c1x2 DB | 4096 | 12288 | 1802 | extremes_m128 | True | 0 | 1.74e-03 | 3.02e-03 |
| main | 27 | 64x128 s8 c1x1 DB | 4096 | 128 | 901 | random | True | 0 | 1.66e-03 | 2.58e-03 |
| main | 28 | 128x128 s5 c1x1 DB | 4096 | 128 | 901 | random | True | 0 | 1.66e-03 | 2.58e-03 |
| main | 29 | 128x128 s5 c1x2 DB | 4096 | 128 | 901 | random | True | 0 | 1.66e-03 | 2.58e-03 |
| main | 27 | 64x128 s8 c1x1 DB | 4096 | 128 | 901 | adversarial_scales | True | 0 | 1.66e-03 | 3.20e-03 |
| main | 28 | 128x128 s5 c1x1 DB | 4096 | 128 | 901 | adversarial_scales | True | 0 | 1.66e-03 | 3.20e-03 |
| main | 29 | 128x128 s5 c1x2 DB | 4096 | 128 | 901 | adversarial_scales | True | 0 | 1.66e-03 | 3.20e-03 |
| main | 27 | 64x128 s8 c1x1 DB | 4096 | 128 | 901 | extremes_pm127 | True | 0 | 1.03e-03 | 9.73e-04 |
| main | 28 | 128x128 s5 c1x1 DB | 4096 | 128 | 901 | extremes_pm127 | True | 0 | 1.03e-03 | 9.73e-04 |
| main | 29 | 128x128 s5 c1x2 DB | 4096 | 128 | 901 | extremes_pm127 | True | 0 | 1.03e-03 | 9.73e-04 |
| main | 27 | 64x128 s8 c1x1 DB | 4096 | 128 | 901 | extremes_m128 | True | 0 | 1.60e-03 | 3.56e-03 |
| main | 28 | 128x128 s5 c1x1 DB | 4096 | 128 | 901 | extremes_m128 | True | 0 | 1.60e-03 | 3.56e-03 |
| main | 29 | 128x128 s5 c1x2 DB | 4096 | 128 | 901 | extremes_m128 | True | 0 | 1.60e-03 | 3.56e-03 |
| main | 27 | 64x128 s8 c1x1 DB | 4096 | 128 | 4096 | random | True | 0 | 1.66e-03 | 2.64e-03 |
| main | 28 | 128x128 s5 c1x1 DB | 4096 | 128 | 4096 | random | True | 0 | 1.66e-03 | 2.64e-03 |
| main | 29 | 128x128 s5 c1x2 DB | 4096 | 128 | 4096 | random | True | 0 | 1.66e-03 | 2.64e-03 |
| main | 27 | 64x128 s8 c1x1 DB | 4096 | 128 | 4096 | adversarial_scales | True | 0 | 1.66e-03 | 3.03e-03 |
| main | 28 | 128x128 s5 c1x1 DB | 4096 | 128 | 4096 | adversarial_scales | True | 0 | 1.66e-03 | 3.03e-03 |
| main | 29 | 128x128 s5 c1x2 DB | 4096 | 128 | 4096 | adversarial_scales | True | 0 | 1.66e-03 | 3.03e-03 |
| main | 27 | 64x128 s8 c1x1 DB | 4096 | 128 | 4096 | extremes_pm127 | True | 0 | 1.54e-03 | 9.73e-04 |
| main | 28 | 128x128 s5 c1x1 DB | 4096 | 128 | 4096 | extremes_pm127 | True | 0 | 1.54e-03 | 9.73e-04 |
| main | 29 | 128x128 s5 c1x2 DB | 4096 | 128 | 4096 | extremes_pm127 | True | 0 | 1.54e-03 | 9.73e-04 |
| main | 27 | 64x128 s8 c1x1 DB | 4096 | 128 | 4096 | extremes_m128 | True | 0 | 1.61e-03 | 3.63e-03 |
| main | 28 | 128x128 s5 c1x1 DB | 4096 | 128 | 4096 | extremes_m128 | True | 0 | 1.61e-03 | 3.63e-03 |
| main | 29 | 128x128 s5 c1x2 DB | 4096 | 128 | 4096 | extremes_m128 | True | 0 | 1.61e-03 | 3.63e-03 |
| main | 27 | 64x128 s8 c1x1 DB | 1024 | 128 | 1802 | random | True | 0 | 1.66e-03 | 2.43e-03 |
| main | 28 | 128x128 s5 c1x1 DB | 1024 | 128 | 1802 | random | True | 0 | 1.66e-03 | 2.43e-03 |
| main | 29 | 128x128 s5 c1x2 DB | 1024 | 128 | 1802 | random | True | 0 | 1.66e-03 | 2.43e-03 |
| main | 27 | 64x128 s8 c1x1 DB | 1024 | 128 | 1802 | adversarial_scales | True | 0 | 1.66e-03 | 3.29e-03 |
| main | 28 | 128x128 s5 c1x1 DB | 1024 | 128 | 1802 | adversarial_scales | True | 0 | 1.66e-03 | 3.29e-03 |
| main | 29 | 128x128 s5 c1x2 DB | 1024 | 128 | 1802 | adversarial_scales | True | 0 | 1.66e-03 | 3.29e-03 |
| main | 27 | 64x128 s8 c1x1 DB | 1024 | 128 | 1802 | extremes_pm127 | True | 0 | 7.15e-04 | 9.73e-04 |
| main | 28 | 128x128 s5 c1x1 DB | 1024 | 128 | 1802 | extremes_pm127 | True | 0 | 7.15e-04 | 9.73e-04 |
| main | 29 | 128x128 s5 c1x2 DB | 1024 | 128 | 1802 | extremes_pm127 | True | 0 | 7.15e-04 | 9.73e-04 |
| main | 27 | 64x128 s8 c1x1 DB | 1024 | 128 | 1802 | extremes_m128 | True | 0 | 1.67e-03 | 2.42e-03 |
| main | 28 | 128x128 s5 c1x1 DB | 1024 | 128 | 1802 | extremes_m128 | True | 0 | 1.67e-03 | 2.42e-03 |
| main | 29 | 128x128 s5 c1x2 DB | 1024 | 128 | 1802 | extremes_m128 | True | 0 | 1.67e-03 | 2.42e-03 |
| main | 27 | 64x128 s8 c1x1 DB | 4096 | 256 | 901 | random | True | 0 | 1.66e-03 | 2.29e-03 |
| main | 28 | 128x128 s5 c1x1 DB | 4096 | 256 | 901 | random | True | 0 | 1.66e-03 | 2.29e-03 |
| main | 29 | 128x128 s5 c1x2 DB | 4096 | 256 | 901 | random | True | 0 | 1.66e-03 | 2.29e-03 |
| main | 27 | 64x128 s8 c1x1 DB | 4096 | 256 | 901 | adversarial_scales | True | 0 | 1.66e-03 | 2.39e-03 |
| main | 28 | 128x128 s5 c1x1 DB | 4096 | 256 | 901 | adversarial_scales | True | 0 | 1.66e-03 | 2.39e-03 |
| main | 29 | 128x128 s5 c1x2 DB | 4096 | 256 | 901 | adversarial_scales | True | 0 | 1.66e-03 | 2.39e-03 |
| main | 27 | 64x128 s8 c1x1 DB | 4096 | 256 | 901 | extremes_pm127 | True | 0 | 9.31e-04 | 4.96e-04 |
| main | 28 | 128x128 s5 c1x1 DB | 4096 | 256 | 901 | extremes_pm127 | True | 0 | 9.31e-04 | 4.96e-04 |
| main | 29 | 128x128 s5 c1x2 DB | 4096 | 256 | 901 | extremes_pm127 | True | 0 | 9.31e-04 | 4.96e-04 |
| main | 27 | 64x128 s8 c1x1 DB | 4096 | 256 | 901 | extremes_m128 | True | 0 | 1.67e-03 | 2.01e-03 |
| main | 28 | 128x128 s5 c1x1 DB | 4096 | 256 | 901 | extremes_m128 | True | 0 | 1.67e-03 | 2.01e-03 |
| main | 29 | 128x128 s5 c1x2 DB | 4096 | 256 | 901 | extremes_m128 | True | 0 | 1.67e-03 | 2.01e-03 |
| main | 27 | 64x128 s8 c1x1 DB | 4096 | 256 | 4096 | random | True | 0 | 1.66e-03 | 1.94e-03 |
| main | 28 | 128x128 s5 c1x1 DB | 4096 | 256 | 4096 | random | True | 0 | 1.66e-03 | 1.94e-03 |
| main | 29 | 128x128 s5 c1x2 DB | 4096 | 256 | 4096 | random | True | 0 | 1.66e-03 | 1.94e-03 |
| main | 27 | 64x128 s8 c1x1 DB | 4096 | 256 | 4096 | adversarial_scales | True | 0 | 1.66e-03 | 2.35e-03 |
| main | 28 | 128x128 s5 c1x1 DB | 4096 | 256 | 4096 | adversarial_scales | True | 0 | 1.66e-03 | 2.35e-03 |
| main | 29 | 128x128 s5 c1x2 DB | 4096 | 256 | 4096 | adversarial_scales | True | 0 | 1.66e-03 | 2.35e-03 |
| main | 27 | 64x128 s8 c1x1 DB | 4096 | 256 | 4096 | extremes_pm127 | True | 0 | 1.40e-03 | 7.26e-04 |
| main | 28 | 128x128 s5 c1x1 DB | 4096 | 256 | 4096 | extremes_pm127 | True | 0 | 1.40e-03 | 7.26e-04 |
| main | 29 | 128x128 s5 c1x2 DB | 4096 | 256 | 4096 | extremes_pm127 | True | 0 | 1.40e-03 | 7.26e-04 |
| main | 27 | 64x128 s8 c1x1 DB | 4096 | 256 | 4096 | extremes_m128 | True | 0 | 1.67e-03 | 2.13e-03 |
| main | 28 | 128x128 s5 c1x1 DB | 4096 | 256 | 4096 | extremes_m128 | True | 0 | 1.67e-03 | 2.13e-03 |
| main | 29 | 128x128 s5 c1x2 DB | 4096 | 256 | 4096 | extremes_m128 | True | 0 | 1.67e-03 | 2.13e-03 |
| main | 27 | 64x128 s8 c1x1 DB | 1024 | 256 | 4096 | random | True | 0 | 1.66e-03 | 2.58e-03 |
| main | 28 | 128x128 s5 c1x1 DB | 1024 | 256 | 4096 | random | True | 0 | 1.66e-03 | 2.58e-03 |
| main | 29 | 128x128 s5 c1x2 DB | 1024 | 256 | 4096 | random | True | 0 | 1.66e-03 | 2.58e-03 |
| main | 27 | 64x128 s8 c1x1 DB | 1024 | 256 | 4096 | adversarial_scales | True | 0 | 1.66e-03 | 2.28e-03 |
| main | 28 | 128x128 s5 c1x1 DB | 1024 | 256 | 4096 | adversarial_scales | True | 0 | 1.66e-03 | 2.28e-03 |
| main | 29 | 128x128 s5 c1x2 DB | 1024 | 256 | 4096 | adversarial_scales | True | 0 | 1.66e-03 | 2.28e-03 |
| main | 27 | 64x128 s8 c1x1 DB | 1024 | 256 | 4096 | extremes_pm127 | True | 0 | 1.10e-03 | 9.91e-04 |
| main | 28 | 128x128 s5 c1x1 DB | 1024 | 256 | 4096 | extremes_pm127 | True | 0 | 1.10e-03 | 9.91e-04 |
| main | 29 | 128x128 s5 c1x2 DB | 1024 | 256 | 4096 | extremes_pm127 | True | 0 | 1.10e-03 | 9.91e-04 |
| main | 27 | 64x128 s8 c1x1 DB | 1024 | 256 | 4096 | extremes_m128 | True | 0 | 1.70e-03 | 2.53e-03 |
| main | 28 | 128x128 s5 c1x1 DB | 1024 | 256 | 4096 | extremes_m128 | True | 0 | 1.70e-03 | 2.53e-03 |
| main | 29 | 128x128 s5 c1x2 DB | 1024 | 256 | 4096 | extremes_m128 | True | 0 | 1.70e-03 | 2.53e-03 |
| main | 27 | 64x128 s8 c1x1 DB | 1024 | 384 | 901 | random | True | 0 | 1.66e-03 | 2.35e-03 |
| main | 28 | 128x128 s5 c1x1 DB | 1024 | 384 | 901 | random | True | 0 | 1.66e-03 | 2.35e-03 |
| main | 29 | 128x128 s5 c1x2 DB | 1024 | 384 | 901 | random | True | 0 | 1.66e-03 | 2.35e-03 |
| main | 27 | 64x128 s8 c1x1 DB | 1024 | 384 | 901 | adversarial_scales | True | 0 | 1.66e-03 | 2.18e-03 |
| main | 28 | 128x128 s5 c1x1 DB | 1024 | 384 | 901 | adversarial_scales | True | 0 | 1.66e-03 | 2.18e-03 |
| main | 29 | 128x128 s5 c1x2 DB | 1024 | 384 | 901 | adversarial_scales | True | 0 | 1.66e-03 | 2.18e-03 |
| main | 27 | 64x128 s8 c1x1 DB | 1024 | 384 | 901 | extremes_pm127 | True | 0 | 4.25e-04 | 6.02e-04 |
| main | 28 | 128x128 s5 c1x1 DB | 1024 | 384 | 901 | extremes_pm127 | True | 0 | 4.25e-04 | 6.02e-04 |
| main | 29 | 128x128 s5 c1x2 DB | 1024 | 384 | 901 | extremes_pm127 | True | 0 | 4.25e-04 | 6.02e-04 |
| main | 27 | 64x128 s8 c1x1 DB | 1024 | 384 | 901 | extremes_m128 | True | 0 | 1.66e-03 | 2.84e-03 |
| main | 28 | 128x128 s5 c1x1 DB | 1024 | 384 | 901 | extremes_m128 | True | 0 | 1.66e-03 | 2.84e-03 |
| main | 29 | 128x128 s5 c1x2 DB | 1024 | 384 | 901 | extremes_m128 | True | 0 | 1.66e-03 | 2.84e-03 |
| main | 19 | 128x128 s5 c1x2 N4096K4096 DB | 4096 | 4096 | 4096 | random | True | 0 | 1.66e-03 | 3.18e-03 |
| main | 20 | 128x128 s5 c2x1 N4096K4096 DB | 4096 | 4096 | 4096 | random | True | 0 | 1.66e-03 | 3.18e-03 |
| main | 21 | 128x128 s5 c1x1 N4096K4096 DB | 4096 | 4096 | 4096 | random | True | 0 | 1.66e-03 | 3.18e-03 |
| main | 22 | 128x128 s4 c1x2 N4096K4096 DB | 4096 | 4096 | 4096 | random | True | 0 | 1.66e-03 | 3.18e-03 |
| main | 27 | 64x128 s8 c1x1 DB | 4096 | 4096 | 4096 | random | True | 0 | 1.66e-03 | 3.18e-03 |
| main | 28 | 128x128 s5 c1x1 DB | 4096 | 4096 | 4096 | random | True | 0 | 1.66e-03 | 3.18e-03 |
| main | 29 | 128x128 s5 c1x2 DB | 4096 | 4096 | 4096 | random | True | 0 | 1.66e-03 | 3.18e-03 |
| main | 19 | 128x128 s5 c1x2 N4096K4096 DB | 4096 | 4096 | 4096 | adversarial_scales | True | 0 | 1.66e-03 | 2.55e-03 |
| main | 20 | 128x128 s5 c2x1 N4096K4096 DB | 4096 | 4096 | 4096 | adversarial_scales | True | 0 | 1.66e-03 | 2.55e-03 |
| main | 21 | 128x128 s5 c1x1 N4096K4096 DB | 4096 | 4096 | 4096 | adversarial_scales | True | 0 | 1.66e-03 | 2.55e-03 |
| main | 22 | 128x128 s4 c1x2 N4096K4096 DB | 4096 | 4096 | 4096 | adversarial_scales | True | 0 | 1.66e-03 | 2.55e-03 |
| main | 27 | 64x128 s8 c1x1 DB | 4096 | 4096 | 4096 | adversarial_scales | True | 0 | 1.66e-03 | 2.55e-03 |
| main | 28 | 128x128 s5 c1x1 DB | 4096 | 4096 | 4096 | adversarial_scales | True | 0 | 1.66e-03 | 2.55e-03 |
| main | 29 | 128x128 s5 c1x2 DB | 4096 | 4096 | 4096 | adversarial_scales | True | 0 | 1.66e-03 | 2.55e-03 |
| main | 19 | 128x128 s5 c1x2 N4096K4096 DB | 4096 | 4096 | 4096 | extremes_pm127 | True | 0 | 1.67e-03 | 2.15e-03 |
| main | 20 | 128x128 s5 c2x1 N4096K4096 DB | 4096 | 4096 | 4096 | extremes_pm127 | True | 0 | 1.67e-03 | 2.15e-03 |
| main | 21 | 128x128 s5 c1x1 N4096K4096 DB | 4096 | 4096 | 4096 | extremes_pm127 | True | 0 | 1.67e-03 | 2.15e-03 |
| main | 22 | 128x128 s4 c1x2 N4096K4096 DB | 4096 | 4096 | 4096 | extremes_pm127 | True | 0 | 1.67e-03 | 2.15e-03 |
| main | 27 | 64x128 s8 c1x1 DB | 4096 | 4096 | 4096 | extremes_pm127 | True | 0 | 1.67e-03 | 2.15e-03 |
| main | 28 | 128x128 s5 c1x1 DB | 4096 | 4096 | 4096 | extremes_pm127 | True | 0 | 1.67e-03 | 2.15e-03 |
| main | 29 | 128x128 s5 c1x2 DB | 4096 | 4096 | 4096 | extremes_pm127 | True | 0 | 1.67e-03 | 2.15e-03 |
| main | 19 | 128x128 s5 c1x2 N4096K4096 DB | 4096 | 4096 | 4096 | extremes_m128 | True | 0 | 1.67e-03 | 3.60e-03 |
| main | 20 | 128x128 s5 c2x1 N4096K4096 DB | 4096 | 4096 | 4096 | extremes_m128 | True | 0 | 1.67e-03 | 3.60e-03 |
| main | 21 | 128x128 s5 c1x1 N4096K4096 DB | 4096 | 4096 | 4096 | extremes_m128 | True | 0 | 1.67e-03 | 3.60e-03 |
| main | 22 | 128x128 s4 c1x2 N4096K4096 DB | 4096 | 4096 | 4096 | extremes_m128 | True | 0 | 1.67e-03 | 3.60e-03 |
| main | 27 | 64x128 s8 c1x1 DB | 4096 | 4096 | 4096 | extremes_m128 | True | 0 | 1.67e-03 | 3.60e-03 |
| main | 28 | 128x128 s5 c1x1 DB | 4096 | 4096 | 4096 | extremes_m128 | True | 0 | 1.67e-03 | 3.60e-03 |
| main | 29 | 128x128 s5 c1x2 DB | 4096 | 4096 | 4096 | extremes_m128 | True | 0 | 1.67e-03 | 3.60e-03 |
| main | 23 | 128x128 s5 c1x2 N12288K4096 DB | 12288 | 4096 | 4096 | random | True | 0 | 1.66e-03 | 2.99e-03 |
| main | 27 | 64x128 s8 c1x1 DB | 12288 | 4096 | 4096 | random | True | 0 | 1.66e-03 | 2.99e-03 |
| main | 28 | 128x128 s5 c1x1 DB | 12288 | 4096 | 4096 | random | True | 0 | 1.66e-03 | 2.99e-03 |
| main | 29 | 128x128 s5 c1x2 DB | 12288 | 4096 | 4096 | random | True | 0 | 1.66e-03 | 2.99e-03 |
| main | 23 | 128x128 s5 c1x2 N12288K4096 DB | 12288 | 4096 | 4096 | adversarial_scales | True | 0 | 1.66e-03 | 2.38e-03 |
| main | 27 | 64x128 s8 c1x1 DB | 12288 | 4096 | 4096 | adversarial_scales | True | 0 | 1.66e-03 | 2.38e-03 |
| main | 28 | 128x128 s5 c1x1 DB | 12288 | 4096 | 4096 | adversarial_scales | True | 0 | 1.66e-03 | 2.38e-03 |
| main | 29 | 128x128 s5 c1x2 DB | 12288 | 4096 | 4096 | adversarial_scales | True | 0 | 1.66e-03 | 2.38e-03 |
| main | 23 | 128x128 s5 c1x2 N12288K4096 DB | 12288 | 4096 | 4096 | extremes_pm127 | True | 0 | 1.62e-03 | 2.32e-03 |
| main | 27 | 64x128 s8 c1x1 DB | 12288 | 4096 | 4096 | extremes_pm127 | True | 0 | 1.62e-03 | 2.32e-03 |
| main | 28 | 128x128 s5 c1x1 DB | 12288 | 4096 | 4096 | extremes_pm127 | True | 0 | 1.62e-03 | 2.32e-03 |
| main | 29 | 128x128 s5 c1x2 DB | 12288 | 4096 | 4096 | extremes_pm127 | True | 0 | 1.62e-03 | 2.32e-03 |
| main | 23 | 128x128 s5 c1x2 N12288K4096 DB | 12288 | 4096 | 4096 | extremes_m128 | True | 0 | 1.69e-03 | 3.69e-03 |
| main | 27 | 64x128 s8 c1x1 DB | 12288 | 4096 | 4096 | extremes_m128 | True | 0 | 1.69e-03 | 3.69e-03 |
| main | 28 | 128x128 s5 c1x1 DB | 12288 | 4096 | 4096 | extremes_m128 | True | 0 | 1.69e-03 | 3.69e-03 |
| main | 29 | 128x128 s5 c1x2 DB | 12288 | 4096 | 4096 | extremes_m128 | True | 0 | 1.69e-03 | 3.69e-03 |
| wi | 9 | 256x128 s3 c1x2 N1024K4096 wi2 | 1024 | 4096 | 901 | random | True | 0 | 1.66e-03 | 3.42e-03 |
| wi | 12 | 256x128 s3 c1x2 wi2 | 1024 | 4096 | 901 | random | True | 0 | 1.66e-03 | 3.42e-03 |
| wi | 9 | 256x128 s3 c1x2 N1024K4096 wi2 | 1024 | 4096 | 901 | adversarial_scales | True | 0 | 1.66e-03 | 2.82e-03 |
| wi | 12 | 256x128 s3 c1x2 wi2 | 1024 | 4096 | 901 | adversarial_scales | True | 0 | 1.66e-03 | 2.82e-03 |
| wi | 9 | 256x128 s3 c1x2 N1024K4096 wi2 | 1024 | 4096 | 901 | extremes_pm127 | True | 0 | 1.82e-03 | 2.32e-03 |
| wi | 12 | 256x128 s3 c1x2 wi2 | 1024 | 4096 | 901 | extremes_pm127 | True | 0 | 1.82e-03 | 2.32e-03 |
| wi | 9 | 256x128 s3 c1x2 N1024K4096 wi2 | 1024 | 4096 | 901 | extremes_m128 | True | 0 | 1.73e-03 | 2.11e-03 |
| wi | 12 | 256x128 s3 c1x2 wi2 | 1024 | 4096 | 901 | extremes_m128 | True | 0 | 1.73e-03 | 2.11e-03 |
| wi | 7 | 256x128 s3 c1x2 N4096K12288 wi2 | 4096 | 12288 | 1802 | random | True | 0 | 1.66e-03 | 2.61e-03 |
| wi | 12 | 256x128 s3 c1x2 wi2 | 4096 | 12288 | 1802 | random | True | 0 | 1.66e-03 | 2.61e-03 |
| wi | 7 | 256x128 s3 c1x2 N4096K12288 wi2 | 4096 | 12288 | 1802 | adversarial_scales | True | 0 | 1.66e-03 | 3.02e-03 |
| wi | 12 | 256x128 s3 c1x2 wi2 | 4096 | 12288 | 1802 | adversarial_scales | True | 0 | 1.66e-03 | 3.02e-03 |
| wi | 7 | 256x128 s3 c1x2 N4096K12288 wi2 | 4096 | 12288 | 1802 | extremes_pm127 | True | 0 | 1.48e-03 | 1.98e-03 |
| wi | 12 | 256x128 s3 c1x2 wi2 | 4096 | 12288 | 1802 | extremes_pm127 | True | 0 | 1.48e-03 | 1.98e-03 |
| wi | 7 | 256x128 s3 c1x2 N4096K12288 wi2 | 4096 | 12288 | 1802 | extremes_m128 | True | 0 | 1.74e-03 | 3.02e-03 |
| wi | 12 | 256x128 s3 c1x2 wi2 | 4096 | 12288 | 1802 | extremes_m128 | True | 0 | 1.74e-03 | 3.02e-03 |
| wi | 12 | 256x128 s3 c1x2 wi2 | 4096 | 128 | 901 | random | True | 0 | 1.66e-03 | 2.58e-03 |
| wi | 12 | 256x128 s3 c1x2 wi2 | 4096 | 128 | 901 | adversarial_scales | True | 0 | 1.66e-03 | 3.20e-03 |
| wi | 12 | 256x128 s3 c1x2 wi2 | 4096 | 128 | 901 | extremes_pm127 | True | 0 | 1.03e-03 | 9.73e-04 |
| wi | 12 | 256x128 s3 c1x2 wi2 | 4096 | 128 | 901 | extremes_m128 | True | 0 | 1.60e-03 | 3.56e-03 |
| wi | 12 | 256x128 s3 c1x2 wi2 | 4096 | 128 | 4096 | random | True | 0 | 1.66e-03 | 2.64e-03 |
| wi | 12 | 256x128 s3 c1x2 wi2 | 4096 | 128 | 4096 | adversarial_scales | True | 0 | 1.66e-03 | 3.03e-03 |
| wi | 12 | 256x128 s3 c1x2 wi2 | 4096 | 128 | 4096 | extremes_pm127 | True | 0 | 1.54e-03 | 9.73e-04 |
| wi | 12 | 256x128 s3 c1x2 wi2 | 4096 | 128 | 4096 | extremes_m128 | True | 0 | 1.61e-03 | 3.63e-03 |
| wi | 12 | 256x128 s3 c1x2 wi2 | 1024 | 128 | 1802 | random | True | 0 | 1.66e-03 | 2.43e-03 |
| wi | 12 | 256x128 s3 c1x2 wi2 | 1024 | 128 | 1802 | adversarial_scales | True | 0 | 1.66e-03 | 3.29e-03 |
| wi | 12 | 256x128 s3 c1x2 wi2 | 1024 | 128 | 1802 | extremes_pm127 | True | 0 | 7.15e-04 | 9.73e-04 |
| wi | 12 | 256x128 s3 c1x2 wi2 | 1024 | 128 | 1802 | extremes_m128 | True | 0 | 1.67e-03 | 2.42e-03 |
| wi | 12 | 256x128 s3 c1x2 wi2 | 4096 | 256 | 901 | random | True | 0 | 1.66e-03 | 2.29e-03 |
| wi | 12 | 256x128 s3 c1x2 wi2 | 4096 | 256 | 901 | adversarial_scales | True | 0 | 1.66e-03 | 2.39e-03 |
| wi | 12 | 256x128 s3 c1x2 wi2 | 4096 | 256 | 901 | extremes_pm127 | True | 0 | 9.31e-04 | 4.96e-04 |
| wi | 12 | 256x128 s3 c1x2 wi2 | 4096 | 256 | 901 | extremes_m128 | True | 0 | 1.67e-03 | 2.01e-03 |
| wi | 12 | 256x128 s3 c1x2 wi2 | 4096 | 256 | 4096 | random | True | 0 | 1.66e-03 | 1.94e-03 |
| wi | 12 | 256x128 s3 c1x2 wi2 | 4096 | 256 | 4096 | adversarial_scales | True | 0 | 1.66e-03 | 2.35e-03 |
| wi | 12 | 256x128 s3 c1x2 wi2 | 4096 | 256 | 4096 | extremes_pm127 | True | 0 | 1.40e-03 | 7.26e-04 |
| wi | 12 | 256x128 s3 c1x2 wi2 | 4096 | 256 | 4096 | extremes_m128 | True | 0 | 1.67e-03 | 2.13e-03 |
| wi | 12 | 256x128 s3 c1x2 wi2 | 1024 | 256 | 4096 | random | True | 0 | 1.66e-03 | 2.58e-03 |
| wi | 12 | 256x128 s3 c1x2 wi2 | 1024 | 256 | 4096 | adversarial_scales | True | 0 | 1.66e-03 | 2.28e-03 |
| wi | 12 | 256x128 s3 c1x2 wi2 | 1024 | 256 | 4096 | extremes_pm127 | True | 0 | 1.10e-03 | 9.91e-04 |
| wi | 12 | 256x128 s3 c1x2 wi2 | 1024 | 256 | 4096 | extremes_m128 | True | 0 | 1.70e-03 | 2.53e-03 |
| wi | 12 | 256x128 s3 c1x2 wi2 | 1024 | 384 | 901 | random | True | 0 | 1.66e-03 | 2.35e-03 |
| wi | 12 | 256x128 s3 c1x2 wi2 | 1024 | 384 | 901 | adversarial_scales | True | 0 | 1.66e-03 | 2.18e-03 |
| wi | 12 | 256x128 s3 c1x2 wi2 | 1024 | 384 | 901 | extremes_pm127 | True | 0 | 4.25e-04 | 6.02e-04 |
| wi | 12 | 256x128 s3 c1x2 wi2 | 1024 | 384 | 901 | extremes_m128 | True | 0 | 1.66e-03 | 2.84e-03 |
| wi | 6 | 256x128 s3 c1x2 N4096K4096 wi2 | 4096 | 4096 | 4096 | random | True | 0 | 1.66e-03 | 3.18e-03 |
| wi | 10 | 256x128 s3 c1x1 N4096K4096 wi2 | 4096 | 4096 | 4096 | random | True | 0 | 1.66e-03 | 3.18e-03 |
| wi | 11 | 256x128 s3 c1x2 N4096K4096 wi1 | 4096 | 4096 | 4096 | random | True | 0 | 1.66e-03 | 3.18e-03 |
| wi | 12 | 256x128 s3 c1x2 wi2 | 4096 | 4096 | 4096 | random | True | 0 | 1.66e-03 | 3.18e-03 |
| wi | 6 | 256x128 s3 c1x2 N4096K4096 wi2 | 4096 | 4096 | 4096 | adversarial_scales | True | 0 | 1.66e-03 | 2.55e-03 |
| wi | 10 | 256x128 s3 c1x1 N4096K4096 wi2 | 4096 | 4096 | 4096 | adversarial_scales | True | 0 | 1.66e-03 | 2.55e-03 |
| wi | 11 | 256x128 s3 c1x2 N4096K4096 wi1 | 4096 | 4096 | 4096 | adversarial_scales | True | 0 | 1.66e-03 | 2.55e-03 |
| wi | 12 | 256x128 s3 c1x2 wi2 | 4096 | 4096 | 4096 | adversarial_scales | True | 0 | 1.66e-03 | 2.55e-03 |
| wi | 6 | 256x128 s3 c1x2 N4096K4096 wi2 | 4096 | 4096 | 4096 | extremes_pm127 | True | 0 | 1.67e-03 | 2.15e-03 |
| wi | 10 | 256x128 s3 c1x1 N4096K4096 wi2 | 4096 | 4096 | 4096 | extremes_pm127 | True | 0 | 1.67e-03 | 2.15e-03 |
| wi | 11 | 256x128 s3 c1x2 N4096K4096 wi1 | 4096 | 4096 | 4096 | extremes_pm127 | True | 0 | 1.67e-03 | 2.15e-03 |
| wi | 12 | 256x128 s3 c1x2 wi2 | 4096 | 4096 | 4096 | extremes_pm127 | True | 0 | 1.67e-03 | 2.15e-03 |
| wi | 6 | 256x128 s3 c1x2 N4096K4096 wi2 | 4096 | 4096 | 4096 | extremes_m128 | True | 0 | 1.67e-03 | 3.60e-03 |
| wi | 10 | 256x128 s3 c1x1 N4096K4096 wi2 | 4096 | 4096 | 4096 | extremes_m128 | True | 0 | 1.67e-03 | 3.60e-03 |
| wi | 11 | 256x128 s3 c1x2 N4096K4096 wi1 | 4096 | 4096 | 4096 | extremes_m128 | True | 0 | 1.67e-03 | 3.60e-03 |
| wi | 12 | 256x128 s3 c1x2 wi2 | 4096 | 4096 | 4096 | extremes_m128 | True | 0 | 1.67e-03 | 3.60e-03 |
| wi | 8 | 256x128 s3 c1x2 N12288K4096 wi2 | 12288 | 4096 | 4096 | random | True | 0 | 1.66e-03 | 2.99e-03 |
| wi | 12 | 256x128 s3 c1x2 wi2 | 12288 | 4096 | 4096 | random | True | 0 | 1.66e-03 | 2.99e-03 |
| wi | 8 | 256x128 s3 c1x2 N12288K4096 wi2 | 12288 | 4096 | 4096 | adversarial_scales | True | 0 | 1.66e-03 | 2.38e-03 |
| wi | 12 | 256x128 s3 c1x2 wi2 | 12288 | 4096 | 4096 | adversarial_scales | True | 0 | 1.66e-03 | 2.38e-03 |
| wi | 8 | 256x128 s3 c1x2 N12288K4096 wi2 | 12288 | 4096 | 4096 | extremes_pm127 | True | 0 | 1.62e-03 | 2.32e-03 |
| wi | 12 | 256x128 s3 c1x2 wi2 | 12288 | 4096 | 4096 | extremes_pm127 | True | 0 | 1.62e-03 | 2.32e-03 |
| wi | 8 | 256x128 s3 c1x2 N12288K4096 wi2 | 12288 | 4096 | 4096 | extremes_m128 | True | 0 | 1.69e-03 | 3.69e-03 |
| wi | 12 | 256x128 s3 c1x2 wi2 | 12288 | 4096 | 4096 | extremes_m128 | True | 0 | 1.69e-03 | 3.69e-03 |

## Verdict (verifier)

**Exactness.** Both variants are exact. Implementer B's double-buffered mainloop (main extension cfg 19-29, including the spilling
cfgs 22/28) is bit-identical (`torch.equal`, 0 differing elements) to the pre-existing default kernel on 164/164 checks, implementer A's
wave-interleave kernels (wi cfg 6-12, incl. the literal two-wave wi1) on 68/68; the default kernel itself equals `kernels/cutlass_int8_bw`
on all 88 input sets (random / adversarial 1e-4-vs-1.0 scales / +-127 / -128 mixes, NaN-filled SFA padding at M=901, K=128/256/384),
rel-L2 vs the fp64 reference 4e-4..1.8e-3 (bf16 output floor), no non-finite outputs. Mainloop review: in the DB kernel every stage is
released (`empty_barrier_arrive_at(prev_stage)`) only after the `wgmma.wait_group 1` that retires that stage's (single) group; the
buffer being re-issued was retired by the previous wait; each 8-k-block region ends in `wait_group 0`, so no group is in flight across a
loop back-edge. In wi2 a stage is released after the wait that retires its 4th quarter-group, i.e. after all four groups that read it
completed. No stage-release race and no read of an in-flight accumulator in either kernel.

**Speed.** B's DB kernels are the new best exact INT8 g128 kernel on 9/12 shapes: +7% at (4096,4096) M=901/4096 (853 / 1140 TFLOPS,
`128x128 s5 c1x1 DB`, vs 798 / 1061), +12% at (1024,4096,901) (477 vs 426, `64x128 s8 c1x1 DB`, now above the pp64x128 tuner's 463
and 1.25x DeepGEMM FP8), +2..+4% at N=12288 and (1024,4096) M>=4096, and a tie (-1.3..+0.5%) at K=12288 and at M=42240 where the
runs are power-throttled (DB SM clocks 1395-1545 MHz vs 1700-1800 for the old kernel: the DB pipeline draws more power per unit
time). A's wi2 is EXACT BUT 13-22% SLOWER than its own reference instantiation on every shape (e.g. 885 vs 1063 at 4096^3), wi1
(240 B spills) 42% slower: ptxas emits C7514 for all 7 wi kernels and the SASS shows every `IGMMA.64x64x32` followed by
`WARPGROUP.DEPBAR.LE gsb0, 0x0` (fully serialized wgmma, 32 ARRIVE/DEPBAR pairs per k-block instead of 8) -- the in-flight quarter-group
carried across the `while` back-edge defeats ptxas' pipeline analysis, whereas B's straight-line 8-block regions (7x `DEPBAR.LE 0x1`,
1x `0x0`, 0 C7514, 0 spills, 168/232 regs) are accepted.

**Did the overlap close the gap?** Only partially. Best exact INT8 g128 vs DeepGEMM FP8 g128 moved from 0.80-0.91x to 0.81-0.90x
per shape (4096^3: 0.81 -> 0.88; (12288,4096,901): 0.81 -> 0.84; K=12288 shapes unchanged at 0.81-0.89) and vs per-tensor CUTLASS INT8
it stays at 0.74-0.87x (per-tensor INT8 itself runs at 1270-1520). Why: the register-resident double buffer only fits with one 64-row
wave per math warpgroup (2 x 64 int32 + 64 fp32 finals = 192 of 232 regs), i.e. BLOCK_M <= 128, and the 128x128 tile has a lower
pipeline ceiling than the 256x128 tile the old kernel uses -- the no-convert experiment (DGINT8_PROMOTE_MODE=2, wrong numerics) reaches
1254-1298 TFLOPS with the 128x128 DB pipeline vs 1405-1525 with 256x128 s3 c1x2 (results/deepgemm_int8_db_pm2_noconvert.json). The
overlap does hide most of the promotion: the exact DB kernel reaches 88% of its own no-convert ceiling (1140/1298) where the old kernel
reached 76% (1061/1405); the remaining 12% is the one exposed promotion + `wait_group 0` drain per 8-block region plus 64 I2FP + 64
FFMA per thread per k-block competing for issue slots with the 4 IGMMAs. A 256x128 double buffer is register-infeasible
(2 waves x 2 x 64 + 128 finals = 384 regs; A's wi1 attempt already spills at 2 x 64 + 128), so with exact fp32 promotion on the CUDA
cores the exact INT8 g128 kernel plateaus at ~1100-1140 TFLOPS, ~10-15% below FP8 g128 (whose scaling is a tensor-core-side fp32 FMA
of the fp32 accumulator, not an I2FP) and 15-25% below per-tensor INT8. Recommended production picks: cfg 21 for (4096,4096) M<=4096,
cfg 27 for (1024,4096,901), cfg 23/25 for N=12288 / (1024,4096) M>=4096, otherwise the old 256x128 s3 c1x2 (ties).
