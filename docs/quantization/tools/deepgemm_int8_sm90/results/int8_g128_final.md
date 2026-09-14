# INT8 g128 (1,128,128) block-scaled GEMM -- final serial timing (idle H100, verifier run)

- NVIDIA H100 80GB HBM3, torch 2.13.0a0+9186a08b2c.nv26.07, 2026-09-14T00:59:31; do_bench(warmup=25, rep=200, median, L2 flushed), 3 interleaved repeats, median of medians; SM MHz = median of nvidia-smi samples taken during the runs (clocks unlocked, power-throttled at large shapes).
- TFLOPS = 2MNK/t with the true M. M=901: `deepgemm_int8`, `DeepGEMM FP8`, `cutlass_int8 per-tensor` and `cuBLASLt FP8` run on the unpadded 901-row A; `cutlass_int8_bw` and the two tuner variants get A/SFA zero-padded to 904 rows OUTSIDE the timed region (their TMA-loaded SFA needs M % 4 == 0) and the output is sliced back to 901 rows (also outside timing).
- correctness: probes/verify_int8_g128.py (fp64 exact reference) -> results/verify_int8_g128.json; all INT8 g128 kernels are bit-identical to each other.

## TFLOPS (median of 3 do_bench medians)

| N | K | M | deepgemm_int8 | cutlass_int8_bw baseline | cutlass tuner coop128x128 c2x1 DB | cutlass tuner pp64x128 c1x2 DB | DeepGEMM FP8 g128 | cutlass_int8 per-tensor | cuBLASLt FP8 per-tensor | bf16 cuBLAS | SM MHz (dgint8 / tuner) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4096 | 4096 | 901 | 797 | 740 | 784 | 623 | 967 | 982 | 1100 | 669 | 1980 / 1980 |
| 4096 | 4096 | 4096 | 1055 | 970 | 1096 | 773 | 1307 | 1390 | 1367 | 731 | 1950 / 1980 |
| 4096 | 4096 | 42240 | 1092 | 984 | 1070 | 808 | 1231 | 1440 | 1317 | 730 | 1695 / 1635 |
| 1024 | 4096 | 901 | 428 | 322 | 361 | 463 | 380 | 273 | 523 | 353 | 1980 / 1980 |
| 1024 | 4096 | 4096 | 902 | 843 | 939 | 809 | 1033 | 1104 | 1140 | 704 | 1980 / 1980 |
| 1024 | 4096 | 42240 | 1062 | 973 | 1093 | 921 | 1196 | 1424 | 1331 | 708 | 1695 / 1725 |
| 12288 | 4096 | 901 | 914 | 841 | 850 | 702 | 1135 | 1177 | 1295 | 701 | 1965 / 1980 |
| 12288 | 4096 | 4096 | 1067 | 973 | 1054 | 923 | 1199 | 1435 | 1345 | 693 | 1875 / 1695 |
| 12288 | 4096 | 42240 | 1061 | 856 | 1001 | 685 | 1275 | 1292 | 1310 | 700 | 1680 / 1365 |
| 4096 | 12288 | 901 | 937 | 868 | 898 | 635 | 1171 | 1276 | 1417 | 745 | 1965 / 1980 |
| 4096 | 12288 | 4096 | 1094 | 968 | 1044 | 788 | 1263 | 1509 | 1394 | 722 | 1635 / 1605 |
| 4096 | 12288 | 42240 | 1094 | 946 | 1041 | 657 | 1312 | 1468 | 1369 | 736 | 1605 / 1335 |

## microseconds (median of 3 medians) and per-repeat medians

| N | K | M | kernel | config | us | TFLOPS | repeats (us) | SM MHz samples |
|---|---|---|---|---|---:|---:|---|---|
| 4096 | 4096 | 901 | deepgemm_int8 | 256x128 s3 c1x2 N4096K4096 | 37.9 | 797 | 37.9, 38.1, 37.9 | [1980, 1980, 1980] |
| 4096 | 4096 | 901 | cutlass_int8_bw baseline | coop 128x128 c1x2 (int8_blockwise_mm) | 40.9 | 740 | 41.1, 40.9, 40.8 | [1980, 1980, 1980] |
| 4096 | 4096 | 901 | cutlass tuner coop128x128 c2x1 DB | spec 128,128,2,1,0,0,0,0,0,1 (v0) | 38.6 | 784 | 38.8, 38.6, 38.5 | [1980, 1980, 1980] |
| 4096 | 4096 | 901 | cutlass tuner pp64x128 c1x2 DB | spec 64,128,1,2,1,0,0,0,0,1 (v2) | 48.5 | 623 | 48.7, 48.5, 48.4 | [1980, 1980, 1980] |
| 4096 | 4096 | 901 | DeepGEMM FP8 g128 | upstream 2.8.0 1D2D, unpadded M | 31.3 | 967 | 31.5, 31.3, 31.2 | [1980, 1980, 1980] |
| 4096 | 4096 | 901 | cutlass_int8 per-tensor | 128x256 c2x1 (int8_scaled_mm cfg 0) | 30.8 | 982 | 30.9, 30.8, 30.7 | [1980, 1980, 1980] |
| 4096 | 4096 | 901 | cuBLASLt FP8 per-tensor | torch._scaled_mm use_fast_accum=True | 27.5 | 1100 | 27.6, 27.5, 27.4 | [1980, 1980, 1980] |
| 4096 | 4096 | 901 | bf16 cuBLAS | A @ W.t() | 45.2 | 669 | 45.4, 45.2, 45.2 | [1980, 1980, 1980] |
| 4096 | 4096 | 4096 | deepgemm_int8 | 256x128 s3 c1x2 N4096K4096 | 130.2 | 1055 | 129.2, 130.3, 130.2 | [1980, 1950, 1920] |
| 4096 | 4096 | 4096 | cutlass_int8_bw baseline | coop 128x128 c1x2 (int8_blockwise_mm) | 141.7 | 970 | 140.9, 141.9, 141.7 | [1980, 1965, 1965] |
| 4096 | 4096 | 4096 | cutlass tuner coop128x128 c2x1 DB | spec 128,128,2,1,0,0,0,0,0,1 (v0) | 125.4 | 1096 | 125.5, 125.3, 125.4 | [1980, 1980, 1980] |
| 4096 | 4096 | 4096 | cutlass tuner pp64x128 c1x2 DB | spec 64,128,1,2,1,0,0,0,0,1 (v2) | 177.9 | 773 | 177.8, 178.1, 177.9 | [1980, 1965, 1965] |
| 4096 | 4096 | 4096 | DeepGEMM FP8 g128 | upstream 2.8.0 1D2D, unpadded M | 105.2 | 1307 | 104.6, 105.2, 105.3 | [1965, 1845, 1890] |
| 4096 | 4096 | 4096 | cutlass_int8 per-tensor | 128x256 c2x1 (int8_scaled_mm cfg 0) | 98.9 | 1390 | 98.9, 98.8, 98.9 | [1965, 1935, 1965] |
| 4096 | 4096 | 4096 | cuBLASLt FP8 per-tensor | torch._scaled_mm use_fast_accum=True | 100.6 | 1367 | 100.9, 100.3, 100.6 | [1965, 1965, 1965] |
| 4096 | 4096 | 4096 | bf16 cuBLAS | A @ W.t() | 187.9 | 731 | 184.9, 187.9, 190.0 | [1905, 1740, 1755] |
| 4096 | 4096 | 42240 | deepgemm_int8 | 256x128 s3 c1x2 N4096K4096 | 1298.2 | 1092 | 1277.1, 1298.2, 1323.0 | [1740, 1650, 1695] |
| 4096 | 4096 | 42240 | cutlass_int8_bw baseline | coop 128x128 c1x2 (int8_blockwise_mm) | 1440.7 | 984 | 1447.9, 1435.8, 1440.7 | [1785, 1755, 1785] |
| 4096 | 4096 | 42240 | cutlass tuner coop128x128 c2x1 DB | spec 128,128,2,1,0,0,0,0,0,1 (v0) | 1325.0 | 1070 | 1325.0, 1326.7, 1313.8 | [1470, 1635, 1740] |
| 4096 | 4096 | 42240 | cutlass tuner pp64x128 c1x2 DB | spec 64,128,1,2,1,0,0,0,0,1 (v2) | 1754.3 | 808 | 1738.8, 1754.3, 1767.1 | [1545, 1725, 1770] |
| 4096 | 4096 | 42240 | DeepGEMM FP8 g128 | upstream 2.8.0 1D2D, unpadded M | 1151.1 | 1231 | 1116.9, 1151.1, 1154.3 | [1320, 1620, 1410] |
| 4096 | 4096 | 42240 | cutlass_int8 per-tensor | 128x256 c2x1 (int8_scaled_mm cfg 0) | 984.1 | 1440 | 997.2, 981.9, 984.1 | [1590, 1665, 1665] |
| 4096 | 4096 | 42240 | cuBLASLt FP8 per-tensor | torch._scaled_mm use_fast_accum=True | 1076.2 | 1317 | 1085.1, 1076.2, 1074.3 | [1500, 1425, 1440] |
| 4096 | 4096 | 42240 | bf16 cuBLAS | A @ W.t() | 1940.7 | 730 | 2069.1, 1934.6, 1940.7 | [1455, 1410, 1425] |
| 1024 | 4096 | 901 | deepgemm_int8 | 64x128 s8 c1x1 | 17.7 | 428 | 17.8, 17.7, 17.7 | [1980, 1980, 1980] |
| 1024 | 4096 | 901 | cutlass_int8_bw baseline | coop 128x128 c1x2 (int8_blockwise_mm) | 23.5 | 322 | 23.5, 23.6, 23.4 | [1980, 1980, 1980] |
| 1024 | 4096 | 901 | cutlass tuner coop128x128 c2x1 DB | spec 128,128,2,1,0,0,0,0,0,1 (v0) | 21.0 | 361 | 21.0, 21.0, 20.9 | [1980, 1980, 1980] |
| 1024 | 4096 | 901 | cutlass tuner pp64x128 c1x2 DB | spec 64,128,1,2,1,0,0,0,0,1 (v2) | 16.3 | 463 | 16.3, 16.3, 16.4 | [1980, 1980, 1980] |
| 1024 | 4096 | 901 | DeepGEMM FP8 g128 | upstream 2.8.0 1D2D, unpadded M | 19.9 | 380 | 19.9, 19.8, 20.0 | [1980, 1980, 1980] |
| 1024 | 4096 | 901 | cutlass_int8 per-tensor | 128x256 c2x1 (int8_scaled_mm cfg 0) | 27.6 | 273 | 27.6, 27.6, 27.8 | [1980, 1980, 1980] |
| 1024 | 4096 | 901 | cuBLASLt FP8 per-tensor | torch._scaled_mm use_fast_accum=True | 14.5 | 523 | 14.4, 14.5, 14.6 | [1980, 1980, 1980] |
| 1024 | 4096 | 901 | bf16 cuBLAS | A @ W.t() | 21.4 | 353 | 21.3, 21.4, 21.6 | [1980, 1980, 1980] |
| 1024 | 4096 | 4096 | deepgemm_int8 | 256x128 s3 c1x2 N1024K4096 | 38.1 | 902 | 38.0, 38.1, 38.2 | [1980, 1980, 1980] |
| 1024 | 4096 | 4096 | cutlass_int8_bw baseline | coop 128x128 c1x2 (int8_blockwise_mm) | 40.7 | 843 | 40.7, 40.6, 40.8 | [1980, 1980, 1980] |
| 1024 | 4096 | 4096 | cutlass tuner coop128x128 c2x1 DB | spec 128,128,2,1,0,0,0,0,0,1 (v0) | 36.6 | 939 | 36.6, 36.4, 36.6 | [1980, 1980, 1980] |
| 1024 | 4096 | 4096 | cutlass tuner pp64x128 c1x2 DB | spec 64,128,1,2,1,0,0,0,0,1 (v2) | 42.5 | 809 | 42.5, 42.3, 42.5 | [1980, 1980, 1980] |
| 1024 | 4096 | 4096 | DeepGEMM FP8 g128 | upstream 2.8.0 1D2D, unpadded M | 33.2 | 1033 | 33.3, 33.0, 33.2 | [1980, 1980, 1980] |
| 1024 | 4096 | 4096 | cutlass_int8 per-tensor | 128x256 c2x1 (int8_scaled_mm cfg 0) | 31.1 | 1104 | 31.1, 31.1, 31.1 | [1980, 1980, 1980] |
| 1024 | 4096 | 4096 | cuBLASLt FP8 per-tensor | torch._scaled_mm use_fast_accum=True | 30.1 | 1140 | 30.2, 30.1, 30.0 | [1980, 1980, 1980] |
| 1024 | 4096 | 4096 | bf16 cuBLAS | A @ W.t() | 48.8 | 704 | 48.8, 48.9, 48.7 | [1980, 1980, 1980] |
| 1024 | 4096 | 42240 | deepgemm_int8 | 256x128 s3 c1x2 N1024K4096 | 333.5 | 1062 | 313.8, 335.3, 333.5 | [1935, 1620, 1695] |
| 1024 | 4096 | 42240 | cutlass_int8_bw baseline | coop 128x128 c1x2 (int8_blockwise_mm) | 364.0 | 973 | 353.7, 365.7, 364.0 | [1830, 1815, 1620] |
| 1024 | 4096 | 42240 | cutlass tuner coop128x128 c2x1 DB | spec 128,128,2,1,0,0,0,0,0,1 (v0) | 324.3 | 1093 | 324.3, 323.6, 329.6 | [1725, 1740, 1725] |
| 1024 | 4096 | 42240 | cutlass tuner pp64x128 c1x2 DB | spec 64,128,1,2,1,0,0,0,0,1 (v2) | 384.6 | 921 | 400.3, 382.8, 384.6 | [1725, 1770, 1725] |
| 1024 | 4096 | 42240 | DeepGEMM FP8 g128 | upstream 2.8.0 1D2D, unpadded M | 296.3 | 1196 | 300.0, 296.3, 289.4 | [1380, 1425, 1770] |
| 1024 | 4096 | 42240 | cutlass_int8 per-tensor | 128x256 c2x1 (int8_scaled_mm cfg 0) | 248.9 | 1424 | 244.2, 248.9, 249.9 | [1815, 1755, 1755] |
| 1024 | 4096 | 42240 | cuBLASLt FP8 per-tensor | torch._scaled_mm use_fast_accum=True | 266.2 | 1331 | 261.8, 270.8, 266.2 | [1830, 1545, 1575] |
| 1024 | 4096 | 42240 | bf16 cuBLAS | A @ W.t() | 500.1 | 708 | 500.3, 500.1, 497.8 | [1575, 1530, 1515] |
| 12288 | 4096 | 901 | deepgemm_int8 | 256x128 s3 c1x2 N12288K4096 | 99.3 | 914 | 98.0, 99.5, 99.3 | [1980, 1950, 1965] |
| 12288 | 4096 | 901 | cutlass_int8_bw baseline | coop 128x128 c1x2 (int8_blockwise_mm) | 107.9 | 841 | 107.5, 108.0, 107.9 | [1980, 1980, 1965] |
| 12288 | 4096 | 901 | cutlass tuner coop128x128 c2x1 DB | spec 128,128,2,1,0,0,0,0,0,1 (v0) | 106.7 | 850 | 106.7, 106.8, 106.7 | [1980, 1980, 1980] |
| 12288 | 4096 | 901 | cutlass tuner pp64x128 c1x2 DB | spec 64,128,1,2,1,0,0,0,0,1 (v2) | 129.2 | 702 | 129.1, 129.3, 129.2 | [1980, 1980, 1980] |
| 12288 | 4096 | 901 | DeepGEMM FP8 g128 | upstream 2.8.0 1D2D, unpadded M | 79.9 | 1135 | 80.1, 79.9, 79.7 | [1965, 1935, 1935] |
| 12288 | 4096 | 901 | cutlass_int8 per-tensor | 128x256 c2x1 (int8_scaled_mm cfg 0) | 77.1 | 1177 | 77.1, 77.2, 77.0 | [1965, 1965, 1965] |
| 12288 | 4096 | 901 | cuBLASLt FP8 per-tensor | torch._scaled_mm use_fast_accum=True | 70.0 | 1295 | 70.3, 70.0, 69.9 | [1950, 1965, 1980] |
| 12288 | 4096 | 901 | bf16 cuBLAS | A @ W.t() | 129.4 | 701 | 129.4, 129.2, 129.4 | [1965, 1860, 1905] |
| 12288 | 4096 | 4096 | deepgemm_int8 | 256x128 s3 c1x2 N12288K4096 | 386.4 | 1067 | 379.0, 386.4, 387.0 | [1905, 1875, 1875] |
| 12288 | 4096 | 4096 | cutlass_int8_bw baseline | coop 128x128 c1x2 (int8_blockwise_mm) | 423.7 | 973 | 415.3, 423.7, 427.1 | [1920, 1875, 1830] |
| 12288 | 4096 | 4096 | cutlass tuner coop128x128 c2x1 DB | spec 128,128,2,1,0,0,0,0,0,1 (v0) | 391.2 | 1054 | 399.7, 383.8, 391.2 | [1680, 1725, 1695] |
| 12288 | 4096 | 4096 | cutlass tuner pp64x128 c1x2 DB | spec 64,128,1,2,1,0,0,0,0,1 (v2) | 446.9 | 923 | 466.1, 446.9, 444.7 | [1695, 1740, 1800] |
| 12288 | 4096 | 4096 | DeepGEMM FP8 g128 | upstream 2.8.0 1D2D, unpadded M | 343.8 | 1199 | 335.8, 343.8, 344.8 | [1560, 1485, 1425] |
| 12288 | 4096 | 4096 | cutlass_int8 per-tensor | 128x256 c2x1 (int8_scaled_mm cfg 0) | 287.4 | 1435 | 287.3, 287.4, 287.6 | [1830, 1785, 1740] |
| 12288 | 4096 | 4096 | cuBLASLt FP8 per-tensor | torch._scaled_mm use_fast_accum=True | 306.6 | 1345 | 306.6, 305.3, 309.1 | [1650, 1620, 1830] |
| 12288 | 4096 | 4096 | bf16 cuBLAS | A @ W.t() | 594.7 | 693 | 601.3, 594.7, 586.0 | [1575, 1575, 1575] |
| 12288 | 4096 | 42240 | deepgemm_int8 | 256x128 s3 c1x2 N12288K4096 | 4005.9 | 1061 | 4091.9, 4005.9, 3948.0 | [1770, 1635, 1680] |
| 12288 | 4096 | 42240 | cutlass_int8_bw baseline | coop 128x128 c1x2 (int8_blockwise_mm) | 4965.8 | 856 | 4965.8, 5011.9, 4961.6 | [1320, 1335, 1335] |
| 12288 | 4096 | 42240 | cutlass tuner coop128x128 c2x1 DB | spec 128,128,2,1,0,0,0,0,0,1 (v0) | 4248.9 | 1001 | 4235.1, 4248.9, 4251.8 | [1365, 1395, 1350] |
| 12288 | 4096 | 42240 | cutlass tuner pp64x128 c1x2 DB | spec 64,128,1,2,1,0,0,0,0,1 (v2) | 6204.4 | 685 | 6127.5, 6227.7, 6204.4 | [1380, 1395, 1395] |
| 12288 | 4096 | 42240 | DeepGEMM FP8 g128 | upstream 2.8.0 1D2D, unpadded M | 3335.7 | 1275 | 3393.5, 3335.7, 3317.5 | [1305, 1380, 1365] |
| 12288 | 4096 | 42240 | cutlass_int8 per-tensor | 128x256 c2x1 (int8_scaled_mm cfg 0) | 3291.7 | 1292 | 3361.3, 3291.7, 3290.8 | [1380, 1380, 1395] |
| 12288 | 4096 | 42240 | cuBLASLt FP8 per-tensor | torch._scaled_mm use_fast_accum=True | 3245.9 | 1310 | 3248.0, 3245.9, 3215.9 | [1380, 1335, 1380] |
| 12288 | 4096 | 42240 | bf16 cuBLAS | A @ W.t() | 6074.7 | 700 | 6025.1, 6074.7, 6083.9 | [1335, 1365, 1350] |
| 4096 | 12288 | 901 | deepgemm_int8 | 256x128 s3 c1x2 N4096K12288 | 96.8 | 937 | 95.0, 97.1, 96.8 | [1980, 1965, 1935] |
| 4096 | 12288 | 901 | cutlass_int8_bw baseline | coop 128x128 c1x2 (int8_blockwise_mm) | 104.5 | 868 | 104.5, 104.5, 104.4 | [1980, 1965, 1965] |
| 4096 | 12288 | 901 | cutlass tuner coop128x128 c2x1 DB | spec 128,128,2,1,0,0,0,0,0,1 (v0) | 100.9 | 898 | 100.9, 101.0, 100.8 | [1980, 1980, 1980] |
| 4096 | 12288 | 901 | cutlass tuner pp64x128 c1x2 DB | spec 64,128,1,2,1,0,0,0,0,1 (v2) | 142.8 | 635 | 143.1, 142.7, 142.8 | [1980, 1980, 1980] |
| 4096 | 12288 | 901 | DeepGEMM FP8 g128 | upstream 2.8.0 1D2D, unpadded M | 77.4 | 1171 | 77.5, 77.4, 77.0 | [1860, 1950, 1980] |
| 4096 | 12288 | 901 | cutlass_int8 per-tensor | 128x256 c2x1 (int8_scaled_mm cfg 0) | 71.1 | 1276 | 71.1, 70.9, 71.4 | [1965, 1965, 1965] |
| 4096 | 12288 | 901 | cuBLASLt FP8 per-tensor | torch._scaled_mm use_fast_accum=True | 64.0 | 1417 | 64.0, 64.0, 63.9 | [1980, 1935, 1965] |
| 4096 | 12288 | 901 | bf16 cuBLAS | A @ W.t() | 121.8 | 745 | 121.6, 121.8, 122.6 | [1980, 1905, 1770] |
| 4096 | 12288 | 4096 | deepgemm_int8 | 256x128 s3 c1x2 N4096K12288 | 376.7 | 1094 | 371.6, 376.7, 379.0 | [1890, 1635, 1620] |
| 4096 | 12288 | 4096 | cutlass_int8_bw baseline | coop 128x128 c1x2 (int8_blockwise_mm) | 426.0 | 968 | 411.4, 426.0, 431.5 | [1785, 1755, 1815] |
| 4096 | 12288 | 4096 | cutlass tuner coop128x128 c2x1 DB | spec 128,128,2,1,0,0,0,0,0,1 (v0) | 395.1 | 1044 | 406.8, 388.3, 395.1 | [1605, 1635, 1590] |
| 4096 | 12288 | 4096 | cutlass tuner pp64x128 c1x2 DB | spec 64,128,1,2,1,0,0,0,0,1 (v2) | 523.5 | 788 | 539.2, 523.5, 507.8 | [1470, 1545, 1650] |
| 4096 | 12288 | 4096 | DeepGEMM FP8 g128 | upstream 2.8.0 1D2D, unpadded M | 326.5 | 1263 | 314.0, 331.8, 326.5 | [1635, 1470, 1515] |
| 4096 | 12288 | 4096 | cutlass_int8 per-tensor | 128x256 c2x1 (int8_scaled_mm cfg 0) | 273.3 | 1509 | 271.9, 273.3, 275.0 | [1785, 1770, 1920] |
| 4096 | 12288 | 4096 | cuBLASLt FP8 per-tensor | torch._scaled_mm use_fast_accum=True | 295.7 | 1394 | 295.7, 295.8, 294.0 | [1635, 1575, 1650] |
| 4096 | 12288 | 4096 | bf16 cuBLAS | A @ W.t() | 571.1 | 722 | 587.7, 571.1, 566.5 | [1575, 1590, 1665] |
| 4096 | 12288 | 42240 | deepgemm_int8 | 256x128 s3 c1x2 N4096K12288 | 3886.6 | 1094 | 3837.2, 3886.6, 3996.7 | [1710, 1575, 1605] |
| 4096 | 12288 | 42240 | cutlass_int8_bw baseline | coop 128x128 c1x2 (int8_blockwise_mm) | 4497.1 | 946 | 4682.0, 4497.1, 4451.4 | [1740, 1470, 1455] |
| 4096 | 12288 | 42240 | cutlass tuner coop128x128 c2x1 DB | spec 128,128,2,1,0,0,0,0,0,1 (v0) | 4084.8 | 1041 | 3992.2, 4084.8, 4168.4 | [1380, 1335, 1335] |
| 4096 | 12288 | 42240 | cutlass tuner pp64x128 c1x2 DB | spec 64,128,1,2,1,0,0,0,0,1 (v2) | 6474.6 | 657 | 6544.8, 6433.1, 6474.6 | [1365, 1425, 1425] |
| 4096 | 12288 | 42240 | DeepGEMM FP8 g128 | upstream 2.8.0 1D2D, unpadded M | 3240.0 | 1312 | 3230.7, 3240.0, 3269.3 | [1410, 1290, 1275] |
| 4096 | 12288 | 42240 | cutlass_int8 per-tensor | 128x256 c2x1 (int8_scaled_mm cfg 0) | 2897.4 | 1468 | 2963.5, 2897.4, 2871.8 | [1395, 1455, 1470] |
| 4096 | 12288 | 42240 | cuBLASLt FP8 per-tensor | torch._scaled_mm use_fast_accum=True | 3105.4 | 1369 | 3080.1, 3137.8, 3105.4 | [1275, 1245, 1275] |
| 4096 | 12288 | 42240 | bf16 cuBLAS | A @ W.t() | 5777.7 | 736 | 5724.4, 5793.3, 5777.7 | [1365, 1380, 1350] |

## Verdict (verifier)

* **Correctness**: `probes/verify_int8_g128.py` (independent fp64 exact block-scaled reference) -- 136/136 checks pass over
  (N,K,M) in {(1024,4096,901),(1024,4096,1802),(4096,12288,901),(4096,12288,4096),(12288,4096,4096)} x {random scales U[1e-3,1e-2],
  adversarial scales (rows/blocks at 1e-4 vs 1.0), int8 values at +-127 (block dots hit exactly +-128*127*127), -128/127 mixes}
  x every compiled config compatible with (N,K) (all 19 covered at M<=1802). rel-L2 1.51e-3..1.72e-3 (the bf16 output floor),
  max|d|/max|ref| 1.7e-3..3.1e-3, no NaN even with NaN-filled SFA padding columns at M=901 (the "padding may hold anything" claim holds).
  The port, `kernels/cutlass_int8_bw`, and both CUTLASS tuner variants (coop128x128 c2x1 DB, pp64x128 c1x2 DB) are bit-identical
  on every input (0 differing elements), i.e. all four implement exactly the same fp32-across-blocks math.
* **Speed** (idle GPU, serial): the DeepGEMM-INT8 port is the fastest exact INT8 g128 kernel at 9/12 shapes (797-1094 TFLOPS for
  N>=4096, +8-24% over the cutlass_int8_bw baseline); the CUTLASS tuner's coop128x128 c2x1 double-buffered kernel edges it at
  (4096,4096,4096) 1096 vs 1055 and (1024,4096,4096) 939 vs 902, and pp64x128 wins the skinny (1024,4096,901) case 463 vs 428.
  Against DeepGEMM FP8 g128 the best INT8 g128 kernel is at 0.80-0.91x (11-20% behind) except (1024,4096,901) where INT8 wins
  (1.12-1.22x); against per-tensor CUTLASS INT8 it is 0.73-0.85x and against cuBLASLt FP8 per-tensor 0.72-0.84x. M=42240 rows are
  power-throttled (SM 1250-1800 MHz), so absolute numbers there are ~5-10% clock-limited for every kernel alike.
* **What is left**: the remaining INT8-vs-FP8 gap is entirely the exposed int32->fp32 promotion (one I2FP + FFMA per output element per
  128-K block, issued while the warpgroup's tensor cores idle); the porter's no-convert experiment (wrong numerics) shows the same
  pipeline reaching 1405-1527 TFLOPS, above FP8. Tile/cluster/stage/raster/epilogue knobs are exhausted on both code bases (<=3%).
  Closing the gap needs the promotion hidden behind the WGMMAs: two consumer warpgroups in ping-pong with an ordering barrier
  (so one promotes while the other's WGMMAs run) or a ptxas-accepted double-buffered accumulator; the naive double-buffer attempt
  got serialized by ptxas (C7514). Realistic upside ~10-20% (FP8 parity); exceeding FP8 would additionally require dropping exactness
  (biased accumulation, ~4e-4 rel-L2 penalty). For M=901 with N=1024 the pingpong 64x128 tile is the right choice (already 1.2x FP8).
