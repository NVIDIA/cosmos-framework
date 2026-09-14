# 1024x4096 with 32 rotating weight copies (128 MB >> L2) so weights are certainly cold; --flush=0 --warmup_ms=1500, cfg3, Gaussian, 120 W, 08:48

| M | FP8 TFLOPS (us) | INT8 TFLOPS (us) | INT8/FP8 |
|---:|---:|---:|---:|
| 901 | 205 (37) | 205 (37) | 1.00 |
| 1517 | 249 (51) | 249 (51) | 1.00 |
| 1802 | 232 (65) | 232 (65) | 1.00 |
| 4096 | 191 (180) | 303 (114) | 1.58 |
