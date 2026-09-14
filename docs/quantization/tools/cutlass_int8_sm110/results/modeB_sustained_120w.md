# Mode B (activations hot, 8 rotating cold weights) re-measured with >= 1.5 s of sustained kernels per case, 120 W mode, cfg3, Gaussian operands, 2026-09-14 08:42

The FP8 power limiter engages ~0.5 s after sustained load starts (see onset section); the 50-iteration sweep rows were partly pre-limiter.

| N x K | M | FP8 TFLOPS (us) | INT8 TFLOPS (us) | INT8/FP8 |
|---|---:|---:|---:|---:|
| 4096x4096 | 901 | 189 (160) | 279 (109) | 1.47 |
| 4096x4096 | 1517 | 204 (250) | 327 (156) | 1.61 |
| 4096x4096 | 1802 | 186 (325) | 292 (207) | 1.57 |
| 4096x4096 | 4096 | 217 (634) | 344 (399) | 1.59 |
| 1024x4096 | 901 | 206 (37) | 205 (37) | 0.99 |
| 1024x4096 | 1517 | 248 (51) | 250 (51) | 1.01 |
| 1024x4096 | 1802 | 231 (65) | 234 (65) | 1.01 |
| 1024x4096 | 4096 | 192 (179) | 308 (112) | 1.61 |
