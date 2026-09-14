# FP8 power-limiter onset (2026-09-14, 120 W mode). cfg3 4096^3, --flush=0 --nw=1 (L2-resident), Gaussian operands, timing started from an idle GPU (--warmup=0 --warmup_ms=0), 4000 back-to-back iterations, per-iteration cudaEvent times.

Tj before 47.9 C; after FP8 run 57.8 C; after INT8 run 56.4 C. GPU clock idles at 315 MHz and ramps within ~20 ms.

| window since start | FP8 mean us -> TFLOPS | INT8 mean us -> TFLOPS |
|---|---:|---:|
| 0-2 ms | 705 -> 195 (n=2) | 1404 -> 98 (n=1) |
| 2-5 ms | 696 -> 197 (n=5) | 1392 -> 99 (n=2) |
| 5-10 ms | 701 -> 196 (n=7) | 1385 -> 99 (n=4) |
| 10-20 ms | 707 -> 194 (n=14) | 1387 -> 99 (n=7) |
| 20-50 ms | 526 -> 261 (n=57) | 436 -> 315 (n=70) |
| 50-100 ms | 505 -> 272 (n=99) | 421 -> 326 (n=118) |
| 100-200 ms | 505 -> 272 (n=198) | 420 -> 327 (n=238) |
| 200-500 ms | 504 -> 273 (n=595) | 422 -> 326 (n=711) |
| 500-1000 ms | 628 -> 219 (n=797) | 421 -> 326 (n=1187) |
| 1000-2000 ms | 605 -> 227 (n=1654) | 409 -> 336 (n=1662) |
| 2000-3000 ms | 604 -> 228 (n=572) |  |

Reading: after the ~20 ms clock ramp FP8 runs at ~272 TFLOPS for ~0.5 s, then steps down to 219-228 TFLOPS and stays there; INT8 runs at 315-336 TFLOPS throughout. A step at ~0.5 s with unchanged GPU/EMC clocks and only +10 C is the signature of a power controller with a ~0.5 s averaging window, not of thermal throttling (seconds) or an instantaneous current limiter (microseconds). Consequence for benchmarking: FP8 numbers from runs shorter than ~0.5 s of sustained load (e.g. 50 back-to-back iterations after a 300 ms warm-up, or any mode with >= 1 ms idle gaps between kernels) are pre-limiter and 20-30 % optimistic versus sustained operation.
