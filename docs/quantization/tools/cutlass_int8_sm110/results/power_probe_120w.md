# FP8 vs INT8 probe, power mode: 120W, GPU max_freq 1386 MHz, 2026-09-14 08:26

cfg3 (2SM 256x256x128 cluster2x1, swizzle auto) for both dtypes. warm = --flush=0 --nw=1 (L2-resident), hotcold = --flush=0 --nw=8 (activations hot, 8 rotating weights), cold = --flush=1. normal = per-tensor-quantized Gaussian operands.

| case | dtype | TFLOPS | median us | gpu clk med/min MHz | emc clk med/min MHz | Tj end C |
|---|---|---:|---:|---:|---:|---:|
| warm 4096^3 normal 3000it | fp8 | 216.9 | 633.66 | 1386/0 | 4266/4266 | 50.1 |
| warm 4096^3 normal 3000it | int8 | 343.4 | 400.22 | 1386/1332 | 4266/4266 | 50.9 |
| warm 4096^3 zeros 3000it | fp8 | 377.8 | 363.78 | 1386/1314 | 4266/4266 | 49.7 |
| warm 4096^3 zeros 3000it | int8 | 381.3 | 360.45 | 1386/1323 | 4266/4266 | 49.8 |
| warm 4096^3 normal 3000it (repeat) | fp8 | 217.2 | 632.90 | 1386/1332 | 4266/4266 | 54.1 |
| warm 4096^3 normal 3000it (repeat) | int8 | 344.3 | 399.20 | 1386/1332 | 4266/4266 | 54.6 |
| hotcold 4096x4096 M=901 normal 200it | fp8 | 213.2 | 141.79 | 1386/1323 | 4266/4266 | 54.5 |
| hotcold 4096x4096 M=901 normal 200it | int8 | 273.6 | 110.50 | 1386/1332 | 4266/4266 | 53.9 |
| hotcold 4096x4096 M=1517 normal 200it | fp8 | 237.8 | 214.05 | 1386/1305 | 4266/4266 | 55.2 |
| hotcold 4096x4096 M=1517 normal 200it | int8 | 327.0 | 155.68 | 1386/1332 | 4266/4266 | 54.5 |
| hotcold 4096x4096 M=1802 normal 200it | fp8 | 223.3 | 270.80 | 1386/1332 | 4266/4266 | 55.9 |
| hotcold 4096x4096 M=1802 normal 200it | int8 | 289.5 | 208.86 | 1386/1323 | 4266/4266 | 55.0 |
| hotcold 1024x4096 M=901 normal 200it | fp8 | 205.6 | 36.77 | 1386/1332 | 4266/4266 | 54.2 |
| hotcold 1024x4096 M=901 normal 200it | int8 | 205.6 | 36.77 | 1386/1287 | 4266/4266 | 53.6 |
| hotcold 1024x4096 M=1517 normal 200it | fp8 | 258.6 | 49.22 | 1386/1296 | 4266/4266 | 55.7 |
| hotcold 1024x4096 M=1517 normal 200it | int8 | 259.1 | 49.12 | 1386/1332 | 4266/4266 | 53.8 |
| cold 4096x4096 M=1517 normal 100it | fp8 | 241.3 | 210.98 | 1386/1296 | 4266/4266 | 54.6 |
| cold 4096x4096 M=1517 normal 100it | int8 | 232.3 | 219.17 | 1386/1341 | 4266/4266 | 54.1 |
