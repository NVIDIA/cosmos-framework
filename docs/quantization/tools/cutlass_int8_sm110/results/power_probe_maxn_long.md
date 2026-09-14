# MAXN, long runs (>= 3 s of back-to-back kernels per case) so the tegrastats VDD_GPU mean is dominated by the kernel. 2026-09-14 08:31

| case (Gaussian operands) | dtype | TFLOPS | median us | GPU W mean | GPU W max | samples | mJ per GEMM (mean W x median us) | TOPS/W |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| hotcold 4096x4096 M=901 | fp8 | 191.8 | 157.7 | 88.8 | 99.0 | 18 | 14.00 | 2.16 |
| hotcold 4096x4096 M=901 | int8 | 310.7 | 97.3 | 93.0 | 98.8 | 18 | 9.05 | 3.34 |
| hotcold 4096x4096 M=1517 | fp8 | 230.2 | 221.2 | 103.8 | 97.7 | 17 | 22.96 | 2.22 |
| hotcold 4096x4096 M=1517 | int8 | 350.5 | 145.2 | 94.5 | 98.8 | 17 | 13.72 | 3.71 |
| hotcold 4096x4096 M=1802 | fp8 | 210.9 | 286.7 | 104.3 | 93.7 | 17 | 29.90 | 2.02 |
| hotcold 4096x4096 M=1802 | int8 | 319.1 | 189.5 | 94.7 | 98.8 | 17 | 17.94 | 3.37 |
| hotcold 1024x4096 M=901 | fp8 | 217.1 | 34.8 | 91.7 | 98.9 | 14 | 3.19 | 2.37 |
| hotcold 1024x4096 M=901 | int8 | 217.7 | 34.7 | 71.1 | 77.2 | 12 | 2.47 | 3.06 |
| hotcold 1024x4096 M=1802 | fp8 | 160.5 | 94.2 | 92.8 | 98.9 | 15 | 8.74 | 1.73 |
| hotcold 1024x4096 M=1802 | int8 | 255.6 | 59.1 | 83.4 | 90.3 | 13 | 4.93 | 3.06 |
| cold(flush) 4096x4096 M=901 | fp8 | 178.9 | 169.0 | 44.4 | 82.6 | 19 | 7.50 | 4.03 |
| cold(flush) 4096x4096 M=901 | int8 | 174.7 | 173.1 | 34.7 | 60.8 | 19 | 6.01 | 5.03 |
| warm 4096^3 | fp8 | 246.3 | 558.0 | 103.9 | 98.6 | 13 | 57.98 | 2.37 |
| warm 4096^3 | int8 | 291.2 | 472.0 | 94.7 | 98.9 | 15 | 44.70 | 3.07 |
| hotcold 4096x4096 M=1517 (cuBLASLt) | bf16 | 128.1 | 397.4 | 95.2 | 98.7 | 29 | 37.83 | 1.35 |

NOTE: the "GPU W max" column is unreliable (string comparison bug, fixed in power_probe.sh since); the mean column is correct and kernel-dominated (>= 3 s of back-to-back kernels per case, 13-29 tegrastats samples at 200 ms). Energy per GEMM = mean W x median us. Reading: compute-bound cases run at the same ~95-104 W power cap for both formats, INT8 finishes 1.35-1.5x sooner -> 35-40 % less energy per GEMM; memory-bound cases (1024x4096 M=901, cold M=901) take the same time and INT8 draws 20-23 % less power.
