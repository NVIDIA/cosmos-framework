# Microbenchmarks

## MUFU.EX2 (exp2) / FFMA throughput per SM per clock

Why: 8-bit attention only pays off where the attention kernel is MMA-bound; when the special-function unit
(MUFU.EX2, one exp2 per softmax element) is the bottleneck, faster Q·Kᵀ MMAs change nothing. Known values:
Hopper and GB200 16 exp2/clk/SM, GB300 32 (Blackwell Ultra doubled the SFU; measured here 29.8 with FFMA at 124.5/128).
Thor (cc 11.0) is not documented publicly — measure it.

CUDA C (JetPack has nvcc):
```bash
nvcc -O3 -use_fast_math -arch=native -o mufu_exp2_bench mufu_exp2_bench.cu && ./mufu_exp2_bench
```
Triton (needs torch + triton):
```bash
python mufu_exp2_bench.py
```
Read the "exp2 scaled by ffma efficiency" column: ~16 = GB200 class, ~32 = GB300 class. FFMA should be close to 128.
The clock used is the nominal max SM clock; if the device throttles, scale by the actual clock (nvidia-smi / tegrastats).

### Measured on GB300 (cc 10.3, 152 SMs, 2026-09-14)

| harness | exp2 /clk/SM | FFMA /clk/SM | exp2 scaled by FFMA efficiency |
| --- | --- | --- | --- |
| Triton (`mufu_exp2_bench.py`, 16 warps x 32 chains, clock read 2017 MHz) | 29.8 | 124.5 | 30.7 |
| CUDA C (`mufu_exp2_bench.cu`, 32 blocks/SM x 256 threads x 32 chains, nominal 2070 MHz) | 24.5 | 112.0 | 28.0 |

Both point to the 32/clk/SM class (GB200 is 16); the CUDA C harness saturates a little less than the Triton one, so read the
scaled column. NVIDIA's own numbers (Blackwell Ultra softmax blog): exp2 FP32 4943 Gop/s on GB200 vs 10024 on GB300.
The CUDA C binary was built with the pip nvcc (nvidia-cuda-nvcc 13.4) on this aarch64 box: `-arch=sm_103a`, link `-l:libcudart.so.13`.
