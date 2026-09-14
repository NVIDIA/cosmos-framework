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
