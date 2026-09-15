# docs/quantization/tools — 目录索引与协作约定

主仓库：**https://github.com/nvidia-cosmos/cosmos-int8-gemm**（私有，`main`）。各平台直接在这里推送代码与结果；结论按节号写进 `../int8_handoff.md`。

| 目录 | 平台 | 内容 |
| --- | --- | --- |
| `cutlass_pertensor_gemm/` | GB200 sm_100a（`ARCH=110` 可编 Thor） | CUTLASS 示例 70 改 INT8 的 per-tensor GEMM + bf16/FP8/INT8 绝对吞吐（handoff §8.5） |
| `cutlass_g128_gemm/` | GB200 sm_100a | INT8 g128 per-col：SM100 blockwise collective 的 shadow 补丁（fp32 scale/提升）、promotion 优化、结果（§8.6） |
| `cutlass_int8_sm110/` | Thor sm_110a | per-tensor 对齐 FP8 + 功耗研究、g128/g256 分组缩放、8-warp 提升 kernel、微基准、汇总报告（§11） |
| `cutlass_int8_sm90/` | H100 sm_90a | 自写 SM90 INT8 per-tensor 与 g128 blockwise kernel（§9） |
| `deepgemm_int8_sm90/` | H100 sm_90a | DeepGEMM 结构的 INT8 g128 移植与提升/MMA 重叠实验（§9.4） |
| `int8_sim_precision/` | 任意（模拟器） | 权重 scale 布局精度研究：per-col / 块 / 可分离 / g256（§3.9） |
| `mufu_exp2_bench.*` | 任意 | 下方的 MUFU.EX2 / FFMA 微基准 |

约定：每个工具目录自带 README（构建、运行、数据口径）与 `results/`；`logs/`、`build*/`、`__pycache__/` 用 `.gitignore` 排除；计时统一为冲 L2 后逐次 cudaEvent 取中位数（Thor 为热 A 冷 W 持续，见 §11）。

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

# INT8 g128 max-performance kernels and precision experiments (H100, 2026-09-14)

Snapshots of the working trees used for handoff sections 3.9 and 9.4. They were run from `users/pzeren/int8_bench` on
aws-iad-cs-002 inside `imaginaire4_v12.1.0.sqsh` (torch 2.13 / CUDA 13.3 / Triton 3.7.1); scripts still contain those
cluster paths (override `W=`/`INT8_WORK` in `run_cfg.sh`; edit the constants at the top of the Python files elsewhere).

## `deepgemm_int8_sm90/` — DeepGEMM-structure INT8 (1,128,128) block-scaled GEMM for sm_90a
`gen_kernel.py` derives `include/deep_gemm/impls/sm90_int8_gemm_1d2d.cuh` from DeepGEMM 66081d4's `sm90_fp8_gemm_1d2d.cuh`
(S8 wgmma via CUTLASS `MMA_64xNx32_S32S8S8_SS_TN`, int32 block accumulator, fp32 promotion `final += (sfa*sfb)*float(acc)`);
`gen_kernel_db.py` / `gen_kernel_pp.py` / `gen_kernel_wi.py` and `occ/` generate the double-buffered (B), warpgroup ping-pong (C),
wave-interleaved (A) and 2-CTA/SM (D) variants; `gen.py`, `pp/gen_pp.py`, `occ/gen_occ.py`, `wi/gen_wi.py` emit the per-config `.cu`
translation units (not committed; regenerate). `launcher*.cuh` build the TMA descriptors and cluster launch; `__init__.py` exposes
`int8_gemm_g128(A, W, sfa_kernel, sfb, cfg)` (+ `prepare_sfa`, `pick_config`). `include/deep_gemm/*` are DeepGEMM headers (MIT,
`LICENSE.DeepGEMM`). `dgint8_smoke.py` = exactness vs fp64 + vs the CUTLASS kernel; `verify_int8_g128.py` = the independent
verifier; `bench_deepgemm_int8.py` = timing. `results/` = the four serial-timing reports (final / overlap / pingpong / occupancy2).
Build needs the CUTLASS 4.2 headers (`CUTLASS_DIR`) for the MMA atom and nvcc 13.x, `TORCH_CUDA_ARCH_LIST=9.0a`.

## `int8_sim_precision/` — simulator-side precision experiments (section 3.9)
`run_cfg.sh <tag> <inference flags>` runs the QDQ simulator on `t2i_set.jsonl` (4 prompts x 3 seeds, per-sample seeds; pass the HF
snapshot dir as `--checkpoint-path`, the model name triggers an offline `hf download`), `psnr.py <bf16_dir> <run_dir>` scores vs bf16
(`psnr_*.json` = the runs quoted in 3.9). `weight_scale_study.py` (+ `weight_scale_study.md`) and `weight_snr_g256.py` are the offline
weight-quantization error studies on the 252 gen-tower linears of the diffusers export (weights are `add_q_proj`/`add_k_proj`/
`add_v_proj`/`to_add_out`/`mlp_moe_gen.*`). Simulator knob: `QDQ_SIM_WEIGHT_SCALE=separable|separable_ls|separable_ls_noclip`
(`quantization.py::fake_quant_weight_separable`).
