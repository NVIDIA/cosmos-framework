# Cosmos3-Nano INT8 量化：结论与实验交接（handoff）

日期：2026-09-12。工作机：GB300 工作站 `pmgb300ws-0083`（单卡 GB300，sm_103，aarch64）。
代码：`~/cosmos-framework`，分支 `pzeren/int8-sim-group-quant`，最新提交 `369427a`
（上一提交 `493b0a1` 已推到 GitHub；`369427a` 是本地提交，含 attention 模拟器、teacher forcing 工具和全部 CLI 接线，
推送命令：`cd ~/cosmos-framework && git push origin pzeren/int8-sim-group-quant`）。
运行笔记（含所有中间数字）：`~/gb300/COSMOS_SETUP.md`。产物在 `~/gb300/outputs`（软链到 NVMe `/var/tmp/pzeren_workdir/outputs`）。

---

## 1. 一句话结论

- **精度**：INT8 g64 只量 7 个 gen 线性层（S0）在 t2i 上 PSNR 26.2 dB（36 图）、保住构图 28/36；官方 FP8 checkpoint 17.8 dB、2/12。
  再加 SageAttention-v1 版式的 attention（Q/K 全部 INT8 + Hadamard + 通道平衡，V/P bf16，Q4a）：25.2 dB、28/36，逐图 11～12/12 胜官方 FP8。
- **速度**：GEMM 部分与官方 per-tensor FP8 同一 MMA 速率（INT8 = FP8 的平台：H100、GB200、Thor、Ada、RTX PRO；A100 只有 INT8），
  额外多出 Q·Kᵀ INT8（t2i 约 +4%，t2v 约 +12%）和可进 CUDA graph 的普通模块路径（官方 FP8 的 torchao subclass 路径在 t2i 上比 bf16 还慢 15%）。
  **GB300 / Rubin 把 INT8 tensor core 砍了约 30 倍，这套方案在这两代上不成立**（本机 `torch._int_mm` 实测 75 TOPS，bf16 1.8 PF，FP8 3.4 PF）。
- **尚未实测**：INT8 全速机器上的真 kernel 速度；视频上的精度；PSNR 之外的指标。

## 2. 推荐方案（S0 + Q4a）

### 2.1 GEMM（S0）
- 范围：gen 塔每层 7 个线性层 × 36 层 = 252 个（`q/k/v/o_proj_moe_gen`, `gate/up/down_proj_moe_gen`），und 塔全部 bf16。
  精确 FQN 列表：`~/gb300/outputs/t2i_int8sim_moe_gen/quantization_matched_fqns.txt`。
- 数学：对称 INT8，qmax 127，round-half-even，scale = 组内 absmax/127（fp32）。激活 A[M,K]：每 token 每 64 个连续 K 一个 scale；
  权重 W[N,K]：每输出通道每 64 个 K 一个 scale（离线，无需校准数据）。Y[m,n] = Σ_g s_a[m,g]·s_w[n,g]·Σ_{k∈g} qa·qw，int32 组内点积，fp32 按组累加，bf16 输出。
- 参考实现：`cosmos_framework/utils/generator/quantization.py::fake_quant_int8(x, per_row=True, group_size=64)`（位精确定义）。
  真 kernel 起点：SGLang `_w8a8_block_int8_matmul`（Triton，`group_n=1, group_k=64, BLOCK_SIZE_K=64`，本机验证与参考位一致；副本在 `~/gb300/tools/ref/sglang_block_int8_kernel_standalone.py`，Apache-2.0）。
  注意 SGLang 自带的激活量化是截断不是四舍五入，要换成 rint + clamp(±127)。cuDNN Graph（block_scale_dequantize×2 + matmul，block 64）也能精确表达但只有 ~63 TOPS，可作正确性参考；cuBLASLt 无 INT8 缩放模式。

### 2.2 Attention（Q4a，只动 gen 塔的 attention）
- Q（gen 查询，32 head）和 K（gen 键 + text 键，8 kv-head）在 RoPE 之后 INT8，每 (token, head) 一个 scale（128 个元素）。V、P、softmax、P·V 全部 bf16，fp32 累加（fp16 累加实测零代价）。
- INT8 舍入前的四步精确变换（softmax 逐位不变，独立复核通过；kernel 里都是逐元素运算，融进 RoPE 之后的量化 kernel）：
  1. K 减每 (kv-head, 通道) 在全部键上的均值（softmax 不变）。
  2. Q 减每 (head, 通道) 在全部查询上的均值，量化后加回：等价于给每个键一个 fp32 偏置 q̄·kⱼ，在 S tile 上与 s_q[i]·s_k[j] 一起乘（<1% MMA 算力）。
  3. 通道平衡：s[c] = (max|K[:,c]| / max|Q[:,c]|)^0.5（每 kv-head，同组 4 个 Q head 共用），Q·s，K/s。
  4. Hadamard：Q、K 各右乘 128 阶归一化 Walsh-Hadamard（7 级蝶形，896 flop/行）。
  1～3 需要每层每步一次跨 token 的规约（预扫描），或直接用上一步/校准的统计（数学上仍精确，精度略降，未测）。Hadamard 无需统计。只留 Hadamard 的纯本地版本未单独测（预计 ≤0.5 dB 代价）。
- text K：与 gen K 同样 INT8（每步随预扫描重量化，2107 键×36 层可忽略）。**不需要 bf16 键段，kernel 只有一条路径。**
- 不需要按步/按层分级（36 图上分级反而少保 2 张）。
- 复现参数：S0 参数 + `--quantization-sim-edges attn_qkv --quantization-attn-k-scope all`，环境变量 `QDQ_SIM_ATTN_OPTS=q_smoothing=1,qk_balance=1,qk_hadamard=1`。
- 真 kernel 起点：SageAttention 仓库的 `sageattn_qk_int8_pv_fp16_triton`（sm80+，INT8 Q·Kᵀ per-block scale + fp16 P·V）；需要把 per-block scale 改成 per-token，并在其量化 kernel 里加均值/平衡/Hadamard。

### 2.3 明确排除的做法（有数据）
- Q/K 用 FP8：单步误差 11.3% 对 INT8 8.6%（同布局），接近官方 FP8 水平。
- V INT8 per-channel 对全部键量化：灾难（18 dB）。排除 text 键 + 1% 离群键后 INT8 per-channel 可用（T6：24.0 dB，9/12），但需要 bf16 键段和按块 P scale，kernel 难写；V 用 FP8 per-channel 即使分段也差（11.3%）。
- P uint8 按 (行, 64 键) 分块比 FP8 固定 scale 好（7.9% 对 8.7%），但需要累加器 promotion。
- 残差流 INT8（−4.4 dB）；GEMM 输出以 INT8 写回（−2.4 dB）；INT16 残差无损但无必要。
- MXFP8/NVFP4 硬件块缩放指令：不可移植，不作为设计选项。

## 3. 证据

### 3.1 compile 路径 e2e PSNR（t2i 960×960，50 步 UniPC，guidance 4，对同 seed bf16；保住构图 = PSNR ≥ 22 dB）
| 配置 | 12 图均值 | 保住/12 | 36 图均值 | 最低 | 保住/36 |
| --- | --- | --- | --- | --- | --- |
| S0 | 25.76 | 10 | 26.20 | 19.9 | 28 |
| 官方 FP8（504 层 per-tensor 静态） | 17.81 | 2 | — | — | — |
| A4 = S0 + Q/K INT8 + Q 去均值（text K bf16） | 22.08 | 7 | | | |
| Q3 = A4 + Hadamard | 24.12 | 9 | | | |
| Q4 = A4 + 平衡 + Hadamard | 24.67 | 8 | | | |
| **Q4a = 全部键 INT8 + 平衡 + Hadamard** | 23.58 | 9 | **25.15** | 18.3 | **28** |
| Q5a = Q4a + L0/L3 bf16 + 前 6 步 bf16 | 23.81 | 7 | 25.13 | 19.5 | 26 |
| T6 = S0 + P·V INT8（V 分段 per-channel，P uint8/64） | 23.99 | 9 | | | |
| T1 = Q/K + P·V 全 INT8 | 21.52 | 5 | | | |
产物：`outputs/grid_c/summary*.txt`（12 图）、`outputs/grid36/summary.txt`（36 图）、`outputs/grid_bf16`、`grid_s0`、`grid_fp8`。
注意：eager 与 compile 两条数值路径各自逐位可复现，但互相差 19.5 dB，且 eager 更接近"换构图"的边界（S0 eager 22.8 / 6 跳）；验收必须在 compile 路径上做。

### 3.2 teacher forcing 单步误差（robot seed 0，eager，v_pred = CFG 合成速度相对 bf16 的相对误差均值；S0 7.73%，官方 FP8 12.83%）
A1 Q/K 全键 8.60 | A2 text bf16 8.54 | A3 +离群键 8.52 | A4 +Q 去均值 8.35 | A5 128-token 块 scale 9.03 | A6 FP8 Q/K 11.28
| Q2 +平衡 8.24 | Q3 +Hadamard 7.99 | Q4 两者 8.06 | Q4a 全键 8.06 | Q5/Q5a 分级 7.88/7.89
| V1 V FP8 全键 11.96 | V2 V FP8 分段 11.32 | V3 V INT8 分段 8.04 | P1 P FP8 8.69 | P2 P uint8/64 7.94 | F1 A4+V2+P1 13.48 | F2 = F1 fp16 累加 13.50
| T1 全 INT8 8.75 | T2 按层 8.30 | T3 前 6 步 8.62 | T4 两者 8.24 | T6 P·V only 8.18
机理：cond/uncond 两支的量化误差几乎不相关（相关系数 0.03），引导项 c−u 只有 |c| 的 4%～20%，1.5% 的单支噪声被 guidance 4 放大成 v_pred 的 7.7%；V per-channel 的误差跨步相干（text V 缓存后每步相同）。
按层敏感度：Q/K 误差集中在 L3、L0；V 误差集中在 L31～L35。Hadamard 把 K 的原位舍入误差从 0.95% 降到 0.46%。
产物：`outputs/tf_t2i/*/tf/sample0.pt`、`attn_err.csv`、`summary*.txt`、`tf_t2i_curves*.png`。

### 3.4 Group size 对比（2026-09-12 补测，compile 路径 36 图 PSNR 对 bf16；保住 = ≥ 22 dB）
| 配置 | 单步误差 | 36 图均值 | 最低 | 保住/36 | robot / desk / street |
| --- | --- | --- | --- | --- | --- |
| S0 g32 | 7.25% | 26.46 | 19.3 | 30 | 25.1 / 30.0 / 24.3 |
| S0 g64 | 7.73% | 26.20 | 19.9 | 28 | 25.7 / 28.1 / 24.9 |
| S0 g128 | 8.37% | 25.01 | 16.3 | 28 | 23.7 / 28.2 / 23.2 |
| S0 g64 + Q4a | 8.06% | 25.15 | 18.3 | 28 | 25.4 / 26.8 / 23.2 |
| S0 g128 + Q4a | 8.60% | 24.28 | 15.1 | 28 | 23.5 / 26.7 / 22.6 |
g128（CUTLASS SM90 blockwise / DeepGEMM 的原生 K 粒度）比 g64 平均低约 1 dB、最差图低 3 dB，保住张数相同；g32 再高 0.26 dB。
如果 kernel 现货只有 128 粒度，精度代价可接受但要接受更差的尾部；g64 是精度/kernel 复杂度的折中点。

### 3.3 速度
- t2i（901 token）：bf16 11.0 ms/forward，GEMM 73%，attention 15%。e2e bf16 13.8 it/s，官方 FP8 11.7 it/s（0.85×，host-bound）；开 CUDA graphs bf16 33.0、FP8 26.9；S0 模拟路径 22～30。
- t2v 720p×189 帧×35 步（约 4.2 万 gen token，text cond 2107 / uncond 24）：bf16 2.29 s/步，官方 FP8 1.96 s/步（1.16×，到落盘 1.14×，与 NIM 表 1.11～1.14× 一致）。kernel 时间：attention 740→731 ms/forward（65%→75%），GEMM 339→160 ms（2.13×），量化 kernel 净增 30 ms（3.1%）。GPU 占用 99.5%。
- attention 与 GEMM 的 FLOPs 交叉点 N = P_layer/(2d) ≈ 2.4 万 token；只量线性层的上限 1/0.65 = 1.54×。
- 单卡稠密 INT8：GB200 5 POPS（=FP8）、GB300 ≈0.19、Rubin 0.25（=FP8 的 1/70）、H100 SXM 1979 TOPS（=FP8）、A100 624（无 FP8）、RTX 6000 Ada 728、RTX PRO 6000 Blackwell ≈1000（推算）、Thor 517。
- g64 GEMM 快速基准（GB300，仅供下界）：Triton g64-INT8 111 TFLOPS，g64-FP8 366，bf16 Triton 1064，cuBLAS bf16 1469，FP8 per-tensor 2264（q_proj，M=901）。脚本 `~/gb300/tools/gemm_g64_bench.py`（agent 生成，未完整复核）。

## 4. 复现方法

### 4.1 环境
`source ~/gb300/cosmos_env.sh`（激活 `~/gb300/workdir/cosmos-venv`：torch 2.10.0+cu130，Triton 3.6，HF 缓存 `~/gb300/data/hf_cache`）。无 HF token，始终加 `--no-guardrails`。
官方 FP8 checkpoint：`~/gb300/data/hf_cache/hub/models--nvidia--Cosmos3-Nano/snapshots/1af44289002ad8afe2567dd85e34fe5dea21e523`（`nvidia/Cosmos3-Nano` 的 fp8 分支）。
其他机器：拉分支后 `uv sync`；x86 上 flash-attn/torchao 可装（aarch64 被 pyproject 排除）。

### 4.2 命令
```bash
cd ~/cosmos-framework
FQNS=~/gb300/outputs/t2i_int8sim_moe_gen/quantization_matched_fqns.txt
S0="--quantization-method int8_sim --quantization-group-size 64 --quantization-target-fqns-file $FQNS"
# S0 12 图 grid（compile 路径）
for p in t2i_robot t2i_street t2i_desk; do python -m cosmos_framework.scripts.inference --parallelism-preset=latency \
  -i ~/gb300/exp/t2i_set/$p.json -o ~/gb300/outputs/grid_s0 --checkpoint-path Cosmos3-Nano --seed=0 --no-guardrails --no-diffusion-cache $S0; done
# Q4a：再加
#   --quantization-sim-edges attn_qkv --quantization-attn-k-scope all   环境变量 QDQ_SIM_ATTN_OPTS=q_smoothing=1,qk_balance=1,qk_hadamard=1
# PSNR 汇总
python ~/gb300/tools/compare_runs.py ~/gb300/outputs/grid_bf16 s0=~/gb300/outputs/grid_s0 q4a=~/gb300/outputs/grid_c/Q4a_c
# teacher forcing：先 dump 后 replay（eager 模式，二者同一数值路径）
COSMOS_TF_MODE=dump   COSMOS_TF_DIR=OUT/bf16/tf python -m cosmos_framework.scripts.inference ... --no-use-torch-compile -i inputs/omni/t2i.json -o OUT/bf16
COSMOS_TF_MODE=replay COSMOS_TF_REF_DIR=OUT/bf16/tf COSMOS_TF_DIR=OUT/x/tf QDQ_SIM_ATTN_OPTS=... python -m cosmos_framework.scripts.inference ... --no-use-torch-compile -o OUT/x $S0 ...
python ~/gb300/tools/tf_local_error.py OUT/bf16/tf x=OUT/x/tf --per-step
```
批处理脚本：`~/gb300/exp/attn_matrix_t2i.sh`、`attn_matrix_t2i_part2.sh`、`attn_e2e_grid_compiled.sh`、`attn_qk_v1.sh`、`attn_qk_v1_all.sh`、`grid36.sh`；prompt 集 `~/gb300/exp/t2i_set{,36}/*.json`。

### 4.3 模拟器选项（`QDQ_SIM_ATTN_OPTS`，逗号分隔）
`outlier_frac`（bf16 离群键比例）、`q_smoothing`、`qk_balance`、`qk_balance_alpha`、`qk_hadamard`、`qk_block`（token 块 scale）、`qk_format=int8|fp8`、
`bf16_first_steps`、`bf16_layers`、`qk_bf16_layers`、`pv_bf16_layers`（用 `-` 分隔层号）、`err_log=path.csv`（每层原位误差，仅 eager）。
CLI：`--quantization-sim-edges {gemm_out,residual,attn_qkv,attn_pv,und_kv}`、`--quantization-attn-k-scope {all,gen,und}`、`--quantization-attn-v-format {int8,fp8,none}`、
`--quantization-attn-p-format {uint8,fp8,none}`、`--quantization-attn-pv-accum {fp32,fp16}`、`--quantization-attn-smoothing`、`--quantization-residual-{bits,group-size}`。
单元测试：`python -m pytest -q cosmos_framework/utils/generator/qdq_sim_edges_test.py`（30 个）。

### 4.4 工具
`~/gb300/tools/`：`compare_images.py`、`compare_runs.py`（grid PSNR 汇总）、`tf_local_error.py`、`attn_err_report.py`、`tf_plot.py`、
`profile_kernels.py`（torch.profiler 按类别统计，已修正 Inductor 融合 kernel 名字里的下游算子导致的误分类）、`speed_report.py`、`profile_categories_t2v.py`、`draw_nano_layer.py`、`gemm_g64_bench.py`（待复核）。
`~/gb300/tools/ref/`：SGLang / vLLM / lmdeploy / TensorRT-LLM / ModelOpt / DeepGEMM 的相关源码副本，cuDNN g64 建图探针 `cudnn_probes/`。

## 5. 到 INT8 全速机器上要做的事（建议顺序）
1. **GEMM 真 kernel**：SGLang block-INT8 Triton kernel（group_n=1, group_k=64）对 cuBLAS bf16、per-tensor FP8 在 7 个模型形状 × M∈{901, 4096, 42240} 上计时；有条件再上 CUTLASS SM90 blockwise（需把 K 粒度 128 的约束改成 64，或接受 128）。目标：确认 INT8 g64 与 per-tensor FP8 的差距在 10% 以内。
2. **Attention 真 kernel**：SageAttention v1 Triton kernel 改 per-token scale，量化 kernel 里加 K/Q 去均值、平衡、Hadamard；对 FA2/FA3 bf16 计时；用 teacher forcing 验证 kernel 输出与模拟一致。
3. **e2e**：S0 + Q4a 的真 kernel 路径开 CUDA graphs，对 bf16 和官方 FP8 计时（t2i 和 t2v）。
4. **视频精度**：在 t2v 上重复 teacher forcing + 若干视频的逐帧 PSNR（模拟器已支持长序列：V-only/QK-only 走融合 kernel，P 量化才走稠密参考）。
5. **补指标**：PSNR 之外加一个感知或分布指标。

## 6. 未解问题
- FP8 模拟 g64 比 per-row 还差（16.0 对 20.3，单图早期实验），原因未查；FP8 路线（GB300/Rubin 需要）若要走必须先弄清。
- 模拟器用的是当步精确统计（K/Q 均值、平衡系数）；单遍 kernel 若用上一步统计，精度有待验证。
- 12 图的"保住构图"计数有 ±2 张噪声，36 图为准；跨 prompt 的敏感度差异大（desk 只保 8/12，即便 S0）。
- Q4a 里 text K 每步重量化的 kernel 成本未计入（预计可忽略）。
