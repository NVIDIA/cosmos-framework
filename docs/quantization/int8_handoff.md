# Cosmos3-Nano INT8 量化：结论与实验交接（handoff）

日期：2026-09-12。工作机：GB300 工作站 `pmgb300ws-0083`（单卡 GB300，sm_103，aarch64）。
代码：分支 `pzeren/int8-sim-group-quant`（GitHub 上已含全部代码提交：`493b0a1` GEMM 模拟 + CLI，`369427a` attention 模拟器、teacher forcing、int8_speed；其后均为 docs 提交）。
2026-09-13 补充：第 8 节为 H100 真 kernel GEMM 基准结果（aws-iad-cs-002，`users/pzeren/int8_bench`）；8.5 为 GB200 上 CUTLASS per-tensor INT8/FP8/bf16 绝对吞吐补测；8.6 为 GB200 上 INT8 g128（per-col）真 kernel 的结果与结构性结论。
2026-09-14 补充：第 11 节为 Thor（sm_110a）上 per-tensor INT8 对齐 FP8 与功耗限流的结论（代码与数据在 `docs/quantization/tools/cutlass_int8_sm110/`）。
运行笔记（含所有中间数字）：`~/gb300/COSMOS_SETUP.md`。产物在 `~/gb300/outputs`（软链到 NVMe `/var/tmp/pzeren_workdir/outputs`）。

---

## 1. 一句话结论

- **精度**：INT8 g64 只量 7 个 gen 线性层（S0）在 t2i 上 PSNR 26.2 dB（36 图）、保住构图 28/36；官方 FP8 checkpoint 17.8 dB、2/12。
  再加 SageAttention-v1 版式的 attention（Q/K 全部 INT8 + Hadamard + 通道平衡，V/P bf16，Q4a）：25.2 dB、28/36，逐图 11～12/12 胜官方 FP8。
- **速度**：GEMM 部分与官方 per-tensor FP8 同一 MMA 速率（INT8 = FP8 的平台：H100、GB200、Thor、Ada、RTX PRO；A100 只有 INT8），
  额外多出 Q·Kᵀ INT8（t2i 约 +4%，t2v 约 +12%）和可进 CUDA graph 的普通模块路径（官方 FP8 的 torchao subclass 路径在 t2i 上比 bf16 还慢 15%）。
  **GB300 / Rubin 把 INT8 tensor core 砍了约 30 倍，这套方案在这两代上不成立**（本机 `torch._int_mm` 实测 75 TOPS，bf16 1.8 PF，FP8 3.4 PF）。
- **H100 真 kernel 实测（2026-09-13，第 8 节）**：现货 INT8 g64 kernel（SGLang Triton）只有 bf16 cuBLAS 的 0.4～0.56×、官方 per-tensor FP8 的 0.21～0.35×；INT8 g128 也只到 bf16 的 0.75～0.96×。
  连 cuBLASLt 的 per-tensor INT8 都只有 bf16 的 1.1～1.4×，低于 FP8 per-tensor 的 1.6～2.0×，"H100 上 INT8 = FP8 速率"的假设不成立。FP8 g128（DeepGEMM）达到 per-tensor FP8 的 0.91～1.03×，是 H100 上唯一兼得分组缩放与速度的现货路径。
- **自写 CUTLASS SM90 INT8 kernel（2026-09-14，第 9 节）**：per-tensor INT8 达到 FP8 per-tensor 的 0.90～1.06×（cuBLASLt 慢是因为没有 Hopper INT8 kernel，不是硬件）；g128 分组缩放做进主循环后约 1000 TFLOPS = bf16 的 1.35×、per-tensor 的 0.70×，H100 上 INT8 分组量化首次拿到正收益。
- **GB200 / Blackwell（2026-09-14，第 8.5/8.6/10/11 节）**：per-tensor INT8 = FP8 = 约 3.4 PTOPS（bf16 的 2 倍）；Thor 上 INT8 更因不受功耗限流而达 FP8 的 1.2～1.6×。
  分组缩放的代价在 Blackwell 上是结构性的（tensor:CUDA core 约 64:1）：GB200 现有实现的 g128 ≈ bf16；Thor 用 8 个提升 warp + 256×256 tile 把 W 块 g128 推到 FP8 pt 的 0.97×，g256 per-col 追平 FP8 pt（1.0～1.3×），per-col g128 仍只有 0.4～0.7×。
- **尚未实测**：视频上的精度；PSNR 之外的指标；FP8 分组量化（g64/g128）的精度（第 6 节的 FP8 g64 异常未查）；g256 kernel 的真实速度。

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
| 官方 FP8（504 层 per-tensor 静态） | 17.81 | 2 | 18.58 | 11.2 | 5 |
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

### 3.5 Policy（动作）任务（2026-09-12 补测）
`inputs/omni/action_policy_robot.json`（bridge 样例，model_mode wam，30 步），8 个 seed，动作块 16×10（平移 3、6D 旋转 6、夹爪 1）。
指标：对同 seed bf16 动作的 MSE；参照 bf16 自身 seed 间 MSE 0.150。工具 `tools/compare_actions.py`，产物 `outputs/policy/`。
| 配置 | MSE 对 bf16 | 占 seed 间差异 | rel-L2 | 夹爪开合一致率 | 最差 seed |
| --- | --- | --- | --- | --- | --- |
| 官方 FP8 | 0.0113 | 7.5% | 18% | 97.7% | 0.0595 |
| S0 g64 | 0.00166 | 1.1% | 7% | 99.2% | 0.0082 |
| S0 g128 | 0.00164 | 1.1% | 6.9% | 99.2% | 0.0071 |
| S0 g64 + Q4a | 0.00180 | 1.2% | 7.2% | 99.2% | 0.0068 |
| S0 g128 + Q4a | 0.00214 | 1.4% | 7.9% | 99.2% | 0.0096 |
S0 比官方 FP8 离 bf16 近约 7 倍；g64 与 g128 在动作上无差别；加上 Q4a attention 几乎不增加误差（非夹爪维 MSE 0.00063 对 0.00064）。
policy 走 two_way attention 路径，1517 个 gen token + 110 个 text 键，无 CFG。样例自带的 golden 动作文件框架并不评估，bf16 对它 MSE 0.2，不可作参照。

### 3.6 权重 scale 布局：per-col g128 对 DeepSeek blockwise 128×128（2026-09-14 补测，激活都是 per-token g128）
| 权重 scale | t2i 36 图均值 | 保住/36 | +Q4a 均值 | +Q4a 保住 | policy MSE 对 bf16 | +Q4a policy MSE |
| --- | --- | --- | --- | --- | --- | --- |
| per-col g128（每通道 × 128 K） | 25.01 | 28 | 24.28 | 28 | 0.00164 | 0.00214 |
| W 128×128 blockwise | 23.05 | 25 | 22.62 | 21 | 0.00875 | 0.00898 |
| 官方 FP8 | 18.58 | 5 | | | 0.0113 | |
每输出通道独立的权重 scale 是精度的支柱：放粗到 128 通道共用一个 scale，t2i 掉约 2 dB，policy 的动作误差回到官方 FP8 的水平（夹爪 MSE 0.036 对 0.039）。
kernel 侧代价：per-col 的 promotion 是 blockwise 的 2 倍 FMA（H100 上 per-col g128 约占 FMA 管线 50%，per-col g64 约 100%）；
现成先例是 CUTLASS SM100 blockwise 的 ScaleGranularityN=1 和 DeepGEMM 1D1D。可分离 scale s_w[n,g]=s_w[n]·c[g]（promotion 与 blockwise 同价）尚未测。

### 3.7 Cosmos3-Edge（2026-09-14 补测）
Edge：28 层、hidden 2048、16 个 Q head × 128、8 个 KV head，gen 塔 168 个线性层（q/k/v/o + MLP up/down，无 gate），公开无 fp8 分支。
t2i 12 图（3 prompt × 4 seed，compile 路径），PSNR 对 Edge bf16：
| 配置（只量 gen 塔） | 平均 | 最低 | 保住/12 |
| --- | --- | --- | --- |
| INT8 per-row 激活 / per-col 权重，无 K 分组 | 20.8 | 14.2 | 5 |
| INT8 g64 | **29.3** | 23.9 | **12** |
| INT8 g128 | 26.2 | 17.9 | 10 |
结论与 Nano 一致且更明显：无分组的 per-row/per-col 不够用，K 分组是精度的来源；Edge 上 g64 全部保住构图，g128 掉 3 dB。

### 3.8 SmoothQuant 对无分组 per-row/per-col INT8 的作用（Cosmos3-Edge，2026-09-14 补测）
校准：3 个 prompt × 1 seed 的 bf16 直通推理，记录 168 个 gen 线性层每个输入通道的 |x| 最大值（通道最大/中位比：中位 6.7，最大 11279）。
迁移 s_k = max|X_k|^α / max|W_k|^(1−α)，激活除以 s、权重列乘 s，再做 per-row/per-col INT8（无 K 分组）。开关 `QDQ_SIM_CALIB_DUMP` / `QDQ_SIM_SMOOTH=stats.pt:alpha`。
| 配置（Edge t2i 12 图，对 bf16） | 平均 | 最低 | 保住/12 |
| --- | --- | --- | --- |
| per-row/per-col，无 SmoothQuant | 20.8 | 14.2 | 5 |
| + SmoothQuant α=0.5 | 24.6 | 15.4 | 8 |
| + SmoothQuant α=0.75 | 22.2 | 15.6 | 8 |
| g128（无 SmoothQuant） | 26.2 | 17.9 | 10 |
| g64（无 SmoothQuant） | 29.3 | 23.9 | 12 |
SmoothQuant 能补回约 4 dB，但仍不及最粗的 K 分组 g128；离群通道迁移不如按 K 分组直接给离群位置独立 scale。
Nano 上同一实验（12 图，对 compile bf16；modules=252 act channel max/median ratio: median 9.6, max 14225.6）：per-row/per-col 无分组 20.0（4/12）→ + SmoothQuant α=0.5 **20.2（4/12）**，几乎无增益；S0 g64 25.8（10/12），官方 FP8 17.8（2/12）。
Edge 上 SmoothQuant 补回 3.8 dB，Nano 上只有 0.15 dB：两个模型的离群结构不同，通道迁移的收益不可迁移，K 分组在两者上都稳定有效。
Nano 上再补 g256（12 图）：g256 22.9（8/12，最低 14.8）→ **g256 + SmoothQuant α=0.5 23.8（10/12，最低 19.8）**；对照 g128 23.9（9/12）、g64 25.8（10/12）。
有 K 分组之后 SmoothQuant 开始起作用（g256 上 +0.9 dB、最差图 +5 dB、多保 2 张），g256+SQ 与 g128 相当；无分组时它救不回来。
零 kernel 代价的两招（Nano，12 图，无分组 per-row/per-col 为基线 20.0 / 4 张）：**权重 scale 的 MSE 裁剪搜索 +1.0 dB（21.0，5 张，最低 14.4→16.5）**，8 bit 下裁剪很少触发，收益小但免费；
**逐通道静态 shift（折进 bias）明显有害**：无分组 + SQ + shift 17.3（0 张），g256 + SQ + shift + 裁剪 18.4（0 张，对照 g256 + SQ 23.8）。原因是激活用的是 per-token 动态 scale，
静态的全局中点 δ 把多数 token 推离零点、抬高了每个 token 的 absmax；且校准显示通道本就近似对称（|max+min|/(max−min) 中位 0.067）。shift 只在静态激活 scale 的方案里才有意义。

### 3.9 可分离权重 scale `s_w[n]·c[g]` 与 g256 布局（H100 模拟器，2026-09-14 补测）
动机：§8.6/§9 表明 kernel 里逐元素提升的成本由权重 scale 的布局决定——W 逐列 g128 每元素要 FMUL+FFMA，W 128×128 块只要一次 FFMA。
若 `s_w[n,g] = s_w[n]·c[g]` 可分离，主循环里只剩每块一个标量 `c[g]`（块缩放成本），`s_w[n]` 放尾声免费。模拟器开关 `QDQ_SIM_WEIGHT_SCALE`
（`quantization.py::fake_quant_weight_separable`）：`separable`（各通道最大比值包络，不裁剪）、`separable_ls`（对数域秩 1 最小二乘，允许裁剪）、
`separable_ls_noclip`（LS 公共 K 轮廓 `c[g]`，再把每个通道的 `s_w[n]` 抬到恰好不裁剪）。脚本、prompt 集、逐图 PSNR：`docs/quantization/tools/int8_sim_precision/`。

t2i 12 图（4 prompt × 3 seed，960×960，50 步 UniPC，compile 路径），PSNR 对同 seed bf16，激活均为 per-token 同组大小：
| 权重 scale 布局 | g64 | g128 | g256 |
| --- | --- | --- | --- |
| 逐列 K 分组（S0 布局） | 26.64（最低 18.9，10/12） | 26.61（19.6，9/12） | 26.10（19.9，10/12） |
| g×g 块（DeepSeek 布局） | — | 23.73（13.7，7/12） | 24.71（19.0，10/12） |
| 可分离 `separable_ls_noclip` | — | **26.28（17.1，10/12）** | 24.09（17.4，8/12） |
| 可分离 `separable`（见下，退化为纯每通道） | — | 25.74（16.8，11/12） | — |
| 可分离 `separable_ls`（允许裁剪） | — | 11.8（8 图） | — |

离线权重量化误差（252 个 gen 线性层真实权重，pooled 权重 SNR dB / 合成激活 GEMM 相对误差；独立复核通过；`weight_scale_study.py`、`weight_snr_g256.py`）：
| 布局 | g64 | g128 | g256 |
| --- | --- | --- | --- |
| 逐列 | 43.60 / 0.0066 | 42.54 / 0.0075 | 41.50 / 0.0084 |
| g×g 块 | — | 35.40 / 0.0170 | 33.11 / 0.0221 |
| 可分离 `ls_noclip` | — | 38.59 / 0.0118 | 38.48 / 0.0119 |
| 纯每通道（不分 K） | 37.89 / 0.0127 | | |
| per-tensor | 23.18 / 0.0694 | | |

结论：
1. **"最大比值包络"的可分离 scale 在数学上退化为纯每通道**：N ≫ K/g 时每个 K 块总有某通道的块最大值等于其行最大，`c[g]≡1`（252 个权重里 c 的离散度平均 0.013 bit）。
   允许裁剪的 LS 拟合则灾难性崩坏（8 位下任何裁剪都不可接受）。可用的是 `ls_noclip`。
2. **absmax 矩阵基本不可分离**：秩 1 对数拟合后的残差 0.28 bit，与它要替代的行效应（0.34 bit）同量级；列效应只有 0.09 bit。所以可分离只拿回逐列相对每通道 4.65 dB 权重 SNR 中的 0.7 dB。
3. **但 e2e 上 `ls_noclip` g128 与逐列 g128 在噪声内持平（26.3 对 26.6），明显好于 128×128 块（23.7）**：端到端误差由激活量化主导，权重侧只要保住"每通道 scale"这根支柱；这与 3.6 节块布局掉 2 dB 一致。
   因此它以块缩放的 kernel 成本拿到接近逐列的精度，是 §8.6 提出的路径的正面验证。
4. **g256**：逐列 g256 相对 g128 掉约 0.5 dB（权重 SNR 掉 1 dB），保住张数不变，与 3.8 节另一 prompt 集上的 g256/g128 差距一致；块与可分离布局在 g256 上退化更快（块 SNR −2.3 dB，可分离 e2e −2.2 dB）。
   逐列 g256 把主循环提升次数减半（int32 组内累加 256·127² < 2²⁴ 仍精确），是精度/速度的新折中点（kernel 侧见 9.4）。
5. 12 图噪声约 ±2 dB（本集 g64 与 g128 打平，36 图上差 1.2 dB；块布局 g256 反高于 g128 与离线方向相反），最终取舍应在 36 图集复核。

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
1. ~~**GEMM 真 kernel**~~ **已完成（2026-09-13，第 8 节）**：结论是差距不在 10% 以内，而是 INT8 g64 慢 2.8～4.8×；CUTLASS SM90 blockwise 未做，因为 cuBLASLt per-tensor INT8 本身就低于 FP8 per-tensor，INT8 分组路线在 H100 上没有速度上限支撑。
2. **Attention 真 kernel**：SageAttention v1 Triton kernel 改 per-token scale，量化 kernel 里加 K/Q 去均值、平衡、Hadamard；对 FA2/FA3 bf16 计时；用 teacher forcing 验证 kernel 输出与模拟一致。
3. **e2e**：S0 + Q4a 的真 kernel 路径开 CUDA graphs，对 bf16 和官方 FP8 计时（t2i 和 t2v）。
4. **视频精度**：在 t2v 上重复 teacher forcing + 若干视频的逐帧 PSNR（模拟器已支持长序列：V-only/QK-only 走融合 kernel，P 量化才走稠密参考）。
5. **补指标**：PSNR 之外加一个感知或分布指标。

## 6. 未解问题
- FP8 模拟 g64 比 per-row 还差（16.0 对 20.3，单图早期实验），原因未查；FP8 路线（GB300/Rubin 需要）若要走必须先弄清。
- 模拟器用的是当步精确统计（K/Q 均值、平衡系数）；单遍 kernel 若用上一步统计，精度有待验证。
- 12 图的"保住构图"计数有 ±2 张噪声，36 图为准；跨 prompt 的敏感度差异大（desk 只保 8/12，即便 S0）。
- Q4a 里 text K 每步重量化的 kernel 成本未计入（预计可忽略）。

## 7. 开源生态对 g64 / g128 的支持（2026-09-12 调研，详见 docs/quantization/g64_oss_survey.md）

- **INT8 g64（双操作数、K 方向分组、mainloop 内缩放）**：没有任何开源栈有现货。唯一能跑的是 SGLang 的 Triton `_w8a8_block_int8_matmul`
  （`block_size=[n,64]`，需显式 `BLOCK_SIZE_N>=16`；数值已验证精确），无 64 的调优配置。cuBLASLt 无 INT8 缩放模式；cuDNN Graph 能表达但走通用 sm80 kernel；
  CUTLASS SM90 blockwise 是 FP8-only 且 K 粒度 >=128，SM100 blockwise 的 scale 类型绑成累加器类型（INT8 → int32）且 kind::i8 只在 sm_100a/101a/110a 开启。
- **INT8 g128**：同样只有 Triton。SGLang dense + fused-MoE kernel 有 A100/A800/H20 的 [128,128] 调优配置，为 DeepSeek-R1 Block-INT8 checkpoint 量产过
  （SGLang PR #3730：R1 在 A100×16×2 上吞吐 +33% vs BF16，GSM8K 95.8 对 95.5）；vLLM 的 dense INT8 块封装已删（MoE Triton 有但未接线）。
  CUTLASS / DeepGEMM / TRT-LLM / FlashInfer / sgl-kernel(CUDA) 的 INT8 全是 per-token/per-channel epilogue 缩放，SM100/SM120 上抛不支持。
- **FP8 g128**：生态标准（CUTLASS SM90/100/120、DeepGEMM SM90、vLLM、FlashInfer、sgl-kernel、TRT-LLM、torch `_scaled_mm` 1x128 仅 SM90），全部硬编码 1×128 / 128×128。
- **FP8 g64**：CUTLASS SM100 blockwise collective 原生支持（K 粒度只需是 32 的倍数且整除 TileK，单测有 64×64×64），但所有上层封装只派发 128；
  FlashInfer 加 64 约 10～20 行。这是 GB300 上 FP8 孪生的最短路径。
- 其他：llama.cpp 的 Q8_0×Q8_1 MMQ 是唯一长期量产的双侧 INT8 K-group kernel（组固定 32，消费卡向）；Atom 有 INT4 组 128 的双侧缩放（研究代码）。
- **精度上 g128 的代价已量过**（3.4 节）：t2i 比 g64 低约 1 dB、最差图低 3 dB、保住张数相同；policy 无差别。
- **建议路径**：目标机上先用 SGLang Triton INT8 g128（小时级接入、有调优配置）拿到相对 per-tensor FP8/INT8 的真实速度；要逼近 per-tensor 速率则 fork CUTLASS
  SM90 blockwise 改 INT8（int32 临时累加器 + fp32 主累加器），g64 与 g128 工作量相近（2～3 人周）。GB300 上用 CUTLASS SM100 FP8 blockwise <1,1,64> 做孪生。

## 8. H100 真 kernel GEMM 基准（2026-09-13）

机器：aws-iad-cs-002 节点 pool0-0977，1× H100 SXM 80GB HBM3，驱动 575.57 + CUDA 13.0 前向兼容 libcuda 580；容器 imaginaire4_v12.1.0（torch 2.13.0a0 nv26.07 / CUDA 13.3 / Triton 3.7.1）。
基准代码与全部数据：`/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/users/pzeren/int8_bench/`（`results/bench_final.{json,md}` 400 行原始数据、`results/tuned_configs.json`、`results/probe_report.md`）；
完整中文报告已随本提交放入 `docs/quantization/h100_gemm_bench_2026-09-13.md`。

### 8.1 方法
- GEMM：Y[M,N] = A[M,K]·W[N,K]ᵀ，bf16 输出。形状 = Nano gen 塔 4 个唯一形状（q/o 4096×4096、k/v 1024×4096、gate/up 12288×4096、down 4096×12288），M ∈ {901, 1802, 4096, 16384, 42240}。
- 计时统一用 `triton.testing.do_bench` 中位数（每次重复前冲掉 L2，warmup 25 ms / rep 100 ms），对所有 kernel 一致；每行记录 nvidia-smi 时钟。
- 正确性：每个 kernel 的输出对**同一组量化后操作数**的 fp64 精确参考 Y = Σ_g s_a s_w (qa·qw) 比较，400 行全部通过（非 fast-accum 路径 rel-L2 1.65e-3～1.70e-3 = bf16 输出舍入下限；fast-accum FP8 2.0e-3～3.9e-3；`_int_mm` int32 位精确）。Triton 激活分组量化 kernel（rint + clamp ±127，absmax/127 fp32）与 `fake_quant_int8` 的数学位精确一致。
- Triton 块缩放 kernel 取自 SGLang @22e4b3a（`_w8a8_block_int8_matmul` / `_w8a8_block_fp8_matmul`，数学未改），每个 (kernel 族, 形状, M) 独立调优 116～198 个配置（约束 BLOCK_K ≤ group_k 且 group_k % BLOCK_K == 0）。
- 对照：cuBLAS bf16；cuBLASLt FP8 per-tensor（`torch._scaled_mm`，fast_accum，= ModelOpt 官方 checkpoint 路径）；cuBLASLt FP8 rowwise（torchao 运行时默认）；cuBLASLt INT8 `torch._int_mm`；cuBLASLt FP8 分块 g128（`torch.nn.functional.scaled_mm`，1×128 / 128×128，SM90 仅此粒度、要求 M % 4 == 0）；DeepGEMM SM90 FP8 g128（commit 66081d4，树内编译）。

### 8.2 主要结果（TFLOPS，中位数；q/o_proj 4096×4096）
| kernel | M=901 | M=4096 | M=42240 |
| --- | --- | --- | --- |
| bf16 cuBLAS | 670 | 729 | 693 |
| FP8 per-tensor cuBLASLt（官方） | 1111 | 1348 | 1347 |
| FP8 rowwise cuBLASLt | 830 | 1318 | 1363 |
| INT8 cuBLASLt `_int_mm`（per-tensor 缩放，int32 输出） | 663 | 839 | 838 |
| FP8 g128 cuBLASLt 1×128 / 128×128 | 894 | 1176 | 1143 |
| FP8 g128 DeepGEMM | 966 | 1301 | 1273 |
| **INT8 g64 Triton，W 逐通道（S0 方案）** | **283** | **342** | **357** |
| INT8 g64 Triton，W 128×64 块 | 325 | 395 | 421 |
| INT8 g128 Triton，W 逐通道 | 410 | 505 | 528 |
| INT8 g128 Triton，W 128×128 块 | 501 | 619 | 646 |
| FP8 g64 / g128 Triton | 286 / 516 | 349 / 647 | 369 / 606 |
| bf16 Triton（校准点，未调优） | 579 | 697 | 648 |

其余三个形状趋势一致（M ≥ 4096 全形状范围：bf16 665～737；FP8 per-tensor 1137～1407；INT8 `_int_mm` 756～975；DeepGEMM FP8 g128 1029～1412；INT8 g64 Triton 323～363；INT8 g128 Triton 553～646）。
激活动态分组量化（Triton 融合 kernel）的额外开销：M=901、K=4096 约 11 µs；M=42240、K=4096 约 193 µs（2.7～2.8 TB/s）。

### 8.3 结论
1. **INT8 g64 在 H100 上是负收益。** 现货 kernel（Triton）为 bf16 的 0.40～0.56×、官方 per-tensor FP8 的 0.21～0.35×（慢 2.8～4.8×）。第 5 节"10% 以内"的目标没有达到。
2. **瓶颈不是 Triton 代码生成。** Triton bf16 与 cuBLAS bf16 相当（≤10% 差距），Triton 3.7.1 的 INT8 `tl.dot` 在 sm90 上确认走 wgmma（SASS 为 IGMMA m64n128k32.s8），裸 INT8 探针 4096³ 达 880～1061 TOPS。慢在块缩放 mainloop 的结构：BLOCK_K 被钉在 ≤64，每个 K tile 做一次 fp32 外积重缩放。g128 比 g64 快 1.4～1.6×（survey 预估的 5%～15% 偏乐观），W 128 块比逐通道快 1.1～1.3×。同结构的 Triton FP8 g128 比 DeepGEMM FP8 g128 又慢约 2×。
3. **cuBLASLt 的 INT8 没有 Hopper 原生 kernel（2026-09-14 更正）。** cuBLASLt per-tensor INT8 只有 bf16 的 1.1～1.4×（838～975 TOPS），低于 FP8 per-tensor 的 1.6～2.0×。第 9 节查明原因：`torch._int_mm` 落到 Ampere 时代的 CUTLASS 2.x sm80 mma.sync kernel，而 bf16/FP8 走 sm90 nvjet wgmma kernel；int32 输出带宽和降频都已排除。自写的 CUTLASS SM90 INT8 kernel 达到 FP8 per-tensor 的 0.90～1.06×，说明"INT8 = FP8 速率"的硬件假设成立，第 1 节的相关表述以第 9 节为准。int8_speed.py 的 `[K,N]` 行主权重布局比 `W.t()` 视图慢 4～7×，如需继续用 `_int_mm` 必须改布局。
4. **FP8 g128 几乎免费。** DeepGEMM 达到 per-tensor FP8 的 0.91～1.03×，cuBLASLt 分块 0.81～0.92×。分组缩放本身不是障碍，缺的是 CUTLASS/DeepGEMM 级 INT8 kernel，而 INT8 在 H100 上又没有速率上限支撑。
5. **FP8 g64 没有快 kernel。** cuBLASLt 无 64 粒度配方；CUTLASS SM90 collective 需改 `ScalePromotionInterval % 4` 断言，预计为 g128 的 85%～90%，未验证。

### 8.4 对方案的影响
- 目标平台若是 H100，INT8 路线（S0 + Q4a 的 GEMM 部分）应放弃，转向 FP8 g128（DeepGEMM）或 FP8 g64（需改 CUTLASS）。前提是先在模拟器中补测 FP8 分组量化的精度，并查清第 6 节"FP8 模拟 g64 比 per-row 还差"的原因。
- INT8 g64 的精度优势只能在 INT8 速率真正 ≥ FP8 或无 FP8 的平台（A100、Thor、Ada）上换成速度；那些平台需要重新做本节测量。
- 注意事项：时钟无法锁定，20 个 case 中 11 个的 bf16 基线在 case 末尾复测漂移 +2.3%～+16.7%，M ≤ 1802 时 5%～10% 以内的差异应视为噪声；冲 L2 的计时在小 M 下偏保守。

### 8.5 GB200 补测：CUTLASS per-tensor INT8 孪生 kernel 的绝对吞吐（2026-09-13/14）

目的：验证 8.3 第 3 条"INT8 没有对 FP8 的速率优势"是库还是硬件的问题。做法：取 CUTLASS `examples/70_blackwell_gemm/70_blackwell_fp8_gemm.cu`
（SM100 tcgen05 2SM UMMA + TMEM，per-tensor scale_a·scale_b 在 epilogue 融合），只把 A/B 换成 int8、累加器换成 int32，其余不动；
代码、脚本与原始数据在 `docs/quantization/tools/cutlass_pertensor_gemm/`（README 有方法与结论；作业 2158916 / 2159012）。
环境：1× GB200（sm_100a，148 SM，2062 MHz 全程不限频），imaginaire4_v12.1.0 容器（torch 2.13.0a0 nv26.07，nvcc 13.3），CUTLASS 4.4.2。
计时口径与 8.1 相同：512 MB memset 冲 L2 后逐次 cudaEvent，50 次中位数；CUTLASS 每行对 fp64 精确参考采样校验 PASS。

结论：
- 同一 tile 下 INT8 与 FP8 的 CUTLASS kernel 同速（20 个 case 的 INT8/FP8 比 0.92～1.07，中位 1.00）；两者大方阵峰值都是约 3.4 PTOPS（名义 5 POPS 的 68%），bf16 约 1.6 PF。
- 最好的 INT8 配置（256x256x128 或 256x128x128，集群 2x1）在 M ≥ 4096 为 cuBLASLt FP8 per-tensor 的 1.02～1.15×，M ≤ 1802 为 0.75～1.01×（N=1024 最差）；
  更小 tile、2x2/4x1 集群都无收益，小 M 需要 stream-K/split-K（SM100 stream-K 归约模式目前挂起，未解）。
- GB200 上 cuBLASLt 有专门的 INT8 IMMA kernel（`_int_mm` 3.1～3.6 PTOPS，与其 FP8 相当或略高）；H100 上 cuBLASLt INT8 只有 FP8 的 0.58～0.73× 是库落到 Ampere 时代 kernel 的问题，§9 已用自写 SM90 kernel 证实（per-tensor INT8 = FP8 的 0.90～1.06×）。
  应是 SM90 库 kernel 的问题而非硬件上限；同一源码 `make ARCH=90` 已能编出 sm_90a 二进制（cooperative / pingpong），待在 H100 上实测。

#### q/o_proj 4096x4096（N×K）

| kernel | M=901 | M=1802 | M=4096 | M=16384 | M=42240 |
|---|---|---|---|---|---|
| bf16 cuBLAS | 1077 | 1359 | 1407 | 1670 | 1771 |
| FP8 cuBLASLt per-tensor (_scaled_mm) | 1380 | 2155 | 2611 | 3059 | 3332 |
| FP8 CUTLASS 128x128x128 c1x1 | 1263 | 1716 | 2422 | 2848 | 3006 |
| FP8 CUTLASS 128x256x128 c1x1 | 1199 | 1498 | 2151 | 2850 | 3094 |
| FP8 CUTLASS 256x128x64 c2x2 | 1263 | 1822 | 2246 | 2762 | 2961 |
| FP8 CUTLASS 256x128x128 c2x1 | 1440 | 1876 | 2611 | 3202 | 3348 |
| FP8 CUTLASS 256x256x128 c2x1 | 1449 | 1942 | 2563 | 3326 | 3610 |
| INT8 cuBLASLt (_int_mm, int32 out) | 1261 | 1667 | 2422 | 3132 | 3535 |
| INT8 CUTLASS 128x128x128 c1x1 | 1318 | 1668 | 2422 | 2824 | 2961 |
| INT8 CUTLASS 128x256x128 c1x1 | 1162 | 1538 | 2118 | 2834 | 3080 |
| INT8 CUTLASS 256x128x64 c2x2 | 1263 | 1769 | 2223 | 2747 | 2924 |
| INT8 CUTLASS 256x128x128 c2x1 | 1379 | 1865 | 2679 | 3202 | 3372 |
| INT8 CUTLASS 256x256x128 c2x1 | 1379 | 1942 | 2576 | 3368 | 3645 |

#### k/v_proj 1024x4096（N×K）

| kernel | M=901 | M=1802 | M=4096 | M=16384 | M=42240 |
|---|---|---|---|---|---|
| bf16 cuBLAS | 480 | 842 | 1141 | 1378 | 1549 |
| FP8 cuBLASLt per-tensor (_scaled_mm) | 645 | 1101 | 1724 | 2466 | 2804 |
| FP8 CUTLASS 128x128x128 c1x1 | 479 | 902 | 1498 | 2260 | 2537 |
| FP8 CUTLASS 128x256x128 c1x1 | 320 | 632 | 1377 | 1970 | 2501 |
| FP8 CUTLASS 256x128x64 c2x2 | 509 | 903 | 1500 | 2185 | 2570 |
| FP8 CUTLASS 256x128x128 c2x1 | 512 | 902 | 1624 | 2513 | 2804 |
| FP8 CUTLASS 256x256x128 c2x1 | 380 | 760 | 1644 | 2422 | 3025 |
| INT8 cuBLASLt (_int_mm, int32 out) | 552 | 848 | 1495 | 2278 | 2850 |
| INT8 CUTLASS 128x128x128 c1x1 | 483 | 902 | 1500 | 2292 | 2518 |
| INT8 CUTLASS 128x256x128 c1x1 | 315 | 640 | 1375 | 1971 | 2500 |
| INT8 CUTLASS 256x128x64 c2x2 | 480 | 903 | 1483 | 2185 | 2519 |
| INT8 CUTLASS 256x128x128 c2x1 | 480 | 900 | 1568 | 2513 | 2827 |
| INT8 CUTLASS 256x256x128 c2x1 | 401 | 724 | 1556 | 2417 | 3025 |

#### gate/up 12288x4096（N×K）

| kernel | M=901 | M=1802 | M=4096 | M=16384 | M=42240 |
|---|---|---|---|---|---|
| bf16 cuBLAS | 1312 | 1287 | 1601 | 1696 | 1461 |
| FP8 cuBLASLt per-tensor (_scaled_mm) | 2368 | 2381 | 2888 | 3238 | 3105 |
| FP8 CUTLASS 128x128x128 c1x1 | 1754 | 2318 | 2713 | 2947 | 2978 |
| FP8 CUTLASS 128x256x128 c1x1 | 1571 | 2176 | 2659 | 3054 | 3186 |
| FP8 CUTLASS 256x128x64 c2x2 | 1867 | 2258 | 2695 | 2953 | 2897 |
| FP8 CUTLASS 256x128x128 c2x1 | 1909 | 2384 | 3018 | 3066 | 3318 |
| FP8 CUTLASS 256x256x128 c2x1 | 2009 | 2381 | 3112 | 3521 | 3405 |
| INT8 cuBLASLt (_int_mm, int32 out) | 1951 | 2288 | 2930 | 3567 | 3474 |
| INT8 CUTLASS 128x128x128 c1x1 | 1722 | 2288 | 2713 | 2900 | 2942 |
| INT8 CUTLASS 128x256x128 c1x1 | 1570 | 2195 | 2676 | 3065 | 3095 |
| INT8 CUTLASS 256x128x64 c2x2 | 1843 | 2231 | 2694 | 2910 | 2683 |
| INT8 CUTLASS 256x128x128 c2x1 | 1909 | 2413 | 3041 | 3054 | 3086 |
| INT8 CUTLASS 256x256x128 c2x1 | 1996 | 2412 | 3136 | 3582 | 3575 |

#### down 4096x12288（N×K）

| kernel | M=901 | M=1802 | M=4096 | M=16384 | M=42240 |
|---|---|---|---|---|---|
| bf16 cuBLAS | 1276 | 1501 | 1589 | 1718 | 1524 |
| FP8 cuBLASLt per-tensor (_scaled_mm) | 2248 | 2707 | 2808 | 3264 | 3152 |
| FP8 CUTLASS 128x128x128 c1x1 | 1703 | 2004 | 2676 | 2982 | 2984 |
| FP8 CUTLASS 128x256x128 c1x1 | 1545 | 1800 | 2270 | 2837 | 2963 |
| FP8 CUTLASS 256x128x64 c2x2 | 1910 | 2177 | 2558 | 3066 | 2931 |
| FP8 CUTLASS 256x128x128 c2x1 | 1940 | 2231 | 2974 | 3402 | 3123 |
| FP8 CUTLASS 256x256x128 c2x1 | 1953 | 2411 | 2875 | 3522 | 3641 |
| INT8 cuBLASLt (_int_mm, int32 out) | 2139 | 2412 | 2808 | 3580 | 3572 |
| INT8 CUTLASS 128x128x128 c1x1 | 1692 | 2012 | 2660 | 2986 | 2749 |
| INT8 CUTLASS 128x256x128 c1x1 | 1539 | 1800 | 2261 | 2808 | 2993 |
| INT8 CUTLASS 256x128x64 c2x2 | 1907 | 2177 | 2554 | 3060 | 3135 |
| INT8 CUTLASS 256x128x128 c2x1 | 1909 | 2260 | 2974 | 3400 | 3082 |
| INT8 CUTLASS 256x256x128 c2x1 | 1952 | 2412 | 2882 | 3506 | 3491 |

#### 大方阵峰值（作业 2159012，热 L2 / 冲 L2，30 次中位数）

| M×N×K | bf16 cuBLAS | FP8 cuBLASLt | FP8 CUTLASS 256x256x128 | FP8 CUTLASS 256x128x128 | INT8 cuBLASLt | INT8 CUTLASS 256x256x128 | INT8 CUTLASS 256x128x128 |
|---|---|---|---|---|---|---|---|
| 8192×8192×8192 | 1624 / 1587 | 3040 / 3134 | 3373 / 3247 | 3148 / 3097 | 3112 / 3349 | 3356 / 3238 | 3100 / 3045 |
| 16384×8192×8192 | 1532 / 1495 | 3097 / 2971 | 3429 / 3427 | 3274 / 3260 | 3098 / 3248 | 3410 / 3416 | 3207 / 3207 |
| 16384×16384×8192 | 1456 / 1401 | 3142 / 2806 | 3048 / 3023 | 2060 / 2046 | 3269 / 3147 | 2907 / 2987 | 2042 / 2030 |

### 8.6 GB200 补测：INT8 g128（per-col）真 kernel（2026-09-14）

按 §2.1/§3.6 的 g128 per-col 定义（A per-token 1×128、W per-col 1×128、fp32 scale、组内 int32 精确点积、组间 fp32 累加、bf16 输出），
以 CUTLASS 示例 81 groupwise（SM100 blockwise-scaling collective）为基础做了 INT8 版：上游把 scale 与提升累加器类型绑成 MMA 累加器类型（int8 时是 int32，会静默按整数位模式乘），
用一份 shadow 头文件解耦为 fp32，并优化了 per-col 的 promotion（SFB 子块载入、TMEM 读取流水、TileK 256）。代码、patch、脚本、原始数据：`docs/quantization/tools/cutlass_g128_gemm/`（README 有方法与全部数字）。

结论：**GB200 上 g128 不是速度方案。** INT8 g128 per-col 优化后 1.09～1.32 PTOPS（M≥4096），是 bf16 cuBLAS 的 0.7～0.85×、同 tile per-tensor INT8（2.7～3.4）的 0.36～0.43×；
W 放粗到 128×128 blockwise 也只到 1.31～1.61（≈bf16）；FP8 孪生同样（per-col 1.24～1.49，blockwise 1.47～1.86）。原因是结构性的：GB200 tensor core 与 CUDA core 吞吐比约 64:1，
每元素每 128-K 的 FMUL+FFMA 提升 ≈ 该 K-tile 的 MMA 时间，Hopper 上这个比值是 ~15:1（所以 DeepGEMM 在 H100 能到 per-tensor 的 0.9～1.0×，见 8.2）。
要在 GB200 上保住 g128 精度并拿到速度，只能把逐元素提升从 CUDA core 拿掉：可分离权重 scale（s_w[n,g]=s_w[n]·c[g]，§3.6 已列、精度未测）或 per-token×per-channel 的 epilogue 缩放（=per-tensor 速率 3.3 PTOPS）。

#### q/o_proj 4096x4096（N×K）

| kernel | M=901 | M=1802 | M=4096 | M=16384 | M=42240 |
|---|---|---|---|---|---|
| bf16 cuBLAS | 1077 | 1359 | 1407 | 1670 | 1771 |
| INT8 per-tensor CUTLASS 256x128x128 | 1379 | 1865 | 2679 | 3202 | 3372 |
| INT8 g128 per-col, stock CUTLASS promotion | 367 | 398 | 524 | 553 | 561 |
| INT8 g128 per-col, opt (TileK 256) | 665 | 804 | 1092 | 1187 | 1224 |
| INT8 g128 W 128x128 blockwise, opt | 769 | 946 | 1311 | 1430 | 1489 |
| FP8 g128 per-col, opt (TileK 256) | 768 | 876 | 1238 | 1337 | 1388 |
| FP8 g128 W 128x128 blockwise (=示例 81 布局) | 859 | 1064 | 1468 | 1640 | 1702 |

#### k/v_proj 1024x4096（N×K）

| kernel | M=901 | M=1802 | M=4096 | M=16384 | M=42240 |
|---|---|---|---|---|---|
| bf16 cuBLAS | 480 | 842 | 1141 | 1378 | 1549 |
| INT8 per-tensor CUTLASS 256x128x128 | 480 | 900 | 1568 | 2513 | 2827 |
| INT8 g128 per-col, stock CUTLASS promotion | 170 | 332 | 417 | 522 | 544 |
| INT8 g128 per-col, opt (TileK 256) | 302 | 559 | 756 | 1079 | 1163 |
| INT8 g128 W 128x128 blockwise, opt | 328 | 606 | 874 | 1298 | 1393 |
| FP8 g128 per-col, opt (TileK 256) | 329 | 605 | 874 | 1216 | 1313 |
| FP8 g128 W 128x128 blockwise (=示例 81 布局) | 362 | 690 | 976 | 1468 | 1592 |

#### gate/up 12288x4096（N×K）

| kernel | M=901 | M=1802 | M=4096 | M=16384 | M=42240 |
|---|---|---|---|---|---|
| bf16 cuBLAS | 1312 | 1287 | 1601 | 1696 | 1461 |
| INT8 per-tensor CUTLASS 256x128x128 | 1909 | 2413 | 3041 | 3054 | 3086 |
| INT8 g128 per-col, stock CUTLASS promotion | 415 | 457 | 546 | 562 | 567 |
| INT8 g128 per-col, opt (TileK 256) | 841 | 970 | 1172 | 1218 | 1232 |
| INT8 g128 W 128x128 blockwise, opt | 1002 | 1155 | 1410 | 1441 | 1489 |
| FP8 g128 per-col, opt (TileK 256) | 948 | 1082 | 1319 | 1380 | 1396 |
| FP8 g128 W 128x128 blockwise (=示例 81 布局) | 1130 | 1308 | 1614 | 1675 | 1714 |

#### down 4096x12288（N×K）

| kernel | M=901 | M=1802 | M=4096 | M=16384 | M=42240 |
|---|---|---|---|---|---|
| bf16 cuBLAS | 1276 | 1501 | 1589 | 1718 | 1524 |
| INT8 per-tensor CUTLASS 256x128x128 | 1909 | 2260 | 2974 | 3400 | 3082 |
| INT8 g128 per-col, stock CUTLASS promotion | 404 | 426 | 553 | 574 | 582 |
| INT8 g128 per-col, opt (TileK 256) | 848 | 939 | 1234 | 1283 | 1315 |
| INT8 g128 W 128x128 blockwise, opt | 1013 | 1145 | 1488 | 1571 | 1609 |
| FP8 g128 per-col, opt (TileK 256) | 980 | 1058 | 1396 | 1458 | 1493 |
| FP8 g128 W 128x128 blockwise (=示例 81 布局) | 1159 | 1315 | 1711 | 1830 | 1860 |

## 9. 自写 CUTLASS SM90 INT8 kernel：per-tensor 对齐 + g128 分组缩放进主循环（2026-09-14）

背景：第 8 节发现 cuBLASLt 的 INT8 只有 FP8 per-tensor 的 0.6～0.7×。profiler 显示原因不在硬件：`torch._int_mm` 在 H100 上落到的是 Ampere 时代的
CUTLASS 2.x kernel `cutlass_80_tensorop_i16832gemm_s8_128x256_128x3_tn_align16`（mma.sync），而 bf16/FP8 走的是 Hopper 的 `nvjet_sm90_*` wgmma kernel。
K 从 4096 扫到 32768（输出大小不变）INT8 cuBLASLt 只从 848 升到 959 TFLOPS，FP8 始终约 1400，排除 int32 输出带宽；INT8 行采样到的 SM 时钟反而更高，排除降频。
一个未调优的 Triton INT8 wgmma kernel（bf16 输出）在同时钟下已达 1100～1160 TOPS。于是用 CUTLASS 3.x 自己写 Hopper 原生 INT8 kernel。
源码随本提交放在 `docs/quantization/tools/cutlass_int8_sm90/`（torch cpp_extension 编译，依赖 CUTLASS 4.2.1 头文件），基准脚本与日志在 `users/pzeren/int8_bench/`（`logs/bw_full.log`）。

### 9.1 per-tensor INT8（CollectiveBuilder，int8×int8→int32，TMA warp-specialized）
- 构造：`CollectiveBuilder<Sm90, OpClassTensorOp, int8_t RowMajor, int8_t ColumnMajor(=W[N,K] 行主), int32_t, TileShape, ClusterShape, StageCountAutoCarveout, KernelTmaWarpSpecializedCooperative>`，
  尾声 Sm90 EVT：D = bf16(acc · sa[m] · sb[n])（Sm90ColBroadcast × Sm90RowBroadcast，per-tensor 即把标量填成向量）。CUTLASS 4.x 的调度器标签是 `cutlass::gemm::PersistentScheduler`。
- 7 个 tile/cluster/调度配置试编（4096×4096，M=4096，TFLOPS）：128×128×128 c1×2 coop 1185；128×256×128 c1×1 coop 1320；128×128×128 c2×1 coop 1257；256×128×128 c1×1 coop 1319；
  64×128×128 pingpong 693；128×128×256 c1×2 coop 1193；**128×256×128 c2×1 coop 1400**。按"只编一个算子"的要求只保留最后一个。
- 结果（下表"INT8 pt"列）：M ≥ 4096 时 1104～1537 TFLOPS，为 cuBLASLt FP8 per-tensor（1155～1463）的 0.90～1.06×，是 cuBLASLt 自带 INT8（838～975）的 1.5～1.7×。
  M=901 时 128×256 的 tile 在 1024×4096 上只发 32 个 CTA（273 对 FP8 528），其他形状为 FP8 的 0.88～0.91×；小 M 需要第二个实例（64×128 pingpong）。
- 结论：**H100 的 INT8 与 FP8 MMA 速率在硬件上确实相同，缺的只是 cuBLASLt 里的 Hopper INT8 kernel。** 第 8.3 节第 3 条的"部分原因是功耗降频"据此更正。

### 9.2 g128 分组缩放进主循环（CUTLASS FP8 blockwise collective 的 INT8 移植）
- 结构：复制 CUTLASS 4.2.1 `sm90_mma_tma_gmma_ss_warpspecialized_fp8_blockwise_scaling.hpp`，新策略类型 `MainloopSm90TmaGmmaWarpSpecializedBlockwiseInt8`；
  `ElementBlockScale` 与累加器类型解耦为 fp32；`GmmaFP8Accumulation` 换成 `GmmaInt8Accumulation`：wgmma s8·s8 累加到 int32 临时累加器，每个 128-K tile（4 条 k32 wgmma）后转 fp32（精确：|和| ≤ 128·127² < 2²⁴）
  乘 `sa[m,kb]·sb[nb,kb]` 累进 fp32 主累加器；主累加器复用 kernel 传入的 int32 寄存器片段（按位重解释），尾声用自定义 EVT 叶子 `Sm90AccFetchBitcastF32` 按位取回。
  builder 断言 FP8 输入，因此 TiledMma / TMA copy / smem atom / stage 数按 builder 逻辑手工拼装。scale 布局 = `Sm90BlockwiseScaleConfig<1,128,128, MN, K>`（A：每行每 128-K 一个，torch `[K/128, M]`；W：每 128×128 块一个，torch `[N/128, K/128]`），
  与 3.6 节新加的 `QDQ_SIM_WEIGHT_BLOCK_N=128` 模拟选项一致；TMA 加载 A 的 scale 要求 M % 4 == 0（Python 侧补零行）。单实例 128×128×128、cluster 1×2、cooperative、6 级流水。
- **踩坑（约 1 小时）**：kernel 首次运行静默挂死。用 CUTLASS 自带 example 67（同一 collective 经 builder）做对照正常，把未修改的 FP8 collective 接进我的封装也挂死，定位到 cooperative kernel 的
  `IsMainloopAuxiliaryLoadNeeded = detail::HasAuxiliaryLoad_v<DispatchPolicy>`：负责 cp.async 加载 scale 的 `MainloopAux` producer warp 只在该 trait 为真时启用，CUTLASS 只对自己的 FP8 blockwise 策略特化了它。
  自定义策略必须补 `template<...> struct kernel::detail::HasAuxiliaryLoad<MyPolicy<...>> : cute::true_type {};`，否则 consumer 永远等不到屏障。
- 正确性：6 个形状（N,K ∈ {1024×4096, 4096×12288}，M ∈ {901, 1802, 4096}）对 fp64 精确分块数学的 rel-L2 全为 1.66e-3（bf16 输出舍入下限），max|d|/max|ref| ≤ 3.4e-3。
- 结果（TFLOPS，do_bench 中位数、冲 L2，同一次运行 `logs/bw_full.log`；时钟未锁定，噪声约 5%）：

| 形状 N×K | M | bf16 | FP8 pt | INT8 pt | INT8 g128 分块 | DeepGEMM FP8 g128 | 分块/bf16 | 分块/FP8 pt | 分块/INT8 pt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4096×4096 | 901 | 686 | 1109 | 1006 | 747 | 887 | 1.09 | 0.67 | 0.74 |
| 4096×4096 | 4096 | 730 | 1391 | 1407 | 977 | 1294 | 1.34 | 0.70 | 0.69 |
| 4096×4096 | 42240 | 729 | 1385 | 1456 | 1001 | 1316 | 1.37 | 0.72 | 0.69 |
| 1024×4096 | 901 | 351 | 528 | 273 | 323 | 329 | 0.92 | 0.61 | 1.18 |
| 1024×4096 | 4096 | 716 | 1155 | 1104 | 844 | 950 | 1.18 | 0.73 | 0.76 |
| 1024×4096 | 42240 | 735 | 1408 | 1464 | 1028 | 1304 | 1.40 | 0.73 | 0.70 |
| 4096×12288 | 901 | 744 | 1438 | 1292 | 884 | 1074 | 1.19 | 0.61 | 0.68 |
| 4096×12288 | 4096 | 747 | 1451 | 1537 | 997 | 1344 | 1.33 | 0.69 | 0.65 |
| 4096×12288 | 42240 | 771 | 1463 | 1479 | 974 | 1353 | 1.26 | 0.67 | 0.66 |
| 12288×4096 | 901 | 713 | 1321 | 1192 | 847 | 1116 | 1.19 | 0.64 | 0.71 |
| 12288×4096 | 4096 | 716 | 1410 | 1451 | 1007 | 1262 | 1.41 | 0.71 | 0.69 |
| 12288×4096 | 42240 | 746 | 1422 | 1278 | 911 | 1278 | 1.22 | 0.64 | 0.71 |

- 解读：**INT8 g128 分块 kernel 达到约 1000 TFLOPS，是 bf16 的 1.35×，per-tensor FP8/INT8 的 0.70×，DeepGEMM FP8 g128 的 0.75×，SGLang Triton INT8 g128（第 8 节 646）的 1.5×。这是 H100 上 INT8 分组量化第一次跑出正收益。**
  与 per-tensor 的 30% 差距由两部分构成：分块结构本身（同为 128×128 c1×2 时 per-tensor 1185 对分块 977，约 16%）和只能用 128×128 tile（fp32 主累加器 + int32 临时累加器把寄存器翻倍，128×256 会溢出 cooperative 的 232 寄存器预算）。
  DeepGEMM FP8 g128 高 25%，但把同一个 collective 编成 FP8 版做对照只有 940～1119（4096×4096 / 4096×12288，M 4096 / 42240），即 INT8 固有开销（提升多一条 I2F）≤ 10%，其余约 20% 是 CUTLASS 通用 collective 对比 DeepGEMM 手写实现的差距，tile/cluster/stage/光栅参数扫遍只有 ≤ 3%（9.4 节据此改用 DeepGEMM 结构；2026-09-14 更正）。

### 9.3 对方案的影响与下一步
- 第 8.4 节"INT8 路线应放弃"需要修正为：**INT8 g128 在 H100 上可以拿到 1.35× bf16 的 GEMM 收益**，精度按 3.4/3.6 节 g128 + 权重 128×128 块的模拟结果评估；仍低于 FP8 per-tensor，但保住 INT8 的精度优势。
- g64：collective 断言 `ScalePromotionInterval % 4 == 0`（g64 对应间隔 2，k32 wgmma × 2），需要改主循环的提升节奏（每 64-K 提升一次，FFMA 数翻倍）；survey 预估相对 g128 再降 10%～15%。
- ~~性能调优：先在 g128 上试 cluster 2×1、stage 数、光栅顺序，目标是 DeepGEMM FP8 g128 的 1300 量级~~ → 已做（9.4 节）：DeepGEMM 结构移植 + 双缓冲/ping-pong/双 CTA 四轮后，精确 INT8 g128 的实测上限约 1085～1140 TFLOPS，FP8 g128 的 0.84～0.89×；剩余差距是暴露的 int32→fp32 提升，结构性。下一个杠杆是 g256（提升减半，精度代价见 3.9）。
- 激活侧：per-token-per-128 的 INT8 动态量化 Triton kernel 已在 int8_bench（约 11 µs @ 901×4096），与本 kernel 的 `[K/128, M]` scale 布局直接对接。

### 9.4 INT8 g128 极致性能：DeepGEMM 结构移植与三种提升/MMA 重叠方案（2026-09-14）

目标：把 9.2 的约 1000 TFLOPS 推到 DeepGEMM FP8 g128 的 1300。源码：`docs/quantization/tools/deepgemm_int8_sm90/`（`gen_kernel*.py` 从 DeepGEMM 66081d4 的
`sm90_fp8_gemm_1d2d.cuh` 生成 INT8 版；`include/deep_gemm/` 为其头文件副本，MIT 许可见 `LICENSE.DeepGEMM`；四份串行计时报告在 `results/`）。所有变体均与默认 kernel **逐位相同**
（独立复核 136～328 组，含 ±127 极值、1e-4/1.0 极端 scale、NaN 填充、K=128/256 边界、5000 次变 M 压力测试无挂死）。

- **DeepGEMM 结构 INT8 移植**（wgmma `m64nNk32.s32.s8.s8`，块内 int32 累加，每 128-K 提升 `final += (sa·sb)·float(acc)`，TMA multicast、持久调度器、19 个编译配置）：
  256×128 s3 cluster 1×2 达 1055～1094（M ≥ 4096，N ≥ 4096），任意 M 直接可用（TMA 越界零填充）。
- **关键实验**：把提升里的 int32→fp32 转换去掉（数值错误，仅探上限）同一流水线跑到 **1405～1527 TFLOPS，高于 FP8**——INT8 的 MMA 侧没有劣势，全部差距是暴露在关键路径上的提升（每元素 I2FP + FFMA，tensor core 空转）。
- **A 单 warpgroup 内 wave 交错**：位精确但慢 13～22%——ptxas C7514：在另一组 wgmma 在飞时读已完成组的累加器会让整个 wgmma 流水串行化（每条 IGMMA 后一个 DEPBAR）。
- **B 双缓冲 int32 累加器**（128×128 tile，直线展开 8 块让缓冲奇偶编译期确定，`wait_group<1>`，0 溢出、无 C7514）：**+7%**（4096³ 1128～1140），瘦形状 1024×4096、M=901 +12%（479，超 FP8 g128 1.26×），是多数形状的新最优。
  256×128 上不可行（主累加 128 + 两组 int32 = 384 寄存器）。
- **C 两个 math warpgroup 顺序屏障 ping-pong**（256×128，不加寄存器）：+3～4%，在 K=12288 与 M=42240 的形状上最优。它同时测出一个硬件事实：**单个 warpgroup 独占 tensor core 只能到 INT8 峰值的 57%**
  （64×128×128 一批约 500 周期，理论 277），必须两组同时发射；而发射顺序上的错位会被 tensor core 的公平仲裁抹平，两组仍同时结束、同时提升；真正互斥则慢 8～15%。
- **D 每 CTA 一个 math warpgroup、每 SM 共驻 2 个 CTA**（寄存器 216/40，smem 91 KB）：占用率 2 实测为真、位精确，但只有最优的 0.65～0.89×——预算只允许 64×128 tile，
  操作数搬运量是大 tile 的 1.5～2 倍，共驻带来的提升重叠（同 tile +20%）被访存吃掉，且与双缓冲不叠加。
- **同一 collective 的 FP8 对照**（9.2 的 CUTLASS 版）：FP8 940～1119 对 INT8 966～1021，INT8 固有开销 ≤ 10%。

串行计时（空闲 H100，do_bench 中位数 ×3 取中位，冲 L2，时钟未锁；`results/int8_g128_occupancy2.md`）：
| 形状 N×K | M | 最优 INT8 g128 精确 kernel | TFLOPS | 默认移植 256×128 | DeepGEMM FP8 g128 | CUTLASS INT8 per-tensor | cuBLASLt FP8 per-tensor | 最优/FP8 g128 | 最优/per-tensor INT8 |
|---|---|---|---|---|---|---|---|---|---|
| 1024×4096 | 901 | B DB cfg 27（64x128 s8 c1x1 DB） | 479 | 223 | 380 | 268 | 516 | 1.26 | 1.79 |
| 1024×4096 | 4096 | B DB cfg 25（128x128 s5 c1x2 N1024K4096 DB） | 930 | 904 | 1047 | 1118 | 1143 | 0.89 | 0.83 |
| 1024×4096 | 42240 | B DB cfg 25（128x128 s5 c1x2 N1024K4096 DB） | 1092 | 1065 | 1223 | 1427 | 1264 | 0.89 | 0.77 |
| 4096×4096 | 901 | B DB cfg 21（128x128 s5 c1x1 N4096K4096 DB） | 848 | 799 | 962 | 978 | 1077 | 0.88 | 0.87 |
| 4096×4096 | 4096 | B DB cfg 21（128x128 s5 c1x1 N4096K4096 DB） | 1122 | 1048 | 1278 | 1371 | 1335 | 0.88 | 0.82 |
| 4096×4096 | 42240 | C pp cfg 5（256x128 s3 c1x2 N4096K4096 pp1） | 1101 | 1071 | 1246 | 1393 | 1269 | 0.88 | 0.79 |
| 4096×12288 | 901 | C pp cfg 11（256x128 s3 c1x2 N4096K12288 pp1） | 977 | 941 | 1164 | 1276 | 1410 | 0.84 | 0.77 |
| 4096×12288 | 4096 | C pp cfg 11（256x128 s3 c1x2 N4096K12288 pp1） | 1132 | 1091 | 1326 | 1497 | 1303 | 0.85 | 0.76 |
| 4096×12288 | 42240 | C pp cfg 11（256x128 s3 c1x2 N4096K12288 pp1） | 1113 | 1067 | 1329 | 1467 | 1309 | 0.84 | 0.76 |
| 12288×4096 | 901 | B DB cfg 23（128x128 s5 c1x2 N12288K4096 DB） | 950 | 914 | 1127 | 1176 | 1293 | 0.84 | 0.81 |
| 12288×4096 | 4096 | B DB cfg 23（128x128 s5 c1x2 N12288K4096 DB） | 1093 | 1066 | 1233 | 1397 | 1237 | 0.89 | 0.78 |
| 12288×4096 | 42240 | C pp cfg 12（256x128 s3 c1x2 N12288K4096 pp1） | 1105 | 1054 | 1254 | 1287 | 1265 | 0.88 | 0.86 |

结论：**精确 INT8 g128 在 H100 上的实测上限约 1085～1140 TFLOPS = bf16 的 1.5×、DeepGEMM FP8 g128 的 0.84～0.89×、per-tensor INT8/FP8 的 0.76～0.87×**（瘦形状 1024×4096、M=901 例外：1.26×/1.79×）。
剩余 11～16% 全部来自两个 lock-step warpgroup 同时做的 int32→fp32 提升，FP8 没有这一步（wgmma 直接输出 fp32、提升仅一条 FFMA）。tile/cluster/stage/光栅/尾声等参数在两套代码上都已穷举（≤ 3%）。
可能的剩余手段：g256 使提升次数减半（3.9 节精度代价约 0.5 dB，预计收回差距的约一半，未实测）；两个 warpgroup 一次性错开半个 k-block（不加逐迭代屏障）；2-CTA cluster 共享 B 的 multicast。
生产选择：4096×4096 与 N=1024/12288 小 M 用 B 双缓冲（cfg 21/23/25/27），K=12288 与 M=42240 用 C pp1 256×128（cfg 5/11/12）。

## 10. 跨架构总结：INT8 GEMM 在 H100 / GB200 / Thor 上的位置（2026-09-14；Thor 列已按 §11 实测更新）

全部数字为 GEMM-only、冲 L2（Thor 为热 A 冷 W 持续）、中位数；"per-tensor" 泛指 scale 在 epilogue 一次性施加的路径（含 per-token×per-channel），"g128/g256" 指 scale 沿 K 每 128/256 元素变化、必须在主循环内逐 K-tile 提升的路径。

| | H100 SXM（sm_90a，§8/§9） | GB200（sm_100a，§8.5/§8.6） | Thor（sm_110a，§11） |
| --- | --- | --- | --- |
| bf16 cuBLAS | 0.67～0.77 PF | 1.4～1.8 PF | 113～160 TF |
| per-tensor FP8 | 1.1～1.46 PF（cuBLASLt） | 3.0～3.4 PF（cuBLASLt / CUTLASS） | cuBLASLt 约 250 TF；FP8 MMA 受 99 W 电源轨限流（真实数据 217，全零 390） |
| per-tensor INT8 | 自写 CUTLASS SM90 = FP8 的 0.90～1.06×；cuBLASLt 只有 0.6～0.7×（Ampere 时代 kernel） | CUTLASS = FP8 的 0.92～1.07×，3.4 PTOPS；cuBLASLt `_int_mm` 亦达 3.1～3.6 | CUTLASS 345 TF，**INT8 基本不受限流** = 同结构 FP8 的 1.44～1.58×、cuBLASLt FP8 的 1.19～1.30×、bf16 的 2.0～2.45× |
| g128 INT8（W 128×128 块） | 1085～1140 TF = bf16 的 1.5×、per-tensor 的 0.76～0.87×（§9.4 DeepGEMM 结构） | 1.3～1.6 PTOPS = bf16 的 0.9～1.0×、per-tensor 的 0.45×（4 个提升 warp、256×128） | 190～237 TF = bf16 的 1.3～1.7×、cuBLASLt FP8 pt 的 0.75～0.97×（8 个提升 warp + 256×256 tile） |
| g128 INT8（W per-col 1×128，S0 布局） | 未做 | 1.1～1.3 PTOPS = bf16 的 0.7～0.85×、per-tensor 的 0.36～0.43× | 133～159 TF = bf16 的 1.0～1.3×（宽 N）/ ≤1（窄 N）、FP8 pt 的 0.4～0.7×；地板 = 每元素 I2F+FMUL+FFMA 三条 |
| g256 INT8（per-col） | 未做 | 未做 | **224～268 TF = cuBLASLt FP8 pt 的 0.98～1.30×、bf16 的 1.8～2.1×**（提升次数减半后被 MMA 藏住） |
| g128 FP8 | DeepGEMM = per-tensor 的 0.91～1.03× | 块 1.5～1.9 / per-col 1.2～1.5 PTOPS | 块 206 / per-col 163 TF |
| 结论 | INT8 g128 有正收益（1.5× bf16），g64 待做 | per-tensor 是唯一实测到 INT8 速率的路径；g128 在现有实现下 ≈ bf16，**Thor 的三项手段尚未在 GB200 试** | per-tensor INT8 与 g256 per-col 都拿到 FP8 速率；g128 per-col 只在宽 N 上胜 bf16 |

原因（§8.6/§11.2）：软件 block-scaling 每个输出元素每 128-K 要在 CUDA core 上发 2（W 块：I2F+FFMA）～3（per-col：I2F+FMUL+FFMA）条指令；Blackwell 的 tensor core 对 CUDA core 吞吐比约 64:1
（Hopper 约 15:1），所以这段提升在 Hopper 上只占 MMA 时间的约 20%，在 Blackwell 上 ≥ 100%，MMA 只能等它。Thor 上把它藏回去的办法有三个，GB200 上应同样有效但未测：
(1) 用闲置的 warp 8-11 做第二个提升 warpgroup（FMA 发射能力翻倍）并把 256×256 tile 的 fp32 全累加器分到两组（§11.2，W 块从 0.76× 到 0.97× FP8 pt）；
(2) g256：提升次数减半，发射地板 209 拍 < MMA 256 拍（§11.2，per-col 追平 FP8 pt；模拟器 §3.9：逐列 g256 比 g128 掉约 0.5 dB，保住张数不变）；
(3) TMEM 预偏置去掉半速 I2F（per-col 3→2 条；已数值验证，因 tcgen05.st 重装造成溢出目前更慢，§11.2）。
Blackwell 的硬件答案是 MXFP8/NVFP4 的块缩放 MMA（32 元素 UE8M0 scale），INT8 没有对应指令。

对 Blackwell 平台（GB200 / Thor）的选择（按 §11.3 与 §3.9 更新）：
1. per-col g128（S0 布局）是唯一被接受的精度布局；它在 Thor 上只有 FP8 pt 的 0.4～0.7×，要么接受，要么走 (3) 预偏置。
2. g256 per-col：Thor 上已达 FP8 pt 速率，模拟器上比 g128 掉约 0.5 dB（12 图，需 36 图复核）——精度若过，是 Blackwell 上性价比最高的分组方案。
3. 可分离 s_w[n]·c[g]（`ls_noclip`）e2e 精度与 per-col g128 在噪声内持平（§3.9：26.28 对 26.61 dB），kernel 成本 = W 块档；§11.2 判其"多半过不了"写在 §3.9 之前，建议按 §3.9 复核后再定。
4. per-token×per-channel（epilogue 缩放）= per-tensor 速率（Thor 1.1～2.6× bf16），精度按 pzeren 判断不达标。
5. GB200 下一步：把 `tools/cutlass_int8_sm110/` 的 8-warp 提升 kernel、256×256、g256 配置以 `ARCH=sm_100a` 编译（README 标注"GB200 未测"）重跑 §8.6 的形状，验证 GB200 是否同样能到 FP8 pt 速率。

## 11. Thor（Jetson AGX Thor，sm_110a）实测：per-tensor INT8 对齐 FP8 + 功耗结论（2026-09-14）

第 10 节对 Thor 的预测（"需实测，但结构相同"）在此实测。代码与全部原始数据：`docs/quantization/tools/cutlass_int8_sm110/`（README 为完整报告）。纯 CUDA C++（无 torch 依赖），CUTLASS 4.8 `70_blackwell_fp8_gemm.cu` 为模板，
同一个模板实例化 FP8（e4m3，fp32 累加）和 INT8（s8，int32 累加），epilogue 用 EVT 融合 `sa*sb` 输出 bf16；cuBLASLt 基线（bf16 / FP8 per-tensor / INT8 s32 输出）、
带宽微基准、功耗探针脚本。机器：20 SM、L2 32 MB、DRAM 235 GB/s、默认 120W 模式 GPU 上限 1386 MHz（MAXN 1575）。目标形状按要求只保留 q/o（4096×4096）和 k/v（1024×4096），
M ∈ {901, 1517, 1802, 4096}；MLP 形状与视频级 M 只有部分数据（`results/full_v1_*`）。

### 11.1 结论
- **per-tensor INT8 已对齐 FP8**：同结构（2SM 256×256×128 tcgen05、cluster 2×1、CLC 调度）下 INT8 ≥ FP8；生产状态（激活热、权重冷、持续运行）q/o_proj 上 INT8 = 同结构 FP8 的 1.44～1.58×、
  cuBLASLt FP8 的 1.19～1.30×、cuBLASLt bf16 的 2.0～2.45×（M=901/1517/1802：110/158/209 µs）。k/v_proj 是纯权重流（4 MB），INT8 = FP8，两者为 cuBLASLt FP8 的 0.79～0.88×（16～64 个 tile 的尾波，stream-K 待做）。
- **FP8 MMA 在 Thor 上受功耗限流，INT8 基本不受**：GPU 电源轨上限约 99 W（root tegrastats VDD_GPU 实测），时钟锁死 1575 MHz 时依然如此；全零输入两者都 ~390 TOPS，真实数据 FP8 217 / INT8 345；
  限流约 0.5 s 后触发（逐次迭代时间序列）。**同功耗下 INT8 每次 GEMM 少 35～40% 能量（算力受限），或同时间下低 18～29% 功率（内存受限）**；不是"功耗低 30%"。
- Thor 特性：cluster 大小 ≤ 2 才能用满 20 SM（4-CTA cluster 只驻留 4 个）；L2→SMEM 约 1.5 TB/s 决定要用最大 tile；权重 > L2 时必须开 raster swizzle（MLP 形状 3×），权重 ≤ 16 MB 时关掉；
  FP8 基准必须 ≥ 1.5 s 持续预热，且不能在迭代间冲 L2（1 ms 空隙让限流恢复，会得到"FP8 = INT8"的假象）。
- cuBLASLt INT8 没有带 scale 的 bf16 epilogue，int32 输出 + 单独 rescale 只有 cuBLASLt FP8 的 0.62～0.76×；与 H100 一样，INT8 可用的前提是自写融合 epilogue 的 kernel。

### 11.2 g128 分组缩放（2026-09-14 下午，`tools/cutlass_int8_sm110/blockwise_gemm.*`）
复用第 10 节 GB200 的 INT8 blockwise collective 补丁，在 Thor 上加了三处通用修正：(1) CUTLASS `arch/reg_reconfig.h` 没有 sm_110a，`setmaxnreg` 被编译成空，
提升 warp 被卡在 168 寄存器（溢出、scale 逐 16B 装载）——shadow 头补上后 per-col 686→371 µs、W 块 301→268 µs；(2) epilogue 子块 128×16 降低 TMEM 片段寄存器占用（per-col 1.48×）；
(3) TMEM 读三缓冲、每两个子块等一次（W 块 1.12×）。Thor 没有 FFMA2（f32x2 被 ptxas 拆开），per-col 每元素每 128-K 三条 FP 指令是硬地板。
结果（热 A 冷 W 持续，4096×4096 M=1520，TFLOPS）：INT8 g128 W 128×128 块 **190**（bf16 的 1.44×，MAXN bf16 160 的 ~1.2×，cuBLASLt FP8 per-tensor 249 的 0.76×，
离该 256×128 tile 结构的天花板 226 只差 8%）；INT8 g128 per-col **135**（≈ bf16，FP8 per-tensor 的 0.54×）；FP8 g128 孪生 206 / 163。1024×4096 上 g128 全部低于 bf16（4 MB 权重流、tile 太少）。
要再上一层只有结构性改动：256×256 tile 需要把 fp32 全累加器分到 8 个提升 warp（kernel 级 fork），per-col 另需可分离权重 scale s_w[n,g]=s_w[n]·c[g]（精度待模拟器评估）。同条件对比表（基线在同样的 M=904/1520/1804 上重测，含加速比）：`results/g128_summary_20260914_101555.md`（脚本 `bench_g128_summary.py`）——
4096×4096 上 INT8 g128 W 块对 bf16 1.61/1.39/1.28×、对 cuBLASLt FP8 per-tensor 0.75/0.74/0.66×；per-col 对 bf16 1.15/0.99/0.91×；1024×4096 上 g128 均不如 bf16（权重流 + tile 太少，应走 per-tensor 或 QKV 合并）。
FP8 g128 W 块与 CUTLASS FP8 per-tensor 256×256 时间完全相同（250/324 µs，同一功耗上限），INT8 g128 W 块与其只差 7%。

可分离权重 scale 的 kernel 已实现（cfg12：W 块主循环 + epilogue 按通道乘 s_w[n]，c[g] 并入激活 scale）：INT8 164/182/168 TFLOPS（4096×4096，M=904/1520/1804），
= per-col 的 1.34×、W 块的 0.96×、bf16 的 1.22～1.53×、cuBLASLt FP8 per-tensor 的 0.63～0.71×；精度（N×K/128 scale 矩阵的秩 1 约束）待模拟器评估。同一 kernel 还能零成本跑更细的组合布局 s_w[n]·c[nb,g]（每通道因子 × 每 (128 通道块, K 块) 因子，把 c[nb,g] 当主循环的 sfb 传入），
自由度 N + (N/128)(K/128)，严格细于可分离和 W 128×128 块；建议三种布局在模拟器里一起评估。表：`results/g128_separable_20260914.md`。

**8-warp 提升 kernel + 256×256 tile（下午晚些，shadow kernel `include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized_mma_transform.hpp`）**：
把闲置的 warp 8-11 变成第二个提升 warpgroup（scale 装载挪到 warp 3，仅支持 C 为 void），两组各管一半 epilogue 子块，tile 结束时第二组经 TMEM 把半个 fp32 全累加器交给 epilogue 组
（两个 named barrier），寄存器 48/216/216（232 恰好占满 64K 会让 `setmaxnreg.inc` 永久等待——挂死；224 反而溢出）。这样 fp32 全累加器每线程 128 个寄存器，256×256 tile 成为可能。
结果（热 A 冷 W 持续，4096×4096，M=904/1520/1804，TFLOPS）：**INT8 g128 W 128×128 块 190/229/210 = cuBLASLt bf16 的 1.62～1.68×、cuBLASLt FP8 per-tensor 的 0.84/0.97/0.86×（M=1520 基本追平）**、
INT8 per-tensor 的 0.70×；INT8 g128 per-col 133/159/146（bf16 的 1.12～1.18×，FP8 pt 的 0.59～0.67×）；可分离布局在 256×128 上 168/187/173（256×256 版溢出待调）。
1024×4096 上 W 块 1.04～1.08× bf16。同条件表：`results/g128_summary_20260914_114605.md`，目标形状表 `results/g128_8warp_targets_20260914.txt`。
hidden_size=1536 的形状（1536×1536 / 6144×1536 / 1536×6144 / 512×1536）：权重全在 L2、tile 数少，W 块 g128 对 bf16 为 0.95～1.42×（MLP 1.1～1.4×）、对 cuBLASLt FP8 per-tensor 0.58～0.83×，
per-col 与 bf16 持平或更慢；这一档最值钱的是 QKV / gate-up 合并成一次 GEMM。表：`results/g128_summary_h1536_*.md`。
合并 GEMM 形状（QKV 6144×4096、gate+up 24576×4096；1536 档 4608×1536、12288×1536）：W 块 g128 对 bf16 1.07～1.89×、对 cuBLASLt FP8 per-tensor 0.66～1.22×（24576×4096 M=1520 达 1.22×）；
per-col g128 对 bf16 0.79～1.30×——合并投影是让 INT8 g128 在所有层都快于 bf16 的模型侧手段。表：`results/g128_summary_fused_*.md`。
接续说明（另一台机器如何编译/运行/未完事项）见 `tools/cutlass_int8_sm110/README.md` 末节。

**TileK=256 与布局取舍（2026-09-14 晚）**：256×256 tile 上把 TileK 提到 256（cfg19 W 块 / cfg18 per-col）：W 块 +4%（4096² M=1520 228→237、M=904 190→198、24576×4096 242→251 TFLOPS），
per-col −2%（158→155，cfg18 还溢出 48 B），所以 per-col 保持 cfg16（TileK=128）。表：`results/g128_tilek256_20260914.md`。
按 pzeren 判断，可分离 s_w[n]·c[g] 与 W 128×128 块两种布局的精度多半过不了，上面 W 块的数字只当 INT8 主循环的上界，后续以 **per-col g128（S0：s_a[m,g]、s_w[n,g]，cfg16）** 为唯一主线。其同条件位置（`results/percol_standing_20260914.md`）：

| N x K | INT8 g128 per-col 256x256 (cfg16) TFLOPS, M=904/1520/1804 | cuBLASLt bf16 TFLOPS | vs bf16 (time ratio) | vs cuBLASLt FP8 per-tensor |
|---|---|---|---|---|
| 4096x4096 | 133 / 159 / 145 | 113 / 142 / 124 | 1.18 / 1.12 / 1.16 | 0.59 / 0.67 / 0.59 |
| 1024x4096 | 106 / 124 / 112 | 143 / 160 / 154 | 0.75 / 0.78 / 0.73 | 0.46 / 0.44 / 0.38 |
| 6144x4096 | 141 / 159 / 143 | 156 / 138 / 132 | 0.91 / 1.16 / 1.08 | 0.51 / 0.60 / 0.62 |
| 24576x4096 | 147 / 167 / 149 | 129 / 128 / 124 | 1.14 / 1.30 / 1.20 | 0.64 / 0.84 / 0.66 |
| 1536x1536 | 91 / 117 / 113 | 123 / 125 / 149 | 0.74 / 0.93 / 0.75 | 0.47 / 0.54 / 0.49 |
| 512x1536 | 70 / 69 / 82 | 88 / 107 / 117 | 0.80 / 0.63 / 0.69 | 0.65 / 0.46 / 0.46 |
| 6144x1536 | 121 / 138 / 125 | 132 / 134 / 151 | 0.91 / 1.03 / 0.82 | 0.51 / 0.59 / 0.62 |
| 1536x6144 | 117 / 148 / 138 | 145 / 139 / 137 | 0.81 / 1.06 / 1.00 | 0.41 / 0.46 / 0.45 |
| 4608x1536 | 112 / 137 / 123 | 142 / 129 / 128 | 0.78 / 1.06 / 0.96 | 0.48 / 0.60 / 0.59 |
| 12288x1536 | 124 / 144 / 129 | 123 / 123 / 109 | 1.01 / 1.17 / 1.18 | 0.69 / 0.68 / 0.67 |

结论：per-col g128 只在宽 N（4096×4096、6144×4096、24576×4096、12288×1536）上稳定快于 bf16（1.0～1.3×），1536 档未合并的层与 bf16 持平（0.8～1.1×），
1024×4096 / 512×1536 / 1536×1536 慢于 bf16；对 cuBLASLt FP8 per-tensor 为 0.4～0.7×。硬地板是提升循环每元素每 128-K 三条 FP 指令（I2F、FMUL、FFMA）且 Thor 无 FFMA2。
1536 档全部 bf16 / INT8 数字（per-tensor、W 块、可分离、per-col、两种 tile）：`results/h1536_bf16_int8_table.md`。

**per-token × per-channel（row × col epilogue 缩放）、指令发射率与 g256（2026-09-14 下午晚些）**：
(1) `pertensor_gemm --scale=rowcol`（D = bf16(acc·s_a[m]·s_w[n])，`Sm90ColBroadcast × Sm90RowBroadcast` EVT，主循环即 per-tensor kernel）在全部 10 个形状上与 per-tensor 速度相同（0.96～1.04×）：
对 cuBLASLt bf16 1.1～2.6×、对 cuBLASLt FP8 per-tensor 0.8～1.6×（只在 1024×4096、512×1536 低于 1）；FP8 孪生在大形状上照样被功耗上限压住（4096² M=1520 204 对 INT8 327）。这是 scale 不进主循环的 INT8 方案的速度上限；精度按 pzeren 判断不达标。表：`results/rowcol_standing_20260914.md`。
(2) 微基准（`tools/cutlass_int8_sm110/microbench/`）：Thor 上 FFMA/FMUL/FADD 128/拍/SM，**I2F（int32→fp32）只有 64/拍/SM（半速）**，F2I 16，FFMA2 被拆成两条；老的寄存器累加 `mma.sync` s8 只有 112.7 TOPS = tcgen05 的 1/4，
所以累加器放不到寄存器（tcgen05.mma 的 D 只能在 TMEM，sm_100/110 无 wgmma），TMEM 也不是瓶颈（每 64 元素一条 tcgen05.ld），瓶颈是每元素每 K 组的 FP 发射数：per-col I2F+FMUL+FFMA 实测 3.27 拍/元素，W 块 I2F+FFMA 2.04。
砍掉 I2F 的办法：把 TMEM int32 累加器预置成 0x4B400000（1.5·2²³ 的位型），读回的位型直接是 float(1.5·2²³+x)（|x| ≤ 128·127² 精确，穷举验证），提升变成 2 条 FFMA（2.06 拍/元素，−37%），W 块无收益；未实现。
(3) **g256 per-col（cfg20：2SM 256×256×256，sf<1,1,256>，每 256 K 提升一次）**：提升次数减半后每块发射地板 418→209 拍，低于 256 拍的 MMA 时间，提升被藏住。同条件实测（TFLOPS，M=904/1520/1804）：
4096×4096 **224/268/248**（g128 per-col 的 1.7×，bf16 的 2.1/1.9/1.8×，**cuBLASLt FP8 per-tensor 的 0.98/1.05/1.30×**，INT8 per-tensor 的 0.81～0.86×）；6144×4096 229/271/244（FP8 pt 0.89/1.07/0.98×）；24576×4096 M=1520 230～262（两次运行，FP8 pt 1.08～1.31×）；
12288×1536 171/202/185（FP8 pt 0.91～0.98×）；6144×1536 181/195/182；1536×1536 130/167/166（bf16 1.06～1.33×，FP8 pt 0.67～0.77×）；1024×4096 168/196/185（bf16 1.1～1.3×）。
全表：`results/g256_percol_standing_20260914.md`。g256（per-token 1×256、per-channel 1×256）的精度需模拟器判定；若必须保 g128，则走 TMEM 预偏置提升（同一发射地板）。

**汇总报告**：当天全部测试的一页结论 + 各方案吞吐 / 加速比表 + 微基准 + 预偏置状态见 `tools/cutlass_int8_sm110/results/THOR_INT8_REPORT_20260914.md`。

### 11.3 下一步
（以 per-col g128 为唯一可接受布局重排）
1. 模拟器精度：先确认 per-col g128（S0）达标，并评估 **g256 per-col**（Thor 上 g256 已追平 cuBLASLt FP8 per-tensor，g128 只有其 0.6×）；可分离 / W 块两种布局预计不达标，不再作为性能主线。
1b. 若必须保 g128：实现 TMEM 预偏置提升（去掉半速 I2F，2 条 FFMA/元素，`microbench/README.md`），预期 per-col g128 从 159 到 ~230～260 TFLOPS。
2. per-col kernel 继续调：MAXN 下重测（g128 不受功耗上限约束，应随 1575/1386 频率放大，FP8 per-tensor 不会）；提升循环的指令级调度（scale 的 ld.shared 与 TMEM 读交错；每 K 块 3 条 FP 指令是地板）；保持 cfg16 / TileK=128。
3. 模型侧：QKV / gate-up 合并成一次 GEMM（per-col g128 只在宽 N 上快于 bf16）；小 N 层（1024×4096、512×1536、1536×1536）若精度允许留 bf16。
4. torch 扩展绑定接入 cosmos-framework；INT8 attention（Q·Kᵀ、P·V）是把 INT8 优势延伸到非 GEMM 部分的下一处。
5. 向 CUTLASS 上游报告 `arch/reg_reconfig.h` 缺 sm_110a（`setmaxnreg` 被编译成空）。

## 12. 方案定稿（2026-09-14，pzeren 决定）

| 角色 | 方案 | 依据 |
| --- | --- | --- |
| **主线** | **per-col g256**（激活 per-token 每 256 K 一个 scale，权重每输出通道每 256 K 一个 scale；SmoothQuant 见 12.1，改为可选） | Thor：g256 per-col 224～268 TFLOPS，追平 cuBLASLt FP8 per-tensor（0.98～1.30×），bf16 的 1.8～2.1×（§11.2）；Nano t2i 12 图 23.8 dB / 保住 10/12，与 g128 持平，比官方 FP8 高 6 dB（§3.8） |
| 拿掉 | g64 | kernel 每 64 K 提升一次，H100 上 FMA 预算约 100%，Thor 上无性能出路；精度（Nano 25.8、Edge 29.3）只作上限参照 |
| 拿掉 | g128 | Thor 上只有 bf16 的 1.1～1.2×、cuBLASLt FP8 pt 的 0.6～0.7×；精度与 g256+SQ 持平，没有保留理由 |
| 拿掉 | W 128×128 块、可分离 s_w[n]·c[g] | 精度不达标（§3.6、§3.9）；只作 INT8 主循环速度上界 |
| 参照 | 官方 FP8 per-tensor | 精度下限（t2i 17.8～18.6 dB，policy 7.5% seed 间差异） |
| 模型侧配合 | QKV / gate-up 合并成宽 GEMM；窄 N 层（k/v_proj 1024×4096、1536 档小层）分组不划算时留 bf16 | Thor 窄 N 层任何分组都不如 bf16（§11.2） |
| attention | Q/K INT8（Sage-v1 版式 + Hadamard + 通道平衡，Q4a）只在 attention 为 MMA 瓶颈的平台（H100、A100）有收益；GB200/Thor 受 exp2 吞吐限制，暂不做 | §3.3、第 10 节 |

### 12.1 主线验证结果（2026-09-14 晚）
| 任务 | g256（无 SQ） | g256 + SmoothQuant α=0.5 | g128 | g64 | 官方 FP8 |
| --- | --- | --- | --- | --- | --- |
| Nano t2i，36 图 PSNR / 保住 | 23.9 / 21 | **24.5 / 26** | 25.0 / 28 | 26.2 / 28 | 18.6 / 5 |
| Edge t2i，12 图 PSNR / 保住 | **26.7 / 12**（最低 22.8） | 25.2 / 9（最低 16.1） | 26.2 / 10 | 29.3 / 12 | — |
| Policy，动作 MSE 对 bf16（占 seed 间差异） | **0.00075（0.5%）** | 0.0031（2.1%，t2i 校准）/ 0.0085（5.7%，policy 校准） | 0.0016（1.1%） | 0.0017（1.1%） | 0.0113（7.5%） |
结论：**SmoothQuant 不稳健**——Nano t2i 上 +0.6 dB、多保 5 张，Edge 上 −1.5 dB、少保 3 张，policy 上误差放大 4～11 倍（用 policy 自身校准更糟）。
因此主线改为 **per-col g256、不做 SmoothQuant**（无校准、无跨任务依赖，kernel 相同）；SmoothQuant 降为按模型/按任务可选、必须逐项验证的开关。
g256 相对 g128 的代价集中在 Nano t2i（−1.1 dB、少保 7/36），Edge 与 policy 上 g256 反而最好。
仍待补：SmoothQuant 为何在 Edge/policy 上有害（推测：s 拉大了权重侧的动态范围，K 分组后权重量化误差反超激活收益）、视频精度。
