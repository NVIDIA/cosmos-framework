# CUTLASS g128 INT8 GEMM（per-token 1×128 激活 scale × per-col 1×128 权重 scale）

目标：把 handoff §2.1/§3.6 定义的 INT8 g128（per-col）GEMM 做成真 kernel，并在 GB200 上对齐它的 FP8 孪生与上限。
起点：CUTLASS 4.4.2 `examples/81_blackwell_gemm_blockwise/81_blackwell_gemm_groupwise.cu`（SM100 blockwise-scaling collective：
MMA 每 128-K 在 TMEM 里累加，4 个 epilogue warp 把 TMEM 读到寄存器、乘 `scale_a·scale_b` 加进 fp32 寄存器累加器）。

```
D[M,N](bf16) = Σ_g SFA[m,g]·SFB[n,g]·( Σ_{k∈g} A[m,k]·W[n,k] ),  g = 128 个连续 K
A[M,K] int8 行主；SFA fp32 [M, K/128]      W[N,K] int8 行主；SFB fp32 [N/SFN, K/128]，SFN=1（per-col）或 128（DeepSeek blockwise）
```

## 文件
| 文件 | 说明 |
| --- | --- |
| `g128_gemm.cu` | 主程序。`-DKIND_INT8`（int8×int8→int32 MMA，fp32 scale/提升）或 `-DKIND_FP8`（e4m3，与上游示例等价）；`-DTM/TN/TK/CM/CN/SFN`。随机 int8 操作数 + 随机 fp32 组 scale，对 fp64 精确公式采样校验；冲 L2 逐次 cudaEvent 取中位数。SFA 是 MN-major（m 最快），要求 M%4==0，程序自动把 M pad 到 4 的倍数并按真实 M 报 TFLOPS。|
| `include/cutlass/gemm/collective/sm100_mma_warpspecialized_blockwise_scaling.hpp` | **shadow 头文件**（`-I./include` 排在 CUTLASS 之前生效）。上游把 scale 类型与提升后累加器都绑成 `TiledMma::ValTypeC`，int8 时是 int32，会把 fp32 scale 的位模式当整数乘（能编译、结果错）。本副本把它们解耦成 fp32（`ElementSF/ElementPromoted`），MMA 仍在 TMEM 里按 int32 累加，提升点做 int32→fp32。`-DG128_OPT_PROMOTION` 再改 promotion：SFB 不再整行拷进寄存器（per-col 时 128～256 个寄存器，256x256 直接溢出），而是按 T2R 子块（≤32 列）从 smem 分批载入；`-DG128_OPT_PIPELINE` 把 TMEM 子块读取与 FMA 双缓冲流水。对 FP8 行为与上游一致。差异见 `patches/0001-*.patch`。|
| `Makefile` | `make -j8 CUTLASS=~/cutlass KINDS='int8 fp8' CONFIGS='256x128x128_c2x1_sfn1 256x128x256_c2x1_sfn1 ...' [EXTRA='-DG128_OPT_PROMOTION -DG128_OPT_PIPELINE']`，config 名 `TMxTNxTK_cCMxCN_sfnS`（TK 为 128 的倍数）。|
| `make_results.py` | 把 `logs/*.log` 的 RESULT 行与 per-tensor 基线拼成 `results/<tag>.md`。|
| `g128_common.py` / `ref/sglang_int8_kernel.py` | 项目量化定义（absmax/127、rint、clamp）、fp64 参考、SGLang Triton block-INT8 kernel（Apache-2.0 副本，"现货"对照）与计时工具。|
| `run_ex81_fp8_baseline.sbatch` / `run_g128.sbatch` / `run_dev_hold.sbatch` | stock 示例 81 基线、本 harness 扫描、长驻开发作业（`srun --overlap --jobid=` 挂载容器编译/测试）。|
| `results/` `logs/` | 数据。|

## 结果（GB200，2026-09-14）

环境：1× GB200（sm_100a），imaginaire4_v12.1.0（nvcc 13.3），CUTLASS 4.4.2；冲 L2 后 50 次中位数；所有 g128 行采样校验 PASS（max_rel ≤ 3.9e-3 = bf16 舍入下限）。
作业：2159753（stock）、2159781（opt）。完整逐配置表 `results/gb200_g128_2026-09-14.md`；原始行在 `logs/`。
"opt" = shadow 头文件 + `-DG128_OPT_PROMOTION -DG128_OPT_PIPELINE`；"TileK 256" = `256x128x256_c2x1`（每 K-tile 两个 128 组，减半 TMA/barrier 开销）。

**q/o_proj 4096x4096（N×K）**

| kernel | M=901 | M=1802 | M=4096 | M=16384 | M=42240 |
|---|---|---|---|---|---|
| bf16 cuBLAS | 1077 | 1359 | 1407 | 1670 | 1771 |
| INT8 per-tensor CUTLASS 256x128x128 | 1379 | 1865 | 2679 | 3202 | 3372 |
| INT8 g128 per-col, stock CUTLASS promotion | 367 | 398 | 524 | 553 | 561 |
| INT8 g128 per-col, opt (TileK 256) | 665 | 804 | 1092 | 1187 | 1224 |
| INT8 g128 W 128x128 blockwise, opt | 769 | 946 | 1311 | 1430 | 1489 |
| FP8 g128 per-col, opt (TileK 256) | 768 | 876 | 1238 | 1337 | 1388 |
| FP8 g128 W 128x128 blockwise (=示例 81 布局) | 859 | 1064 | 1468 | 1640 | 1702 |

**k/v_proj 1024x4096（N×K）**

| kernel | M=901 | M=1802 | M=4096 | M=16384 | M=42240 |
|---|---|---|---|---|---|
| bf16 cuBLAS | 480 | 842 | 1141 | 1378 | 1549 |
| INT8 per-tensor CUTLASS 256x128x128 | 480 | 900 | 1568 | 2513 | 2827 |
| INT8 g128 per-col, stock CUTLASS promotion | 170 | 332 | 417 | 522 | 544 |
| INT8 g128 per-col, opt (TileK 256) | 302 | 559 | 756 | 1079 | 1163 |
| INT8 g128 W 128x128 blockwise, opt | 328 | 606 | 874 | 1298 | 1393 |
| FP8 g128 per-col, opt (TileK 256) | 329 | 605 | 874 | 1216 | 1313 |
| FP8 g128 W 128x128 blockwise (=示例 81 布局) | 362 | 690 | 976 | 1468 | 1592 |

**gate/up 12288x4096（N×K）**

| kernel | M=901 | M=1802 | M=4096 | M=16384 | M=42240 |
|---|---|---|---|---|---|
| bf16 cuBLAS | 1312 | 1287 | 1601 | 1696 | 1461 |
| INT8 per-tensor CUTLASS 256x128x128 | 1909 | 2413 | 3041 | 3054 | 3086 |
| INT8 g128 per-col, stock CUTLASS promotion | 415 | 457 | 546 | 562 | 567 |
| INT8 g128 per-col, opt (TileK 256) | 841 | 970 | 1172 | 1218 | 1232 |
| INT8 g128 W 128x128 blockwise, opt | 1002 | 1155 | 1410 | 1441 | 1489 |
| FP8 g128 per-col, opt (TileK 256) | 948 | 1082 | 1319 | 1380 | 1396 |
| FP8 g128 W 128x128 blockwise (=示例 81 布局) | 1130 | 1308 | 1614 | 1675 | 1714 |

**down 4096x12288（N×K）**

| kernel | M=901 | M=1802 | M=4096 | M=16384 | M=42240 |
|---|---|---|---|---|---|
| bf16 cuBLAS | 1276 | 1501 | 1589 | 1718 | 1524 |
| INT8 per-tensor CUTLASS 256x128x128 | 1909 | 2260 | 2974 | 3400 | 3082 |
| INT8 g128 per-col, stock CUTLASS promotion | 404 | 426 | 553 | 574 | 582 |
| INT8 g128 per-col, opt (TileK 256) | 848 | 939 | 1234 | 1283 | 1315 |
| INT8 g128 W 128x128 blockwise, opt | 1013 | 1145 | 1488 | 1571 | 1609 |
| FP8 g128 per-col, opt (TileK 256) | 980 | 1058 | 1396 | 1458 | 1493 |
| FP8 g128 W 128x128 blockwise (=示例 81 布局) | 1159 | 1315 | 1711 | 1830 | 1860 |

### 数字怎么读
- **INT8 g128 per-col（S0 目标布局）在 GB200 上慢于 bf16 cuBLAS**：opt 后 1.09～1.32 PTOPS（M≥4096），是 bf16 的 0.7～0.85×、同 tile per-tensor INT8 的 0.36～0.43×。stock 代码只有 0.52～0.58 PTOPS。
- 权重放粗到 128×128 blockwise 也只到 1.31～1.61 PTOPS（≈ bf16），FP8 孪生同样：per-col 1.24～1.49、blockwise 1.47～1.86。**g128 在 GB200 上不是速度方案，无论 INT8 还是 FP8。**
- INT8 与 FP8 的 g128 差距 8～12%，来自每元素多一次 int32→fp32 转换；per-tensor 时两者相同。
- 优化做了什么：(1) SFB 从"整行拷进寄存器"改成按 32 列子块从 smem 分批载入——per-col 的寄存器溢出 272 B→60 B，吞吐 ×2；(2) TMEM 读取与 FMA 双缓冲流水——+1～4%；(3) TileK 256——+4～7%；(4) magic-number 替代 I2F——更慢，撤回。ptxas：168 寄存器，60 B 溢出（与 blockwise 相同），即剩余差距不再是寄存器问题。

## 结构性结论
- GB200 tensor core : CUDA core 吞吐比约 64:1（8192 MAC/clk vs 128 FMA/clk）。每个输出元素每 128-K 至少 1 FFMA（blockwise，scale 乘积可按行复用）或 FMUL+FFMA（per-col），
  在 128×128 CTA tile 上 = 128～256 条 warp 指令/线程 ≈ 256～512 clk，等于或超过该 K-tile 的 MMA 时间（256 clk）。软件 block-scaling 的上限因此约为 per-tensor 的 50%（blockwise）/33%（per-col），
  实测 45% / 36%。Hopper 上这个比值是 ~15:1，所以 DeepGEMM 在 H100 能贴近 per-tensor（H100 报告 0.91～1.03×）。Blackwell 对此的硬件答案是 MXFP8/NVFP4 的块缩放 MMA（32 元素 UE8M0 scale），INT8 没有对应指令。
- 要在 GB200 上既保 g128 精度又要速度，必须把逐元素提升从 CUDA core 里拿掉，例如：权重 scale 可分离 s_w[n,g]=s_w[n]·c[g]（c[g] 折进激活 scale，per-col 退化为 blockwise 成本 1.3～1.6 PF，精度未测，handoff §3.6 已列）；
  或 per-token × per-channel 的 epilogue 缩放（3.3 PTOPS，即 per-tensor 路径，精度是 S0 之外的另一档）。


## 已知限制 / 待办
- M 需为 4 的倍数（MN-major SFA 的 16B cp.async）；改用 K-major scale 布局（`Sm100BlockwiseScaleConfig<1,SFN,128,Major::K,Major::K>`）可去掉 pad 且直接吃 torch 的 `[M,K/128]` 布局，未做。
- 校验是采样 + 1.5e-2 容差（bf16 ULP 级），不是逐位；bit-exact 参考需按 kernel 的 FMUL→FFMA、K 顺序模拟。
- shadow 头文件靠 -I 顺序生效；CUTLASS 升级会静默回退到错误的 int32 scale 运算，需重打 patch 并重跑校验。array（grouped）变体未打补丁。
- INT8 tcgen05 只在 sm_100a/101a/110a 开启（GB300 sm_103a 没有）。
