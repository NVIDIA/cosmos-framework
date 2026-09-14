# CUTLASS per-tensor INT8 GEMM（FP8 per-tensor 的 INT8 孪生）

目的：回答 handoff 第 8 节遗留的问题——"INT8 per-tensor 的速率上限是否与 FP8 per-tensor 相同"。
做法：取 CUTLASS 的 FP8 per-tensor GEMM 示例，把 A/B 换成 int8、累加器换成 int32，其余（TMA、warp-specialized
流水、epilogue 里的 scale_a·scale_b 融合、bf16 输出）不动，然后在相同 tile 配置下对比 INT8 与 FP8 两个 kernel，
并以 cuBLASLt FP8 per-tensor（`torch._scaled_mm`，官方 ModelOpt checkpoint 路径）为外部基线。

```
D[M,N] (bf16) = scale_a * scale_b * (A[M,K] @ W[N,K]^T)      # A 行主（K 连续），W 行主 [N,K] 即 K-major B
```

## 文件

| 文件 | 说明 |
| --- | --- |
| `pertensor_gemm.cu` | 单一源码。`-DKIND_INT8` / `-DKIND_FP8` 选 MMA 类型；`-DARCH=100`（GB200/B200，派生自 `examples/70_blackwell_gemm/70_blackwell_fp8_gemm.cu`，tcgen05 1SM/2SM UMMA + TMEM）或 `-DARCH=90`（H100，派生自 `examples/54_hopper_fp8_warp_specialized_gemm`，wgmma cooperative / `-DPINGPONG`）；`-DTM/-DTN/-DTK/-DCM/-DCN` 给 MMA tile 与集群；`-DSTREAMK` 换 stream-K 调度器（`--decomposition=Heuristic|StreamK|SplitK|DataParallel --splits=`）。每次运行都对随机采样的 (m,n) 用 fp64 精确参考校验（`--verify_samples`），并用 512 MB memset 冲 L2 后逐次 cudaEvent 计时取中位数（`--flush_l2=1`，与 H100 报告方法一致）。|
| `Makefile` | 一个 (kind, config) 一个二进制：`make -j10 CUTLASS=~/cutlass ARCH=100 CONFIGS="256x128x128_c2x1 ..." KINDS="int8 fp8"`；config 名 `TMxTNxTK_cCMxCN[_sk|_pp]`。|
| `bench.py` | 在容器里跑：`torch` 基线（bf16 cuBLAS、FP8 per-tensor `_scaled_mm`、INT8 `_int_mm`）+ 全部二进制，4 个 Nano gen 塔形状 × M 列表，同一冲 L2/中位数口径；输出 `results/<tag>.{json,md}`（含对 cuBLASLt FP8 的比值表）。|
| `run.sbatch` / `run_smallm.sbatch` / `run_sk.sbatch` | GB200 作业脚本（imaginaire4_v12.1.0 容器，nvcc 13.3，CUTLASS 4.4.2 @ `~/cutlass`）。|
| `results/` `logs/` | 原始数据。|

## 结果（GB200，2026-09-13，作业 2158916 / 2158958 / 2159012 / 2159046）

环境：1× GB200（sm_100a），imaginaire4_v12.1.0（torch 2.13.0a0 nv26.07，CUDA 13.3），CUTLASS 4.4.2（baea077e）。
计时：冲 L2 后 50 次中位数；SM 时钟多为 2062 MHz，M=42240 时偶见功耗限频（记录在 json 的 `clocks` 列）。
所有 CUTLASS 行的采样校验全部 PASS（max_rel 3.8e-3～3.9e-3 = bf16 输出舍入下限）。

### 1. 同一 tile 下 INT8 与 FP8 的 CUTLASS kernel 同速

INT8/FP8 TFLOPS 比（20 个 case：4 形状 × M∈{901,1802,4096,16384,42240}）：

| tile / cluster | 最小 | 最大 | 中位 |
| --- | --- | --- | --- |
| 128x128x128 c1x1（1SM） | 0.92 | 1.04 | 0.99 |
| 128x256x128 c1x1（1SM） | 0.97 | 1.03 | 1.00 |
| 256x128x128 c2x1（2SM） | 0.93 | 1.03 | 1.00 |
| 256x128x64 c2x2（2SM，示例 70 原配置） | 0.93 | 1.07 | 0.99 |
| 256x256x128 c2x1（2SM） | 0.95 | 1.06 | 1.00 |

结论：在 GB200 上 INT8 与 FP8 的 tcgen05 MMA 速率相同，INT8 的 int32→fp32 缩放在 epilogue 里无额外代价。

### 2. 对 cuBLASLt FP8 per-tensor（官方路径）的比值——按 case 取最好的 INT8 配置

| 形状 (N,K) | M=901 | M=1802 | M=4096 | M=16384 | M=42240 |
| --- | --- | --- | --- | --- | --- |
| q/o_proj 4096x4096 | 1.00 | 0.90 | 1.03 | 1.10 | 1.09 |
| k/v_proj 1024x4096 | 0.75 | 0.82 | 0.91 | 1.02 | 1.08 |
| gate/up 12288x4096 | 0.84 | 1.01 | 1.09 | 1.11 | 1.15 |
| down 4096x12288 | 0.87 | 0.89 | 1.06 | 1.07 | 1.11 |

- M ≥ 4096：INT8 CUTLASS（256x256x128 或 256x128x128，2SM）为 cuBLASLt FP8 的 1.02～1.15×。绝对值：q/o 2.7～3.6 PTOPS，gate/up 3.1～3.6 PTOPS。
- M ≤ 1802：0.75～1.01×。缺口出现在 tile 数少的 case（k/v_proj N=1024：M=901 时 256x128 tile 只有 32 个 2SM 集群，148 个 SM 大半空闲）。
- 对照：cuBLASLt 自己的 INT8（`_int_mm`，int32 输出、无缩放）对 FP8 是 0.77～1.13×，小 M 同样落后，即厂商库在 GB200 上也没有把 INT8 小 M 调好。
- 更小的 tile（128x64 / 64x128 / 64x64，1SM）没有帮助：全部 0.58～0.92×，比大 tile 更慢。小 M 的缺口不是 CTA 数量问题，需要 split-K/stream-K 沿 K 切分（见第 3 节）。

- 集群形状 2x2 / 4x1（TMA 多播更多 CTA）对 256x128x128 与 256x256x128 都没有稳定收益（±3%，噪声级；gate/up M=42240 的 256x128x128 c4x1 反而掉到 0.57×），
  作业 2159055。集群保持 2x1。
- **推荐配置**：M ≥ 4096 或 N ≥ 4096 用 `int8 256x256x128 c2x1`；N=1024（k/v_proj）或 M ≤ 1802 用 `int8 256x128x128 c2x1`。
  两者按形状二选一即可覆盖表中最好值。

完整逐 kernel 表：`results/gb200_2026-09-13_2158916.md`（主配置）、`results/gb200_smallM_2026-09-13_2158958.md`（小 tile）、
`results/gb200_cluster_2026-09-13_2159055.md`（集群形状）。

### 3. 上限（大方阵峰值，作业 2159012，`results/gb200_ceiling_2159012.md`）

1× GB200，SM 时钟全程 2062 MHz、功耗 225～260 W（上限 1200 W，无限频），30 次中位数。

| 形状 M×N×K | CUTLASS INT8 256x256x128 | CUTLASS FP8 256x256x128 | cuBLASLt INT8 `_int_mm` | cuBLASLt FP8 `_scaled_mm` | cuBLAS bf16 |
| --- | --- | --- | --- | --- | --- |
| 8192³ 热 L2 | 3356 | 3373 | 3112 | 3040 | 1624 |
| 8192³ 冲 L2 | 3238 | 3247 | 3349 | 3134 | 1587 |
| 16384×8192×8192 热 / 冲 | 3410 / 3416 | 3429 / 3427 | 3098 / 3248 | 3097 / 2971 | 1532 / 1495 |
| 16384×16384×8192 热 / 冲 | 2907 / 2987 | 3048 / 3023 | 3269 / 3147 | 3142 / 2806 | 1456 / 1401 |

- GB200 上 INT8 与 FP8 的实测峰值相同：约 **3.4 PTOPS**（名义稠密 5 POPS 的 ~68%，即 2062 MHz × 148 SM × 16384 op/clk），bf16 约 1.6 PF。
- cuBLASLt 在 GB200 上确有专门的 INT8 IMMA kernel（`_int_mm` 3.1～3.35 PTOPS，与其 FP8 相当或略高）；H100 报告里 cuBLASLt INT8 只有 FP8 的 0.58～0.73× 应是 SM90 库 kernel 的问题，不是硬件上限——需要用本目录的 `ARCH=90` 二进制在 H100 上实测确认。
- 256x128x128 在 16384² 上掉到 2.0 PTOPS（两种精度一样），256x256x128 是大形状的首选。

### 4. Stream-K / split-K（小 M 的补救，未完成）

`-DSTREAMK` 二进制（显式 `KernelTmaWarpSpecialized1Sm/2SmSm100` 调度 + `hw_info.sm_count` 已设置）只有 `--decomposition=DataParallel`
能跑（结果正确）；Heuristic / StreamK / SplitK 这些需要归约 workspace 的模式仍然不终止（GPU 100%，作业 2158983 / 2159005）。
示例 74 在同一环境可跑 256×256×16384。差异尚未定位（怀疑与 epilogue 融合或 C=nullptr 的 fixup 路径有关）。小 M 的 0.75～0.9× 缺口暂时保留。

## H100（SM90）

本集群（gcp-iad-cs-001）只有 GB200，H100 版本只做了编译验证：`make ARCH=90` 生成 sm_90a 二进制
（cooperative 128x128x128 c1x2 / 128x256x128 c1x2 / 128x128x128 c2x1，pingpong 64x128x128 c1x2 / 64x256x128 c1x2；
FP8 版用 `KernelTmaWarpSpecializedCooperativeFP8FastAccum`，与 cuBLASLt `use_fast_accum` 口径一致）。
在 H100 机器上运行：容器内 `make -j10 CUTLASS=<cutlass> ARCH=90 CONFIGS="..."` 后
`python3 bench.py --bin-dir build_sm90 --tag h100_<date>`。handoff §9 已用独立的 SM90 kernel 证实 H100 上 INT8 = FP8 速率（cuBLASLt 慢是库落到 Ampere 时代 kernel）；
这组二进制可作交叉验证。Thor（sm_110a）用 `make ARCH=110`。

## 已知问题

- SM100 stream-K：`KernelScheduleAuto` + `StreamKScheduler` 编译通过但不终止；改成示例 74 的显式 1Sm/2Sm 调度并设置 `hw_info.sm_count` 后，
  DataParallel 模式可跑，需要归约的模式仍不终止（见第 4 节）。
- `--swizzle`（`scheduler.max_swizzle_size`，raster swizzle）在 SM100 CLC 调度器 + 256x256x128 c2x1 上不可用：
  swizzle=2/4 报 `illegal instruction`，swizzle=8 在 M=1802 挂起（作业 2159046）。默认 0 正常；本目录的所有数字都是 swizzle=0。
- 任何新调度器/参数的校验循环都要包 `timeout`，否则一次挂起会吃掉整个作业。
