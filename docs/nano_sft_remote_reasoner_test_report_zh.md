# Nano SFT Reasoner 解耦与远程服务测试报告（中文）

日期：2026-09-22

分支：`feat/nano-sft-remote-reasoner`

对应英文报告：`docs/nano_sft_remote_reasoner_test_report.md`

## 结论摘要

本阶段已经完成 Nano SFT 冻结 Reasoner 与可训练 Generator 的结构性解耦，
并实现了两种可实际使用的外部条件输入方式：

- `offline`：预先抽取每层 Reasoner K/V，训练时从本地 safetensors cache 读取；
- `remote`：由独立的单 GPU gRPC 服务计算 Reasoner K/V，Generator 训练进程只接收结果。

`vision_sft_nano` 实际训练的是 405 个 Generator tensor，包括 397 个
`moe_gen` tensor，以及 `time_embedder`、`vae2llm`、`llm2vae` 共 8 个
tensor。外部 backend 会在模型实例化、FSDP 包装和 EMA 创建之前裁掉冻结的
UND/Reasoner 路径，因此 Generator rank 不再承担以下开销：

- Reasoner 参数实例化与 FSDP shard；
- Reasoner 参数 all-gather；
- Reasoner forward；
- FP32 EMA 中的第二份 Reasoner 权重。

目前的核心测试结论如下：

1. Offline 8xH20 短测中，平均 step time 下降 17.05%，每张 Generator GPU
   的 peak allocated memory 下降 15.63 GiB。
2. Remote localhost correctness smoke 中，36 层全部 K/V tensor 与直接抽取
   结果逐 bit 相等。
3. Remote 真实训练中，第二个 optimizer call 使用非零 LR `2e-6`，405/405
   个 Generator tensor 全部发生变化。
4. 同一 job 从 iteration 2 恢复后，以 LR `4e-6` 完成 iteration 3，405/405
   个 Generator tensor 再次发生变化。
5. 所有成功运行中均未出现 CUDA OOM、RPC、NCCL 或训练异常，任务结束后
   8 张 GPU 均恢复到 0 MiB 占用。

当前结果证明外部 Reasoner 条件输入可以完成真实 Generator 更新和
model/optimizer/scheduler/trainer 的续训，但还不代表已经完成生产部署验证或
与 joint/inline/offline 的严格数值等价验证。

## 已实现的能力

### Provider-neutral 条件接口

实现了 `joint`、`inline`、`offline`、`remote` backend：

- `joint` 保持现有训练行为，是默认值；
- `inline` 在本进程先运行 Reasoner，再用静态 K/V 执行 Generator，是数值
  对照路径，但不会节省 Reasoner 显存；
- `offline` 从不可变 cache 读取 K/V；
- `remote` 从独立服务异步获取 K/V；
- `read_through` 仅保留配置契约，目前仍然 fail closed。

外部 backend 使用已经完成 RoPE 的 Reasoner K/V。Generator Q、Generator
K/V、attention、MLP 和 residual 路径仍然在线计算并参与反向传播。

### Offline backend

- Reasoner-only DCP loader，不构建 Generator、VAE 或 EMA；
- 与训练一致的 caption/document framing；
- 可恢复、可分布式运行的抽取 CLI；
- layer-major safetensors shard；
- 原子发布、SHA-256 校验、fingerprint 和完整性检查；
- Generator-only warm start、checkpoint 和 EMA 初始化处理。

### Remote backend

- versioned protobuf 和 gRPC server-streaming 协议；
- 启动 handshake、feature signature 和 fingerprint 校验；
- per-chunk 与完整 response checksum；
- 异步请求、绝对 deadline、有限重试和取消；
- request/token admission、bounded queue 和 backpressure；
- OOM 或 runtime invariant 错误后的 unhealthy replica 传播；
- Generator 各 rank 在进入对应 FSDP collective 前同步失败状态；
- success、dry-run 和 failure 路径上的 provider 清理。

## 自动化验证

最终相关测试集结果：**186 passed**。

同时通过：

- Ruff check 与 format check；
- Pyrefly，0 errors；
- `git diff --check`；
- pre-commit；
- `uv lock --check`。

测试覆盖 cache 完整性、tensor codec、协议损坏、checksum/order/shape/dtype
校验、retry、deadline、cancellation、queue admission、OOM 传播、分布式失败
同步、checkpoint load，以及资源清理。

## H20 实测结果

### 1. Remote K/V 正确性

使用发布的 `Cosmos3-Nano` regular DCP，在两个独立 H20 进程中分别运行直接
`ReasonerFeatureRuntime` 和 localhost gRPC service。输入为 10-token BF16
framed prompt。

| 指标                        |                    结果 |
| --------------------------- | ----------------------: |
| 比较的 Reasoner 层数        |                      36 |
| K/V 一致性                  | 每层 K、V 均逐 bit 相等 |
| 直接抽取                    |              0.293149 s |
| localhost remote round trip |              0.326337 s |
| 服务端 Reasoner compute     |              0.288081 s |
| 服务端 D2H                  |              0.002015 s |
| CPU stream preparation      |              0.003000 s |
| 服务 load 后 allocated      |               15.26 GiB |
| 服务 load peak reserved     |               15.40 GiB |

这是 transport/runtime correctness smoke，不是容量或并发吞吐测试。

### 2. 第一次 7-rank smoke 的 zero-LR 修正

第一次端到端运行使用 GPU 0 作为 Reasoner service，GPU 1--7 作为
Generator FSDP ranks。该运行完成了 forward、backward、gradient clipping、
optimizer/EMA 路径和 checkpoint 写入。

但 recipe 的 50-step warmup 从 multiplier 0 开始，因此第一个 optimizer
call 的有效 LR 为 0。完整 DCP 对比结果是：

- gradient norm：0.49541；
- 0/405 个 live Generator tensor 发生变化；
- `max_abs_delta=0`。

因此，这次运行只证明 remote execution、梯度、Adam state、EMA/checkpoint
plumbing 和资源路径正常，不能作为“Generator 权重已经学习更新”的证据。

### 3. Fresh 2-step 非零 LR 更新

使用同样的 1 Reasoner + 7 Generator 拓扑重新运行两个 iteration：

| 指标                                 |                         结果 |
| ------------------------------------ | ---------------------------: |
| optimizer calls / 非零 LR updates    |                        2 / 1 |
| 两次 optimizer LR                    |                  `0`, `2e-6` |
| Iteration 1：loss / grad norm / time |   0.2496 / 0.54616 / 41.21 s |
| Iteration 2：loss / grad norm / time |   0.2131 / 0.35519 / 52.28 s |
| Iteration 2 checkpoint save          |                      32.02 s |
| `torchrun` wall time                 |                     120.62 s |
| Reasoner physical peak               |                   18.577 GiB |
| Generator physical peak min/mean/max | 30.007 / 30.922 / 31.718 GiB |

CPU-only 全量 DCP 对比结果：

- 405/405 个 live Generator tensor 发生变化；
- 6,714,184,413 / 6,965,486,784 个元素变化，即 96.3922%；
- maximum absolute delta：`2.026557922e-6`；
- mean absolute delta：`2.529132027e-7`；
- relative L2 delta：`1.500317368e-5`；
- base 与 candidate 均无 NaN/Inf；
- 405 个 optimizer step state 全部为 2；
- scheduler `last_epoch=2`，下一步 LR 为 `4e-6`。

该运行使用 recipe 默认的 CFG dropout 0.1 和 T2V/I2V/V2V 分布
0.7/0.2/0.1，与第一次固定 T2V、无 dropout 的 smoke 不是严格 paired A/B，
两次运行的 loss 和 time 不应直接用于性能对比。

### 4. Iteration 2 到 iteration 3 的续训

重启独立 Reasoner service 后，同一训练 job 从 `iter_000000002` 恢复：

- model、optimizer、scheduler 和 trainer state 恢复成功；
- checkpoint load：10.49 s；
- iteration 3 使用恢复后的 LR `4e-6`；
- loss：0.2138；
- gradient norm：0.55963；
- iteration time：73.23 s，其中 checkpoint save 为 30.29 s；
- process wall time：102.831 s；
- 保存 `iter_000000003`。

Iter-2 与 iter-3 的 CPU-only 全量对比结果：

- 405/405 个 live Generator tensor 再次变化；
- 6,763,395,246 / 6,965,486,784 个元素变化，即 97.0987%；
- maximum absolute delta：`4.053115845e-6`；
- relative L2 delta：`2.647678690e-5`；
- 两个 checkpoint 均无 NaN/Inf；
- 405 个 optimizer step state 从 2 全部推进到 3；
- scheduler `last_epoch=3`，下一步 LR 为 `6e-6`；
- iter-3 model 仍然只有 405 个 live + 405 个 EMA Generator leaves，
  Reasoner leaves 为 0。

Resume 运行的 physical memory peak：

| 角色                       |                     峰值显存 |
| -------------------------- | ---------------------------: |
| Reasoner GPU               |                   18.386 GiB |
| Generator GPU min/mean/max | 30.019 / 31.104 / 32.972 GiB |

重要限制：checkpoint loader 请求了 dataloader state，但 iteration-2 checkpoint
中不存在 `dataloader` subtree，因此所有 rank 都明确跳过了 dataloader cursor
恢复。本测试证明 model/optimizer/scheduler/trainer 的 operational
resume-and-continue，不证明 datastream position 恢复，也不证明与 uninterrupted
3-step run 数值等价。

### 5. Offline 与原始 Joint 的短测对比

该 8xH20 A/B 使用完整 Nano、FSDP、full activation checkpointing、FP32 EMA、
eager mode 和同一个 8-video BridgeData sample。统计 steady steps 5--10：

| Backend                      | Mean step time | Mean peak allocated / Generator GPU |
| ---------------------------- | -------------: | ----------------------------------: |
| Joint Reasoner + Generator   |       57.887 s |                          52.128 GiB |
| Offline K/V + Generator only |       48.015 s |                          36.498 GiB |

Offline 的变化：

- mean step time：-17.05%；
- aggregate token throughput：+20.56%；
- mean peak allocated memory：-15.630 GiB/GPU。

这是短时 hot-cache benchmark，不代表完整数据集抽取、共享文件系统带宽或长期
训练吞吐。

## 运行中遇到的问题

### Reasoner service 不应解析训练专用环境变量

最初 service 启动会在通用 SFT config resolution 阶段解析 dataset、VAE 和
training checkpoint 环境变量。实现已经在 service loader 中裁掉这些无关
subtree，并增加无相关环境变量的回归测试。

### `torchrun` 必须使用 module entrypoint

一次 resume 尝试将 `cosmos_framework/scripts/train.py` 文件路径直接传给
`torchrun`，导致同目录的 `hydra.py` 遮蔽安装的 Hydra package，出现：

```text
No module named 'hydra.core'; 'hydra' is not a package
```

正确启动方式是：

```bash
torchrun --nproc_per_node=<N> --module cosmos_framework.scripts.train ...
```

该问题发生在 CUDA 初始化之前，不是模型、checkpoint、OOM 或 RPC 故障。

## 当前限制与后续 gate

合入生产训练前仍需完成：

1. 使用完全相同的数据、dropout、RNG 和配置，对 remote 与
   joint/inline/offline 的 loss、gradient 和 post-update weights 做严格 A/B。
2. 保存并恢复 dataloader cursor，对比 uninterrupted 与 resumed update。
3. 使用 representative 和 maximum-length prompt 做 remote parity 与吞吐测试。
4. 测试一个 Reasoner replica 服务 1/2/4/8 个 Generator ranks 时的 queue、
   compute、D2H、network/client wait 和 step latency percentile。
5. 为 7-view/11-view per-view caption 与 specialized multiview attention 增加
   multi-document 支持；protocol v1 当前对此明确 fail closed。
6. 实现或验证 read-through cache、dynamic batching、layerwise H2D 和持久化
   service cache。
7. 从实际 checkpoint/tokenizer 内容自动派生 digest，而不是依赖人工标签。
8. 跨节点生产部署前增加 TLS、authentication、health check 和 metrics。
9. 完成 full-corpus extraction、共享存储压力和 compile-enabled 长跑测试。

## 本地测试产物

原始日志和 checkpoint 位于 gitignored `outputs/`，不会随 PR 提交：

```text
outputs/nano_sft_reasoner_benchmark/remote_smoke_7gpu_1step_20260922_v1
outputs/nano_sft_reasoner_benchmark/remote_smoke_7gpu_2step_20260922_v2
```

第二次运行中的主要证据：

```text
logs/generator_train.log
logs/generator_resume_iter2_to_iter3.log
logs/reasoner_service.log
logs/reasoner_service_resume_iter3.log
logs/nvidia_smi.csv
logs/nvidia_smi_resume_iter3.csv
logs/gpu_telemetry_summary.json
logs/training_state_audit_iter2.json
logs/training_state_audit_iter3.json
logs/generator_weight_delta_iter2_vs_base.json
logs/generator_weight_delta_iter3_vs_iter2.json
```

两个 Generator-only checkpoint 分别约 104 GiB：

```text
train/cosmos3/sft/nano_remote_nonzero_7gpu/checkpoints/iter_000000002
train/cosmos3/sft/nano_remote_nonzero_7gpu/checkpoints/iter_000000003
```

## 最终判断

当前实现已经证明：冻结 Reasoner 可以从 Nano Generator SFT rank 中结构性移除，
offline 和 remote backend 均能驱动真实的 Generator 参数更新，并显著降低训练
rank 显存。Offline 方案已经显示明确的显存与 step-time 收益；Remote 方案已经
通过 K/V 正确性、非零 LR 更新和 operational resume gate。

建议当前阶段定位为可 review、可继续扩展的 MVP。正式生产化仍应以严格数值
A/B、dataloader resume、代表性并发吞吐、multiview 支持和安全部署能力为准。
