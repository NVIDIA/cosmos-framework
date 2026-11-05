# V1.2 continuation parameters

The executable source of truth is
`cosmos_framework/configs/tokenizer/lidar_v1p2.py`. This is a fixed recipe;
`--stop-after` bounds execution while retaining the schedule. Scientific changes
should be explicit recipe edits on a new branch/run and must be requalified.

| Setting                              | Value                                                                                                              |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| Network channels                     | input/output 3, latent 128, base 128                                                                               |
| Canvas / patch                       | 128×1808 / 2×2                                                                                                     |
| Depths / heads                       | (3,3,3,3) / (4,4,8,8)                                                                                              |
| Neighborhood / dilation              | (5,45) / all 1                                                                                                     |
| Temporal down/up sampling            | all false; nine input/latent frames                                                                                |
| Latent shape                         | `[B,128,9,8,113]`                                                                                                  |
| Precision / activation checkpointing | FP32 / enabled                                                                                                     |
| Batch / accumulation                 | 2 clips per GPU / 1                                                                                                |
| Optimizer                            | AdamW, betas (0.9,0.999), eps 1e-8, weight decay 0                                                                 |
| LR / schedule                        | 1e-5; 1000-step linear warmup then linear decay to 1e-6 at 30k                                                     |
| Gradient clipping                    | global parameter norm 1.0                                                                                          |
| Loss                                 | reconstruction 2; equal range/intensity weights; KL 1e-6; validity BCE 1; metric point L1 1; multiscale gradient 1 |
| Reconstruction reduction             | mask error, then mean over all rays                                                                                |
| Gradient convention                  | add the multiscale gradient term twice                                                                             |
| Latent stats                         | channelwise globally reduced moments; EMA momentum 0.999                                                           |
| Parameter averaging                  | power EMA relative width s=0.1, not fixed decay 0.1                                                                |
| Iterations / saves                   | 30k / every 5k, plus explicit stop                                                                                 |
| Parallelism                          | DDP, static graph, buffer broadcast, CP=1                                                                          |

The recipe uses FP32 without a gradient scaler to keep range reconstruction and
metric geometric losses numerically stable. BF16 and FP16 need separate
qualification. Effective global batch is `2 × world_size`; validate memory on
the selected hardware before changing batch size. EMA needs an extra FP32
parameter copy per rank. The low learning rate limits changes to a converged
initializer, and gradient clipping bounds each update. Do not automatically scale
LR with rank count; qualify any new rate against held-out reconstruction and
validity metrics.

The runner checks finite losses and gradients before optimizer updates. Failures
stop the run; no scaler or automatic skipped-update policy is implied. Checkpoints
contain complete training state and data-content identity. They are written
synchronously and atomically on rank zero. Keep enough local/shared disk space.
Only load trusted PyTorch checkpoint files.

Network-only warm start intentionally resets latent moments and EMA age. Full
resume restores both. Native resume requires matching input contents, output
identity, topology and recipe, and advances to an absolute endpoint. Optimizer
and scheduler state from another training runner are not interchangeable with
this runner's state. Changed topology is a new continuation experiment.

Serving export preserves the chosen network weights, range bounds and accumulated
latent moments, and removes optimizer/EMA copies and input paths. Use the
matching metadata for loading. Publication requires held-out reconstruction
qualification of **both** live and EMA candidates, including validity IoU.
