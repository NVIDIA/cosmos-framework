# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Reproduction of Akash's 8B wallace T2I DPO (wandb nvidia-dir/cosmos3_vfm/20260514100000,
`wallace_ebs2048_b5000_500iter`, finished; anchors: step500 loss 2.184 / acc 0.902 / margin 48.0).

Reference setup (from the run's wandb config):
  base   midtraining_v5 8B `t2w_mot_exp406_.../checkpoints/iter_000010000` (DCP, experiments bucket)
  data   HPDv3 80K offset-jsonl + images.tar (agokul's lustre paths), 720p T=1, cfg_dropout 0.1
  ebs    2048 = 32 ranks (8 GB200 nodes x 4) x bs2 x grad_accum 32
  opt    fusedadam lr 5e-6 CONSTANT, betas (0.9, 0.99), wd 1e-4, FULL net (no key subset)
  extras beta 5000, grad clip 1.0 + force_finite, EMA 0.1 (EMA not replicated — eval uses
         regular weights per Akash's guide), seed 0, guidance 0

Known deltas vs the reference (documented, believed curve-neutral): no EMA branch,
our env's DistributedSampler-style sharding vs their DataLoader worker sharding (data
ORDER differs; per-step stats over ebs2048 are statistically tight), fsdp width 32 vs
their fsdp4xreplicate8 (gradient math identical).

Optimizer MUST be fusedadam (fp32 master + fp32 moments): the first repro attempt
(2026-07-06, wandb 0ebc91efba372036, killed at step ~30) ran plain AdamW on the bf16
net — bf16 moments + bf16 updates round away most of lr=5e-6, giving ~7x slower
loser-side drift and a margin plateau at ~3 vs the reference's climb to 48. Same
trainable set (14.53B, everything but lm_head), same data, same objective — the
optimizer numerics alone did that. See utils/fused_adam.py.

Run (8 nodes, ray backend; init_from must point at a LOCAL copy of the base DCP):
    python3 cosmos_framework/posttrain/utils/launch_multinode.py --mode torchrun \\
      --run "-m cosmos_framework.posttrain.entrypoints.dpo -- experiment=wallace_8b_hpdv3_repro_v1 \\
             model.config.parallelism.data_parallel_shard_degree=32 rl.init_from=<local-base-dcp>" \\
      --nodes 8 --gpus-per-node 4 ...
"""

from hydra.core.config_store import ConfigStore

from cosmos_framework.utils.lazy_config import LazyDict
from cosmos_framework.posttrain.config.rl import DPOConfig, OptimizerConfig, PreferenceEnvConfig, RefConfig

_AGOKUL = "/path/to/shared-storage"

wallace_8b_hpdv3_repro_v1 = LazyDict(
    dict(
        defaults=[
            "/experiment/t2w_mot_exp406_000_qwen3_vl_8b_multires_recipe_midtraining_v5",
            "_self_",
        ],
        algorithm="dpo",
        rl=DPOConfig(
            beta=5000.0,
            max_steps=500,  # optimizer steps (the loop's outer axis == wallace's iteration)
            pairs_per_step=2,  # bs per rank
            grad_accum=32,  # x 32 ranks x bs2 = ebs 2048
            save_every=25,
            keep_optim_last=2,
            optimizer=OptimizerConfig(
                name="fusedadam",  # fp32 master+moments — REQUIRED on bf16 nets (docstring)
                lr=5e-6,
                weight_decay=1e-4,
                betas=[0.9, 0.99],
                train_keys=[],  # FULL net — cosmos-rl trains every requires_grad param
                grad_clip_norm=1.0,
                force_finite_grads=True,
            ),
            ref=RefConfig(update_every=0),  # frozen anchor
            data=PreferenceEnvConfig(
                jsonl_path=f"{_AGOKUL}/dpo_eval_2026_05_12/hpdv3_full_train.jsonl",
                tar_path=f"{_AGOKUL}/hpdv3_raw/images.tar",
                media="video",  # wallace parity: T=1 video samples, VIDEO res table
                resolution="720",
                num_frames=1,
                cfg_dropout_rate=0.1,
                caption_key="prompt",
                shuffle=True,
            ),
        ),
        job=dict(group="posttrain_dpo", name="wallace_8b_hpdv3_repro_v1"),
    ),
    flags={"allow_objects": True},
)

ConfigStore.instance().store(
    group="experiment", package="_global_", name="wallace_8b_hpdv3_repro_v1", node=wallace_8b_hpdv3_repro_v1
)
