# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Reproduction of Akash's 32B (64bm32b) T2I DPO — wandb nvidia-dir/cosmos3_vfm/20260523175740
(`ga_64bm32b_v1p5_hq_dpo_b5000_lr5e6_ebs256_500iter`, CRASHED at step ~152; we align the
first 152 steps against his curve and then complete the 500 he never finished).

Reference setup (from the run's wandb config, pulled 2026-07-07):
  base   `cosmos3_ga_image_only/cosmos3_ga_64bm32b_from_v2_image_only_v1p5_hq_distill/
         checkpoints/iter_000108000` (DCP, preserved in the permanent bucket)
  data   the SAME HPDv3 80K offset-jsonl + images.tar as the 8B wallace runs, 720p T=1,
         cfg_dropout 0.1, gaps 0, seed 0
  ebs    256 = 64 ranks (16 GB200 nodes x 4) x bs1 x grad_accum 4
  opt    fusedadam (fp32 master + moments) lr 5e-6 CONSTANT, betas (0.9, 0.99), wd 1e-4,
         FULL net; dpo_beta 5000; EMA 0.1 (not replicated — eval uses regular weights)

Known deltas vs the reference (same class as the VALIDATED 8B repro, which landed at
HPSv3 +0.305 vs his +0.172): no EMA branch, data order (sampler impl), pure FSDP
dpsd=64 vs his fsdp32 x replicate2 (gradient math identical).

32B-specific choices (framework features validated on the 8B run):
  checkpoint_format=dcp   consolidated would gather ~800GB to rank-0 host RAM (OOM);
                          DCP shards the write across all 64 ranks, and the reference
                          checkpoints/eval tooling are DCP-native anyway
  upload.enabled          each save lands in the experiments bucket (mirror layout)
  keep_model_last=2       local disk holds the working set only (~1.5TB instead of ~7TB)

Run (16 nodes, batch_long single window — no relay needed at MaxTime=7d):
    python3 cosmos_framework/posttrain/utils/launch_multinode.py --mode torchrun \\
      --run "-m cosmos_framework.posttrain.entrypoints.dpo -- experiment=ga_32b_hpdv3_repro_v1 \\
             model.config.parallelism.data_parallel_shard_degree=64 rl.init_from=<local-base-dcp>" \\
      --nodes 16 --gpus-per-node 4 --partition batch_long --wckey p0 --time 2-00:00:00 ...
"""

from hydra.core.config_store import ConfigStore

from cosmos_framework.utils.lazy_config import LazyDict
from cosmos_framework.posttrain.config.rl import (
    CheckpointUploadConfig,
    DPOConfig,
    OptimizerConfig,
    PreferenceEnvConfig,
    RefConfig,
)

_AGOKUL = "/path/to/shared-storage"

ga_32b_hpdv3_repro_v1 = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos3_ga_64bm32b_from_v2_image_only_v1p5_hq_distill",
            "_self_",
        ],
        algorithm="dpo",
        rl=DPOConfig(
            beta=5000.0,
            max_steps=500,
            pairs_per_step=1,  # bs per rank
            grad_accum=4,  # x 64 ranks x bs1 = ebs 256
            save_every=50,
            checkpoint_format="dcp",
            keep_optim_last=2,
            keep_model_last=2,
            upload=CheckpointUploadConfig(enabled=True),
            optimizer=OptimizerConfig(
                name="fusedadam",  # fp32 master+moments — REQUIRED on bf16 nets
                lr=5e-6,
                weight_decay=1e-4,
                betas=[0.9, 0.99],
                train_keys=[],  # FULL net (reference trains every requires_grad param)
                grad_clip_norm=1.0,
                force_finite_grads=True,
            ),
            ref=RefConfig(update_every=0),  # frozen anchor
            data=PreferenceEnvConfig(
                jsonl_path=f"{_AGOKUL}/dpo_eval_2026_05_12/hpdv3_full_train.jsonl",
                tar_path=f"{_AGOKUL}/hpdv3_raw/images.tar",
                media="video",  # reference parity: T=1 video samples, VIDEO res table
                resolution="720",
                num_frames=1,
                cfg_dropout_rate=0.1,
                caption_key="prompt",
                shuffle=True,
            ),
        ),
        job=dict(group="posttrain_dpo", name="ga_32b_hpdv3_repro_v1"),
    ),
    flags={"allow_objects": True},
)

ConfigStore.instance().store(
    group="experiment", package="_global_", name="ga_32b_hpdv3_repro_v1", node=ga_32b_hpdv3_repro_v1
)
