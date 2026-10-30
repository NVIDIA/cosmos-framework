# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Example posttrain DPO experiment — inherits an upstream experiment + adds the RL section.

This is the researcher-facing shape (DESIGN: config system, requirement #1): inherit the
full model/data tree from an upstream experiment via `defaults`, then add only the typed `rl`
section. Launch + ablate via the unified loader:

    PYTHONPATH=. torchrun --nproc_per_node=4 -m cosmos_framework.posttrain.entrypoints.dpo -- \
        experiment=cosmos3_ga_16bm8b_dpo_v1 rl.beta=1000 rl.optimizer.lr=5e-7 \
        model.config.parallelism.data_parallel_shard_degree=4

Here it inherits the 8B *test* experiment (random backbone, no ckpt) so it runs without
data/weights; swap the `defaults` entry to `cosmos3_ga_16bm8b_v2_midtrain` + point
`rl.data.jsonl_path` at real preference pairs for a convergence run.
"""

from __future__ import annotations

from hydra.core.config_store import ConfigStore

from cosmos_framework.utils.lazy_config import LazyDict
from cosmos_framework.posttrain.config.rl import DPOConfig, OptimizerConfig, PreferenceEnvConfig, RefConfig

# In the unified path the model/data tree comes from the inherited experiment (config.model)
# and parallelism is overridden via `model.config.parallelism.*`; the `rl.model` subtree is
# unused here (it only sources the model in the self-contained runners). So `rl` carries just
# the RL hyperparameters: objective, optimizer (RL-owned), reference, and preference data.
cosmos3_ga_16bm8b_dpo_v1 = LazyDict(
    dict(
        defaults=[
            "/experiment/t2w_mot_exp302_000_qwen3_vl_8b_multires",  # inherit the upstream model/data (test exp)
            "_self_",  # this file's keys override the inherited ones
        ],
        rl=DPOConfig(
            beta=500.0,
            grad_accum=1,
            max_steps=4,
            pairs_per_step=1,
            save_every=2,
            # freeze backbone, train only the OmniMoT gen-pathway params
            optimizer=OptimizerConfig(lr=1e-6, train_keys=["moe_gen", "time_embedder", "vae2llm", "llm2vae"]),
            ref=RefConfig(build_ref=True, update_every=0),
            data=PreferenceEnvConfig(synthetic_pairs=64, media="video", resolution="256", num_frames=9),
        ),
        job=dict(group="posttrain_dpo", name="cosmos3_ga_16bm8b_dpo_v1"),
    ),
    flags={"allow_objects": True},
)

cs = ConfigStore.instance()
cs.store(group="experiment", package="_global_", name="cosmos3_ga_16bm8b_dpo_v1", node=cosmos3_ga_16bm8b_dpo_v1)
