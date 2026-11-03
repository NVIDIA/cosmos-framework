# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Text-to-image Diffusion-DPO experiment (T=1) — the image sibling of the video DPO.

Same unified shape: inherit an upstream experiment via `defaults` + a typed `rl` section, but
`rl.data.media="image"` (→ single latent frame → rectified_flow_image) with the T2I
hyperparameters from the cosmos-rl recipe (beta=5000, lr=1e-6). Run via the unified loader:

    PYTHONPATH=. torchrun --nproc_per_node=1 -m cosmos_framework.posttrain.entrypoints.dpo -- \
        experiment=cosmos3_ga_16bm8b_t2i_dpo_v1 rl.max_steps=4

Real run: point `rl.data.jsonl_path` at HPD image pairs to reproduce the +0.144 HPSv3 trend.
"""

from __future__ import annotations

from hydra.core.config_store import ConfigStore

from cosmos_framework.utils.lazy_config import LazyDict
from cosmos_framework.posttrain.config.rl import DPOConfig, OptimizerConfig, PreferenceEnvConfig, RefConfig

cosmos3_ga_16bm8b_t2i_dpo_v1 = LazyDict(
    dict(
        defaults=[
            "/experiment/t2w_mot_exp302_000_qwen3_vl_8b_multires",  # inherit the upstream model/data (test exp)
            "_self_",
        ],
        rl=DPOConfig(
            beta=5000.0,  # T2I Diffusion-DPO convention
            grad_accum=1,
            max_steps=4,
            pairs_per_step=1,
            save_every=2,
            # freeze backbone, train only the OmniMoT gen-pathway params
            optimizer=OptimizerConfig(lr=1e-6, train_keys=["moe_gen", "time_embedder", "vae2llm", "llm2vae"]),
            ref=RefConfig(build_ref=True, update_every=0),
            data=PreferenceEnvConfig(synthetic_pairs=64, media="image", resolution="256", num_frames=1),
        ),
        job=dict(group="posttrain_dpo", name="cosmos3_ga_16bm8b_t2i_dpo_v1"),
    ),
    flags={"allow_objects": True},
)

cs = ConfigStore.instance()
cs.store(group="experiment", package="_global_", name="cosmos3_ga_16bm8b_t2i_dpo_v1", node=cosmos3_ga_16bm8b_t2i_dpo_v1)
