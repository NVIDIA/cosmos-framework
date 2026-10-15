# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Single-node Nano FD+ID adaptation to measured YAM reel trajectories."""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_fd_droid_posttrain import (
    action_fd_droid_posttrain,
)
from cosmos_framework.utils.lazy_config import LazyCall as L
from failuregen.yam_training.dataset import YamReelDataset

cfg = copy.deepcopy(action_fd_droid_posttrain)
cfg["job"].update(project="cosmos3_yam", group="fd_id", name="yam_reel_nano", wandb_mode="disabled")
model = cfg["model"]["config"]
model["num_embodiment_domains"] = 33
model["resolution"] = "720"
model["diffusion_expert_config"]["max_vae_latent_side_after_patchify"] = 30
# First-run portability; compilation can be profiled after the GPU smoke passes.
model["compile"]["enabled"] = False
model["tokenizer"]["encode_exact_durations"] = [17]
# OmniMoTModel.validation_step is a stub in the reference framework. Evaluate
# saved checkpoints with held-out replay instead of entering that trainer path.
cfg["trainer"].update(logging_iter=1, grad_accum_iter=1, run_validation=False, run_validation_on_start=False)
cfg["trainer"]["callbacks"]["compile_tokenizer"]["enabled"] = False
# The launcher deletes sigma_loss_analysis after Hydra composes all defaults.
cfg["trainer"]["straggler_detection"]["enabled"] = False
cfg["checkpoint"]["save_iter"] = 100
cfg["trainer"]["max_iter"] = "${oc.decode:${oc.env:YAM_MAX_ITER}}"
cfg["scheduler"]["cycle_lengths"] = ["${trainer.max_iter}"]

train = cfg["dataloader_train"]
train["dataset_name"] = "yam_reel"
# This checkout requires count-packing OR token-packing, not both.
train["max_sequence_length"] = None
train["max_samples_per_batch"] = "${oc.decode:${oc.env:YAM_SAMPLES_PER_RANK,4}}"
train["dataloader"]["num_workers"] = 2
train["dataloader"]["in_order"] = True
train["dataloader"]["datasets"] = dict(
    yam=dict(
        ratio=1,
        dataset=L(YamReelDataset)(
            prepared_root="${oc.env:YAM_PREPARED_ROOT}",
            dataset_root="${oc.env:YAM_DATASET_ROOT}",
            split="train",
            seed=42,
            tokenizer_config="${model.config.vlm_config.tokenizer}",
            max_action_dim="${model.config.max_action_dim}",
        ),
    )
)
cfg["dataloader_val"] = None

ConfigStore.instance().store(group="experiment", package="_global_", name="failuregen_yam_joint_fd_id", node=cfg)

eef_cfg = copy.deepcopy(cfg)
eef_cfg["job"]["name"] = "yam_eef_nano"
eef_cfg["model"]["config"]["num_embodiment_domains"] = 32
eef_cfg["dataloader_train"]["dataloader"]["datasets"]["yam"]["dataset"]["action_space"] = "eef"
ConfigStore.instance().store(group="experiment", package="_global_", name="failuregen_yam_eef_fd_id", node=eef_cfg)
