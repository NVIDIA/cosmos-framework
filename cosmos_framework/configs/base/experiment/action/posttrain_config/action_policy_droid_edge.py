# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``action_policy_droid_edge`` — Cosmos3-Edge DROID action-policy post-training recipe.

Edge-tier (Nemotron-2B-Dense-VL) variant of ``action_policy_droid_nano``,
matching the internal reference run ``droid_lerobot_2b_ga_midtrain_policy``:
same DROID pipeline (``joint_pos`` 8D + ``use_state``, ``concat_view`` @ 480p,
chunk length 32, JSON action prompts), lr 5e-4 with a linear decay to 0.4x,
global batch 8192. Trains from the public ``nvidia/Cosmos3-Edge`` base
(converted via ``convert_model_to_dcp``); the generation heads and the
``k_norm_und_for_gen`` qk-norm fix keep training, the action heads init fresh.

Usage (1 node, 8 GPU)::

    DROID_ROOT=/path/to/droid_lerobot_640x360 \\
    BASE_CHECKPOINT_PATH=<Cosmos3-Edge DCP dir> \\
    WAN_VAE_PATH=<Wan2.2_VAE.pth> \\
    torchrun --nproc_per_node=8 -m cosmos_framework.scripts.train \\
        --sft-toml examples/toml/sft_config/action_policy_droid_edge.toml
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_droid_nano import (
    action_policy_droid_nano,
)
from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG

cs = ConfigStore.instance()


action_policy_droid_edge = copy.deepcopy(action_policy_droid_nano)
action_policy_droid_edge["job"]["name"] = "action_policy_droid_edge"


# Swap the Nano backbone for the Edge model config (loss_scale=10.0 is already the
# EDGE_MODEL_CONFIG default) and re-apply the DROID deltas on the fresh deep copy.
action_policy_droid_edge["model"]["config"] = copy.deepcopy(EDGE_MODEL_CONFIG)

# This recipe trains the action heads (vision SFT overrides action_gen to False).
action_policy_droid_edge["model"]["config"]["action_gen"] = True

# Generator qk-norm fix: the GA-midtrain base trained k_norm_und_for_gen, so enable
# it to load and keep training those weights.
action_policy_droid_edge["model"]["config"]["vlm_config"]["model_instance"]["config"]["use_und_k_norm_for_gen"] = True

# chunk_length=32 -> 33 observation frames; pin the VAE encode duration to match.
action_policy_droid_edge["model"]["config"]["tokenizer"]["encode_exact_durations"] = [33]

# Uncap the packed-sequence length (the EDGE default 45056 would truncate long
# DROID windows); processes the full vision sequence per step.
action_policy_droid_edge["model"]["config"]["max_num_tokens_after_packing"] = -1


# Edge reference-run optimization deltas.
_opt = action_policy_droid_edge["optimizer"]
_opt["lr"] = 5.0e-04  # for the 8192 global batch (8 samples/rank x 1024 ranks)
_opt["keys_to_select"].insert(_opt["keys_to_select"].index("llm2vae") + 1, "k_norm_und_for_gen")
action_policy_droid_edge["scheduler"]["f_start"] = [1.0e-06]


# gbs-8192 geometry and the Edge reference dataloader shape.
_dl = action_policy_droid_edge["dataloader_train"]
_dl["max_samples_per_batch"] = 8  # per rank; x ranks x grad_accum_iter -> global batch 8192
_dl["dataloader"]["batch_size"] = 16
_dl["dataloader"]["num_workers"] = 16
_dl["dataloader"]["prefetch_factor"] = 2
_dl["dataloader"]["datasets"]["droid"]["dataset"]["append_idle_frames"] = True  # Edge reference run


for _item in [action_policy_droid_edge]:
    _name = [k for k, v in globals().items() if v is _item][0]
    cs.store(group="experiment", package="_global_", name=_name, node=_item)
