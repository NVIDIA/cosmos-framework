# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""VideoPhy-2 SFT recipe (Cosmos3-Super tier): Qwen3-VL-32B full fine-tune.

Super-tier counterpart of ``videophy2_sft_nano``. Reuses that recipe's VideoPhy-2
``LocalSFTDataset`` + ``CosmosDataLoader`` dataflow verbatim (imported + deepcopied,
the same variant idiom ``llava_ov_vlm.py`` uses) and changes only what the larger
backbone needs:

  * ``vlm_policy`` ``qwen3_vl_8b_instruct`` -> ``qwen3_vl_32b_instruct``
    (``Qwen/Qwen3-VL-32B-Instruct``) — the visual tower + config the Cosmos3-Super
    Reasoner is built on, mirroring how the nano recipe rides the 8B tower.
  * FSDP full-shard across every rank (``data_parallel_shard_degree=-1``) so the 32B
    weights + optimizer state fit, instead of the nano recipe's fixed dp=8. This is
    the same super-tier sharding switch ``vision_sft_super`` makes, and lets the recipe
    run unchanged on a 4-GPU (e.g. GB200x4) or 8-GPU allocation.

Still a full fine-tune (no LoRA): the freeze config is inherited from the nano recipe
(vision encoder frozen, LM + mm_projector trained).

Launch via ``examples/launch_sft_videophy2_super.sh`` after
``prepare_videophy2_from_hf`` populates ``$VIDEOPHYSICS_ROOT``, supplying the merged
Cosmos3-Super Reasoner checkpoint through ``VLM_SAFETENSORS_PATH`` (see the launch shell).
"""

from __future__ import annotations

import copy

from hydra.core.config_store import ConfigStore

# Importing the nano module registers `videophy2_sft_nano` and pulls in the shared
# VideoPhy-2 dataflow helpers; we clone its LazyDict rather than re-declaring them.
from cosmos_framework.configs.base.reasoner.experiment.videophy2_sft_nano import videophy2_sft_nano

cs = ConfigStore.instance()


videophy2_sft_super = copy.deepcopy(videophy2_sft_nano)

# Backbone: nano 8B -> super 32B. The vlm_policy override lives in the Hydra
# `defaults` list; rewrite that one entry's value in place, leaving the rest of the
# recipe (checkpoint backend, callbacks, dataflow) untouched.
for _default in videophy2_sft_super["defaults"]:
    if not isinstance(_default, str) and "override /vlm_policy" in _default:
        _default["override /vlm_policy"] = "qwen3_vl_32b_instruct"

# 32B full fine-tune: shard model + optimizer state across every rank (FSDP full
# shard, auto-sized from WORLD_SIZE) instead of the nano recipe's fixed dp_shard=8,
# so the recipe fits on 4- or 8-GPU nodes.
# examples/toml/sft_config/videophy2_sft_super.toml is authoritative at launch and
# repeats these; keeping them here lets `experiment=videophy2_sft_super` run standalone.
videophy2_sft_super.model.config.parallelism.data_parallel_shard_degree = -1
videophy2_sft_super.model.config.parallelism.data_parallel_replicate_degree = 1


for _item in [videophy2_sft_super]:
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]
    if "job" not in _item:
        _item["job"] = dict(name=experiment_name + "_${now:%Y-%m-%d}_${now:%H-%M-%S}")
    else:
        _item["job"]["name"] = experiment_name + "_${now:%Y-%m-%d}_${now:%H-%M-%S}"

    cs.store(group="experiment", package="_global_", name=experiment_name, node=_item)
