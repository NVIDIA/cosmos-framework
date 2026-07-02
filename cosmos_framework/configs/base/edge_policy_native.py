# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Native Cosmos3 Edge policy-validation configuration."""

import copy
import os
from typing import Any

import attrs
from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.defaults.reasoner import Cosmos3EdgeReasoner_VLM_GCP_Config_9b4c028
from cosmos_framework.configs.base.defaults.tokenizer import Wan2pt2VAEConfig
from cosmos_framework.configs.base.experiment.sft.models.nano_model_config import NANO_MODEL_CONFIG
from cosmos_framework.utils.config import Config
from cosmos_framework.utils.lazy_config import LazyDict


@attrs.define(slots=False)
class EdgeConfig(Config):
    defaults: list[Any] = attrs.field(factory=lambda: ["_self_", {"model": "mot_fsdp"}, {"experiment": None}])


def _model_config():
    model = copy.deepcopy(NANO_MODEL_CONFIG)
    edge_vlm = copy.deepcopy(Cosmos3EdgeReasoner_VLM_GCP_Config_9b4c028)
    edge_vlm.pretrained_weights.enabled = False
    edge_vlm.tokenizer["config_variant"] = "hf"
    model["vlm_config"] = edge_vlm
    tokenizer_options = model["tokenizer"]
    tokenizer = copy.deepcopy(Wan2pt2VAEConfig)
    for key, value in tokenizer_options.items():
        tokenizer[key] = value
    tokenizer["bucket_name"] = ""
    tokenizer["object_store_credential_path_pretrained"] = ""
    tokenizer["vae_path"] = os.environ["WAN_VAE_PATH"]
    tokenizer["encode_exact_durations"] = [33]
    model["tokenizer"] = tokenizer
    model["max_num_tokens_after_packing"] = -1
    model["ema"]["enabled"] = False
    model["activation_checkpointing"]["mode"] = "none"
    return model


EDGE_POLICY_NATIVE = LazyDict(
    {
        "defaults": ["_self_"],
        "job": {
            "project": "cosmos3_edge_native_pytorch",
            "group": "thor_benchmark",
            "name": "edge_policy_native",
            "wandb_mode": "disabled",
        },
        "model": {"config": _model_config()},
        "trainer": {"distributed_parallelism": "fsdp", "run_validation": False, "seed": 0},
        "dataloader_train": None,
        "dataloader_val": None,
        "upload_reproducible_setup": False,
    },
    flags={"allow_objects": True},
)


def make_config() -> EdgeConfig:
    cfg = EdgeConfig(
        model=None,
        optimizer=None,
        scheduler=None,
        dataloader_train=None,
        dataloader_val=None,
    )
    cfg.trainer.straggler_detection.enabled = False
    cfg.trainer.run_validation = False
    cfg.upload_reproducible_setup = False

    from cosmos_framework.configs.base.defaults.model import register_model

    register_model()
    ConfigStore.instance().store(
        group="experiment",
        package="_global_",
        name="edge_policy_native",
        node=EDGE_POLICY_NATIVE,
    )
    return cfg
