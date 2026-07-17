# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.defaults.reasoner import VLMConfig
from cosmos_framework.configs.base.reasoner.defaults.policy_config import PolicyConfig

# Each entry replaces cfg.model.config.policy via package="model.config.policy".
# Sibling to the VFM vlm_config group at
# cosmos_framework/configs/base/defaults/reasoner.py: that group binds VLMConfig
# SKUs onto OmniMoTModelConfig.vlm_config; this group binds PolicyConfig SKUs
# onto VLMModelConfig.policy. Both groups compose the same VLMConfig: VFM as
# vlm_config: VLMConfig, VLM as policy.backbone: VLMConfig.

qwen2_5_vl_7b = PolicyConfig(backbone=VLMConfig(model_name="Qwen/Qwen2.5-VL-7B-Instruct"))

eagle_er_1p7b = PolicyConfig(
    backbone=VLMConfig(model_name="eagle_er_qwen3_1p7b_siglip_400m"),
    model_max_length=16000,
)

internvl3_5_1b = PolicyConfig(
    backbone=VLMConfig(model_name="OpenGVLab/InternVL3_5-1B-HF"),
    model_max_length=16000,  # 40960 is the max length by default.
)

internvl3_5_2b = PolicyConfig(
    backbone=VLMConfig(model_name="OpenGVLab/InternVL3_5-2B-HF"),
    model_max_length=16000,  # 40960 is the max length by default.
)

qwen3_vl_2b = PolicyConfig(backbone=VLMConfig(model_name="Qwen/Qwen3-VL-2B-Init"))

qwen3_vl_30b_a3b_instruct = PolicyConfig(backbone=VLMConfig(model_name="Qwen/Qwen3-VL-30B-A3B-Instruct"))

qwen3_vl_30b_a3b_thinking = PolicyConfig(backbone=VLMConfig(model_name="Qwen/Qwen3-VL-30B-A3B-Thinking"))

qwen3_vl_235b_a22b_thinking = PolicyConfig(backbone=VLMConfig(model_name="Qwen/Qwen3-VL-235B-A22B-Thinking"))

qwen3_vl_8b_thinking = PolicyConfig(backbone=VLMConfig(model_name="Qwen/Qwen3-VL-8B-Thinking"))

qwen3_vl_8b_instruct = PolicyConfig(backbone=VLMConfig(model_name="Qwen/Qwen3-VL-8B-Instruct"))

qwen3_vl_2b_instruct = PolicyConfig(backbone=VLMConfig(model_name="Qwen/Qwen3-VL-2B-Instruct"))

qwen3_vl_2b_thinking = PolicyConfig(backbone=VLMConfig(model_name="Qwen/Qwen3-VL-2B-Thinking"))

qwen3_vl_4b_instruct = PolicyConfig(backbone=VLMConfig(model_name="Qwen/Qwen3-VL-4B-Instruct"))

qwen3_vl_4b_thinking = PolicyConfig(backbone=VLMConfig(model_name="Qwen/Qwen3-VL-4B-Thinking"))

qwen3_vl_32b_instruct = PolicyConfig(backbone=VLMConfig(model_name="Qwen/Qwen3-VL-32B-Instruct"))

nemotron_nano_12b_v2_vl_bf16 = PolicyConfig(backbone=VLMConfig(model_name="nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"))


def register_vlm_policy():
    cs = ConfigStore.instance()
    cs.store(
        group="vlm_policy",
        package="model.config.policy",
        name="qwen2_5_vl_7b",
        node=qwen2_5_vl_7b,
    )
    cs.store(
        group="vlm_policy",
        package="model.config.policy",
        name="eagle_er_1p7b",
        node=eagle_er_1p7b,
    )
    cs.store(
        group="vlm_policy",
        package="model.config.policy",
        name="internvl3_5_1b",
        node=internvl3_5_1b,
    )
    cs.store(
        group="vlm_policy",
        package="model.config.policy",
        name="internvl3_5_2b",
        node=internvl3_5_2b,
    )
    cs.store(
        group="vlm_policy",
        package="model.config.policy",
        name="qwen3_vl_2b",
        node=qwen3_vl_2b,
    )
    cs.store(
        group="vlm_policy",
        package="model.config.policy",
        name="qwen3_vl_30b_a3b_instruct",
        node=qwen3_vl_30b_a3b_instruct,
    )
    cs.store(
        group="vlm_policy",
        package="model.config.policy",
        name="qwen3_vl_30b_a3b_thinking",
        node=qwen3_vl_30b_a3b_thinking,
    )
    cs.store(
        group="vlm_policy",
        package="model.config.policy",
        name="qwen3_vl_235b_a22b_thinking",
        node=qwen3_vl_235b_a22b_thinking,
    )
    cs.store(
        group="vlm_policy",
        package="model.config.policy",
        name="qwen3_vl_8b_thinking",
        node=qwen3_vl_8b_thinking,
    )
    cs.store(
        group="vlm_policy",
        package="model.config.policy",
        name="qwen3_vl_8b_instruct",
        node=qwen3_vl_8b_instruct,
    )
    cs.store(
        group="vlm_policy",
        package="model.config.policy",
        name="qwen3_vl_2b_instruct",
        node=qwen3_vl_2b_instruct,
    )
    cs.store(
        group="vlm_policy",
        package="model.config.policy",
        name="qwen3_vl_2b_thinking",
        node=qwen3_vl_2b_thinking,
    )
    cs.store(
        group="vlm_policy",
        package="model.config.policy",
        name="qwen3_vl_4b_instruct",
        node=qwen3_vl_4b_instruct,
    )
    cs.store(
        group="vlm_policy",
        package="model.config.policy",
        name="qwen3_vl_4b_thinking",
        node=qwen3_vl_4b_thinking,
    )
    cs.store(
        group="vlm_policy",
        package="model.config.policy",
        name="qwen3_vl_32b_instruct",
        node=qwen3_vl_32b_instruct,
    )
    cs.store(
        group="vlm_policy",
        package="model.config.policy",
        name="nemotron_nano_12b_v2_vl_bf16",
        node=nemotron_nano_12b_v2_vl_bf16,
    )
