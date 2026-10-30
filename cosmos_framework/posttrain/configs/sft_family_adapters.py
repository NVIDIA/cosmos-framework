# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Pinned local HF/native snapshots, shared SFT lifecycle, explicit components.

SFT_MODEL_PATH and SFT_DATA_ROOT select staged inputs. Text families consume text
rows; VLM families consume text/image/video rows. This recipe is not a claim that
every checkpoint in a bound family has passed real-model qualification.
"""

import json
import os
from pathlib import Path

from cosmos_framework.posttrain.config.adapters import AdapterComponentConfig as Component
from cosmos_framework.posttrain.config.adapters import AdapterConfig
from cosmos_framework.posttrain.config.model import HFModelConfig
from cosmos_framework.posttrain.configs.sft_cosmos3_nano_adapters import make_config as root_config
from cosmos_framework.posttrain.model.adapter_families import adapter_family


def make_config():
    config = root_config()
    path = os.environ["SFT_MODEL_PATH"]
    metadata = json.loads((Path(path) / "config.json").read_text())
    family = adapter_family(metadata["model_type"])
    is_text = family.hf_auto_class == "AutoModelForCausalLM"
    config.job.project = "adapter_families"
    config.job.group = family.name
    config.job.name = "adapter_sft_seed42"
    # The native MoE loader owns expert tensor layout and grouped-mm kernels.
    # It must not be dispatched through the dense HF loader.
    native = family.name in ("qwen3_vl", "qwen3_vl_moe") or os.environ.get("SFT_MODEL_BACKEND") == "i4"
    if native:
        if family.name not in ("qwen3_vl", "qwen3_vl_moe"):
            raise ValueError("this native recipe binds only Qwen3-VL dense/MoE")
    else:
        config.model_backend = "hf"
        config.model_hf = HFModelConfig(name_or_path=path, dtype="bfloat16", attn_impl="sdpa", grad_ckpt=True)
    common = dict(
        rank=8,
        alpha=16,
        method=os.environ.get("SFT_ADAPTER_METHOD", "lora"),
        use_rslora=os.environ.get("SFT_ADAPTER_RSLORA", "0") == "1",
    )
    prefix = "model.layers" if is_text else "model.language_model.layers"
    projections = (
        ["mlp.up_proj", "mlp.down_proj"] if family.name == "qwen3_5" else ["self_attn.q_proj", "self_attn.v_proj"]
    )
    groups = [Component(component="language", targets=[f"{prefix}.*.{leaf}" for leaf in projections], **common)]
    if not is_text:
        merger = ["model.visual.merger.linear_fc1", "model.visual.merger.linear_fc2"]
        if family.name in ("qwen2_vl", "qwen2_5_vl"):
            merger = ["model.visual.merger.mlp.0", "model.visual.merger.mlp.2"]
        if family.name in ("qwen3_vl", "qwen3_vl_moe"):
            merger += [
                "model.visual.deepstack_merger_list.*.linear_fc1",
                "model.visual.deepstack_merger_list.*.linear_fc2",
            ]
        layer = int(os.environ.get("SFT_VISION_LAYER", str(metadata["vision_config"]["depth"] - 1)))
        groups += [
            Component(component="projector", targets=merger, lr_multiplier=0.5, **common),
            Component(
                component="vision",
                targets=["model.visual.blocks.*.attn.qkv"],
                layers=[layer],
                lr_multiplier=0.1,
                **common,
            ),
        ]
    else:
        config.rl.processor_kwargs = {}
    config.rl.adapters = AdapterConfig(components=groups)
    return config
