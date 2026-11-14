# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Lightweight identities shared by export metadata and Qwen3-VL inference.

No model construction, execution kernels, or training imports belong here.
"""

PROFILE = "qwen3_vl_fp32_fixed_mma_v1"
BASE_FORMAT = "cosmos-unadapted-qwen3-vl-deployment-v1"
BASE_SHA256 = "9ab1478ba4659a470e3758fff33ba7418d3d99c12890d976b956bf5a9eaf5deb"
PROCESSOR_SHA256 = "13e4f0d99ce470581140ff7562c9d0d0ee22632aa3ff222b77a58dae6bab4066"
DEPENDENCIES = {"torch": "2.10.0", "transformers": "5.8.1", "triton": "3.6.0"}
EXPORT_EQUIVALENCE = "versioned_fixed_projection_fp32_boundaries_requires_qualification"
TEXT_ARCHITECTURE = {
    "model_type": "qwen3_vl_text",
    "hidden_size": 2048,
    "intermediate_size": 6144,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-6,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "max_position_embeddings": 262144,
    "tie_word_embeddings": True,
}
VISION_ARCHITECTURE = {
    "model_type": "qwen3_vl_vision",
    "hidden_size": 1024,
    "intermediate_size": 4096,
    "depth": 24,
    "num_heads": 16,
    "out_hidden_size": 2048,
    "patch_size": 16,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
    "in_channels": 3,
    "num_position_embeddings": 2304,
    "hidden_act": "gelu_pytorch_tanh",
    "deepstack_visual_indexes": [5, 11, 17],
}
ROOT_ARCHITECTURE = {
    "model_type": "qwen3_vl",
    "tie_word_embeddings": True,
    "image_token_id": 151655,
    "video_token_id": 151656,
    "vision_start_token_id": 151652,
    "vision_end_token_id": 151653,
}
ROPE_PARAMETERS = {
    "mrope_interleaved": True,
    "mrope_section": [24, 20, 20],
    "rope_type": "default",
    "rope_theta": 5000000,
}
CACHE_CONFIGURATION = {
    "sliding_window": None,
    "attention_chunk_size": None,
    "layer_types": None,
    "num_kv_shared_layers": None,
}


def validate_configuration(config):
    value = config if isinstance(config, dict) else config.to_dict()
    text, vision = value.get("text_config", {}), dict(value.get("vision_config", {}))
    # The immutable 4.57-era snapshot stores qwen3_vl here; TF5.8.1
    # serializes the same stock vision config as qwen3_vl_vision.
    if vision.get("model_type") == "qwen3_vl":
        vision["model_type"] = "qwen3_vl_vision"
    for section, expected in ((value, ROOT_ARCHITECTURE), (text, TEXT_ARCHITECTURE), (vision, VISION_ARCHITECTURE)):
        if any(section.get(key) != wanted for key, wanted in expected.items()):
            raise ValueError(f"{PROFILE} requires the recorded dense Qwen3-VL-2B configuration")
        if section.get("quantization_config") or section.get("auto_map"):
            raise ValueError(f"{PROFILE} does not admit quantization or remote model code")
    rope = text.get("rope_parameters")
    if rope is None:
        rope = {**text.get("rope_scaling", {}), "rope_theta": text.get("rope_theta")}
    if rope != ROPE_PARAMETERS or any(text.get(key) != wanted for key, wanted in CACHE_CONFIGURATION.items()):
        raise ValueError(f"{PROFILE} requires the recorded MRoPE and ordinary full-attention cache policy")


def validate_identity(identity):
    resolved = identity.get("resolved", {})
    targets = resolved.get("targets", [])
    if (
        identity.get("format") != "cosmos-posttrain-adapter-v3"
        or identity.get("architecture") != "qwen3_vl"
        or identity.get("base_storage_dtype") != "bfloat16"
        or identity.get("base", {}).get("sha256") != BASE_SHA256
        or identity.get("processor", {}).get("sha256") != PROCESSOR_SHA256
        or resolved.get("model_execution") != "qwen3_vl_hf_stock_v1"
        or not targets
        # The registered Qwen3-VL binding excludes output-head adapters. Its
        # frozen head remains tied to the embedding and is still projected by
        # this inference profile; admitting new head factors would expand scope.
        or any(row.get("name", "").split(".")[0] == "lm_head" for row in targets)
        or any(
            row.get("settings", {}).get("method") not in ("lora", "dora", "pissa")
            or row["settings"].get("execution", "factorized_v1") != "factorized_v1"
            or row["settings"].get("dtype") != "float32"
            for row in targets
        )
        or any(identity.get("dependencies", {}).get(k) != DEPENDENCIES[k] for k in ("torch", "transformers"))
    ):
        raise ValueError(f"{PROFILE} requires the exact BF16-base/FP32-factorized stock Qwen3-VL-2B artifact")


def validate_base_identity(identity):
    if (
        set(identity)
        != {
            "format",
            "architecture",
            "base_storage_dtype",
            "base",
            "processor",
            "dependencies",
            "execution_profile",
            "dtype",
        }
        or identity.get("format") != BASE_FORMAT
        or identity.get("architecture") != "qwen3_vl"
        or identity.get("base_storage_dtype") != "bfloat16"
        or identity.get("base", {}).get("sha256") != BASE_SHA256
        or identity.get("processor", {}).get("sha256") != PROCESSOR_SHA256
        or identity.get("dependencies") != DEPENDENCIES
        or identity.get("execution_profile") != PROFILE
        or identity.get("dtype") != "float32"
    ):
        raise ValueError(f"{PROFILE} requires its exact unadapted base deployment identity")
