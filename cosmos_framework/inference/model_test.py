# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
from pathlib import Path

import attrs
import hydra
import safetensors.torch
import torch
import torch.distributed.checkpoint as dcp

from cosmos_framework.configs.base.defaults.compile import CompileConfig
from cosmos_framework.configs.base.defaults.parallelism import ParallelismConfig
from cosmos_framework.inference.args import _CHECKPOINTS, DEFAULT_CHECKPOINT
from cosmos_framework.inference.common.args import CheckpointType
from cosmos_framework.inference.common.config import structure_config
from cosmos_framework.inference.model import (
    Cosmos3OmniConfig,
    _diffusers_to_net_key,
    _diffusers_weight_map,
    _DiffusersHuggingFaceStorageReader,
    _DiffusersLoadPlanner,
    _is_diffusers_checkpoint,
    _normalize_diffusers_target_key,
)


def test_config():
    parallelism = ParallelismConfig(
        data_parallel_shard_degree=2,
        context_parallel_shard_degree=2,
        cfg_parallel_shard_degree=2,
    )
    compile = CompileConfig(enabled=True, use_cuda_graphs=True)
    checkpoint_path = DEFAULT_CHECKPOINT.download()
    config = Cosmos3OmniConfig.from_pretrained(
        checkpoint_path,
        parallelism=attrs.asdict(parallelism),
        compile=attrs.asdict(compile),
    )
    assert hydra.utils.instantiate(structure_config(config.parallelism, ParallelismConfig)) == parallelism
    assert hydra.utils.instantiate(structure_config(config.compile, CompileConfig)) == compile


def test_checkpoint_type_from_path_hf_index(tmp_path: Path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text("{}", encoding="utf-8")

    assert CheckpointType.from_path(tmp_path) == CheckpointType.HF


def test_normalize_diffusers_target_key():
    assert (
        _normalize_diffusers_target_key(
            "model.net._orig_mod.language_model.model.layers.0._checkpoint_wrapped_module.input_layernorm.weight"
        )
        == "language_model.model.layers.0.input_layernorm.weight"
    )


def test_diffusers_to_net_key():
    cases = {
        "lm_head.weight": "language_model.lm_head.weight",
        "embed_tokens.weight": "language_model.model.embed_tokens.weight",
        "norm_moe_gen.weight": "language_model.model.norm_moe_gen.weight",
        "layers.18.self_attn.to_q.weight": "language_model.model.layers.18.self_attn.q_proj.weight",
        "layers.18.self_attn.to_out.weight": "language_model.model.layers.18.self_attn.o_proj.weight",
        "layers.18.self_attn.norm_q.weight": "language_model.model.layers.18.self_attn.q_norm.weight",
        "layers.18.self_attn.add_k_proj.weight": "language_model.model.layers.18.self_attn.k_proj_moe_gen.weight",
        "layers.18.self_attn.to_add_out.weight": "language_model.model.layers.18.self_attn.o_proj_moe_gen.weight",
        "layers.18.self_attn.norm_added_k.weight": "language_model.model.layers.18.self_attn.k_norm_moe_gen.weight",
        "language_model.model.layers.18.self_attn.to_q.weight": "language_model.model.layers.18.self_attn.q_proj.weight",
        "proj_in.weight": "vae2llm.weight",
        "proj_out.bias": "llm2vae.bias",
        "time_embedder.linear_1.weight": "time_embedder.mlp.0.weight",
        "time_embedder.linear_2.bias": "time_embedder.mlp.2.bias",
        "audio_proj_in.weight": "sound2llm.weight",
        "audio_proj_out.bias": "llm2sound.bias",
        "audio_modality_embed": "sound_modality_embed",
        "action_proj_in.fc.weight": "action2llm.fc.weight",
        "action_proj_out.bias.weight": "llm2action.bias.weight",
        "action_modality_embed": "action_modality_embed",
    }
    for diffusers_key, net_key in cases.items():
        assert _diffusers_to_net_key(diffusers_key, "transformer/diffusion_pytorch_model.safetensors") == net_key

    assert (
        _diffusers_to_net_key("blocks.0.attn.qkv.weight", "vision_encoder/model.safetensors")
        == "language_model.visual.blocks.0.attn.qkv.weight"
    )
    assert _diffusers_to_net_key("decoder.conv.weight", "vae/diffusion_pytorch_model.safetensors") is None


def test_diffusers_dcp_load_remaps_nested_safetensors(tmp_path: Path):
    shard_rel_path = "transformer/diffusion_pytorch_model.safetensors"
    shard_path = tmp_path / shard_rel_path
    shard_path.parent.mkdir(parents=True)

    source = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    safetensors.torch.save_file({"proj_in.weight": source}, shard_path)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {
                    "proj_in.weight": shard_rel_path,
                    "decoder.conv.weight": "vae/diffusion_pytorch_model.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )

    target = {"model.net._orig_mod.vae2llm.weight": torch.empty_like(source)}
    dcp.load(
        state_dict=target,
        storage_reader=_DiffusersHuggingFaceStorageReader(tmp_path),
        planner=_DiffusersLoadPlanner(tmp_path),
    )

    torch.testing.assert_close(target["model.net._orig_mod.vae2llm.weight"], source)


def test_diffusers_weight_map_registered_checkpoint():
    checkpoint_path = Path(_CHECKPOINTS["Cosmos3-Nano"].hf.download())

    assert (checkpoint_path / "model_index.json").exists()
    assert (checkpoint_path / "model.safetensors.index.json").exists()
    assert CheckpointType.from_path(checkpoint_path) == CheckpointType.HF
    assert _is_diffusers_checkpoint(checkpoint_path)

    weight_map = _diffusers_weight_map(checkpoint_path)
    assert weight_map["proj_in.weight"].startswith("transformer/")
    assert weight_map["blocks.0.attn.qkv.weight"] == "vision_encoder/model.safetensors"
    assert _diffusers_to_net_key("proj_in.weight", weight_map["proj_in.weight"]) == "vae2llm.weight"
    assert (
        _diffusers_to_net_key("blocks.0.attn.qkv.weight", weight_map["blocks.0.attn.qkv.weight"])
        == "language_model.visual.blocks.0.attn.qkv.weight"
    )
