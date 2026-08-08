# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import attrs
import hydra
import pytest
import safetensors.torch
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.metadata import Metadata, TensorProperties, TensorStorageMetadata

from cosmos_framework.configs.base.defaults.compile import CompileConfig
from cosmos_framework.configs.base.defaults.parallelism import ParallelismConfig
from cosmos_framework.configs.base.defaults.quantization import QuantizationConfig
from cosmos_framework.inference.args import _CHECKPOINTS, DEFAULT_CHECKPOINT
from cosmos_framework.inference.common.args import CheckpointType
from cosmos_framework.inference.common.config import structure_config
from cosmos_framework.inference.model import (
    Cosmos3OmniConfig,
    Cosmos3OmniModel,
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


def test_diffusers_weight_map_transformer_only(tmp_path: Path) -> None:
    transformer_path = tmp_path / "transformer"
    transformer_path.mkdir()
    (tmp_path / "model_index.json").write_text("{}", encoding="utf-8")
    (transformer_path / "config.json").write_text("{}", encoding="utf-8")
    (transformer_path / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {
                    "proj_in.weight": "diffusion_pytorch_model-00001-of-00002.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )

    assert CheckpointType.from_path(tmp_path) == CheckpointType.HF
    assert _is_diffusers_checkpoint(tmp_path)
    assert _diffusers_weight_map(tmp_path) == {
        "proj_in.weight": "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
    }


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


class LightweightDcpModel(Cosmos3OmniModel):
    """Skip the hydra model build so the checkpoint-loading plumbing can be tested on CPU."""

    model: Any

    def __init__(self, config: Cosmos3OmniConfig, *args: object, **kwargs: object) -> None:
        object.__setattr__(self, "model", SimpleNamespace(net=object()))


def test_diffusers_load_planner_skips_modelopt_fp8_weights(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    source_key = "transformer.time_embedder.linear_1.weight"
    target_key = "time_embedder.mlp.0.weight"
    (checkpoint_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {source_key: "transformer/model.safetensors"}}),
        encoding="utf-8",
    )
    metadata = Metadata(
        state_dict_metadata={
            source_key: TensorStorageMetadata(
                properties=TensorProperties(dtype=torch.float8_e4m3fn),
                size=torch.Size([1]),
                chunks=[],
            )
        }
    )
    planner = _DiffusersLoadPlanner(checkpoint_path, defer_modelopt_fp8_weights_loading=True)

    planner.set_up_planner({target_key: torch.empty(1)}, metadata)
    plan = planner.create_local_plan()

    assert planner.state_dict == {}
    assert planner.skipped_source_keys == {source_key}
    assert plan.items == []


def test_from_pretrained_dcp_installs_modelopt_fp8_after_load(tmp_path: Path) -> None:
    events: list[str] = []
    converted_targets: list[object] = []
    merged_weight_map = {"selected.weight": "transformer/model.safetensors"}

    def record_load(**kwargs: Any) -> None:
        assert kwargs["planner"].defer_modelopt_fp8_weights_loading
        events.append("load")

    def record_modelopt_conversion(
        model: object, checkpoint_path: Path, *, key_mapper: object, weight_map: dict[str, str]
    ) -> list[str]:
        del checkpoint_path, key_mapper
        events.append("modelopt")
        converted_targets.append(model)
        assert weight_map is merged_weight_map
        return ["selected"]

    with (
        patch.object(CheckpointType, "from_path", return_value=CheckpointType.HF),
        patch("cosmos_framework.inference.model._is_diffusers_checkpoint", return_value=True),
        patch("cosmos_framework.inference.model.is_modelopt_fp8_checkpoint", return_value=True),
        patch("cosmos_framework.inference.model._diffusers_weight_map", return_value=merged_weight_map),
        patch("cosmos_framework.inference.model.plan_modelopt_fp8_targets", return_value=["selected"]),
        patch("cosmos_framework.inference.model.get_model_state_dict", return_value={}),
        patch("cosmos_framework.inference.model.dcp.load", side_effect=record_load),
        patch(
            "cosmos_framework.inference.model.apply_modelopt_fp8_checkpoint_inplace",
            side_effect=record_modelopt_conversion,
        ),
    ):
        model = LightweightDcpModel.from_pretrained_dcp(checkpoint_path=tmp_path, config=Cosmos3OmniConfig())

    assert events == ["load", "modelopt"]
    assert converted_targets == [model.model.net]


def test_from_pretrained_dcp_rejects_modelopt_fp8_with_runtime_quantization(tmp_path: Path) -> None:
    with patch("cosmos_framework.inference.model.is_modelopt_fp8_checkpoint", return_value=True):
        with pytest.raises(ValueError, match="already quantized"):
            LightweightDcpModel.from_pretrained_dcp(
                checkpoint_path=tmp_path,
                config=Cosmos3OmniConfig(),
                quantization_config=QuantizationConfig(method="fp8"),
            )


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
