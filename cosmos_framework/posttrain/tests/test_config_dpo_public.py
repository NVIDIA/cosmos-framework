# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The public T2I DPO recipe is catalog-free and uses synthetic preference pairs."""

from __future__ import annotations

import pytest

from cosmos_framework.posttrain.configs.dpo_t2i_public import make_config

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_public_dpo_recipe_is_self_contained(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve operator artifacts without importing an inherited experiment."""
    monkeypatch.setenv("DPO_TOKENIZER_PATH", "/models/Wan2.2_VAE.pth")
    monkeypatch.setenv("DPO_BASE_CHECKPOINT", "/models/cosmos3-dpo-base")

    config = make_config()

    assert config.defaults == ["_self_"]
    assert config.algorithm == "dpo"
    assert config.grid.backend == "local"
    assert config.grid.grouping == "single"
    assert config.grid.ranks == 4
    assert config.trainer.distributed_parallelism == "fsdp"

    model = config.model.config
    assert model.resolution == "256"
    assert model.state_t == 1
    assert model.sound_gen is False
    assert model.action_gen is False
    assert model.parallelism.data_parallel_shard_degree == 4
    assert model.tokenizer.bucket_name == ""
    assert model.tokenizer.object_store_credential_path_pretrained is None
    assert model.tokenizer.vae_path == "/models/Wan2.2_VAE.pth"
    assert model.vlm_config.model_name == "Qwen/Qwen3-VL-8B-Instruct"
    assert model.vlm_config.tokenizer.config_variant == "hf"
    assert model.vlm_config.pretrained_weights.enabled is False
    assert model.vlm_config.pretrained_weights.backbone_path == ""
    assert model.vlm_config.pretrained_weights.credentials_path == ""
    # Cosmos3-Nano carries the Qwen3-VL vision tower; build it so the base DCP loads key for key.
    assert model.vlm_config.model_instance.config.include_visual is True

    assert config.rl.init_from == "/models/cosmos3-dpo-base"
    # The sound and action heads exist only in the base checkpoint; T2I DPO does not build them.
    assert config.rl.init_keys_to_skip_loading == [
        "llm2sound.weight",
        "llm2sound.bias",
        "sound2llm.weight",
        "sound2llm.bias",
        "sound_modality_embed",
        "action2llm.bias.weight",
        "action2llm.fc.weight",
        "action_modality_embed",
        "llm2action.bias.weight",
        "llm2action.fc.weight",
    ]
    assert config.rl.max_steps == 1000
    assert config.rl.save_every == 0
    assert config.rl.optimizer.name == "adamw"
    assert config.rl.ref.build_ref is True
    assert config.rl.ref.update_every == 0
    assert config.rl.data.synthetic_pairs == 64
    assert config.rl.data.media == "image"
    assert config.rl.data.num_frames == 1
    assert config.rl.data.shuffle is False


def test_dpo_init_keys_to_skip_loading_must_be_unique_non_empty_strings() -> None:
    from cosmos_framework.posttrain.config.rl import DPOConfig

    with pytest.raises(TypeError):
        DPOConfig(init_keys_to_skip_loading="llm2sound.weight")
    with pytest.raises(TypeError):
        DPOConfig(init_keys_to_skip_loading=[""])
    with pytest.raises(ValueError):
        DPOConfig(init_keys_to_skip_loading=["llm2sound.weight", "llm2sound.weight"])
