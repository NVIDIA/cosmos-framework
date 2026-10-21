# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Public T2I NFT recipe stays self-contained and uses only operator-provided artifacts."""

from __future__ import annotations

import pytest

from cosmos_framework.posttrain.config.config_nft import PosttrainNFTConfig
from cosmos_framework.posttrain.config.experiment.nft import (
    NFT_T2I_PROFILES,
    NFT_T2I_PUBLIC_HPSV3_CONTRACT,
    NFT_T2I_PUBLIC_HPSV3_LORA_720P_PROFILE,
)
from cosmos_framework.posttrain.config.experiment.nft.cosmos3_nft_t2i_public import cosmos3_nft_t2i_public

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _build_public_config() -> PosttrainNFTConfig:
    from cosmos_framework.utils.config_helper import import_module, override
    from cosmos_framework.posttrain.config.config import build_root

    import_module(
        "cosmos_framework.posttrain.config.experiment.nft.cosmos3_nft_t2i_public",
        reload=True,
    )
    root = build_root(PosttrainNFTConfig, "posttrain_nft", register_experiment_configs=False)
    return override(root, ["--", "experiment=cosmos3_nft_t2i_public"])


def test_public_t2i_recipe_has_no_internal_experiment_parent() -> None:
    defaults = list(cosmos3_nft_t2i_public["defaults"])

    assert all("/experiment/" not in str(default) for default in defaults)
    assert {"override /model": "mot_fsdp"} in defaults
    assert {"override /tokenizer": "wan2pt2_tokenizer"} in defaults
    assert {"override /vlm_config": "qwen3_vl_mot_vlm_32b_instruct"} in defaults


def test_public_t2i_profile_is_registered_without_the_private_backbone() -> None:
    contract = NFT_T2I_PUBLIC_HPSV3_CONTRACT

    assert contract.profile == NFT_T2I_PUBLIC_HPSV3_LORA_720P_PROFILE
    assert contract.base_checkpoint == ""
    assert contract.backbone_path == ""
    assert NFT_T2I_PROFILES[contract.profile] is contract


def test_public_t2i_recipe_resolves_external_artifacts_and_passes_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cosmos_framework.utils.lazy_config import instantiate
    from cosmos_framework.posttrain.entrypoints.nft import validate_nft_runtime_config

    monkeypatch.setenv("NFT_BASE_CHECKPOINT", "/models/cosmos3-nft-base")
    monkeypatch.setenv("NFT_TOKENIZER_PATH", "/models/Wan2.2_VAE.pth")
    monkeypatch.setenv("NFT_T2I_PROMPTS_JSONL", "/data/prompts.jsonl")
    monkeypatch.setenv("NFT_T2I_PROMPTS_SHA256", "a" * 64)
    monkeypatch.setenv("NFT_REWARD_REVISION", "hpsv3@git:0123456789abcdef")
    environment = {
        "REMOTE_REWARD_ENQUEUE_URL": "https://reward/enqueue",
        "REMOTE_REWARD_FETCH_URL": "https://reward/fetch",
        "REMOTE_REWARD_TOKEN": "secret",
        "WORLD_SIZE": "8",
    }

    config = _build_public_config()
    rl = instantiate(config.rl)

    assert rl.init_from == "/models/cosmos3-nft-base"
    assert config.model.config.tokenizer.vae_path == "/models/Wan2.2_VAE.pth"
    assert config.model.config.vlm_config.pretrained_weights.backbone_path == ""
    assert config.dataloader_train.path == "/data/prompts.jsonl"
    assert rl.reward.provider_revision == "hpsv3@git:0123456789abcdef"
    assert validate_nft_runtime_config(config, rl, environ=environment) == []
