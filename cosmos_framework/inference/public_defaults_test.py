# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from types import SimpleNamespace

import pytest

from cosmos_framework.inference.args import _CHECKPOINTS, OmniSampleOverrides

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_edge_defaults_follow_architecture_for_local_exports() -> None:
    vlm = SimpleNamespace(
        model_instance={"_target_": "cosmos_framework.model.generator.mot.unified_mot.Nemotron3DenseVLTextForCausalLM"},
        model_name="example/local-backbone",
    )
    key = OmniSampleOverrides._vlm_model_key(vlm)
    assert OmniSampleOverrides._VLM_MODEL_SIZE[key] == "2B"
    assert OmniSampleOverrides._NUM_FRAMES_DEFAULTS[key] == 121


def test_named_model_defaults_resolve_through_hub() -> None:
    assert _CHECKPOINTS
    for checkpoint in _CHECKPOINTS.values():
        assert not checkpoint.s3_uri
        assert checkpoint.hf.repository.startswith("nvidia/")
        assert checkpoint.hf.revision


def test_custom_tokenizer_requires_a_checkpoint_before_instantiation() -> None:
    import copy

    from omegaconf.errors import MissingMandatoryValue

    from cosmos_framework.configs.base.defaults import tokenizer
    from cosmos_framework.utils.lazy_config import instantiate

    config = copy.deepcopy(
        tokenizer.UniAE4x16x16C48T16to160MixpFpsMixEncoderNoncausalDecoderNoncausalNoganS3Nemotron2bVAEConfig
    )
    with pytest.raises(MissingMandatoryValue, match="vae_path"):
        instantiate(config)
    config.vae_path = "/data/my-tokenizer.pt"
    assert config.vae_path == "/data/my-tokenizer.pt"


@pytest.mark.parametrize("suffix", ["", "/"])
def test_public_backbone_uri_preserves_registered_revision(monkeypatch, tmp_path, suffix):
    import huggingface_hub

    from cosmos_framework.inference.common.checkpoints import register_checkpoints
    from cosmos_framework.model.generator.utils.safetensors_loader import _download_hf_checkpoint
    from cosmos_framework.utils.checkpoint_db import CheckpointConfig

    register_checkpoints()
    repo = "Qwen/Qwen3-VL-8B-Instruct"
    registered = CheckpointConfig.from_uri(f"s3://bucket/cosmos3/pretrained/huggingface/{repo}")
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    assert _download_hf_checkpoint(f"hf://{repo}{suffix}") == str(tmp_path)
    assert calls[0]["repo_id"] == repo
    assert calls[0]["revision"] == registered.hf.revision
    assert "*.safetensors" in calls[0]["allow_patterns"]


@pytest.mark.parametrize(
    "alias,repository",
    [
        ("s3://example-bucket/Cosmos3-Edge", "nvidia/Cosmos3-Edge"),
        ("s3://example-bucket/cosmos3/pretrained/huggingface/Qwen/Qwen3-VL-8B-Instruct", "Qwen/Qwen3-VL-8B-Instruct"),
    ],
)
def test_oss_alias_downloads_the_public_hub_snapshot(monkeypatch, tmp_path, alias, repository):
    from cosmos_framework.inference.common.checkpoints import register_checkpoints
    from cosmos_framework.utils import checkpoint_db

    register_checkpoints()
    monkeypatch.setattr(checkpoint_db, "INTERNAL", False)
    calls = []

    def download(config):
        calls.append((config.repository, config.revision))
        return str(tmp_path)

    monkeypatch.setattr(checkpoint_db.CheckpointDirHf, "download", download)
    checkpoint_db.download_checkpoint_v2.cache_clear()
    try:
        assert checkpoint_db.download_checkpoint_v2(alias) == str(tmp_path)
        assert len(calls) == 1 and calls[0][0] == repository and calls[0][1]
    finally:
        checkpoint_db.download_checkpoint_v2.cache_clear()
