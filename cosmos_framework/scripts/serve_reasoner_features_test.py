# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

from pathlib import Path

import pytest

from cosmos_framework.model.generator.reasoner_runtime import prepare_reasoner_model_config
from cosmos_framework.scripts.serve_reasoner_features import (
    _UNUSED_REASONER_SERVICE_PATH,
    _load_reasoner_service_config,
    _parse_args,
)

_RECIPE = Path(__file__).parents[2] / "examples" / "toml" / "sft_config" / "vision_sft_nano.toml"


def test_service_config_does_not_require_training_only_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("DATASET_PATH", "WAN_VAE_PATH", "BASE_CHECKPOINT_PATH"):
        monkeypatch.delenv(name, raising=False)
    args = _parse_args(
        [
            "--sft-toml",
            str(_RECIPE),
            "--checkpoint",
            "/checkpoints/Cosmos3-Nano",
            "--reasoner-fingerprint",
            "reasoner",
            "--tokenizer-fingerprint",
            "tokenizer",
            "--framing-fingerprint",
            "framing",
            "--",
            "model.config.vlm_config.model_instance.config.qk_norm_for_text=false",
        ]
    )

    assert args.host == "127.0.0.1"
    config = _load_reasoner_service_config(args)

    model_instance = prepare_reasoner_model_config(config)
    assert str(model_instance["_target_"]).endswith("Qwen3VLTextForCausalLM")
    assert model_instance["config"]["qk_norm_for_text"] is False
    assert config.dataloader_train is None
    assert config.dataloader_val is None
    assert config.checkpoint.load_path == _UNUSED_REASONER_SERVICE_PATH
    assert config.model.config.tokenizer.vae_path == _UNUSED_REASONER_SERVICE_PATH
