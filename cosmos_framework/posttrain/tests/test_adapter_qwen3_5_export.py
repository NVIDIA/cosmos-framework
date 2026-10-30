# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Public Qwen3.5 sources include MTP weights outside the trained HF model."""

import pytest
import torch
import transformers
from safetensors.torch import load_file, save_file

from cosmos_framework.posttrain.checkpoint.adapters import save_adapter_artifact
from cosmos_framework.posttrain.checkpoint.tools.adapters import export_merged_adapter
from cosmos_framework.posttrain.config.adapters import AdapterComponentConfig, AdapterConfig
from cosmos_framework.posttrain.config.model import HFModelConfig
from cosmos_framework.posttrain.model.hf import build_hf_model
from cosmos_framework.posttrain.tests.test_model_qwen3_5 import config


@pytest.mark.parametrize("extra", ["mtp.fc.weight", "unreviewed.weight"])
def test_export_explicitly_accounts_for_unused_source_head(tmp_path, monkeypatch, extra):
    base = tmp_path / "base"
    net = transformers.AutoModelForImageTextToText.from_config(config.__wrapped__())
    net.save_pretrained(base)
    path = base / "model.safetensors"
    state = load_file(str(path))
    state[extra] = torch.randn(4, 4)
    save_file(state, str(path))
    model = build_hf_model(
        HFModelConfig(name_or_path=str(base), dtype="float32", attn_impl="sdpa"),
        adapters=AdapterConfig(
            components=[AdapterComponentConfig(targets=["model.language_model.layers.0.mlp.up_proj"], rank=2)]
        ),
    )
    portable = tmp_path / "adapter"
    save_adapter_artifact(model.net, str(portable))
    captured = {}

    def materialize(**kwargs):
        captured.update(kwargs)
        merged = torch.load(kwargs["ckpt_dir"] + "/model.pt", weights_only=True)["net"]
        assert extra not in merged
        net.load_state_dict(merged, strict=True)
        return kwargs["out"]

    monkeypatch.setattr("cosmos_framework.posttrain.checkpoint.tools.materialize_hf.run_request", materialize)
    if extra.startswith("mtp."):
        export_merged_adapter(str(portable), str(base), str(tmp_path / "merged"))
        assert captured["export_metadata"]["excluded_base_tensors"] == {
            extra: {"shape": [4, 4], "reason": "unused_source_mtp_head"}
        }
    else:
        with pytest.raises(ValueError, match="absent from the bound export architecture"):
            export_merged_adapter(str(portable), str(base), str(tmp_path / "merged"))
        assert not captured
