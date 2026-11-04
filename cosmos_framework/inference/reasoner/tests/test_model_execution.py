# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Stock MoE loader activation and artifact/profile refusals."""

import copy
import json

import pytest
import torch

from cosmos_framework.inference.reasoner.model_execution import PROFILE, model_execution_profile, validate_loaded_model
from cosmos_framework.inference.reasoner.runtime import CosmosFrameworkRuntime


def artifact(path, config=None):
    config = config or {"model_type": "qwen3_vl_moe"}
    config.update(dtype="float32", torch_dtype="float32", cosmos_model_execution=PROFILE)
    marker = {
        "identity": {
            "output_dtype": "float32",
            "export_metadata": {
                "format": "cosmos-merged-adapter-v1",
                "adapter_identity": {
                    "format": "cosmos-posttrain-adapter-v3",
                    "architecture": "qwen3_vl_moe",
                    "resolved": {"model_execution": PROFILE},
                    "dependencies": {"torch": "2.10.0", "transformers": "5.8.1"},
                },
            },
        }
    }
    (path / "config.json").write_text(json.dumps(config))
    (path / ".materialized").write_text(json.dumps(marker))
    return config, marker


def test_model_execution_binds_config_transaction_and_runtime(tmp_path, monkeypatch):
    config, _ = artifact(tmp_path)
    assert model_execution_profile(str(tmp_path)) == PROFILE
    config.pop("cosmos_model_execution")
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="conflicting"):
        model_execution_profile(str(tmp_path))
    artifact(tmp_path)
    monkeypatch.setattr("cosmos_framework.inference.reasoner.model_execution.version", lambda name: "4.57.6")
    with pytest.raises(ValueError, match="Transformers 5.8.1"):
        model_execution_profile(str(tmp_path))


@pytest.mark.parametrize("dtype,attention", [("bfloat16", "sdpa"), ("float32", "eager")])
def test_runtime_refuses_conflicting_stock_contract_before_loading(tmp_path, dtype, attention):
    artifact(tmp_path)
    runtime = CosmosFrameworkRuntime.__new__(CosmosFrameworkRuntime)
    runtime.model_path, runtime.dtype, runtime.attn_implementation = str(tmp_path), dtype, attention
    with pytest.raises(ValueError, match="stock FP32 MoE"):
        runtime._load()


@pytest.mark.parametrize("marker", [None, "old marker", "null", "[]", "{}"])
def test_untagged_model_keeps_its_existing_contract(tmp_path, marker):
    (tmp_path / "config.json").write_text('{"model_type":"qwen3_vl_moe"}')
    if marker is not None:
        (tmp_path / ".materialized").write_text(marker)
    assert model_execution_profile(str(tmp_path)) is None


def test_actual_runtime_selects_stock_fp32_sdpa_grouped_loader(tmp_path):
    from transformers import AutoModelForImageTextToText, Qwen3VLMoeConfig

    from cosmos_framework.posttrain.tests.helpers.sft_packer import local_processor, local_tokenizer
    from cosmos_framework.posttrain.tests.test_model_qwen3_vl_gpu import _moe_config

    config = Qwen3VLMoeConfig.from_dict(_moe_config().to_dict())
    config.text_config.vocab_size = len(local_tokenizer())
    model = AutoModelForImageTextToText.from_config(config, attn_implementation="sdpa").float()
    model.save_pretrained(tmp_path)
    local_processor(local_tokenizer()).save_pretrained(tmp_path)
    artifact(tmp_path, json.loads((tmp_path / "config.json").read_text()))
    runtime = CosmosFrameworkRuntime.__new__(CosmosFrameworkRuntime)
    runtime.model_path, runtime.dtype, runtime.attn_implementation = str(tmp_path), "auto", None
    runtime.model_type, runtime.device, runtime.device_map, runtime.rank = "qwen3_vl_moe", "cpu", "auto", 0
    runtime._load()
    assert runtime.dtype == "float32" and runtime.attn_implementation == "sdpa"
    assert all(p.dtype == torch.float32 for p in runtime.model.parameters())
    assert runtime.model.config.text_config._experts_implementation == "grouped_mm"
    experts = runtime.model.model.language_model.layers[0].mlp.experts
    experts.config = copy.deepcopy(experts.config)
    experts.config._experts_implementation = "eager"
    with pytest.raises(ValueError, match="differs from"):
        validate_loaded_model(runtime.model)
