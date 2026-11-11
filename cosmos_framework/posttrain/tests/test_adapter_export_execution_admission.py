# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The exporter must not emit a precision that its declared loader rejects."""

import hashlib

import pytest
import torch

from cosmos_framework.inference.reasoner.model_execution import model_execution_profile
from cosmos_framework.posttrain.checkpoint.adapters import save_adapter_artifact
from cosmos_framework.posttrain.checkpoint.tools.adapters import export_merged_adapter
from cosmos_framework.posttrain.config.model import HFModelConfig
from cosmos_framework.posttrain.model.hf import build_hf_model
from cosmos_framework.posttrain.tests.test_adapters_native import adapter_config, tiny_snapshot


@pytest.mark.parametrize("model_execution", ["qwen3_vl_hf_stock_v1", "qwen_vl_hf_ordered_v1"])
@pytest.mark.parametrize("deployment", [None, "qwen3_vl_bf16_residual_fp32_v1"])
def test_fp32_execution_refuses_bf16_before_materialization(tmp_path, monkeypatch, model_execution, deployment):
    from cosmos_framework.posttrain.checkpoint.tools import materialize_hf

    base = tmp_path / "base"
    tiny_snapshot(base, tied=True)
    config = adapter_config("lora")
    for component in config.components:
        component.use_rslora = True
    config.model_execution = model_execution
    config.__post_init__()
    policy = build_hf_model(HFModelConfig(name_or_path=str(base), dtype="bfloat16", attn_impl="sdpa"), adapters=config)
    portable = tmp_path / "portable"
    save_adapter_artifact(policy.net, str(portable))
    source = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in portable.iterdir()}
    parameters = {name: (parameter, parameter.detach().clone()) for name, parameter in policy.net.named_parameters()}

    def unexpected_materialization(*args, **kwargs):
        pytest.fail("refused export reached the materializer transaction")

    destination = tmp_path / "refused-bf16"
    with monkeypatch.context() as patch:
        patch.setattr(materialize_hf, "run_request", unexpected_materialization)
        with pytest.raises(ValueError, match="defines FP32 merged loading"):
            export_merged_adapter(
                str(portable), str(base), str(destination), output_dtype="bfloat16", execution_profile=deployment
            )
    assert not destination.exists()
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in portable.iterdir()} == source
    assert all(
        p is parameters[name][0] and torch.equal(p, parameters[name][1]) for name, p in policy.net.named_parameters()
    )

    # The same artifact remains exportable under its existing FP32 contract.
    merged = tmp_path / "merged-fp32"
    export_merged_adapter(str(portable), str(base), str(merged))
    assert model_execution_profile(merged) == model_execution
