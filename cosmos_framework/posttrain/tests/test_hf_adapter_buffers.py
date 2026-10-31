# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""BF16 Gemma training must preserve config constants in its FP32 merged model."""

import copy

import pytest
import torch
import transformers

from cosmos_framework.posttrain.checkpoint.adapters import save_adapter_artifact
from cosmos_framework.posttrain.checkpoint.tools.adapters import export_merged_adapter, validate_export_targets
from cosmos_framework.posttrain.config.adapters import AdapterComponentConfig, AdapterConfig
from cosmos_framework.posttrain.config.model import HFModelConfig
from cosmos_framework.posttrain.model.hf import build_hf_model
from cosmos_framework.posttrain.tests.test_adapter_families import text_snapshot


@pytest.mark.parametrize("family", ["gemma", "gemma2"])
@pytest.mark.parametrize("method,rslora", [("lora", False), ("lora", True), ("dora", False), ("pissa", False)])
def test_bf16_training_and_fp32_stock_merge_keep_embedding_scale(tmp_path, family, method, rslora):
    torch.manual_seed(13)
    base = tmp_path / "base"
    text_snapshot(base, family)
    cfg = HFModelConfig(name_or_path=str(base), dtype="bfloat16", attn_impl="sdpa")
    disabled = build_hf_model(cfg).net.eval()
    ids = torch.tensor([[11, 13, 19, 21, 17]])
    before = disabled(input_ids=ids, use_cache=False).logits.detach()
    old_buffer = disabled.model.embed_tokens.embed_scale.clone()
    adapters = AdapterConfig(
        components=[
            AdapterComponentConfig(
                targets=["model.layers.0.self_attn.q_proj"], rank=2, method=method, use_rslora=rslora
            )
        ]
    )
    model = build_hf_model(cfg, adapters=adapters).net.eval()
    # The native forward casts the constant to the embedding dtype, so fixing
    # its storage cannot change the initial BF16 model or the disabled recipe.
    torch.testing.assert_close(model(input_ids=ids, use_cache=False).logits, before, atol=0, rtol=0)
    assert torch.equal(disabled.model.embed_tokens.embed_scale, old_buffer)
    assert model.model.embed_tokens.embed_scale.dtype == torch.float32
    assert "model.embed_tokens.embed_scale" not in model.state_dict()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    for _ in range(3):
        optimizer.zero_grad()
        loss = model(input_ids=ids, labels=ids, use_cache=False).loss
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
    expected_bf16 = model(input_ids=ids, use_cache=False).logits.detach()
    portable = tmp_path / "portable"
    save_adapter_artifact(model, str(portable))
    adapters.warm_start = str(portable)
    restored = build_hf_model(cfg, adapters=adapters).net.eval()
    torch.testing.assert_close(restored(input_ids=ids, use_cache=False).logits, expected_bf16, atol=0, rtol=0)
    export_merged_adapter(str(portable), str(base), str(tmp_path / "merged"))
    merged = transformers.AutoModelForCausalLM.from_pretrained(tmp_path / "merged", dtype=torch.float32).eval()
    assert merged.get_input_embeddings().weight is merged.get_output_embeddings().weight
    reference = model.float()(input_ids=ids, use_cache=False).logits
    torch.testing.assert_close(merged(input_ids=ids, use_cache=False).logits, reference, atol=3e-6, rtol=3e-5)
    # Older BF16 artifacts did not bind this regenerated buffer and must not
    # silently pass the new merged export contract.
    identity = copy.deepcopy(model._posttrain_adapter_identity)
    del identity["model_contract"]
    with pytest.raises(ValueError, match="embedding-scale contract"):
        validate_export_targets(str(base), identity)
