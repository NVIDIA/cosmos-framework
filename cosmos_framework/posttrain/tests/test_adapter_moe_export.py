# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Actual HF MoE export oracle with native checkpoint expert layout.

CPU tests exercise conversion and stock FP32 execution, not the native CUDA
grouped expert training kernel. The latter remains a separate Slurm gate.
"""

import json

import pytest
import torch
import transformers
from safetensors.torch import load_file, save_file

from cosmos_framework.posttrain.checkpoint.adapters import read_adapter_artifact, save_adapter_artifact
from cosmos_framework.posttrain.checkpoint.tools.adapters import export_merged_adapter
from cosmos_framework.posttrain.config.model import Qwen3VLModelConfig
from cosmos_framework.posttrain.model.adapters import assert_trainability
from cosmos_framework.posttrain.model.qwen3_vl import build_qwen3_vl_model
from cosmos_framework.posttrain.tests.adapter_numerics import HFProbePolicy
from cosmos_framework.posttrain.tests.helpers.sft_packer import local_processor, local_tokenizer
from cosmos_framework.posttrain.tests.qualify_adapters import logits
from cosmos_framework.posttrain.tests.reload_moe_adapters import hf_portable, verify_serialized
from cosmos_framework.posttrain.tests.test_adapters_native import adapter_config, batches
from cosmos_framework.posttrain.tests.test_model_qwen3_vl_gpu import _moe_config


def native_moe_snapshot(path):
    config = transformers.Qwen3VLMoeConfig.from_dict(_moe_config().to_dict())
    # A square gate_up tensor cannot expose an accidental missing transpose.
    config.text_config.moe_intermediate_size = 24
    config.text_config.vocab_size = len(local_tokenizer())
    config.architectures = ["Qwen3VLMoeForConditionalGeneration"]
    net = transformers.AutoModelForImageTextToText.from_config(config, attn_implementation="sdpa")
    net.bfloat16()
    path.mkdir()
    net.config.save_pretrained(path)
    local_processor(local_tokenizer()).save_pretrained(path)
    state = net.state_dict()
    for name, tensor in state.items():
        if ".mlp.experts." in name:
            state[name] = tensor.transpose(1, 2).contiguous()
    shard = "model-00001-of-00001.safetensors"
    save_file(state, str(path / shard))
    (path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": sum(t.numel() * t.element_size() for t in state.values())},
                "weight_map": {n: shard for n in state},
            }
        )
    )


@pytest.mark.parametrize("method,rslora", [("lora", False), ("lora", True), ("dora", False), ("pissa", False)])
def test_moe_native_manifest_stock_fp32_reload_and_every_export_tensor(tmp_path, method, rslora):
    torch.manual_seed(23)
    base = tmp_path / "base"
    native_moe_snapshot(base)
    cfg = adapter_config(method)
    for group in cfg.components:
        group.use_rslora = rslora
    native = build_qwen3_vl_model(
        Qwen3VLModelConfig(name_or_path=str(base), dtype="bfloat16", activation_checkpointing="none"),
        adapters=cfg,
    )
    initial = tmp_path / "initial"
    save_adapter_artifact(native.net, str(initial))
    identity = native.net._posttrain_adapter_identity
    data = batches()
    native.net.float().eval()
    native_fp32 = logits(native, data)
    del native
    # This asserts exact native/HF placement, component and artifact agreement.
    model = hf_portable(str(base), str(initial), identity).float()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    assert_trainability(model, opt)
    for batch in data:
        tokens = torch.zeros_like(batch["input_ids"])
        tokens[batch["input_ids"] == model.config.image_token_id] = 1
        tokens[batch["input_ids"] == model.config.video_token_id] = 2
        batch["mm_token_type_ids"] = tokens
    # MoEExperts already dispatches FP32 to native.naive. Its independent
    # native model forward must agree with stock HF, not only HF with itself.
    for actual, expected in zip(logits(HFProbePolicy(model), data), native_fp32, strict=True):
        torch.testing.assert_close(actual, expected, atol=5e-6, rtol=5e-5)
    model.train()
    for batch in data[1:] * 2:
        opt.zero_grad(set_to_none=True)
        loss = model(**batch).logits.float().square().mean()
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
        opt.step()
    for name, parameter in model.named_parameters():
        assert torch.equal(before[name], parameter) is not parameter.requires_grad, name
    model.eval()
    with torch.no_grad():
        expected = [model(**b).logits.clone() for b in data]
    portable = tmp_path / "portable"
    save_adapter_artifact(model, str(portable))
    _, adapter_state = read_adapter_artifact(str(portable))
    merged = tmp_path / "merged"
    export_merged_adapter(str(portable), str(base), str(merged))
    assert verify_serialized(base, merged, identity, adapter_state) == len(
        load_file(str(base / "model-00001-of-00001.safetensors"))
    )
    loaded = transformers.AutoModelForImageTextToText.from_pretrained(merged, dtype=torch.float32).eval()
    with torch.no_grad():
        for batch, reference in zip(data, expected, strict=True):
            torch.testing.assert_close(loaded(**batch).logits, reference, atol=5e-6, rtol=5e-5)
    # Tampering with a frozen expert must fail the exhaustive tensor check.
    index = json.loads((merged / "model.safetensors.index.json").read_text())["weight_map"]
    name = next(n for n in index if ".mlp.experts." in n)
    file = merged / index[name]
    state = load_file(str(file))
    state[name].flatten()[0] += 1
    save_file(state, str(file))
    with pytest.raises(ValueError, match="serialized merged tensor differs"):
        verify_serialized(base, merged, identity, adapter_state)
