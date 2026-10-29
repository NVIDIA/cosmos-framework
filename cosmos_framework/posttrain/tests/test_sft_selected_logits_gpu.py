# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Real native VLM/BF16/NF4 projection parity; no model/data downloads."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from cosmos_framework.posttrain.model.qwen3_vl import (
    Qwen3VLBuildHooks,
    build_qwen3_vl_policy,
    override_qwen3_vl_forward,
)
from cosmos_framework.posttrain.tests.test_model_qwen3_vl_gpu import _batch, _dense_config
from cosmos_framework.posttrain.tests.test_model_qwen3_vl_qlora_gpu import restore_determinism_state  # noqa: F401

pytestmark = [pytest.mark.GPU, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA kernels")]


@pytest.mark.parametrize("quantized", [False, True], ids=["bf16", "nf4"])
def test_native_selected_head_preserves_quantized_backbone_and_adapter_gradients(quantized):
    from cosmos_framework.posttrain.config.model import NativeLoRAConfig, QuantizedBaseTrainingConfig
    from cosmos_framework.posttrain.model.quantization.loader import quantized_state_record
    from cosmos_framework.posttrain.utils.determinism import enable_determinism
    from cosmos_framework.utils.generator.lora import LoraInjectedLinear

    enable_determinism(strict=True)
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(37)
    policy = build_qwen3_vl_policy(
        model_type="qwen3_vl",
        model_config=_dense_config(),
        dtype=torch.bfloat16,
        initialize_on_meta=False,
        attn_implementation="sdpa",
        hooks=Qwen3VLBuildHooks(prepare_model=lambda net: override_qwen3_vl_forward(net.model)),
        sft_logits_mode="supervised",
    )
    policy.net.requires_grad_(False)
    attention = policy.net.model.language_model.layers[0].self_attn
    source = attention.q_proj
    if quantized:
        from cosmos_framework.posttrain.model.quantization.bnb import NativeLoraLinear4bit

        adapter = NativeLoraLinear4bit(
            nn.Linear(source.in_features, source.out_features, bias=False, device="meta"),
            NativeLoRAConfig(rank=2, alpha=2),
            QuantizedBaseTrainingConfig(),
        )
        adapter.lora_A.weight = nn.Parameter(torch.empty(2, source.in_features, device=device))
        adapter.lora_B.weight = nn.Parameter(torch.empty(source.out_features, 2, device=device))
        adapter.load_dense_weight_once(source.weight.detach().cpu(), device)
        frozen_state = quantized_state_record(adapter)
    else:
        adapter = LoraInjectedLinear(source, rank=2, alpha=2, adapter_dtype=torch.float32)
        frozen_state = source.weight.detach().clone()
    adapter.reset_adapter_parameters(generator=torch.Generator(device=device).manual_seed(38))
    # Model a restored, already-updated adapter so both A and B must receive gradients.
    with torch.no_grad():
        adapter.lora_B.weight.fill_(0.01)
    attention.q_proj = adapter
    policy.net.model.language_model.layers[0] = checkpoint_wrapper(policy.net.model.language_model.layers[0])
    batch = policy.prepare_policy_batch(_batch())
    valid = torch.tensor([[False, False, True, True]], device=device)
    labels = torch.tensor([3, 4], device=device)
    shapes = []
    hook = policy.net.lm_head.register_forward_pre_hook(lambda _module, args: shapes.append(tuple(args[0].shape)))
    try:
        full = policy.forward_policy_batch(batch).logits[:, :-1][valid].float()
        full_loss = torch.nn.functional.cross_entropy(full, labels)
        full_loss.backward()
        reference_gradients = {name: p.grad.clone() for name, p in adapter.adapter_parameters().items()}
        policy.net.zero_grad(set_to_none=True)
        selected = policy.forward_sft_batch(batch, valid).logits.float()
        selected_loss = torch.nn.functional.cross_entropy(selected, labels)
        selected_loss.backward()
    finally:
        hook.remove()
    assert shapes == [(1, 5, 32), (2, 32)]
    # BF16 GEMMs can choose a different tile for the narrower output projection.
    # These gates compare the same base and hidden states, not independently trained runs.
    torch.testing.assert_close(selected, full, rtol=0.02, atol=0.002)
    torch.testing.assert_close(selected_loss, full_loss, rtol=0.002, atol=0.002)
    for name, parameter in adapter.adapter_parameters().items():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.count_nonzero() > 0
        torch.testing.assert_close(parameter.grad, reference_gradients[name], rtol=0.03, atol=1e-5)
    assert policy.net.lm_head.weight.grad is None
    assert policy.net.model.language_model.embed_tokens.weight.grad is None
    if quantized:
        assert quantized_state_record(adapter) == frozen_state
    else:
        torch.testing.assert_close(adapter.weight, frozen_state, rtol=0, atol=0)
