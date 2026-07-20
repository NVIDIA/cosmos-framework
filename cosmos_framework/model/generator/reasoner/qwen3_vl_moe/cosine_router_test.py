# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Unit tests for gen-tower routing on ``Qwen3VLMoeTextSparseMoeBlock``.

CPU tests cover routing math, configuration, and EMA input-centering state.
GPU tests cover full-block forward and backward through the grouped-mm experts.
"""

import math

import pytest
import torch

from cosmos_framework.model.generator.reasoner.qwen3_vl_moe.configuration_qwen3_vl_moe import (
    Qwen3VLMoeTextConfig,
)
from cosmos_framework.model.generator.reasoner.qwen3_vl_moe.qwen3_vl_moe import (
    CosineRouter,
    CosineRouterConfig,
    Qwen3VLMoeTextSparseMoeBlock,
)

# Fast, CPU-only routing-math tests: run on every commit.
pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _make_config() -> Qwen3VLMoeTextConfig:
    return Qwen3VLMoeTextConfig(
        hidden_size=64,
        moe_intermediate_size=32,
        num_experts=16,
        num_experts_per_tok=4,
        hidden_act="silu",
    )


def _router_logits(block: Qwen3VLMoeTextSparseMoeBlock, hidden_states: torch.Tensor) -> torch.Tensor:
    return block.cosine_router(hidden_states, block.gate)  # [N,E]


def test_cosine_router_creates_temperature_param() -> None:
    config = _make_config()
    cosine = Qwen3VLMoeTextSparseMoeBlock(config, cosine_router_config=CosineRouterConfig(use_cosine_similarity=True))
    standard = Qwen3VLMoeTextSparseMoeBlock(
        config, cosine_router_config=CosineRouterConfig(use_cosine_similarity=False)
    )

    assert cosine.cosine_router.use_cosine_similarity is True
    assert cosine.cosine_router.input_centering == "none"
    assert hasattr(cosine.cosine_router, "log_temperature")
    assert not hasattr(cosine.cosine_router, "router_bias")
    # log_temperature is a learnable per-channel vector [hidden_size] (a vector,
    # not a numel-1 scalar, so FSDP shards it cleanly) initialized uniformly to
    # log(init temperature).
    assert cosine.cosine_router.log_temperature.requires_grad
    assert cosine.cosine_router.log_temperature.shape == torch.Size([config.hidden_size])
    # log_temperature is initialized uniformly to the scaled init value.
    expected_log_temp = CosineRouter.log_temperature_init(config.hidden_size)
    torch.testing.assert_close(
        cosine.cosine_router.log_temperature.detach(),
        torch.full((config.hidden_size,), expected_log_temp),
    )
    # Init T = sqrt(hidden_size) (matches the pretrained warm-start router's logit scale).
    torch.testing.assert_close(
        cosine.cosine_router.log_temperature.exp().max().item(),
        math.sqrt(config.hidden_size),
        atol=1e-3,
        rtol=1e-4,
    )

    # The und-tower (standard) block must not grow the gen-only param.
    assert standard.cosine_router.use_cosine_similarity is False
    assert standard.cosine_router.input_centering == "none"
    assert not hasattr(standard.cosine_router, "log_temperature")


def test_cosine_router_ema_bias_buffer() -> None:
    config = _make_config()
    block = Qwen3VLMoeTextSparseMoeBlock(
        config,
        cosine_router_config=CosineRouterConfig(use_cosine_similarity=True, input_centering="ema"),
    )

    assert block.cosine_router.use_cosine_similarity is True
    assert block.cosine_router.input_centering == "ema"
    # router_bias is a gradient-free per-channel BUFFER (not a Parameter), zero-init.
    assert not isinstance(block.cosine_router.router_bias, torch.nn.Parameter)
    assert block.cosine_router.router_bias.requires_grad is False
    assert block.cosine_router.router_bias.shape == torch.Size([config.hidden_size])
    torch.testing.assert_close(block.cosine_router.router_bias.detach(), torch.zeros(config.hidden_size))
    # router_bias is persistent (checkpointed); the accumulators are transient (not saved).
    persisted = set(block.state_dict().keys())
    assert "cosine_router.log_temperature" in persisted
    assert "cosine_router.router_bias" in persisted
    assert not any(k.endswith("router_bias_sum") for k in persisted)
    assert not any(k.endswith("router_bias_count") for k in persisted)

    restored = Qwen3VLMoeTextSparseMoeBlock(
        config,
        cosine_router_config=CosineRouterConfig(use_cosine_similarity=True, input_centering="ema"),
    )
    restored.load_state_dict(block.state_dict(), strict=True)
    torch.testing.assert_close(restored.cosine_router.log_temperature, block.cosine_router.log_temperature)
    torch.testing.assert_close(restored.cosine_router.router_bias, block.cosine_router.router_bias)


def test_cosine_router_without_input_centering_is_batch_independent() -> None:
    torch.manual_seed(0)
    config = _make_config()
    block = Qwen3VLMoeTextSparseMoeBlock(
        config,
        cosine_router_config=CosineRouterConfig(use_cosine_similarity=True, input_centering="none"),
    )
    block.eval()
    hidden_states = torch.randn(48, config.hidden_size)

    batch_logits = _router_logits(block, hidden_states)
    single_logits = _router_logits(block, hidden_states[:1])

    assert block.cosine_router.input_centering == "none"
    assert not hasattr(block.cosine_router, "router_bias")
    torch.testing.assert_close(single_logits, batch_logits[:1], atol=1e-5, rtol=1e-5)


def test_ema_bias_subtracts_and_is_batch_independent() -> None:
    """At inference the EMA buffer is frozen and subtracted from every token, and — unlike
    mean-centering — a single token's logits do not depend on the rest of the batch
    (batch-independent, identical at train and inference)."""
    torch.manual_seed(0)
    config = _make_config()
    block = Qwen3VLMoeTextSparseMoeBlock(
        config,
        cosine_router_config=CosineRouterConfig(use_cosine_similarity=True, input_centering="ema"),
    )
    block.eval()  # inference: buffer frozen, no accumulation
    # Give the buffer a non-trivial value.
    with torch.no_grad():
        block.cosine_router.router_bias.copy_(torch.randn(config.hidden_size) * 3.0)

    x = torch.randn(48, config.hidden_size)

    # 1) Subtracting router_bias then routing with a zero-buffer block == routing (x - b).
    logits = _router_logits(block, x)
    zero_bias = Qwen3VLMoeTextSparseMoeBlock(
        config,
        cosine_router_config=CosineRouterConfig(use_cosine_similarity=True, input_centering="ema"),
    )
    zero_bias.eval()
    zero_bias.gate.load_state_dict(block.gate.state_dict())
    zero_bias.cosine_router.log_temperature.data.copy_(block.cosine_router.log_temperature.data)
    # zero_bias.router_bias stays at its zero init.
    torch.testing.assert_close(logits, _router_logits(zero_bias, x - block.cosine_router.router_bias))

    # 2) Batch-independence: a token's logits are unchanged whether routed alone or
    #    as part of a larger batch (mean-centering would NOT satisfy this).
    single = _router_logits(block, x[:1])
    torch.testing.assert_close(single, logits[:1], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("use_cosine_similarity", [False, True])
def test_ema_bias_is_gradient_free_and_accumulates_in_train(use_cosine_similarity: bool) -> None:
    """In train mode forward() accumulates the local token sum + count (for the deferred
    EMA step) and the buffer receives no gradient."""
    torch.manual_seed(0)
    config = _make_config()
    block = Qwen3VLMoeTextSparseMoeBlock(
        config,
        cosine_router_config=CosineRouterConfig(
            use_cosine_similarity=use_cosine_similarity,
            input_centering="ema",
        ),
    )
    block.train()
    x = torch.randn(48, config.hidden_size)

    _router_logits(block, x).square().mean().backward()
    # It's a buffer, so no gradient is produced for it.
    assert getattr(block.cosine_router.router_bias, "grad", None) is None
    # The local per-step token stats are accumulated.
    torch.testing.assert_close(block.cosine_router.router_bias_sum, x.sum(dim=0), atol=1e-4, rtol=1e-4)
    assert block.cosine_router.router_bias_count.item() == 48


@pytest.mark.parametrize("use_cosine_similarity", [False, True])
def test_update_router_bias_ema_step_tracks_batch_mean(use_cosine_similarity: bool) -> None:
    """update_router_bias() applies the EMA of the accumulated global batch mean and resets
    the accumulators; a second step continues the EMA from the first value."""
    torch.manual_seed(0)
    config = _make_config()
    block = Qwen3VLMoeTextSparseMoeBlock(
        config,
        cosine_router_config=CosineRouterConfig(
            use_cosine_similarity=use_cosine_similarity,
            input_centering="ema",
            ema_momentum=0.9,
        ),
    )
    block.train()

    x1 = torch.randn(48, config.hidden_size)
    _router_logits(block, x1)
    block.update_router_bias(device_mesh=None)  # single-process: no cross-rank reduce
    # router_bias = m·0 + (1-m)·mean(x1) with m=0.9.
    torch.testing.assert_close(block.cosine_router.router_bias, 0.1 * x1.mean(dim=0), atol=1e-5, rtol=1e-5)
    # Accumulators reset after the update.
    assert block.cosine_router.router_bias_count.item() == 0
    torch.testing.assert_close(block.cosine_router.router_bias_sum, torch.zeros(config.hidden_size))

    prev = block.cosine_router.router_bias.clone()
    x2 = torch.randn(48, config.hidden_size)
    _router_logits(block, x2)
    block.update_router_bias(device_mesh=None)
    expected = 0.9 * prev + 0.1 * x2.mean(dim=0)
    torch.testing.assert_close(block.cosine_router.router_bias, expected, atol=1e-5, rtol=1e-5)


def test_temperature_clamp_default_on_and_caps_at_init() -> None:
    torch.manual_seed(0)
    config = _make_config()
    block = Qwen3VLMoeTextSparseMoeBlock(config, cosine_router_config=CosineRouterConfig(use_cosine_similarity=True))
    # Safety clamp is a default-on cosine-router property.
    assert block.cosine_router.clamp_temperature is True
    init_log = CosineRouter.log_temperature_init(config.hidden_size)
    assert block.cosine_router.initial_log_temperature == pytest.approx(init_log)

    x = torch.randn(48, config.hidden_size)
    # Drive log_temperature far ABOVE its init on every channel → the clamp must cap the
    # effective temperature at T_init, so logits equal those of a block sitting at init.
    with torch.no_grad():
        block.cosine_router.log_temperature.fill_(init_log + 10.0)
    at_init = Qwen3VLMoeTextSparseMoeBlock(config, cosine_router_config=CosineRouterConfig(use_cosine_similarity=True))
    at_init.gate.load_state_dict(block.gate.state_dict())  # log_temperature stays at init
    torch.testing.assert_close(_router_logits(block, x), _router_logits(at_init, x), atol=1e-4, rtol=1e-4)


def test_temperature_clamp_noop_below_init_and_off_lets_it_grow() -> None:
    torch.manual_seed(0)
    config = _make_config()
    init_log = CosineRouter.log_temperature_init(config.hidden_size)
    x = torch.randn(48, config.hidden_size)

    # Below init: clamp is a no-op, so clamp-on and clamp-off agree.
    on = Qwen3VLMoeTextSparseMoeBlock(
        config,
        cosine_router_config=CosineRouterConfig(use_cosine_similarity=True, clamp_temperature=True),
    )
    off = Qwen3VLMoeTextSparseMoeBlock(
        config,
        cosine_router_config=CosineRouterConfig(use_cosine_similarity=True, clamp_temperature=False),
    )
    off.gate.load_state_dict(on.gate.state_dict())
    with torch.no_grad():
        on.cosine_router.log_temperature.fill_(init_log - 2.0)
        off.cosine_router.log_temperature.fill_(init_log - 2.0)
    torch.testing.assert_close(_router_logits(on, x), _router_logits(off, x))

    # Above init: with the clamp OFF the larger temperature is used (logits scale up), so the
    # clamped and unclamped logits must DIFFER — proving the clamp is what caps them.
    with torch.no_grad():
        on.cosine_router.log_temperature.fill_(init_log + 10.0)
        off.cosine_router.log_temperature.fill_(init_log + 10.0)
    assert _router_logits(off, x).abs().max() > _router_logits(on, x).abs().max() + 1.0
    assert off.cosine_router.clamp_temperature is False


def test_standard_router_matches_plain_gate() -> None:
    torch.manual_seed(0)
    config = _make_config()
    block = Qwen3VLMoeTextSparseMoeBlock(config, cosine_router_config=CosineRouterConfig(use_cosine_similarity=False))
    x = torch.randn(32, config.hidden_size)

    torch.testing.assert_close(_router_logits(block, x), block.gate(x))


def test_dot_product_router_applies_batch_mean_centering() -> None:
    torch.manual_seed(0)
    config = _make_config()
    block = Qwen3VLMoeTextSparseMoeBlock(
        config,
        cosine_router_config=CosineRouterConfig(
            use_cosine_similarity=False,
            input_centering="batch_mean",
        ),
    )
    hidden_states = torch.randn(32, config.hidden_size)  # [N,D]
    centered_hidden_states = hidden_states - hidden_states.mean(dim=0, keepdim=True)  # [N,D]

    router_logits = _router_logits(block, hidden_states)  # [N,E]

    torch.testing.assert_close(router_logits, block.gate(centered_hidden_states))


def test_dot_product_router_supports_ema_centering() -> None:
    torch.manual_seed(0)
    config = _make_config()
    block = Qwen3VLMoeTextSparseMoeBlock(
        config,
        cosine_router_config=CosineRouterConfig(
            use_cosine_similarity=False,
            input_centering="ema",
        ),
    )
    block.eval()
    hidden_states = torch.randn(32, config.hidden_size)  # [N,D]
    with torch.no_grad():
        block.cosine_router.router_bias.copy_(torch.randn(config.hidden_size))  # [D]
    centered_hidden_states = hidden_states - block.cosine_router.router_bias  # [N,D]

    router_logits = _router_logits(block, hidden_states)  # [N,E]

    assert not hasattr(block.cosine_router, "log_temperature")
    assert "cosine_router.router_bias" in block.state_dict()
    torch.testing.assert_close(router_logits, block.gate(centered_hidden_states))


def test_cosine_logits_match_temperature_scaled_cosines() -> None:
    torch.manual_seed(0)
    config = _make_config()
    block = Qwen3VLMoeTextSparseMoeBlock(config, cosine_router_config=CosineRouterConfig(use_cosine_similarity=True))
    x = torch.randn(48, config.hidden_size)

    logits = _router_logits(block, x)  # [N,E]
    normalized_x = torch.nn.functional.normalize(x.to(torch.float32), dim=-1)  # [N,D]
    temperature = block.cosine_router.log_temperature.exp().to(torch.float32)  # [D]
    normalized_gate_weight = torch.nn.functional.normalize(
        block.gate.weight.to(torch.float32),
        dim=-1,
    )  # [E,D]
    expected_logits = torch.nn.functional.linear(
        normalized_x * temperature,
        normalized_gate_weight,
    ).to(x.dtype)  # [N,E]

    assert logits.shape == (48, config.num_experts)
    torch.testing.assert_close(logits, expected_logits)


def test_cosine_router_invariant_to_token_constant_offset() -> None:
    """Mean-centering removes any per-channel token-constant ("sink") component,
    so adding the same vector to every token must not change the router logits."""
    torch.manual_seed(0)
    config = _make_config()
    block = Qwen3VLMoeTextSparseMoeBlock(
        config,
        cosine_router_config=CosineRouterConfig(
            use_cosine_similarity=True,
            input_centering="batch_mean",
        ),
    )
    x = torch.randn(48, config.hidden_size)

    sink = torch.randn(1, config.hidden_size) * 25.0  # large token-constant offset
    logits_base = _router_logits(block, x)
    logits_shifted = _router_logits(block, x + sink)

    torch.testing.assert_close(logits_base, logits_shifted, atol=1e-4, rtol=1e-4)


def test_cosine_router_invariant_to_positive_input_scale() -> None:
    """L2-normalization makes logits depend on input direction only, so a global
    positive rescale of the input must not change them."""
    torch.manual_seed(0)
    config = _make_config()
    block = Qwen3VLMoeTextSparseMoeBlock(config, cosine_router_config=CosineRouterConfig(use_cosine_similarity=True))
    x = torch.randn(48, config.hidden_size)

    logits_base = _router_logits(block, x)
    logits_scaled = _router_logits(block, x * 7.5)

    torch.testing.assert_close(logits_base, logits_scaled, atol=1e-4, rtol=1e-4)


def test_init_weights_resets_temperature() -> None:
    config = _make_config()
    block = Qwen3VLMoeTextSparseMoeBlock(config, cosine_router_config=CosineRouterConfig(use_cosine_similarity=True))
    with torch.no_grad():
        block.cosine_router.log_temperature.fill_(123.0)

    block.init_weights()
    expected_log_temp = CosineRouter.log_temperature_init(config.hidden_size)
    torch.testing.assert_close(
        block.cosine_router.log_temperature.detach(),
        torch.full((config.hidden_size,), expected_log_temp),
    )


@pytest.mark.parametrize("use_cosine_similarity", [False, True])
def test_init_weights_sets_up_ema_bias_buffers(use_cosine_similarity: bool) -> None:
    config = _make_config()
    block = Qwen3VLMoeTextSparseMoeBlock(
        config,
        cosine_router_config=CosineRouterConfig(
            use_cosine_similarity=use_cosine_similarity,
            input_centering="ema",
        ),
    )
    # Dirty the buffers, then re-materialize via init_weights.
    with torch.no_grad():
        block.cosine_router.router_bias.fill_(5.0)
        block.cosine_router.router_bias_sum.fill_(3.0)
        block.cosine_router.router_bias_count.fill_(7.0)

    block.init_weights()
    # Buffers are zero-reinitialized and still gradient-free / correctly shaped.
    torch.testing.assert_close(block.cosine_router.router_bias, torch.zeros(config.hidden_size))
    torch.testing.assert_close(block.cosine_router.router_bias_sum, torch.zeros(config.hidden_size))
    assert block.cosine_router.router_bias_count.item() == 0.0
    assert not isinstance(block.cosine_router.router_bias, torch.nn.Parameter)


@pytest.mark.GPU
@pytest.mark.skipif(not torch.cuda.is_available(), reason="grouped_mm experts require CUDA")
def test_cosine_router_forward_backward_train_mode() -> None:
    """Full-block forward+backward on GPU with cosine_router AND noisy_gating both
    on (train mode). Covers the grouped_mm expert path and, critically, that the
    learnable temperature receives a finite, non-zero gradient end-to-end — which
    the CPU tests and the eval-mode inference sanity run cannot exercise."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    config = Qwen3VLMoeTextConfig(
        hidden_size=512,
        moe_intermediate_size=256,
        num_experts=16,
        num_experts_per_tok=4,
        hidden_act="silu",
    )
    block = Qwen3VLMoeTextSparseMoeBlock(
        config,
        noisy_gating=True,
        cosine_router_config=CosineRouterConfig(
            use_cosine_similarity=True,
            input_centering="batch_mean",
        ),
    )
    block.init_weights()
    block = block.to(device=device, dtype=torch.bfloat16)
    block.train()

    hidden_states = torch.randn(512, config.hidden_size, device=device, dtype=torch.bfloat16)
    # Add a large token-constant offset; centering must keep routing well-behaved.
    hidden_states = hidden_states + torch.randn(1, config.hidden_size, device=device, dtype=torch.bfloat16) * 20.0

    routed_out, metadata = block(hidden_states)

    assert routed_out.shape == hidden_states.shape
    assert torch.isfinite(routed_out.float()).all()
    assert metadata.num_tokens_per_expert.sum().item() == 512 * config.num_experts_per_tok

    routed_out.float().square().mean().backward()
    assert block.cosine_router.log_temperature.grad is not None
    assert block.cosine_router.log_temperature.grad.shape == (config.hidden_size,)
    assert torch.isfinite(block.cosine_router.log_temperature.grad).all()
    assert block.cosine_router.log_temperature.grad.abs().max().item() > 0.0


@pytest.mark.GPU
@pytest.mark.skipif(not torch.cuda.is_available(), reason="grouped_mm experts require CUDA")
def test_cosine_router_ema_bias_forward_backward_and_update() -> None:
    """Full-block forward+backward on GPU with cosine_router AND the EMA de-sink bias in
    train mode: covers the grouped_mm expert path, that forward accumulates the token stats,
    that the buffer is gradient-free while log_temperature still gets a finite gradient, and
    that update_router_bias moves the buffer toward the batch mean."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    config = Qwen3VLMoeTextConfig(
        hidden_size=512,
        moe_intermediate_size=256,
        num_experts=16,
        num_experts_per_tok=4,
        hidden_act="silu",
    )
    block = Qwen3VLMoeTextSparseMoeBlock(
        config,
        noisy_gating=True,
        cosine_router_config=CosineRouterConfig(
            use_cosine_similarity=True,
            input_centering="ema",
            ema_momentum=0.9,
        ),
    )
    block.init_weights()
    block = block.to(device=device, dtype=torch.bfloat16)
    block.train()
    assert block.cosine_router.input_centering == "ema"

    hidden_states = torch.randn(512, config.hidden_size, device=device, dtype=torch.bfloat16)
    # Large token-constant offset the EMA de-sink is meant to track/remove.
    sink = torch.randn(1, config.hidden_size, device=device, dtype=torch.bfloat16) * 20.0
    hidden_states = hidden_states + sink

    routed_out, metadata = block(hidden_states)
    assert routed_out.shape == hidden_states.shape
    assert torch.isfinite(routed_out.float()).all()
    assert metadata.num_tokens_per_expert.sum().item() == 512 * config.num_experts_per_tok
    # forward accumulated the local token stats for the deferred EMA step.
    assert block.cosine_router.router_bias_count.item() == 512

    routed_out.float().square().mean().backward()
    # Buffer is gradient-free; the temperature parameter still learns.
    assert getattr(block.cosine_router.router_bias, "grad", None) is None
    assert block.cosine_router.log_temperature.grad is not None
    assert torch.isfinite(block.cosine_router.log_temperature.grad).all()

    # The EMA update moves the buffer toward the accumulated batch mean (which carries the sink).
    # (The block was cast to bf16 here, so compare in fp32 — real fp32-master training keeps the
    # buffer in fp32 and accumulates exactly.)
    expected_mean = hidden_states.to(torch.float32).sum(dim=0) / 512
    block.update_router_bias(device_mesh=None)
    torch.testing.assert_close(block.cosine_router.router_bias.float(), 0.1 * expected_mean, atol=1e-1, rtol=2e-2)
    assert block.cosine_router.router_bias_count.item() == 0
    # The buffer now points along the sink direction it will subtract at inference.
    cos = torch.nn.functional.cosine_similarity(
        block.cosine_router.router_bias.float(),
        sink.squeeze(0).float(),
        dim=0,
    )
    assert cos.item() > 0.5
