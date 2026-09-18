# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Activation checkpointing of the VFM-level modules.

``parallelize_unified_mot.apply_ac`` only reaches the repeated decoder blocks,
so the modules hanging directly off ``Cosmos3VFMNetwork`` -- ``time_embedder``
above all -- used to keep every intermediate alive from their forward all the
way to their backward, across the whole transformer stack.
``parallelize_vfm_network.apply_ac`` closes that gap; these tests pin the two
properties that make the wrap safe to turn on mid-run (checkpoint keys survive
it, and the recompute is bit-exact under the float32 autocast the timestep path
runs in) alongside the memory it is there to buy.
"""

import pytest
import torch
from torch import nn

from cosmos_framework.utils.helper_test import RunIf
from cosmos_framework.configs.base.defaults.activation_checkpointing import ActivationCheckpointingConfig
from cosmos_framework.model.generator.mot.modeling_utils import TimestepEmbedder
from cosmos_framework.model.generator.mot.parallelize_vfm_network import apply_ac

pytestmark = [pytest.mark.L0]

# Shapes for the GPU memory/numerics tests. Small next to the real model
# (hidden_size=4096 over ~0.6M noised tokens), but large enough that one
# float32 [N, HIDDEN] activation is ~200 MiB and so unmistakable in the
# allocator against fixed overheads.
HIDDEN = 1024
NUM_ROWS = 50_000
TRUNK_DEPTH = 4


class _StubVFM(nn.Module):
    """The shape of the real network that matters here.

    ``time_embedder`` runs early and under a float32 autocast, its output then
    crosses a deep trunk that is itself already checkpointed, and only then is
    anything backpropagated -- which is exactly the span over which the
    timestep activations would otherwise be pinned.
    """

    def __init__(self, hidden: int = HIDDEN, depth: int = TRUNK_DEPTH, with_time_embedder: bool = True):
        super().__init__()
        if with_time_embedder:
            self.time_embedder = TimestepEmbedder(hidden, bias=True)
        self.trunk = nn.ModuleList(nn.Linear(hidden, hidden, bias=False) for _ in range(depth))

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", enabled=True, dtype=torch.float32):
            hidden_states = self.time_embedder(timesteps)  # [N,hidden], float32
        hidden_states = hidden_states.to(torch.bfloat16)
        for layer in self.trunk:
            hidden_states = torch.utils.checkpoint.checkpoint(layer, hidden_states, use_reentrant=False)
        return hidden_states


def _build(device: str = "cpu", **kwargs) -> _StubVFM:
    torch.manual_seed(0)
    model = _StubVFM(**kwargs).to(device)
    if device != "cpu":
        model = model.to(torch.bfloat16)
    return model


def _config(mode: str) -> ActivationCheckpointingConfig:
    return ActivationCheckpointingConfig(mode=mode)


@pytest.mark.CPU
@pytest.mark.parametrize("mode", ["full", "selective"])
def test_wraps_the_time_embedder(mode: str) -> None:
    model = _build()
    apply_ac(model, _config(mode))
    assert type(model.time_embedder).__name__ == "CheckpointWrapper"
    # The wrapper must delegate, or every ``self.time_embedder.<attr>`` read in
    # the network breaks.
    assert isinstance(model.time_embedder.mlp, nn.Sequential)


@pytest.mark.CPU
def test_mode_none_leaves_the_module_untouched() -> None:
    model = _build()
    original = model.time_embedder
    apply_ac(model, _config("none"))
    assert model.time_embedder is original


@pytest.mark.CPU
def test_tolerates_a_network_without_a_time_embedder() -> None:
    """``time_embedder`` only exists when ``config.vision_gen`` is set."""
    model = _build(with_time_embedder=False)
    apply_ac(model, _config("full"))
    assert not hasattr(model, "time_embedder")


@pytest.mark.CPU
@pytest.mark.parametrize("mode", ["full", "selective"])
def test_checkpoint_keys_survive_the_wrap(mode: str) -> None:
    """A wrapped run must stay loadable from, and into, an unwrapped checkpoint.

    ``ptd_checkpoint_wrapper`` inserts a ``_checkpoint_wrapped_module`` level
    into ``named_parameters()``; the state-dict hooks strip it again. If that
    ever stopped holding, turning AC on would silently invalidate every
    existing checkpoint rather than fail loudly.
    """
    plain = _build()
    wrapped = _build()
    apply_ac(wrapped, _config(mode))

    assert sorted(wrapped.state_dict()) == sorted(plain.state_dict())
    assert plain.load_state_dict(wrapped.state_dict(), strict=True)
    assert wrapped.load_state_dict(plain.state_dict(), strict=True)


@pytest.mark.CPU
def test_rejects_an_unknown_mode() -> None:
    config = _config("full")
    object.__setattr__(config, "mode", "bogus")
    with pytest.raises(ValueError, match="Invalid AC mode"):
        apply_ac(_build(), config)


@RunIf(min_gpus=1)
@pytest.mark.GPU
@pytest.mark.parametrize("mode", ["full", "selective"])
def test_recompute_is_bit_exact_under_float32_autocast(mode: str) -> None:
    """Recompute happens outside the caller's ``autocast`` block.

    ``torch.utils.checkpoint`` has to restore the autocast state itself for the
    timestep MLP to be replayed in float32 rather than the surrounding bf16. A
    regression there would not raise -- it would quietly change gradients.
    """
    timesteps = torch.rand(4096, device="cuda")

    def run(model: _StubVFM) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        out = model(timesteps)
        out.float().pow(2).mean().backward()
        return out, {name: p.grad.clone() for name, p in model.time_embedder.named_parameters()}

    baseline = _build("cuda")
    apply_ac(baseline, _config("none"))
    expected_out, expected_grads = run(baseline)

    model = _build("cuda")
    apply_ac(model, _config(mode))
    actual_out, actual_grads = run(model)

    assert torch.equal(actual_out, expected_out)
    assert len(actual_grads) == len(expected_grads)
    for name, expected in expected_grads.items():
        # Names match without adjustment: ``run`` reads ``named_parameters`` off
        # ``time_embedder`` itself, and ``ActivationWrapper`` overrides that to
        # strip ``_checkpoint_wrapped_module``. (Reading it off the *root*
        # module would keep the prefix -- that traversal does not go through the
        # wrapper's override.)
        assert torch.equal(actual_grads[name], expected), name


@RunIf(min_gpus=1)
@pytest.mark.GPU
@pytest.mark.parametrize("mode", ["full", "selective"])
def test_frees_the_timestep_activations_across_the_trunk(mode: str) -> None:
    """The point of the wrap: measure what stays resident until the backward.

    Peak memory during the embedder's own forward is unchanged -- the
    activations have to exist while they are being produced. What AC removes is
    their residency across everything that runs in between, so the measurement
    is taken with the forward complete and the autograd graph still alive.
    """
    timesteps = torch.rand(NUM_ROWS, device="cuda")
    one_activation = NUM_ROWS * HIDDEN * torch.finfo(torch.float32).bits // 8

    def retained(mode: str) -> int:
        model = _build("cuda")
        apply_ac(model, _config(mode))
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        before = torch.cuda.memory_allocated()
        out = model(timesteps)
        held = torch.cuda.memory_allocated() - before
        out.float().pow(2).mean().backward()  # keep the graph honest, then drop it
        return held

    saved = retained("none") - retained(mode)
    # The float32 [N, HIDDEN] intermediates inside the MLP (the first Linear's
    # output and the SiLU's) are what stop being pinned; assert at least both.
    assert saved >= 2 * one_activation, f"only freed {saved / 2**20:.0f} MiB"
