# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""LoRA merging in HFExportCallback._gather_weights.

The property under test throughout: an export taken from a LoRA run must be
indistinguishable from an export of a full fine-tune that reached the same
effective weights. Concretely — no ``lora_*`` keys, and every adapted
``<path>.weight`` already carrying ``W + (alpha / r) * B @ A``.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from cosmos_framework.callbacks.hf_export import HFExportCallback
from cosmos_framework.utils.generator.lora import LoraInjectedLinear

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _lora_linear(in_features: int, out_features: int, rank: int, alpha: int, *, bias: bool = False):
    """A materialized LoraInjectedLinear with a non-zero adapter.

    ``LoraInjectedLinear.__init__`` allocates lora_A / lora_B on the meta device
    (production materializes them after the FSDP wrap, in
    ``init_lora_weights_post_materialization``); swap in real CPU Linears here.
    lora_B is deliberately non-zero — at its trained-from-zero init the merge
    would be a no-op and could not distinguish a working merge from none at all.
    """
    base = nn.Linear(in_features, out_features, bias=bias)
    module = LoraInjectedLinear(base, rank, alpha)
    module.lora_A = nn.Linear(in_features, rank, bias=False)
    module.lora_B = nn.Linear(rank, out_features, bias=False)
    nn.init.normal_(module.lora_A.weight, std=0.02)
    nn.init.normal_(module.lora_B.weight, std=0.02)
    return module


class _CheckpointWrapper(nn.Module):
    """Mimics the attribute name gradient checkpointing injects into paths."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self._checkpoint_wrapped_module = inner


def _fake_vlm(net: nn.Module):
    """_gather_weights reaches the HF transformer at ``model.model.model``."""
    return SimpleNamespace(model=SimpleNamespace(model=net))


def _gather(net: nn.Module, dtype: str = "float32") -> dict[str, torch.Tensor]:
    callback = HFExportCallback(dtype=dtype)
    cpu_chunks, manifest, total_size = callback._gather_weights(_fake_vlm(net))
    flat = {name: tensor for chunk in cpu_chunks for name, tensor in chunk.items()}
    assert set(flat) == set(manifest), "manifest and shard contents disagree"
    assert total_size == sum(t.element_size() * t.numel() for t in flat.values())
    return flat


# ---------------------------------------------------------------- merged_weight


def test_merged_weight_matches_explicit_formula():
    module = _lora_linear(8, 6, rank=4, alpha=16)
    merged = module.merged_weight(module.weight, module.lora_A.weight, module.lora_B.weight)

    expected = module.weight + (16 / 4) * (module.lora_B.weight @ module.lora_A.weight)
    torch.testing.assert_close(merged, expected)


def test_merged_weight_reproduces_the_lora_forward():
    """The merge is only correct if it is invisible to the forward pass."""
    module = _lora_linear(8, 6, rank=4, alpha=8, bias=True)
    x = torch.randn(3, 8)

    merged = module.merged_weight(module.weight, module.lora_A.weight, module.lora_B.weight)
    torch.testing.assert_close(F.linear(x, merged, module.bias), module(x))


def test_merged_weight_is_identity_at_zero_init():
    """lora_B starts at zero, so an export before any step must be the base model."""
    module = _lora_linear(8, 6, rank=4, alpha=16)
    nn.init.zeros_(module.lora_B.weight)

    merged = module.merged_weight(module.weight, module.lora_A.weight, module.lora_B.weight)
    torch.testing.assert_close(merged, module.weight)


def test_merged_weight_preserves_base_dtype():
    module = _lora_linear(8, 6, rank=4, alpha=16)
    base = module.weight.to(torch.bfloat16)
    merged = module.merged_weight(
        base, module.lora_A.weight.to(torch.bfloat16), module.lora_B.weight.to(torch.bfloat16)
    )
    assert merged.dtype == torch.bfloat16


def test_merged_weight_accumulates_in_float32():
    """A bfloat16 B @ A loses enough of the delta to be measurably worse.

    rank=64 gives the matmul enough terms for bfloat16 accumulation to drift;
    the merge must land nearer the float32 result than the bfloat16 one.
    """
    torch.manual_seed(0)
    module = _lora_linear(64, 64, rank=64, alpha=64)
    base_bf16 = module.weight.to(torch.bfloat16)
    a_bf16 = module.lora_A.weight.to(torch.bfloat16)
    b_bf16 = module.lora_B.weight.to(torch.bfloat16)

    merged = module.merged_weight(base_bf16, a_bf16, b_bf16).to(torch.float32)
    reference = base_bf16.float() + (b_bf16.float() @ a_bf16.float())
    naive_bf16 = (base_bf16 + (b_bf16 @ a_bf16)).float()

    assert (merged - reference).abs().max() <= (naive_bf16 - reference).abs().max()


# ---------------------------------------------------------------- _gather_weights


def test_gather_weights_merges_and_drops_adapter_keys():
    net = nn.Module()
    net.q_proj = _lora_linear(8, 6, rank=4, alpha=16)
    net.mlp = nn.Linear(6, 6)
    expected = net.q_proj.merged_weight(net.q_proj.weight, net.q_proj.lora_A.weight, net.q_proj.lora_B.weight)

    flat = _gather(net)

    assert not [k for k in flat if "lora_" in k], f"adapter keys leaked into the export: {sorted(flat)}"
    assert set(flat) == {"q_proj.weight", "mlp.weight", "mlp.bias"}
    torch.testing.assert_close(flat["q_proj.weight"], expected)


def test_gather_weights_exports_lora_bias_unchanged():
    net = nn.Module()
    net.q_proj = _lora_linear(8, 6, rank=4, alpha=16, bias=True)

    flat = _gather(net)

    assert set(flat) == {"q_proj.weight", "q_proj.bias"}
    torch.testing.assert_close(flat["q_proj.bias"], net.q_proj.bias)


def test_gather_weights_merges_under_a_checkpoint_wrapper():
    """Adapters under a gradient-checkpointing wrapper must still be found.

    The module path and the parameter path both carry the wrapper segment; the
    merge only fires if the two are stripped consistently.
    """
    lora = _lora_linear(8, 6, rank=4, alpha=16)
    net = nn.Module()
    net.layer = _CheckpointWrapper(lora)
    expected = lora.merged_weight(lora.weight, lora.lora_A.weight, lora.lora_B.weight)

    flat = _gather(net)

    assert set(flat) == {"layer.weight"}
    torch.testing.assert_close(flat["layer.weight"], expected)


def test_gather_weights_leaves_a_full_finetune_untouched():
    """No LoraInjectedLinear anywhere means the pre-existing path is unchanged."""
    net = nn.Sequential(nn.Linear(8, 6), nn.Linear(6, 4))
    reference = {name: param.detach().clone() for name, param in net.named_parameters()}

    flat = _gather(net)

    assert set(flat) == set(reference)
    for name, tensor in flat.items():
        torch.testing.assert_close(tensor, reference[name])


def test_gather_weights_casts_to_the_export_dtype():
    net = nn.Module()
    net.q_proj = _lora_linear(8, 6, rank=4, alpha=16)

    flat = _gather(net, dtype="bfloat16")

    assert all(t.dtype == torch.bfloat16 for t in flat.values())
