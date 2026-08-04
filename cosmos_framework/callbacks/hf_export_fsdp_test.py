# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Distributed counterpart to ``hf_export_test.py`` — LoRA merging under FSDP2.

``hf_export_test.py`` runs single-process on CPU with plain ``nn.Linear``, so it
cannot cover the part that actually carries risk: ``lora_A`` / ``lora_B`` are
DTensors sharded across ranks, and ``_gather_weights`` all-gathers them from
inside the *base weight's* loop iteration rather than at their own
``named_parameters()`` entries. That reordering is safe only because every rank
walks the module tree identically — a property worth asserting rather than
arguing.

World size must be 2. Launch with::

    torchrun --nproc_per_node=2 -m pytest cosmos_framework/callbacks/hf_export_fsdp_test.py

Under plain pytest (no ``RANK``) every test skips, matching ``cfgp_ar_test`` and
``context_parallel_test``.
"""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.fsdp import fully_shard

from cosmos_framework.callbacks.hf_export import HFExportCallback
from cosmos_framework.utils.generator.lora import LoraInjectedLinear

_WORLD_SIZE = 2


def setup_distributed_environment() -> tuple[int, int]:
    if "RANK" not in os.environ:
        pytest.skip("requires distributed environment (run with: torchrun --nproc_per_node=2)")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    if world_size != _WORLD_SIZE:
        pytest.skip(f"requires world_size={_WORLD_SIZE}, got {world_size}")
    torch.cuda.set_device(rank)
    return rank, world_size


def _lora_linear(in_f: int, out_f: int, rank: int, alpha: int, *, bias: bool = False) -> LoraInjectedLinear:
    """Materialized adapter — ``__init__`` puts lora_A / lora_B on the meta device."""
    base = nn.Linear(in_f, out_f, bias=bias)
    module = LoraInjectedLinear(base, rank, alpha)
    module.lora_A = nn.Linear(in_f, rank, bias=False)
    module.lora_B = nn.Linear(rank, out_f, bias=False)
    return module


class _Block(nn.Module):
    """Four adapted projections plus an unadapted MLP."""

    _ADAPTED = ("q_proj", "k_proj", "v_proj", "o_proj")

    def __init__(self, dim: int = 64, rank: int = 8, alpha: int = 16) -> None:
        super().__init__()
        for name in self._ADAPTED:
            setattr(self, name, _lora_linear(dim, dim, rank, alpha, bias=(name == "o_proj")))
        self.mlp = nn.Linear(dim, dim * 2)


def _build_sharded_block() -> tuple[nn.Module, dict[str, torch.Tensor]]:
    """Return an FSDP2-sharded block and the merged weights computed BEFORE sharding.

    The reference is the whole point: it is what a single-process export of the
    same model would produce, so comparing against it catches any way the
    sharded path could diverge.
    """
    torch.manual_seed(1234)  # identical init on every rank
    model = _Block().cuda()
    for name in _Block._ADAPTED:
        module = getattr(model, name)
        # lora_B ships zero-initialized; a zero adapter makes the merge a no-op
        # and would let a broken merge pass.
        nn.init.normal_(module.lora_A.weight, std=0.02)
        nn.init.normal_(module.lora_B.weight, std=0.02)

    reference: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, LoraInjectedLinear):
            reference[f"{name}.weight"] = module.merged_weight(
                module.weight.detach(), module.lora_A.weight.detach(), module.lora_B.weight.detach()
            ).clone()
    for name, param in model.named_parameters():
        # Adapters are folded into the base weight, so a correct export does not
        # carry them and neither does the reference.
        if name.endswith(("lora_A.weight", "lora_B.weight")):
            continue
        reference.setdefault(name, param.detach().clone())

    # Gradient checkpointing on one projection puts a real
    # _checkpoint_wrapped_module segment in the module tree — the exact shape
    # that broke an earlier revision's path stripping.
    model.k_proj = checkpoint_wrapper(model.k_proj)

    for child in list(model.children()):
        fully_shard(child)
    fully_shard(model)
    return model, reference


def _gather(model: nn.Module) -> tuple[dict[str, torch.Tensor], dict[str, str], int]:
    callback = HFExportCallback(dtype="float32")
    chunks, manifest, total = callback._gather_weights(SimpleNamespace(model=SimpleNamespace(model=model)))
    return {k: v for c in chunks for k, v in c.items()}, manifest, total


def test_adapters_are_actually_sharded():
    """Guards the test itself: without DTensors the rest proves nothing."""
    setup_distributed_environment()
    model, _ = _build_sharded_block()

    sharded = [n for n, p in model.named_parameters() if isinstance(p, torch.distributed.tensor.DTensor)]
    assert sharded, "FSDP2 did not shard anything; the remaining assertions would be vacuous"
    assert len([n for n in sharded if "lora_" in n]) == 8, f"expected 8 sharded adapter tensors, got {sharded}"


def test_merged_export_matches_the_unsharded_reference():
    """The whole contract: sharded export == single-process export, key for key."""
    rank, _ = setup_distributed_environment()
    model, reference = _build_sharded_block()

    flat, manifest, total = _gather(model)
    # Returning on every rank is itself the assertion that the reordered
    # all-gathers stay in lockstep; a mismatch hangs here instead.
    dist.barrier()

    if rank != 0:
        return

    assert not [k for k in flat if "lora_" in k], f"adapter keys leaked into the export: {sorted(flat)}"
    assert set(flat) == set(reference), (
        f"extra={sorted(set(flat) - set(reference))} missing={sorted(set(reference) - set(flat))}"
    )
    assert set(manifest) == set(flat)
    assert total == sum(t.element_size() * t.numel() for t in flat.values())
    for key, tensor in flat.items():
        torch.testing.assert_close(tensor.cuda(), reference[key], rtol=0, atol=1e-5, msg=f"mismatch at {key}")


def test_merge_completeness_guard_fires_under_fsdp():
    """The guard must abort on every rank, not just where the manifest lives."""
    setup_distributed_environment()
    model, _ = _build_sharded_block()

    callback = HFExportCallback(dtype="float32")
    real_plan = callback._lora_merge_plan
    # A target that no parameter name can match — i.e. the plan and the loop
    # disagreeing, which is how a silently unmerged export would arise.
    callback._lora_merge_plan = lambda root: ({"bogus.path.weight": None}, real_plan(root)[1])

    with pytest.raises(RuntimeError, match="LoRA merge incomplete"):
        callback._gather_weights(SimpleNamespace(model=SimpleNamespace(model=model)))
