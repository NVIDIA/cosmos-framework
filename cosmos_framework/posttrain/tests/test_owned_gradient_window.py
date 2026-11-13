# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU algebra/ownership fixtures; these do not qualify real FSDP collectives."""

import copy
import hashlib
import inspect
import math
from pathlib import Path

import pytest
import torch

from cosmos_framework.posttrain.roles.trainer.gradient_window import (
    TORCH_SOURCE_SHA256,
    OwnedGradientWindow,
    Owner,
    count_from_native_dp_mean,
)


def owner(name, values, shards=1, replicas=1, global_shape=None, aliases=()):
    p = torch.nn.Parameter(torch.as_tensor(values, dtype=torch.float32).clone())
    return Owner(name, p, global_shape or tuple(p.shape), shards, replicas, aliases)


def dispatch(window, group, values):
    active = [o for o in window.groups[group] if o.name in values]
    pieces = []
    for o in active:
        raw = values[o.name].double().reshape(-1)
        pieces.append(torch.cat([raw, torch.zeros(o.padded_numel - raw.numel(), dtype=torch.float64)]))
    packed = torch.cat(pieces)
    window.hook(group)(packed)
    assert torch.count_nonzero(packed) == 0
    for o in active:
        o.parameter.grad = torch.zeros_like(o.parameter)
    return packed


def test_intermittenly_unused_owner_keeps_earlier_gradients_and_shared_alias_once():
    a = owner("a", [1.0, 2.0], aliases=("left.weight", "right.weight"))
    b = owner("b", [3.0, 4.0])
    never = owner("never", [5.0, 6.0])
    w = OwnedGradientWindow({"root": [a, b, never]}, microbatches=3, world_size=1)
    w.begin_window()
    w.begin_micro(2)
    dispatch(w, "root", {"a": torch.tensor([2.0, 4.0]), "b": torch.tensor([6.0, 8.0])})
    w.after_backward()
    w.begin_micro(0)
    w.after_backward()  # entire group unused; previous numerators remain
    w.begin_micro(1)
    dispatch(w, "root", {"a": torch.tensor([1.0, 2.0])})
    w.after_backward()
    w.finalize()
    assert torch.equal(a.parameter.grad, torch.tensor([1.0, 2.0]))
    assert torch.equal(b.parameter.grad, torch.tensor([2.0, 8 / 3]))
    assert never.parameter.grad is None
    assert [id(o.parameter) for o in w.owners].count(id(a.parameter)) == 1


@pytest.mark.parametrize("shards,replicas", [(4, 1), (2, 2)])
def test_fsdp_hsdp_global_mean_tail_and_clipping_owner_accounting(shards, replicas):
    full = torch.arange(1, 9, dtype=torch.float32)
    windows = []
    for rank in range(4):
        slot = rank % shards
        local = full.chunk(shards)[slot].clone()
        o = owner("p", local, shards, replicas, global_shape=(8,))
        w = OwnedGradientWindow({"root": [o]}, microbatches=3, world_size=4)
        w.begin_window()
        windows.append(w)
    for micro, (count, scale) in enumerate([(3, 2.0), (1, 1.0), (0, 0.0)]):
        for rank, w in enumerate(windows):
            w.begin_micro(count)
            slot = rank % shards
            dispatch(w, "root", {"p": (full * scale).chunk(shards)[slot]})
            w.after_backward()
    for w in windows:
        w.finalize()
    expected = full * 3 / 4
    squared = sum((w.owners[0].parameter.grad.double().square().sum() / replicas for w in windows))
    norm = math.sqrt(sum(float(v) ** 2 for v in expected))
    coefficient = torch.tensor(1 / (norm + 1e-6), dtype=torch.float32)
    for rank, w in enumerate(windows):
        r = w.clip(1, reduce_sum=lambda x: squared.clone())
        assert float(r["norm"]) == norm
        assert torch.equal(w.owners[0].parameter.grad, (expected * coefficient).chunk(shards)[rank % shards])
        with pytest.raises(RuntimeError):
            w.clip(1, reduce_sum=lambda x: squared.clone())


def test_real_autograd_shared_parameter_and_independent_uneven_objective():
    p = torch.nn.Parameter(torch.tensor([0.5, -0.25], dtype=torch.float32))
    o = Owner("p", p, (2,), 1, 1, ("path0", "path1"))
    independent = torch.nn.Parameter(p.detach().clone())
    oracle = GlobalObjectiveOracle({"p": independent})
    w = OwnedGradientWindow({"root": [o]}, microbatches=3, world_size=1)
    w.begin_window()
    numerators = []
    for x, count in [
        (torch.tensor([[1.0, 2.0], [2.0, 1.0]]), 2),
        (torch.tensor([[3.0, -1.0]]), 1),
        (torch.tensor([[1.0, 1.0]]), 0),
    ]:
        # Same owner used twice; count is independent number of valid scalar targets.
        n = ((x @ p + p.sum()).square()).sum() * (count > 0)
        native = torch.autograd.grad(n, p)[0]
        n2 = ((x @ independent + independent.sum()).square()).sum() * (count > 0)
        oracle.add(n2, count)
        numerators.append(float(n.detach()))
        w.begin_micro(count)
        dispatch(w, "root", {"p": native})
        w.after_backward()
    w.finalize()
    result = oracle.finish(max_norm=1)
    r = w.clip(1)
    assert result["loss"] == math.fsum(numerators) / 3 and result["global_valid_elements"] == 3
    assert torch.equal(p.grad, result["gradients"]["p"])
    assert float(r["norm"]) == result["norm"]
    assert independent.grad is None
    old = p.detach().clone()
    opt = torch.optim.AdamW([p], lr=0.01, weight_decay=0)
    opt.step()
    w.finish_step()
    assert not torch.equal(old, p)
    assert p.grad is None and w.state_dict()["completed_updates"] == 1


def test_uneven_last_shard_padding_is_not_a_second_parameter():
    a = owner("a", [1.0], 2, 1, global_shape=(3,))
    b = owner("b", [2.0, 3.0], 2, 1, global_shape=(4,))
    w = OwnedGradientWindow({"g": [a, b]}, microbatches=1, world_size=2)
    w.begin_window()
    w.begin_micro(1)
    dispatch(w, "g", {"a": torch.tensor([4.0]), "b": torch.tensor([5.0, 6.0])})
    w.after_backward()
    w.finalize()
    assert torch.equal(a.parameter.grad, torch.tensor([4.0]))
    assert torch.equal(b.parameter.grad, torch.tensor([5.0, 6.0]))


@pytest.mark.parametrize(
    "failure",
    ["nonfinite", "wrong_dtype", "nonzero_placeholder", "layout", "missing_hook", "duplicate_hook", "zero_padding"],
)
def test_bad_backward_never_becomes_an_optimizer_window(failure):
    a = owner("a", [1.0, 2.0])
    w = OwnedGradientWindow({"g": [a]}, microbatches=1, world_size=1)
    w.begin_window()
    w.begin_micro(0 if failure == "zero_padding" else 1)
    with pytest.raises(RuntimeError):
        if failure == "wrong_dtype":
            w.hook("g")(torch.ones(2))
        elif failure == "missing_hook":
            a.parameter.grad = torch.ones_like(a.parameter)
            w.after_backward()
        else:
            v = torch.ones(1 if failure == "layout" else 2, dtype=torch.float64)
            if failure == "nonfinite":
                v[0] = float("nan")
            w.hook("g")(v)
            if failure == "duplicate_hook":
                w.hook("g")(v)
            a.parameter.grad = (
                torch.ones_like(a.parameter) if failure == "nonzero_placeholder" else torch.zeros_like(a.parameter)
            )
            w.after_backward()
    w.abort()
    assert w.phase == "idle" and not w.buffers and not w.pending and a.parameter.grad is None
    w.begin_window()
    w.begin_micro(1)
    dispatch(w, "g", {"a": torch.tensor([2.0, 3.0])})
    w.after_backward()
    w.finalize()
    assert torch.equal(a.parameter.grad, torch.tensor([2.0, 3.0]))


def test_resume_identity_and_partial_state_rejection():
    a = owner("a", [1.0, 2.0])
    w = OwnedGradientWindow({"g": [a]}, microbatches=1, world_size=1)
    state = w.state_dict()
    state["completed_updates"] = 9
    w.load_state_dict(state)
    assert w.completed_updates == 9
    altered = copy.deepcopy(state)
    altered["identity"]["normalization"] = "local_mean"
    with pytest.raises(ValueError):
        w.load_state_dict(altered)
    for phase in ["open", "micro", "ready"]:
        w.phase = phase
        with pytest.raises(RuntimeError):
            w.state_dict()
    w.abort()
    assert w.completed_updates == 9


def test_native_forward_preparation_checks_owner_before_count_is_known():
    a = owner("a", [1.0, 2.0])
    calls = []
    w = OwnedGradientWindow({"g": [a]}, microbatches=1, world_size=1, before_micro=lambda: calls.append("ownership"))
    w.begin_window()
    w.prepare_forward()
    assert calls == ["ownership"] and w.forward_prepared
    with pytest.raises(RuntimeError):
        w.prepare_forward()
    with pytest.raises(RuntimeError):
        w.state_dict()
    with pytest.raises(RuntimeError):
        w.finalize()
    # An invalid finalization aborts the interrupted window. Reopen explicitly.
    assert w.phase == "idle"
    w.begin_window()
    w.prepare_forward()
    w.begin_micro(2)
    assert calls == ["ownership", "ownership"] and not w.forward_prepared
    dispatch(w, "g", {"a": torch.tensor([4.0, 6.0])})
    w.after_backward()
    w.finalize()
    assert torch.equal(a.parameter.grad, torch.tensor([2.0, 3.0]))
    w.abort()
    assert not w.forward_prepared


def test_failed_forward_preparation_never_becomes_a_checkpoint():
    a = owner("a", [1.0])
    w = OwnedGradientWindow({"g": [a]}, microbatches=1, world_size=1)
    w.begin_window()
    w.prepare_forward()
    w.abort()
    assert w.phase == "idle" and not w.forward_prepared and w.state_dict()["completed_updates"] == 0


@pytest.mark.parametrize("count", [True, -1, 0.5, float("nan"), float("inf"), 2**53])
def test_invalid_denominator_rejected(count):
    w = OwnedGradientWindow({"g": [owner("p", [1.0])]}, microbatches=1, world_size=1)
    w.begin_window()
    with pytest.raises(ValueError):
        w.begin_micro(count)
    assert count_from_native_dp_mean(17.5, 4) == 70


def test_empty_window_rejects_and_abort_preserves_parameters():
    a = owner("a", [1.0, 2.0])
    before = a.parameter.detach().clone()
    w = OwnedGradientWindow({"g": [a]}, microbatches=1, world_size=1)
    w.begin_window()
    w.begin_micro(0)
    w.after_backward()
    with pytest.raises(RuntimeError):
        w.finalize()
    assert w.phase == "idle" and torch.equal(a.parameter, before)


def test_alias_and_cross_group_ownership_conflict():
    a = owner("a", [1.0], aliases=("alias",))
    b = owner("b", [2.0], aliases=("alias",))
    with pytest.raises(ValueError):
        OwnedGradientWindow({"g": [a, b]}, microbatches=1, world_size=1)
    with pytest.raises(ValueError):
        OwnedGradientWindow({"g": [a], "other": [a]}, microbatches=1, world_size=1)


def test_torch_source_contract_and_public_hook_order():
    from torch.distributed.fsdp import FSDPModule

    root = Path(inspect.getsourcefile(FSDPModule.set_all_reduce_hook)).parent
    for name, h in TORCH_SOURCE_SHA256.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == h
    src = (root / "_fsdp_collectives.py").read_text()
    start = src.index("def foreach_reduce(")
    body = src[start : src.index("def foreach_reduce_scatter_copy_in", start)]
    assert (
        body.index("all_reduce_hook(reduce_output)")
        < body.index("_to_dtype_if_needed(reduce_output, orig_dtype)")
        < body.index("fsdp_param.sharded_param.grad =")
    )
    assert "fsdp_params_with_grad.append(fsdp_param)" in (root / "_fsdp_param_group.py").read_text()
    for method in ["set_all_reduce_hook", "set_gradient_divide_factor"]:
        assert callable(getattr(FSDPModule, method))


def test_independent_oracle_rejects_wrong_counts_and_never_uses_distributed_state():
    p = torch.nn.Parameter(torch.tensor([1.0]))
    oracle = GlobalObjectiveOracle({"p": p})
    with pytest.raises(ValueError):
        oracle.add(p.sum(), 0)
    with pytest.raises(ValueError):
        oracle.add(p.sum(), True)
    with pytest.raises(ValueError):
        oracle.finish()
    assert p.grad is None


def test_native_denominator_owner_uses_integer_global_sum_not_rank_mean():
    a = owner("p", [1.0, 2.0], 2, 2, global_shape=(4,))
    events = []
    w = OwnedGradientWindow(
        {"g": [a]},
        microbatches=2,
        world_size=4,
        count_sum=lambda local: events.append(("local_count", local)) or [3, 2][len(events) - 1],
        count_consensus=lambda count: None,
    )
    w.begin_window()
    assert w.begin_native_micro(1) == 3
    dispatch(w, "g", {"p": torch.tensor([6.0, 12.0])})
    w.after_backward()
    assert w.begin_native_micro(0) == 2
    dispatch(w, "g", {"p": torch.tensor([4.0, 8.0])})
    w.after_backward()
    w.finalize()
    assert (
        events == [("local_count", 1), ("local_count", 0)]
        and w.count == 5
        and torch.equal(a.parameter.grad, torch.tensor([2.0, 4.0]))
    )
    assert w.identity()["denominator_owner"] == "native_valid_mask_int64_global_sum"


def test_denominator_disagreement_or_late_policy_change_rejects_before_backward():
    a = owner("p", [1.0])
    events = []

    def guard():
        raise RuntimeError("no-sync not admitted")

    w = OwnedGradientWindow(
        {"g": [a]}, microbatches=1, world_size=1, before_micro=guard, count_consensus=lambda count: events.append(count)
    )
    w.begin_window()
    with pytest.raises(RuntimeError, match="no-sync"):
        w.begin_micro(1)
    assert w.phase == "open" and not events and not w.pending
    w.before_micro = lambda: None

    def disagree(count):
        raise RuntimeError("global denominator differs")

    w.count_consensus = disagree
    with pytest.raises(RuntimeError, match="denominator"):
        w.begin_micro(1)
    assert w.phase == "open" and not w.pending


def test_positive_count_without_any_owned_derivative_is_not_a_step():
    w = OwnedGradientWindow({"g": [owner("p", [1.0])]}, microbatches=1, world_size=1)
    w.begin_window()
    w.begin_micro(1)
    w.after_backward()
    with pytest.raises(RuntimeError, match="no owned gradients"):
        w.finalize()
    assert w.phase == "idle" and w.completed_updates == 0


def test_cross_stream_capture_lifetime_is_explicitly_recorded():
    source = inspect.getsource(OwnedGradientWindow._wait)
    assert source.index("wait_event(event)") < source.index("record_stream(stream)") < source.index("return tensor")


def test_alias_cannot_shadow_another_canonical_owner_name():
    a = owner("a", [1.0], aliases=("b",))
    b = owner("b", [2.0], aliases=("other",))
    with pytest.raises(ValueError, match="alias"):
        OwnedGradientWindow({"g": [a, b]}, microbatches=1, world_size=1)


class GlobalObjectiveOracle:
    def __init__(self, named_parameters):
        self.parameters = dict(named_parameters)
        if not self.parameters or len({id(p) for p in self.parameters.values()}) != len(self.parameters):
            raise ValueError("unique independent trainable owners required")
        if any(not p.requires_grad or p.dtype != torch.float32 for p in self.parameters.values()):
            raise ValueError("oracle requires native FP32 trainable leaves")
        self.numerators = []
        self.counts = []
        self.derivatives = {}
        self.closed = False

    def add(self, numerator, valid_elements):
        if self.closed or type(valid_elements) is not int or valid_elements < 0:
            raise ValueError("invalid independent sample count")
        if numerator.ndim or not numerator.requires_grad or not torch.isfinite(numerator):
            raise ValueError("finite differentiable scalar numerator required")
        value = float(numerator.detach().double())
        if valid_elements == 0 and value != 0:
            raise ValueError("padding numerator must be zero")
        derivatives = torch.autograd.grad(numerator, tuple(self.parameters.values()), allow_unused=True)
        for (name, p), grad in zip(self.parameters.items(), derivatives):
            if grad is None:
                continue
            if grad.dtype != torch.float32 or not bool(torch.isfinite(grad).all()):
                raise ValueError("invalid native numerator derivative")
            if valid_elements == 0 and torch.count_nonzero(grad):
                raise ValueError("padding derivative must be zero")
            if name not in self.derivatives:
                self.derivatives[name] = torch.zeros_like(p, dtype=torch.float64)
            self.derivatives[name].add_(grad.detach().double())
        self.numerators.append(value)
        self.counts.append(valid_elements)

    def finish(self, max_norm=None):
        count = sum(self.counts)
        if self.closed or count <= 0:
            raise ValueError("empty or already finalized objective")
        gradients = {name: (value / count).float() for name, value in self.derivatives.items()}
        # Independent scalar fsum of all unique owner squares. This deliberate CPU
        # oracle is a correctness reference, not a proposed production norm kernel.
        norm = math.sqrt(
            math.fsum(
                float(x) * float(x) for value in gradients.values() for x in value.detach().cpu().reshape(-1).tolist()
            )
        )
        coefficient = 1.0
        if max_norm is not None:
            if isinstance(max_norm, bool) or not math.isfinite(max_norm) or max_norm <= 0:
                raise ValueError("invalid independent clip limit")
            coefficient = float(torch.tensor(min(1.0, max_norm / (norm + 1e-6)), dtype=torch.float32))
            gradients = {name: value * coefficient for name, value in gradients.items()}
        self.closed = True
        return {
            "loss": math.fsum(self.numerators) / count,
            "global_valid_elements": count,
            "gradients": gradients,
            "norm": norm,
            "coefficient": coefficient,
            "scope": "independent native per-example numerator derivatives, no distributed state or update rebasing",
        }
