# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""L0/CPU tests for the policy->rollout transfer dtype (`rl.transfer_dtype`).

The trainer's STORAGE dtype need not be the inference engine's: with `rl.master_dtype="float32"` the
policy holds fp32 parameters while a vLLM engine is bf16. Without a declared transfer dtype the
manifest reports fp32, which (a) sizes the buckets at 2x and (b) hands the engine fp32 bytes for
bf16 receive buffers.

The plan was ALREADY authoritative for the receiver (`alloc_recv_buffers` allocates from the plan's
dtype) and for bucket sizing (`_dtype_bytes`); the PRODUCER was the one end ignoring it —
`bucket_full_tensors` destructured the declared dtype as `_dtype` and never used it. That was
invisible only because the declared dtype always WAS the model's own.

These tests exist because that path is otherwise unexercised: every parity run to date had
`master == param`, or used the HF rollout flavor where weight sync is a shared-model no-op. cosmos-rl
applies the same cast on both ends of this hop (`train.transfer_dtype`).
"""

from __future__ import annotations

import pytest
import torch

from cosmos_framework.posttrain.comm.weight_transfer import (
    bucket_full_tensors,
    bucket_plan,
    model_manifest,
)


def _net(dtype: torch.dtype) -> torch.nn.Module:
    net = torch.nn.Linear(8, 4, bias=True)
    return net.to(dtype)


@pytest.mark.L0
@pytest.mark.CPU
def test_manifest_reports_the_models_dtype_when_unset():
    """`transfer_dtype=None` ⇒ report each parameter's actual dtype — the pre-change behaviour."""
    manifest = model_manifest(_net(torch.float32))
    assert manifest and all(d == "torch.float32" for _n, _s, d in manifest)


@pytest.mark.L0
@pytest.mark.CPU
def test_manifest_declares_the_transfer_dtype_when_set():
    """A declared dtype overrides the model's own for every FLOATING parameter."""
    manifest = model_manifest(_net(torch.float32), transfer_dtype="bfloat16")
    assert manifest and all(d == "torch.bfloat16" for _n, _s, d in manifest)


@pytest.mark.L0
@pytest.mark.CPU
def test_bare_and_prefixed_dtype_spellings_are_both_accepted():
    """`"bfloat16"` and `"torch.bfloat16"` must resolve identically.

    The manifest dtype string is consumed by `_dtype_bytes` and by `alloc_recv_buffers`, and is
    compared in existing tests, so the EMITTED spelling stays the legacy `torch.<name>` form
    regardless of which spelling was supplied.
    """
    a = model_manifest(_net(torch.float32), transfer_dtype="bfloat16")
    b = model_manifest(_net(torch.float32), transfer_dtype="torch.bfloat16")
    assert a == b and all(d == "torch.bfloat16" for _n, _s, d in a)


@pytest.mark.L0
@pytest.mark.CPU
def test_declared_dtype_halves_the_bucket_byte_accounting():
    """Bucket sizing must follow the WIRE dtype, not the model's.

    An fp32 manifest for a bf16 transfer would pack half as many params per bucket — the 2x
    over-sizing this field exists to prevent.
    """
    net = _net(torch.float32)
    # A cap that splits fp32 (4 bytes/elem) but not bf16 (2 bytes/elem) for this model.
    cap = 8 * 4 * 4 + 8  # a little over half of the 8x4 weight in fp32
    fp32_buckets = bucket_plan(model_manifest(net), bucket_bytes=cap)
    bf16_buckets = bucket_plan(model_manifest(net, transfer_dtype="bfloat16"), bucket_bytes=cap)
    assert len(bf16_buckets) <= len(fp32_buckets)


@pytest.mark.L0
@pytest.mark.CPU
def test_producer_casts_to_the_dtype_the_plan_declares():
    """THE POINT: `bucket_full_tensors` must emit the plan's dtype, not the parameter's.

    The receiver allocates from the plan, so a producer that ignored it would send fp32 bytes into
    bf16 buffers — a silent size/format mismatch rather than an error.
    """
    net = _net(torch.float32)
    plan = bucket_plan(model_manifest(net, transfer_dtype="bfloat16"))
    for bucket in plan:
        tensors = bucket_full_tensors(net, bucket)
        assert tensors, "bucket produced no tensors"
        for name, _shape, _dtype in bucket:
            assert tensors[name].dtype is torch.bfloat16, f"{name} was not cast to the declared dtype"


@pytest.mark.L0
@pytest.mark.CPU
def test_cast_is_value_preserving_within_the_target_precision():
    """The cast must be a plain dtype conversion — not a reinterpret, not a zero-fill."""
    net = _net(torch.float32)
    plan = bucket_plan(model_manifest(net, transfer_dtype="bfloat16"))
    src = dict(net.state_dict())
    for bucket in plan:
        for name, tensor in bucket_full_tensors(net, bucket).items():
            assert torch.equal(tensor, src[name].to(torch.bfloat16)), f"{name} value changed"
    # Guard the degenerate pass: an all-zero model would satisfy the loop above trivially.
    assert any(v.abs().sum().item() > 0 for v in src.values())


@pytest.mark.L0
@pytest.mark.CPU
def test_matching_dtype_is_not_copied():
    """When the declared dtype already matches, the tensor is passed through unchanged.

    `bucket_full_tensors` skips the `.to()` in that case; this pins that the no-op path stays a
    no-op (every recipe with `master == param` takes it).
    """
    net = _net(torch.float32)
    plan = bucket_plan(model_manifest(net, transfer_dtype="float32"))
    for bucket in plan:
        for name, tensor in bucket_full_tensors(net, bucket).items():
            assert tensor.dtype is torch.float32
            assert torch.equal(tensor, dict(net.state_dict())[name])


def _net_with_buffers() -> torch.nn.Module:
    """A float module carrying persistent NON-float buffers, as real models do.

    Integer/bool buffers are ordinary in production nets — step counters, token-type ids, causal
    masks, quantization zero-points — and `state_dict()` returns them alongside the parameters.
    """
    net = torch.nn.Linear(8, 4, bias=True)
    # 12345 is deliberately NOT representable in bf16 (8-bit mantissa); it rounds to 12352, so a
    # wrongly-declared cast shows up as a value change rather than only a dtype change.
    net.register_buffer("step_counter", torch.tensor([12345, 7, 99999], dtype=torch.int64))
    net.register_buffer("causal_mask", torch.tensor([True, False, True], dtype=torch.bool))
    # A FLOATING buffer — BatchNorm running statistics are the canonical case. This one survives a
    # "is it floating?" check while still being wrong to rewrite: it is model state, not a weight,
    # and a bf16 round trip truncates it permanently. 1/3 is chosen because it is not representable
    # in bf16, so a regression shows as a value change.
    net.register_buffer("running_var", torch.tensor([1.0 / 3.0, 2.0 / 3.0], dtype=torch.float32))
    return net


@pytest.mark.L0
@pytest.mark.CPU
def test_manifest_keeps_buffer_dtypes_including_floating_ones():
    """The transfer dtype is a precision choice about WEIGHTS; it must not touch ANY buffer.

    The declared dtype is authoritative end to end — `bucket_full_tensors` casts to it — so
    declaring bf16 for an int64 buffer would round its values on the way out and load the rounded
    result back on the way in, corrupting state silently.
    """
    manifest = dict((n, d) for n, _s, d in model_manifest(_net_with_buffers(), transfer_dtype="bfloat16"))
    assert manifest["weight"] == "torch.bfloat16"
    assert manifest["bias"] == "torch.bfloat16"
    assert manifest["step_counter"] == "torch.int64"
    assert manifest["causal_mask"] == "torch.bool"
    assert manifest["running_var"] == "torch.float32", "a floating BUFFER is still not a weight"


@pytest.mark.L0
@pytest.mark.CPU
def test_bucket_transfer_preserves_integer_buffer_values_bit_for_bit():
    """The producer-side cast must leave non-float buffers alone, values and dtype both."""
    net = _net_with_buffers()
    manifest = model_manifest(net, transfer_dtype="bfloat16")
    out = bucket_full_tensors(net, manifest)
    assert out["step_counter"].dtype == torch.int64
    assert torch.equal(out["step_counter"], torch.tensor([12345, 7, 99999], dtype=torch.int64))
    assert out["causal_mask"].dtype == torch.bool
    assert torch.equal(out["causal_mask"], torch.tensor([True, False, True], dtype=torch.bool))
    assert out["running_var"].dtype == torch.float32
    assert torch.equal(out["running_var"], torch.tensor([1.0 / 3.0, 2.0 / 3.0], dtype=torch.float32))
    assert out["weight"].dtype == torch.bfloat16  # floats still honour the declared wire dtype


@pytest.mark.L0
@pytest.mark.CPU
def test_tied_parameter_aliases_take_the_transfer_dtype_too():
    """A tied weight appears under TWO names in `state_dict()` but once in `named_parameters()`.

    With the default `remove_duplicate=True` the alias falls out of the parameter set, keeps its
    storage dtype in the plan, and is sent uncast — the same tensor on the wire under two names at
    two dtypes, and an unnecessary fp32 transfer of one of the largest matrices in the model
    (`lm_head.weight` is the classic case).
    """
    net = torch.nn.Module()
    net.embed = torch.nn.Linear(4, 8, bias=False)
    net.lm_head = torch.nn.Linear(4, 8, bias=False)
    net.lm_head.weight = net.embed.weight  # tie them

    sd_names = set(net.state_dict().keys())
    assert {"embed.weight", "lm_head.weight"} <= sd_names, "fixture did not produce a tied alias"

    manifest = dict((n, d) for n, _s, d in model_manifest(net, transfer_dtype="bfloat16"))
    assert manifest["embed.weight"] == "torch.bfloat16"
    assert manifest["lm_head.weight"] == "torch.bfloat16", "tied alias was left at its storage dtype"


@pytest.mark.L0
@pytest.mark.CPU
def test_a_validating_mapper_is_refused_at_plan_time_not_mid_sync():
    """A mapper that checks its exports against storage dtype must be refused where the knob is named.

    `Qwen3VLWeightMapper.export_bucket` raises "wire metadata for X changed from ... to ..." when a
    bucket's dtype differs from the spec it recorded, and its witness carries the storage dtype too
    (`vllm_weights.py` re-checks it on the receive side). Restating the plan for such a mapper
    produces a failure several layers down whose message never mentions `rl.transfer_dtype`.

    The refusal is deliberately conditional on the manifest ACTUALLY changing, so this cannot break
    a working configuration: `transfer` defaults to `master`, so every shipped recipe restates
    nothing and never reaches the raise.
    """
    from cosmos_framework.posttrain.comm.weight_transfer import model_manifest

    class _ValidatingMapper:  # no `accepts_wire_dtype` -- exactly Qwen3VLWeightMapper's shape
        def wire_manifest(self, model):
            return [(n, tuple(t.shape), str(t.dtype)) for n, t in sorted(model.state_dict().items())]

    net = _net(torch.float32)
    # Same dtype: nothing is restated, so the mapper is left alone.
    same = model_manifest(net, _ValidatingMapper(), transfer_dtype="float32")
    assert {d for _n, _s, d in same} == {"torch.float32"}

    with pytest.raises(ValueError, match=r"does not declare `accepts_wire_dtype`"):
        model_manifest(net, _ValidatingMapper(), transfer_dtype="bfloat16")


@pytest.mark.L0
@pytest.mark.CPU
def test_acceptance_requires_a_declaration_on_the_mappers_own_class():
    """Only the concrete class's own declaration counts -- not inheritance, not an instance, not shape.

    Three earlier forms of this predicate were each shown to have a way around them, and every one
    of those routes is pinned here:

      * `getattr` -- any subclass inherits the base's True;
      * an MRO walk -- a subclass of a DECLARED-compatible subclass inherits its True before any
        structural check runs;
      * "did it override `export_bucket`" -- an instance attribute replaces the callable that
        `bucket_full_tensors` actually invokes without touching the class, and a mapper can keep the
        identity export while recording a storage dtype in `wire_manifest` or a witness instead.

    Hence: declared on `type(mapper)` itself, or refused. A subclass that changes nothing relevant
    pays one line for a rule with no way around it.
    """
    from cosmos_framework.posttrain.comm.weight_transfer import WeightMapper, accepts_wire_dtype

    class _Duck:  # not a WeightMapper at all -- Qwen3VLWeightMapper's shape
        def wire_manifest(self, model):
            return []

        def export_bucket(self, model, bucket):
            return {}

    class _PlainSubclass(WeightMapper):
        """Changes nothing, and still must declare: no structural signal is trusted."""

    class _Declared(WeightMapper):
        accepts_wire_dtype = True

    class _ChildOfDeclared(_Declared):
        """Inherits a True it never declared, and validates on export -- the MRO-walk bypass."""

        def export_bucket(self, model, bucket):
            return {}

    assert accepts_wire_dtype(WeightMapper()) is True
    assert accepts_wire_dtype(_Declared()) is True
    assert accepts_wire_dtype(_PlainSubclass()) is False
    assert accepts_wire_dtype(_ChildOfDeclared()) is False
    assert accepts_wire_dtype(_Duck()) is False

    # An INSTANCE attribute cannot buy acceptance. The predicate reads `type(mapper).__dict__`, so a
    # flag set on the object is not consulted at all.
    flagged_instance = _PlainSubclass()
    flagged_instance.accepts_wire_dtype = True
    assert accepts_wire_dtype(flagged_instance) is False

    # ... and swapping the exported callable per instance -- the case that defeated the "did it
    # override `export_bucket`" form, because `bucket_full_tensors` invokes `mapper.export_bucket`
    # while that check inspected `type(mapper).export_bucket` -- is likewise not a route in: the
    # declaration is what is read, and this instance's class never made one.
    swapped_export = WeightMapper()
    assert accepts_wire_dtype(swapped_export) is True  # its class declares, so it starts accepted
    swapped_instance = _PlainSubclass()
    swapped_instance.export_bucket = lambda model, bucket: {}
    assert accepts_wire_dtype(swapped_instance) is False


@pytest.mark.L0
@pytest.mark.CPU
def test_an_undeclared_subclass_is_refused_with_a_message_describing_the_rule():
    """The refusal must fire through `model_manifest`, not only through the predicate."""
    from cosmos_framework.posttrain.comm.weight_transfer import WeightMapper, model_manifest

    class _Undeclared(WeightMapper):
        pass

    net = _net(torch.float32)
    with pytest.raises(ValueError, match=r"does not declare `accepts_wire_dtype`") as caught:
        model_manifest(net, _Undeclared(), transfer_dtype="bfloat16")
    # The message must describe the RULE, not assume the mapper is incompatible: `_Undeclared` is
    # inert and refused purely for not having been reviewed, so wording that told its author to fix
    # an export guard and a witness would be false guidance for a legitimate refusal.
    message = str(caught.value)
    assert "never inherited" in message and "accepts_wire_dtype = True" in message
    assert "rl.transfer_dtype" in message and "rl.master_dtype" in message
    # ... and is untouched when the declaration restates nothing.
    assert {d for _n, _s, d in model_manifest(net, _Undeclared(), transfer_dtype="float32")} == {"torch.float32"}
