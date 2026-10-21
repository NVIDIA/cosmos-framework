# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""L0/CPU tests for the `HFRunConfig` precision block (storage vs compute dtype split).

Two layers, both torch-free:
  - SCHEMA — where the fields live, that they default to a no-op, that a bad value fails at config
    load, and that the `hf`-backend algorithms carry them while the `i4`-backend one does not.
  - RESOLUTION — `config/build.py::resolve_precision`, the ONE place the `""`-means-inherit chain
    and the `model_hf` deprecation live. The load-bearing case is the FIRST one: everything unset
    must reproduce the pre-split behaviour exactly, or the port is not the no-op it claims to be.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from cosmos_framework.posttrain.config.build import Precision, resolve_precision
from cosmos_framework.posttrain.config.model import HFModelConfig
from cosmos_framework.posttrain.config.rl import (
    PRECISION_DTYPES,
    DPOConfig,
    GRPOConfig,
    HFRunConfig,
    OPDConfig,
    RunConfig,
)

_PRECISION_FIELDS = ("param_dtype", "master_dtype", "reduce_dtype", "transfer_dtype")


def _root(rl=None, model_hf=None):
    """A minimal stand-in for the root config: `resolve_precision` reads only these two sections."""
    return SimpleNamespace(rl=rl if rl is not None else GRPOConfig(), model_hf=model_hf or HFModelConfig())


@pytest.mark.L0
@pytest.mark.CPU
def test_precision_fields_default_to_empty_so_the_block_is_a_no_op():
    """Every precision field defaults to `""` — "inherit the deprecated `model_hf` value".

    This is what makes the schema addition behaviour-preserving: a config that sets none of them
    must resolve exactly as it did before the split existed.
    """
    cfg = GRPOConfig()
    for name in _PRECISION_FIELDS:
        assert getattr(cfg, name) == "", f"{name} must default to '' (inherit), got {getattr(cfg, name)!r}"


@pytest.mark.L0
@pytest.mark.CPU
def test_hf_backend_algorithms_carry_precision_and_the_i4_one_does_not():
    """`GRPOConfig`/`OPDConfig` (backend=hf) own the block; `DPOConfig` (backend=i4) must not.

    DPO's model is the inherited i4 LazyConfig tree, which carries its own dtype — precision fields
    there would be silently ignored. Excluding them makes `rl.master_dtype=...` on a DPO run an
    unknown-key error at load instead of a no-op.
    """
    assert issubclass(GRPOConfig, HFRunConfig)
    assert issubclass(OPDConfig, GRPOConfig)  # OPD inherits the block through GRPO
    assert not issubclass(DPOConfig, HFRunConfig)
    assert issubclass(DPOConfig, RunConfig)

    dpo_fields = {f.name for f in dataclasses.fields(DPOConfig)}
    for name in _PRECISION_FIELDS:
        assert name not in dpo_fields, f"DPOConfig must not carry {name} (its backend is i4)"
        assert name in {f.name for f in dataclasses.fields(OPDConfig)}


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("name", _PRECISION_FIELDS)
def test_bad_dtype_name_fails_at_config_load_not_model_build(name):
    """A typo'd dtype raises here, not later inside `getattr(torch, ...)`.

    Without this the failure lands during model construction — after placement and after GPUs are
    already held — as an opaque AttributeError.
    """
    with pytest.raises(ValueError, match=name):
        GRPOConfig(**{name: "bflaot16"})


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("dtype", PRECISION_DTYPES)
@pytest.mark.parametrize("name", _PRECISION_FIELDS)
def test_every_accepted_dtype_is_accepted_on_every_field(name, dtype):
    """The accepted set is uniform across the four fields and matches cosmos-rl's three choices."""
    assert getattr(GRPOConfig(**{name: dtype}), name) == dtype


@pytest.mark.L0
@pytest.mark.CPU
def test_reference_recipe_combination_is_expressible():
    """fp32 storage + bf16 compute — the combination `model_hf.dtype` alone cannot express.

    This is the whole point of the split: it is the configuration both this framework and cosmos-rl
    actually want to run, and the reason `rl.optimizer.name="fusedadam"` existed (apex rebuilding an
    fp32 master because the model layer could not provide one).
    """
    cfg = GRPOConfig(master_dtype="float32", param_dtype="bfloat16", reduce_dtype="float32")
    assert (cfg.master_dtype, cfg.param_dtype, cfg.reduce_dtype) == ("float32", "bfloat16", "float32")


@pytest.mark.L0
@pytest.mark.CPU
def test_run_config_validation_still_runs_on_the_subclass():
    """`HFRunConfig.__post_init__` must not shadow `RunConfig`'s checks.

    It overrides `__post_init__` to validate the dtype names, so a missing `super()` call would
    silently drop `grad_accum`/`checkpoint_format` validation for every hf-backend algorithm.
    """
    with pytest.raises(ValueError, match="grad_accum"):
        GRPOConfig(grad_accum=0)
    with pytest.raises(ValueError, match="checkpoint_format"):
        GRPOConfig(checkpoint_format="not-a-format")


# --------------------------------------------------------------------------- resolve_precision


@pytest.mark.L0
@pytest.mark.CPU
def test_everything_unset_reproduces_the_pre_split_behaviour():
    """THE gate for the port: nothing set ⇒ storage == compute == `model_hf.dtype`.

    Before the split one field drove both the parameter normalization and `ShardPlan.param_dtype`.
    If this case ever resolved `master != param`, every existing recipe would silently change
    optimizer precision — which is the one thing steps A1-A4 must not do.
    """
    assert resolve_precision(_root()) == Precision(
        master="bfloat16", param="bfloat16", reduce="float32", transfer="bfloat16"
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_deprecated_model_hf_fields_are_still_honoured_as_the_fallback():
    """Existing recipes and dotlists keep working — with ONE deliberate behaviour change.

    `model_hf.dtype` still sets BOTH storage and compute, so that half is a strict no-op.

    `model_hf.fsdp_reduce_dtype` is NOT: it was DEAD before this change (nothing read it;
    `model/hf.py` hardcoded `reduce_dtype=torch.float32`) and now actually reaches
    `ShardPlan.reduce_dtype`. So a config that set it to anything other than "float32" previously
    got fp32 reduction regardless, and now gets what it asked for. That is a real behaviour change
    for such a config, and the port is therefore NOT a strict no-op in that one respect — see the
    commit message and CHANGE_LEDGER section A. No in-repo config is affected (the field defaults to
    "float32" and the only explicit setting, in `configs/opd_qwen1p5b_countdown.py`, is "float32"),
    and honouring a configured value is the correct behaviour, so this is asserted rather than
    reverted.
    """
    p = resolve_precision(_root(model_hf=HFModelConfig(dtype="float32", fsdp_reduce_dtype="bfloat16")))
    assert (p.master, p.param, p.reduce) == ("float32", "float32", "bfloat16")


@pytest.mark.L0
@pytest.mark.CPU
def test_rl_fields_override_the_deprecated_ones():
    """`rl.*` wins; the deprecated field is only consulted when the new one is `""`."""
    p = resolve_precision(_root(GRPOConfig(param_dtype="bfloat16"), HFModelConfig(dtype="float32")))
    assert p.param == "bfloat16"


@pytest.mark.L0
@pytest.mark.CPU
def test_reference_recipe_resolves_to_fp32_master_bf16_compute():
    """The combination `model_hf.dtype` alone could not express, and the reason apex was needed."""
    p = resolve_precision(_root(GRPOConfig(master_dtype="float32", param_dtype="bfloat16")))
    assert (p.master, p.param) == ("float32", "bfloat16")


@pytest.mark.L0
@pytest.mark.CPU
def test_transfer_dtype_defaults_to_master_like_cosmos_rl():
    """cosmos-rl defaults `transfer_dtype` to `master_dtype`; so do we.

    It matters once master != param: the policy->rollout hop would otherwise ship the trainer's
    fp32 buffers into a bf16 engine at double the wire bytes.
    """
    p = resolve_precision(_root(GRPOConfig(master_dtype="float32", param_dtype="bfloat16")))
    assert p.transfer == "float32"
    explicit = resolve_precision(_root(GRPOConfig(master_dtype="float32", transfer_dtype="bfloat16")))
    assert explicit.transfer == "bfloat16"


@pytest.mark.L0
@pytest.mark.CPU
def test_an_rl_without_precision_fields_still_resolves():
    """DPO's `rl` predates `HFRunConfig` and has no precision fields at all.

    `resolve_precision` must fall back entirely to `model_hf` rather than raising — `build_backbone`
    calls it for whatever backend the root selects.
    """
    p = resolve_precision(_root(DPOConfig(), HFModelConfig(dtype="float32")))
    assert (p.master, p.param) == ("float32", "float32")


# --------------------------------------------------------------------------- mixed_precision_mode


@pytest.mark.L0
@pytest.mark.CPU
def test_mixed_precision_mode_defaults_to_fsdp():
    """Unset ⇒ "fsdp", the MixedPrecisionPolicy cast every existing recipe already gets. `Precision` carries
    the same default, so untouched constructions (`build_hf_model`'s fallback, the tests) resolve identically."""
    assert GRPOConfig().mixed_precision_mode == ""
    assert resolve_precision(_root()).mode == "fsdp"
    assert Precision(master="bfloat16", param="bfloat16", reduce="float32", transfer="bfloat16").mode == "fsdp"
    assert "mixed_precision_mode" not in {f.name for f in dataclasses.fields(DPOConfig)}


@pytest.mark.L0
@pytest.mark.CPU
def test_mixed_precision_mode_autocast_resolves_and_bad_values_fail_at_config_load():
    assert resolve_precision(_root(GRPOConfig(mixed_precision_mode="autocast"))).mode == "autocast"
    with pytest.raises(ValueError, match="mixed_precision_mode"):
        GRPOConfig(mixed_precision_mode="amp")


@pytest.mark.L0
@pytest.mark.CPU
def test_fp16_autocast_is_refused_at_resolution():
    """autocast without a GradScaler underflows fp16 gradients. The check sits in `resolve_precision`, not
    `__post_init__`, because an unset `param_dtype` inherits `model_hf.dtype`: only the resolved value is
    known to be fp16."""
    with pytest.raises(ValueError, match="float16"):
        resolve_precision(_root(GRPOConfig(mixed_precision_mode="autocast", param_dtype="float16")))
    with pytest.raises(ValueError, match="float16"):
        resolve_precision(_root(GRPOConfig(mixed_precision_mode="autocast"), HFModelConfig(dtype="float16")))
