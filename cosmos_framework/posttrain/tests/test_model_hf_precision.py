# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""L0/GPU tests for the storage-vs-compute dtype split in `HFModel` (`config/rl.py::HFRunConfig`).

The schema and resolver are covered torch-free in `test_config_precision.py`. What can ONLY be
checked against a real model is the mechanism itself:

  1. the default path still stores parameters in `dtype` (pre-split behaviour, bit-exact);
  2. `master_dtype=float32` with `dtype=bfloat16` actually yields fp32 parameters AND a bf16
     `ShardPlan.param_dtype` — the combination the single-knob world could not express;
  3. `load_weights` stripes the checkpoint's **bf16** safetensors into **fp32** shards without
     corrupting them. This was the flagged risk of A2: the loader could have assumed matching
     dtypes, and a silent mis-cast would look like a working model that trains from garbage.

`Qwen/Qwen2.5-0.5B-Instruct` is used because it is small, already cached in the training image, and
ships bf16 weights — so case 3 is a genuine cross-dtype load rather than a no-op copy.
"""

from __future__ import annotations

import pytest

from cosmos_framework.utils.helper_test import RunIf
from cosmos_framework.posttrain.config.build import Precision
from cosmos_framework.posttrain.config.model import HFModelConfig
from cosmos_framework.posttrain.model.hf import build_hf_model

_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _cfg() -> HFModelConfig:
    # eager attention: the GRPO stack's default, and it avoids a flash-attn dependency here.
    return HFModelConfig(name_or_path=_MODEL, dtype="bfloat16", attn_impl="eager")


def _param_dtypes(model) -> set:
    return {p.dtype for p in model.net.parameters() if p.is_floating_point()}


@pytest.mark.L0
@pytest.mark.GPU
@RunIf(min_gpus=1)
def test_default_path_stores_params_in_dtype_unchanged():
    """`precision=None` ⇒ storage == compute == `hf_cfg.dtype`, exactly as before the split.

    This is the no-op guarantee every existing recipe and every untouched call site relies on.
    """
    import torch

    model = build_hf_model(_cfg(), pdims=None)
    assert _param_dtypes(model) == {torch.bfloat16}
    assert model.shard_plan.param_dtype is torch.bfloat16
    assert model.shard_plan.reduce_dtype is torch.float32


@pytest.mark.L0
@pytest.mark.GPU
@RunIf(min_gpus=1)
def test_master_dtype_drives_storage_independently_of_hf_cfg_dtype():
    """`master_dtype` — not `model_hf.dtype` — decides what parameters are STORED in.

    That is the whole point of the split: storage sets the optimizer's moment dtype
    (`torch.optim.AdamW` allocates state with `zeros_like(p)`), which is what apex's
    `master_weights` was previously emulating.

    Uses master == param == float32 so the mesh guard does not fire: `param_dtype` is only
    deliverable under an FSDP wrap, and this test runs unwrapped (`pdims=None`). The
    storage-vs-compute SPLIT is exercised by the resolver tests and by the guard test below;
    what is checked here is that `master_dtype` overrides `_cfg().dtype` ("bfloat16").
    """
    import torch

    model = build_hf_model(
        _cfg(),
        pdims=None,
        precision=Precision(master="float32", param="float32", reduce="float32", transfer="float32"),
    )
    assert _param_dtypes(model) == {torch.float32}, "parameters must be STORED in master_dtype"
    assert model.shard_plan.param_dtype is torch.float32
    assert model.shard_plan.reduce_dtype is torch.float32


@pytest.mark.L0
@pytest.mark.GPU
@RunIf(min_gpus=1)
def test_bf16_checkpoint_loads_into_fp32_shards_without_corruption():
    """`load_weights` must stripe bf16 safetensors into fp32 parameters losslessly.

    bf16 -> fp32 is an EXACT widening (every bf16 value is representable in fp32), so the fp32
    build cast back down must equal the bf16 build bit-for-bit. Anything else — a truncated cast,
    a reinterpreted buffer, an untouched random init — shows up here as a mismatch.
    """
    import torch

    bf16 = build_hf_model(_cfg(), pdims=None)
    fp32 = build_hf_model(
        _cfg(),
        pdims=None,
        # master == param so the unwrapped-mesh guard does not fire; the point here is the LOAD
        # (bf16 safetensors -> fp32 parameters), which is independent of the compute dtype.
        precision=Precision(master="float32", param="float32", reduce="float32", transfer="float32"),
    )
    a = dict(bf16.net.named_parameters())
    b = dict(fp32.net.named_parameters())
    assert a.keys() == b.keys() and a, "both builds must expose the same (non-empty) parameter set"

    checked = 0
    for name, p_bf16 in a.items():
        if not p_bf16.is_floating_point():
            continue
        p_fp32 = b[name]
        assert torch.equal(p_fp32.to(torch.bfloat16), p_bf16.to(p_fp32.device)), (
            f"{name}: fp32-stored weights disagree with the bf16 build — the checkpoint was "
            f"mis-cast on the way into fp32 shards"
        )
        checked += 1
    assert checked > 0

    # Non-zero somewhere: an all-zero match would satisfy the loop above while proving nothing
    # (two models that both failed to load are trivially equal).
    assert any(p.abs().sum().item() > 0 for p in b.values())


@pytest.mark.L0
@pytest.mark.GPU
@RunIf(min_gpus=1)
def test_unsatisfiable_param_dtype_is_refused_without_an_fsdp_wrap():
    """`param_dtype` != `master_dtype` on an unwrapped mesh must RAISE, not silently downgrade.

    `param_dtype` is delivered only by `MixedPrecisionPolicy`, which `model/parallelize.py` attaches
    inside `fully_shard` — and that returns early when `pdims is None` or the mesh is
    `dp_shard=1, dp_replicate=1`. So with no wrapper there is no cast, and the forward runs in the
    STORAGE dtype.

    This test originally PINNED that as a defect: it asserted the fp32-storage build produced
    `torch.float32` logits despite `param_dtype=bfloat16`, i.e. the requested compute dtype was
    ignored. Measured, that is a real precision difference against a reference that wraps
    unconditionally (cosmos-rl does), and it is invisible whenever storage == compute — which is
    every recipe that leaves `rl.master_dtype` unset.

    It is refused in mode "fsdp" rather than emulated: `torch.autocast` applies a PER-OP policy
    (layernorm/softmax stay fp32) where FSDP casts the PARAMETER, so silently substituting one for the
    other would make the same config compute differently at dp_shard=1 than at dp_shard=2. The explicit
    alternative is `mixed_precision_mode="autocast"` (next test), which means the same thing at every
    world size.
    """
    with pytest.raises(ValueError, match="cannot be honoured without an FSDP wrap"):
        build_hf_model(
            _cfg(),
            pdims=None,
            precision=Precision(master="float32", param="bfloat16", reduce="float32", transfer="bfloat16", mode="fsdp"),
        )


@pytest.mark.L0
@pytest.mark.GPU
@RunIf(min_gpus=1)
def test_autocast_mode_builds_unwrapped_with_fp32_storage_and_bf16_compute():
    """`mode="autocast"`: parameters stay in `master_dtype`, `ShardPlan.param_dtype` is `None` (the FSDP wrap
    delivers no compute dtype and casts neither parameters nor forward inputs, so wrapped == unwrapped at every
    world size), and the forward runs under `torch.autocast(param_dtype)` — the fp32-master/bf16-compute recipe on
    ONE GPU, which mode "fsdp" refuses above.
    """
    import torch

    model = build_hf_model(
        _cfg(),
        pdims=None,
        precision=Precision(master="float32", param="bfloat16", reduce="float32", transfer="bfloat16", mode="autocast"),
    )
    assert _param_dtypes(model) == {torch.float32}
    assert model.shard_plan.param_dtype is None
    assert model.shard_plan.reduce_dtype is torch.float32
    ids = torch.tensor([[1, 2, 3, 4, 5]], device="cuda")
    with torch.no_grad():
        # Through `forward`, where the context is applied; `model.net(...)` would bypass it.
        out = model(input_ids=ids)
    assert out.dtype is torch.bfloat16 and torch.isfinite(out).all()


@pytest.mark.L0
@pytest.mark.GPU
@RunIf(min_gpus=1)
def test_matched_storage_and_compute_still_builds_unwrapped():
    """The guard must fire ONLY on the unsatisfiable combination.

    storage == compute is the overwhelmingly common case (every recipe with `rl.master_dtype`
    unset), and there is nothing to cast, so the unwrapped path stays legal and unchanged.
    """
    import torch

    ids = torch.tensor([[1, 2, 3, 4, 5]], device="cuda")
    model = build_hf_model(
        _cfg(),
        pdims=None,
        precision=Precision(master="bfloat16", param="bfloat16", reduce="float32", transfer="bfloat16"),
    )
    with torch.no_grad():
        out = model.net(ids).logits
    assert out.dtype is torch.bfloat16 and torch.isfinite(out).all()


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("declared", ["float32", "bfloat16"])
def test_precision_fallback_honours_the_config_reduce_dtype(monkeypatch, declared: str) -> None:
    """Without a `Precision`, the reduce dtype comes from the CONFIG, not from a hardcode.

    The previous `build_hf_model` passed no `reduce_dtype` at all, so every caller got `HFModel`'s
    default fp32 and `HFModelConfig.fsdp_reduce_dtype` was a dead knob -- a field a recipe could set
    with no effect. Folding `precision` in changed that, and the change is deliberate rather than
    incidental, so both halves are pinned here: `"float32"` reproduces the old behaviour exactly,
    and a non-default value now reaches the model.

    Asserting on the kwargs `HFModel` is CONSTRUCTED with keeps this on CPU -- the mechanism is the
    handoff, and `test_master_dtype_*` already covers what the model then does with it on GPU.
    """
    import torch

    from cosmos_framework.posttrain.model import hf as hf_mod

    seen: dict = {}

    class _Recording:
        def __init__(self, *args, **kwargs):
            seen.update(kwargs)

        def setup(self, pdims):
            del pdims

    monkeypatch.setattr(hf_mod, "HFModel", _Recording)
    cfg = HFModelConfig(name_or_path=_MODEL, dtype="bfloat16", attn_impl="eager", fsdp_reduce_dtype=declared)
    hf_mod.build_hf_model(cfg)

    assert seen["reduce_dtype"] == getattr(torch, declared), (
        f"the config declared fsdp_reduce_dtype={declared!r}; the model was built with {seen['reduce_dtype']}"
    )
    # master == param on this path, which is what the single-knob world did.
    assert seen["master_dtype"] == seen["dtype"] == torch.bfloat16


@pytest.mark.L0
@pytest.mark.CPU
def test_explicit_precision_overrides_the_config_dtypes(monkeypatch) -> None:
    """A resolved `Precision` wins over `hf_cfg`'s deprecated single knob, on every axis."""
    import torch

    from cosmos_framework.posttrain.model import hf as hf_mod

    seen: dict = {}

    class _Recording:
        def __init__(self, *args, **kwargs):
            seen.update(kwargs)

        def setup(self, pdims):
            del pdims

    monkeypatch.setattr(hf_mod, "HFModel", _Recording)
    cfg = HFModelConfig(name_or_path=_MODEL, dtype="bfloat16", attn_impl="eager", fsdp_reduce_dtype="bfloat16")
    hf_mod.build_hf_model(
        cfg, None, Precision(master="float32", param="bfloat16", reduce="float32", transfer="bfloat16")
    )

    assert seen["master_dtype"] == torch.float32  # STORAGE, from rl.master_dtype
    assert seen["dtype"] == torch.bfloat16  # COMPUTE, from rl.param_dtype
    assert seen["reduce_dtype"] == torch.float32  # beats the config's "bfloat16"
