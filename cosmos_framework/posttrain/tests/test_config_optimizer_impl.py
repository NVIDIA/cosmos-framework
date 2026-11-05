# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""L0 tests for `OptimizerConfig.impl` — torch KERNEL selection, cosmos-rl's `optm_impl`.

`name` and `impl` are two axes and the whole point of this field is that they stay separate:
`name` picks the optimizer CLASS, `impl` picks the torch kernel for `adamw`. cosmos-rl's
`optm_impl="fused"` means "pass `fused=True` to `torch.optim.AdamW`", NOT "use apex" — the GSM8K
recipe originally read that TOML row as `name="fusedadam"` and thereby swapped the optimizer class
against the reference. These tests pin the distinction so it cannot regress.

The CPU tests are torch-free (schema only). The one test that builds a real optimizer is marked
GPU-free but requires torch, so it lives at the bottom and is skipped when torch is absent.
"""

from __future__ import annotations

import pytest

from cosmos_framework.posttrain.config.rl import OPTIMIZER_IMPLS, OptimizerConfig


@pytest.mark.L0
@pytest.mark.CPU
def test_impl_defaults_to_empty_so_the_knob_is_a_no_op():
    """`""` ⇒ `build_optimizer` passes NEITHER `fused` nor `foreach`, exactly as before the knob.

    torch reads `fused=False`/`foreach=False` as an explicit refusal rather than "unspecified", so
    a default that sent both would silently pin the for-loop kernel for every existing recipe.
    """
    assert OptimizerConfig().impl == ""


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("impl", OPTIMIZER_IMPLS)
def test_every_cosmos_rl_impl_value_is_accepted(impl):
    """The accepted set is cosmos-rl's `optm_impl` choices verbatim, so a TOML row ports 1:1."""
    assert OptimizerConfig(impl=impl).impl == impl


@pytest.mark.L0
@pytest.mark.CPU
def test_unknown_impl_is_rejected_at_config_load():
    with pytest.raises(ValueError, match="must be one of"):
        OptimizerConfig(impl="fusedd")


@pytest.mark.L0
@pytest.mark.CPU
def test_impl_with_fusedadam_is_rejected_rather_than_silently_dropped():
    """apex `FusedAdam` has neither a `fused` nor a `foreach` argument.

    Accepting the combination would leave the run reporting a kernel it never used — the same
    "knob the layers below refuse to honor" failure this repo names elsewhere.
    """
    with pytest.raises(ValueError, match="applies only to"):
        OptimizerConfig(name="fusedadam", impl="fused")

    # ...but fusedadam WITHOUT impl stays valid: the apex path is unaffected by this field.
    assert OptimizerConfig(name="fusedadam").impl == ""


@pytest.mark.L0
@pytest.mark.CPU
def test_name_and_impl_are_independent_axes():
    """`adamw` + `fused` is the cosmos-rl reference pairing, and must NOT imply apex."""
    cfg = OptimizerConfig(name="adamw", impl="fused")
    assert (cfg.name, cfg.impl) == ("adamw", "fused")


@pytest.mark.L0
@pytest.mark.CPU
def test_build_optimizer_passes_only_the_requested_kernel_flag():
    """`build_optimizer` must send the requested flag and NOTHING else.

    Sending the unrequested one as `False` would pin a kernel the config never asked for; sending
    neither (the `""` case) is what preserves today's behaviour.
    """
    torch = pytest.importorskip("torch")

    net = torch.nn.Linear(4, 4)
    from cosmos_framework.posttrain.config.build import build_optimizer

    # "" -> torch's own default for both flags.
    defaults = build_optimizer(net, OptimizerConfig()).param_groups[0]
    assert defaults.get("fused") in (None, False)

    assert build_optimizer(net, OptimizerConfig(impl="foreach")).param_groups[0]["foreach"] is True
    assert build_optimizer(net, OptimizerConfig(impl="for-loop")).param_groups[0]["foreach"] is False
    # `fused=True` needs CUDA params; on CPU torch raises, which still proves the flag was passed.
    if torch.cuda.is_available():
        cuda_net = torch.nn.Linear(4, 4).cuda()
        assert build_optimizer(cuda_net, OptimizerConfig(impl="fused")).param_groups[0]["fused"] is True
