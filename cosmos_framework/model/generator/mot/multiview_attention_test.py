# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Which multiview attention a config and a host resolve to between them.

The decision is a config-level one taken once per run, so these are CPU tests: the only thing
the device is consulted for is whether FA4 is usable, and a CPU device stands in for any host
where it is not. What each backend then computes is covered by ``flex_attention_test`` for the
mask and ``attention_test`` for the folds.
"""

import pytest
import torch

from cosmos_framework.configs.base.defaults.multiview_attention import (
    AttentionScope,
    MultiviewAttentionConfig,
    MultiviewAttentionMaskConfig,
)
from cosmos_framework.model.generator.mot.flex_attention import triton_backend_block_size
from cosmos_framework.model.generator.mot.multiview_attention import resolve_multiview_backend
from cosmos_framework.model.generator.mot.multiview_maskless_attention import (
    MASKLESS_ATTENTION_SCOPES,
    maskless_unavailable_reason,
)


def _config(scope: AttentionScope, **mask_kwargs) -> MultiviewAttentionConfig:
    """A config "maskless" can serve, unless a keyword here takes that away.

    Whether multiview attention runs at all is the pathway's business, not this config's, so
    there is nothing here to turn on -- only the description of how its GEN pass runs.
    """
    return MultiviewAttentionConfig(mask=MultiviewAttentionMaskConfig(attention_scope=scope, **mask_kwargs))


@pytest.mark.L0
def test_resolve_multiview_backend_auto_never_takes_the_folds() -> None:
    """ "auto" ranks the masks first, so it chooses kernels and never which attention runs.

    The config here is one the folds *could* serve -- decomposed scope, no window -- which is
    exactly the case where the old folds-first ordering would have switched what the run trains.
    Triton always resolves, so the folds rank last and are never reached: "maskless" is opt-in,
    by name.
    """
    backend, geometry = resolve_multiview_backend(torch.device("cpu"), "auto", config=_config("decomposed"))

    assert backend == "flex_triton"
    assert geometry is not None, "A mask carries the block it is built at."


@pytest.mark.L0
def test_resolve_multiview_backend_auto_takes_a_mask_when_the_config_rules_maskless_out() -> None:
    """The verdict changes nothing under "auto", which was taking a mask either way."""
    backend, _ = resolve_multiview_backend(
        torch.device("cpu"), "auto", config=_config("decomposed", decomposed_temporal_window_seconds=0.4)
    )

    assert backend == "flex_triton"


@pytest.mark.L0
def test_resolve_multiview_backend_demanding_maskless_reports_why_the_config_rules_it_out() -> None:
    """Pinning "maskless" and silently getting a mask would train a different distribution."""
    with pytest.raises(ValueError, match="asks for the maskless folds, but decomposed_temporal_window_seconds"):
        resolve_multiview_backend(
            torch.device("cpu"), "maskless", config=_config("decomposed", decomposed_temporal_window_seconds=0.4)
        )


@pytest.mark.L0
def test_resolve_multiview_backend_demanding_flash_reports_why_it_is_unavailable() -> None:
    """Availability of FA4 is the host's answer, and pinning it wants the reason."""
    with pytest.raises(ValueError, match="FlashAttention-4 backend, but .*CUDA device"):
        resolve_multiview_backend(torch.device("cpu"), "flex_flash", config=_config("all_views"))


@pytest.mark.L0
def test_resolve_multiview_backend_pins_a_mask_even_where_maskless_is_available() -> None:
    """An explicit flex backend is a choice of attention, not merely of kernels."""
    backend, geometry = resolve_multiview_backend(torch.device("cpu"), "flex_triton", config=_config("decomposed"))

    assert backend == "flex_triton"
    # A mask does need its geometry: the block it is built at, and the padding that block wants.
    assert geometry is not None
    assert geometry.block_size == triton_backend_block_size()


@pytest.mark.L0
def test_resolve_multiview_backend_rejects_an_unknown_preference() -> None:
    with pytest.raises(ValueError, match="Unknown multiview attention backend 'MASKLESS'"):
        resolve_multiview_backend(torch.device("cpu"), "MASKLESS", config=_config("decomposed"))


@pytest.mark.L0
@pytest.mark.parametrize("scope", MASKLESS_ATTENTION_SCOPES)
def test_maskless_is_available_for_the_scopes_the_folds_express(scope: AttentionScope) -> None:
    """Both are a partition of the GEN stream, which is what an unmasked pass needs."""
    assert maskless_unavailable_reason(_config(scope)) is None


@pytest.mark.L0
def test_maskless_is_unavailable_for_all_views() -> None:
    """The default scope, and the one a "maskless" config is most likely to land on by accident.

    Without this the folds would run and quietly ignore the scope, training the decomposed
    pattern under a config that asked for the full square -- the silent substitution every other
    condition here refuses.
    """
    reason = maskless_unavailable_reason(_config("all_views"))

    assert reason is not None
    assert "all_views" in reason


@pytest.mark.L0
def test_auto_takes_a_mask_for_a_scope_the_folds_do_not_express() -> None:
    """The default scope keeps its mask, as every config does under "auto"."""
    backend, _ = resolve_multiview_backend(torch.device("cpu"), "auto", config=_config("all_views"))

    assert backend == "flex_triton"


@pytest.mark.L0
def test_demanding_maskless_under_all_views_reports_the_scope_as_the_reason() -> None:
    with pytest.raises(ValueError, match="all_views"):
        resolve_multiview_backend(torch.device("cpu"), "maskless", config=_config("all_views"))


@pytest.mark.L0
@pytest.mark.parametrize("scope", MASKLESS_ATTENTION_SCOPES)
@pytest.mark.parametrize("control_attends_sensor", [True, False])
def test_maskless_serves_either_control_rule(scope: AttentionScope, control_attends_sensor: bool) -> None:
    """The flag used to rule the folds out and no longer does, at either scope.

    A control item shares its target's view group, so with the flag off a control query needs a
    narrower key set than a sensor query on the same view -- which one varlen segment over that
    group cannot give. The folds cut the group into two segments instead, a sensor one keyed
    against the whole group and a control one keyed against its control tokens, so both values
    of the flag are an unmasked pass. See ``build_multiview_maskless_plan``.
    """
    config = _config(scope, control_attends_sensor=control_attends_sensor)

    assert maskless_unavailable_reason(config) is None


@pytest.mark.L0
def test_auto_keeps_a_mask_without_control_attends_sensor() -> None:
    """The default config -- all_views, flag off -- is a mask run, as any config is under "auto".

    The scope is what keeps it one now; the flag no longer rules anything out on its own.
    """
    backend, _ = resolve_multiview_backend(
        torch.device("cpu"),
        "auto",
        config=MultiviewAttentionConfig(),
    )

    assert backend == "flex_triton"


@pytest.mark.L0
def test_demanding_maskless_without_control_attends_sensor_is_served() -> None:
    """Pinning "maskless" with the flag off is a config the folds express, not a refusal."""
    backend, geometry = resolve_multiview_backend(
        torch.device("cpu"),
        "maskless",
        config=_config("decomposed", control_attends_sensor=False),
    )

    assert backend == "maskless"
    # The folds build no mask, so they carry no block geometry -- with the flag off as without.
    assert geometry is None
