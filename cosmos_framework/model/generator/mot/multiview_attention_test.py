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
from omegaconf import OmegaConf

from cosmos_framework.utils.lazy_config import LazyCall, instantiate
from cosmos_framework.configs.base.defaults.multiview_attention import (
    AttentionScope,
    MultiviewAttentionConfig,
    MultiviewAttentionMaskConfig,
    TemporalWindow,
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
def test_resolve_multiview_backend_auto_keeps_a_mask_with_a_window() -> None:
    """Supporting windows does not change auto's preference for exact masked attention."""
    backend, _ = resolve_multiview_backend(
        torch.device("cpu"), "auto", config=_config("decomposed", decomposed_temporal_window_seconds=(-0.4, 0.0))
    )

    assert backend == "flex_triton"


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("window", [(0.0, 0.0), (-0.1, 0.0), (-0.4, 0.0), (-1.0, 0.0), (-0.4, 0.4), (-0.4, 0.2)])
@pytest.mark.parametrize("deduplicate_cross_view", [True, False])
def test_maskless_accepts_temporal_window(window: TemporalWindow, deduplicate_cross_view: bool) -> None:
    """Pinning maskless must preserve its requested counting mode rather than fall back."""
    config = _config("decomposed", decomposed_temporal_window_seconds=window)
    config.deduplicate_cross_view = deduplicate_cross_view
    assert maskless_unavailable_reason(config) is None
    backend, geometry = resolve_multiview_backend(torch.device("cpu"), "maskless", config=config)
    assert backend == "maskless"
    assert geometry is None


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("window", [None, 0, 0.4, 1, (-0.4, 0.0), (-0.4, 0.4), (-0.4, 0.2)])
def test_temporal_window_config_roundtrips_through_yaml(window: TemporalWindow | float | None) -> None:
    config = LazyCall(MultiviewAttentionMaskConfig)(decomposed_temporal_window_seconds=window)
    restored = instantiate(OmegaConf.create(OmegaConf.to_yaml(config)))
    expected = (-window, 0.0) if isinstance(window, (int, float)) else window
    assert restored.decomposed_temporal_window_seconds == expected


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("window", [None, 0, 0.4, 1, (-0.4, 0.0), (-0.2, 0.2), (-0.4, 0.2)])
def test_temporal_window_structured_config_roundtrips(window: TemporalWindow | float | None) -> None:
    # Production configs embed attrs instances, not only LazyCall dictionaries.
    # OmegaConf validates every field annotation here, including unused defaults.
    config = OmegaConf.structured(MultiviewAttentionConfig(), flags={"allow_objects": True})
    config = OmegaConf.merge(config, {"mask": {"decomposed_temporal_window_seconds": window}})
    restored = OmegaConf.to_object(OmegaConf.merge(config, OmegaConf.create(OmegaConf.to_yaml(config))))
    expected = (-window, 0.0) if isinstance(window, (int, float)) else window
    assert restored.mask.decomposed_temporal_window_seconds == expected


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("window", [(0.1, -0.1), (-0.4, float("nan")), (-float("inf"), 0.0), (-0.4,), (-0.4, 0.0, 0.4)])
def test_temporal_window_rejects_invalid_bounds(window: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="decomposed_temporal_window_seconds"):
        _config("decomposed", decomposed_temporal_window_seconds=window)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("window", [-0.1, float("nan"), float("inf")])
def test_temporal_window_rejects_invalid_legacy_config(window: float) -> None:
    with pytest.raises(ValueError, match="legacy scalar must be finite and non-negative"):
        _config("decomposed", decomposed_temporal_window_seconds=window)


@pytest.mark.L0
@pytest.mark.parametrize("window", [(-8, 8), (-16, 0)])
def test_attention_accepts_supported_neighborhood_windows(window: tuple[int, int]) -> None:
    """Centered ``(-N, N)`` and past-looking ``(-N, 0)`` windows are supported."""
    config = _config("decomposed", sensor_to_sensor_window=window)

    assert maskless_unavailable_reason(config) is None
    assert resolve_multiview_backend(torch.device("cpu"), "maskless", config=config) == ("maskless", None)


@pytest.mark.L0
@pytest.mark.parametrize(
    "field_name",
    (
        "sensor_to_sensor_window",
        "sensor_to_control_window",
        "control_to_control_window",
        "control_to_sensor_window",
    ),
)
def test_frame_window_experiment_overrides_accept_two_integer_lists(field_name: str) -> None:
    """Each directional edge accepts the same two-bound list representation."""
    mask = MultiviewAttentionMaskConfig(**{field_name: [-8, 8]})

    assert getattr(mask, field_name) == (-8, 8)


@pytest.mark.L0
@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("-8:+8", "must be null or a two-integer list"),  # String instead of a sequence.
        ("full", "must be null or a two-integer list"),  # String instead of None.
        ([-8], "needs exactly two bounds"),  # Missing the upper bound.
        ([-8, 8.0], "bounds must be integers"),  # Float upper bound.
        ([8, -8], "lower bound must not exceed"),  # Reversed interval.
    ],
)
def test_frame_window_experiment_overrides_reject_malformed_values(value: object, message: str) -> None:
    """Malformed values fail with an error that identifies the violated contract."""
    with pytest.raises((TypeError, ValueError), match=message):
        MultiviewAttentionMaskConfig(sensor_to_control_window=value)


@pytest.mark.L0
def test_attention_rejects_unsupported_asymmetric_neighborhood_window() -> None:
    # (-12, 3) is a valid inclusive temporal window, but the current NATTEN call
    # can describe a neighborhood only with its size and a causal flag. That
    # represents centered (-N, N) and causal (-N, 0) windows, not a window with
    # an independently shifted anchor. Reject it instead of silently attending
    # to a different set of frames.
    config = _config("decomposed", sensor_to_sensor_window=(-12, 3))

    assert "not centered" in str(maskless_unavailable_reason(config))
    with pytest.raises(ValueError, match="not centered"):
        resolve_multiview_backend(torch.device("cpu"), "maskless", config=config)


@pytest.mark.L0
@pytest.mark.parametrize("backend", ["auto", "flex_triton", "flex_flash"])
def test_flex_attention_rejects_temporal_frame_windows(backend: str) -> None:
    """Every FlexAttention selection fails rather than silently ignoring a window."""
    config = _config("decomposed", sensor_to_control_window=(-4, 4))

    with pytest.raises(NotImplementedError, match="not implemented for FlexAttention"):
        resolve_multiview_backend(torch.device("cpu"), backend, config=config)


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


@pytest.mark.L0
@pytest.mark.CPU
def test_exact_maskless_is_explicit_and_cannot_be_ignored_by_flex() -> None:
    config = _config("decomposed")
    config.deduplicate_cross_view = True
    assert resolve_multiview_backend(torch.device("cpu"), "maskless", config=config) == ("maskless", None)
    for preference in ("auto", "flex_triton", "flex_flash"):
        with pytest.raises(ValueError, match="requires backend='maskless'"):
            resolve_multiview_backend(torch.device("cpu"), preference, config=config)
