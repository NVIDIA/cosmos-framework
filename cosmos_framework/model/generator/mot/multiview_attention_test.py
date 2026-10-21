# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Which multiview attention a config and a host resolve to between them.

The decision is a config-level one taken once per run, so these are CPU tests: the only thing
the device is consulted for is whether FA4 is usable, and a CPU device stands in for any host
where it is not. What each backend then computes is covered by ``flex_attention_test`` for the
mask and ``attention_test`` for the folds.
"""

from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

from cosmos_framework.utils.lazy_config import LazyCall, instantiate
from cosmos_framework.configs.base.defaults.multiview_attention import (
    AttentionScope,
    CrossViewTemporalWindowConfig,
    MultiviewAttentionConfig,
    MultiviewAttentionMaskConfig,
    SameViewTemporalWindowConfig,
    TemporalWindow,
)
from cosmos_framework.model.generator.mot import multiview_attention as multiview_module
from cosmos_framework.model.generator.mot.flex_attention import FlexBackend, triton_backend_block_size
from cosmos_framework.model.generator.mot.multiview_attention import resolve_multiview_backend
from cosmos_framework.model.generator.mot.multiview_maskless_attention import (
    MASKLESS_ATTENTION_SCOPES,
    maskless_unavailable_reason,
)

pytestmark = pytest.mark.CPU


def _config(scope: AttentionScope, **mask_kwargs: Any) -> MultiviewAttentionConfig:
    """A config "maskless" can serve, unless a keyword here takes that away.

    Whether multiview attention runs at all is the pathway's business, not this config's, so
    there is nothing here to turn on -- only the description of how its GEN pass runs.
    """
    return MultiviewAttentionConfig(mask=MultiviewAttentionMaskConfig(attention_scope=scope, **mask_kwargs))


@pytest.mark.L0
@pytest.mark.parametrize("scope", MASKLESS_ATTENTION_SCOPES)
@pytest.mark.parametrize("window", [None, (-0.4, 0.0)])
def test_resolve_multiview_backend_auto_prefers_maskless(
    monkeypatch: pytest.MonkeyPatch, scope: AttentionScope, window: TemporalWindow | None
) -> None:
    """A supported config uses exact maskless attention even when FA4 is available."""
    monkeypatch.setattr(multiview_module, "flash_backend_unavailable_reason", lambda device: None)
    backend, geometry = resolve_multiview_backend(
        torch.device("cpu"),
        "auto",
        config=_config(scope, cross_view_temporal_window=CrossViewTemporalWindowConfig(bounds=window)),
    )

    assert backend == "maskless"
    assert geometry is None


@pytest.mark.L0
@pytest.mark.parametrize("flash_available", [True, False])
def test_resolve_multiview_backend_auto_falls_back_to_flex(
    monkeypatch: pytest.MonkeyPatch, flash_available: bool
) -> None:
    """Unsupported scopes prefer FA4, then Triton."""
    monkeypatch.setattr(
        multiview_module, "flash_backend_unavailable_reason", lambda device: None if flash_available else "no FA4"
    )
    resolved_preferences: list[str] = []
    geometry = multiview_module.resolve_flex_backend(torch.device("cpu"), "flex_triton")

    def resolve_geometry(device: torch.device, preference: str) -> FlexBackend:
        resolved_preferences.append(preference)
        return geometry

    monkeypatch.setattr(multiview_module, "resolve_flex_backend", resolve_geometry)
    backend, actual_geometry = resolve_multiview_backend(torch.device("cpu"), "auto", config=_config("all_views"))
    expected = "flex_flash" if flash_available else "flex_triton"
    assert backend == expected
    assert resolved_preferences == [expected]
    assert actual_geometry is geometry


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("window", [(0.0, 0.0), (-0.1, 0.0), (-0.4, 0.0), (-1.0, 0.0), (-0.4, 0.4), (-0.4, 0.2)])
def test_maskless_accepts_temporal_window(window: TemporalWindow) -> None:
    """Pinning maskless preserves the requested temporal window without a fallback."""
    config = _config("decomposed", cross_view_temporal_window=CrossViewTemporalWindowConfig(bounds=window))
    assert maskless_unavailable_reason(config) is None
    backend, geometry = resolve_multiview_backend(torch.device("cpu"), "maskless", config=config)
    assert backend == "maskless"
    assert geometry is None


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("window", [None, (0.0, 0.0), (-0.4, 0.0), (-0.4, 0.4), (-0.4, 0.2)])
def test_temporal_window_config_roundtrips_through_yaml(window: TemporalWindow | None) -> None:
    # Spell out nested config nodes for YAML without resolving arbitrary attrs factories.
    config = LazyCall(MultiviewAttentionMaskConfig)(
        cross_view_temporal_window=LazyCall(CrossViewTemporalWindowConfig)(bounds=window),
        same_view_temporal_window=LazyCall(SameViewTemporalWindowConfig)(),
    )
    restored = instantiate(OmegaConf.create(OmegaConf.to_yaml(config)))
    assert restored.cross_view_temporal_window.bounds == window
    assert restored.same_view_temporal_window == SameViewTemporalWindowConfig()


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("window", [None, (0.0, 0.0), (-0.4, 0.0), (-0.2, 0.2), (-0.4, 0.2)])
def test_temporal_window_structured_config_roundtrips(window: TemporalWindow | None) -> None:
    # Production configs embed attrs instances, not only LazyCall dictionaries.
    # OmegaConf validates every field annotation here, including unused defaults.
    config = OmegaConf.structured(MultiviewAttentionConfig(), flags={"allow_objects": True})
    config = OmegaConf.merge(config, {"mask": {"cross_view_temporal_window": {"bounds": window}}})
    restored = OmegaConf.to_object(OmegaConf.merge(config, OmegaConf.create(OmegaConf.to_yaml(config))))
    # Reassigning restored bounds must not accumulate the runtime comparison tolerance.
    restored.mask.cross_view_temporal_window.bounds = restored.mask.cross_view_temporal_window.bounds
    assert restored.mask.cross_view_temporal_window.bounds == window


@pytest.mark.L0
def test_nested_temporal_windows_roundtrip_fractional_seconds() -> None:
    config = OmegaConf.structured(MultiviewAttentionConfig(), flags={"allow_objects": True})
    cross_view = {"bounds": [-0.25, 0.125], "includes_first_frame": True}
    same_view = {
        "sensor_to_sensor_bounds": [-0.15, 0.15],
        "sensor_to_control_bounds": [-0.75, 0.0],
        "control_to_control_bounds": [-0.625, 0.625],
        "control_to_sensor_bounds": [-0.125, 0.0],
    }
    config = OmegaConf.merge(
        config, {"mask": {"cross_view_temporal_window": cross_view, "same_view_temporal_window": same_view}}
    )
    restored = OmegaConf.to_object(OmegaConf.merge(config, OmegaConf.create(OmegaConf.to_yaml(config))))

    assert isinstance(restored.mask.cross_view_temporal_window, CrossViewTemporalWindowConfig)
    assert restored.mask.cross_view_temporal_window.bounds == (-0.25, 0.125)
    assert restored.mask.cross_view_temporal_window.includes_first_frame
    assert isinstance(restored.mask.same_view_temporal_window, SameViewTemporalWindowConfig)
    for field_name, bounds in same_view.items():
        setattr(
            restored.mask.same_view_temporal_window,
            field_name,
            getattr(restored.mask.same_view_temporal_window, field_name),
        )
        assert getattr(restored.mask.same_view_temporal_window, field_name) == tuple(bounds)


@pytest.mark.L0
def test_nested_temporal_window_defaults_are_independent() -> None:
    first = MultiviewAttentionConfig()
    second = MultiviewAttentionConfig()
    first.mask.cross_view_temporal_window.bounds = (-0.25, 0.0)
    first.mask.cross_view_temporal_window.includes_first_frame = True
    first.mask.same_view_temporal_window.sensor_to_sensor_bounds = (-0.5, 0.5)

    assert second.mask.cross_view_temporal_window.bounds is None
    assert not second.mask.cross_view_temporal_window.includes_first_frame
    assert second.mask.same_view_temporal_window.sensor_to_sensor_bounds is None


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("window", [(0.1, -0.1), (-0.4, float("nan")), (-float("inf"), 0.0), (-0.4,), (-0.4, 0.0, 0.4)])
def test_temporal_window_rejects_invalid_bounds(window: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="Temporal window"):
        _config("decomposed", cross_view_temporal_window=CrossViewTemporalWindowConfig(bounds=window))


@pytest.mark.L0
@pytest.mark.CPU
def test_cross_view_temporal_window_rejects_scalar_duration() -> None:
    with pytest.raises(ValueError, match="requires two finite bounds"):
        CrossViewTemporalWindowConfig(bounds=0.4)


@pytest.mark.L0
@pytest.mark.parametrize("window", [(-0.8, 0.8), (-1.6, 0.0)])
def test_attention_accepts_supported_neighborhood_windows(window: TemporalWindow) -> None:
    """Centered ``(-N, N)`` and past-looking ``(-N, 0)`` windows are supported."""
    config = _config(
        "decomposed", same_view_temporal_window=SameViewTemporalWindowConfig(sensor_to_sensor_bounds=window)
    )

    assert maskless_unavailable_reason(config) is None
    assert resolve_multiview_backend(torch.device("cpu"), "maskless", config=config) == ("maskless", None)


@pytest.mark.L0
@pytest.mark.parametrize(
    "field_name",
    (
        "sensor_to_sensor_bounds",
        "sensor_to_control_bounds",
        "control_to_control_bounds",
        "control_to_sensor_bounds",
    ),
)
def test_same_view_window_experiment_overrides_accept_fractional_seconds(field_name: str) -> None:
    """Each directional edge accepts the same two-bound list representation."""
    window = SameViewTemporalWindowConfig(**{field_name: [-0.8, 0.8]})

    assert getattr(window, field_name) == (-0.8, 0.8)


@pytest.mark.L0
@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("-0.8:+0.8", "two finite bounds"),  # String instead of a sequence.
        ("full", "two finite bounds"),  # String instead of None.
        ([-0.8], "two finite bounds"),  # Missing the upper bound.
        ([-0.8, float("inf")], "two finite bounds"),  # Non-finite upper bound.
        ([0.8, -0.8], "start <= end"),  # Reversed interval.
    ],
)
def test_same_view_window_experiment_overrides_reject_malformed_values(value: object, message: str) -> None:
    """Malformed values fail with an error that identifies the violated contract."""
    with pytest.raises((TypeError, ValueError), match=message):
        SameViewTemporalWindowConfig(sensor_to_control_bounds=value)


@pytest.mark.L0
def test_attention_rejects_unsupported_asymmetric_neighborhood_window() -> None:
    # (-1.2, 0.3) seconds is a valid inclusive temporal window, but the current NATTEN call
    # can describe a neighborhood only with its size and a causal flag. That
    # represents centered (-N, N) and causal (-N, 0) windows, not a window with
    # an independently shifted anchor. Reject it instead of silently attending
    # to a different set of frames.
    config = _config(
        "decomposed",
        same_view_temporal_window=SameViewTemporalWindowConfig(sensor_to_sensor_bounds=(-1.2, 0.3)),
    )

    assert "not centered" in str(maskless_unavailable_reason(config))
    with pytest.raises(ValueError, match="not centered"):
        resolve_multiview_backend(torch.device("cpu"), "maskless", config=config)


@pytest.mark.L0
@pytest.mark.parametrize("backend", ["flex_triton", "flex_flash"])
def test_flex_attention_rejects_same_view_temporal_windows(backend: str) -> None:
    """Every FlexAttention selection fails rather than silently ignoring a window."""
    config = _config(
        "decomposed",
        same_view_temporal_window=SameViewTemporalWindowConfig(sensor_to_control_bounds=(-0.4, 0.4)),
    )

    with pytest.raises(NotImplementedError, match="not implemented for FlexAttention"):
        resolve_multiview_backend(torch.device("cpu"), backend, config=config)


@pytest.mark.L0
def test_resolve_multiview_backend_demanding_flash_reports_why_it_is_unavailable() -> None:
    """Availability of FA4 is the host's answer, and pinning it wants the reason."""
    with pytest.raises(ValueError, match="FlashAttention-4 backend, but .*CUDA device"):
        resolve_multiview_backend(torch.device("cpu"), "flex_flash", config=_config("all_views"))


@pytest.mark.L0
def test_resolve_multiview_backend_pins_a_mask_even_where_maskless_is_available() -> None:
    """An explicit flex backend remains honored when maskless supports the same config."""
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
    """The all_views scope uses a Flex mask because maskless cannot express it."""
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
    """The default config -- all_views, flag off -- requires a Flex mask.

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
def test_auto_serves_supported_same_view_temporal_windows() -> None:
    config = _config(
        "decomposed",
        same_view_temporal_window=SameViewTemporalWindowConfig(sensor_to_control_bounds=(-0.4, 0.4)),
    )

    assert resolve_multiview_backend(torch.device("cpu"), "auto", config=config) == ("maskless", None)


@pytest.mark.L0
def test_auto_serves_first_frame_temporal_window() -> None:
    config = _config(
        "decomposed",
        cross_view_temporal_window=CrossViewTemporalWindowConfig(bounds=(-0.4, 0.0), includes_first_frame=True),
    )

    assert resolve_multiview_backend(torch.device("cpu"), "auto", config=config) == ("maskless", None)


@pytest.mark.L0
def test_auto_rejects_same_view_windows_when_no_backend_supports_them() -> None:
    config = _config(
        "decomposed",
        same_view_temporal_window=SameViewTemporalWindowConfig(sensor_to_control_bounds=(-0.4, 0.2)),
    )

    with pytest.raises(NotImplementedError, match="not implemented for FlexAttention"):
        resolve_multiview_backend(torch.device("cpu"), "auto", config=config)
