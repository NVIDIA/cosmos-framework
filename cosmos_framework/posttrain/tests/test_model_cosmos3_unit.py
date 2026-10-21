# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU unit tests for Cosmos3's FlowMatchingPolicy adapter guards."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from cosmos_framework.posttrain.model.cosmos3 import (
    Cosmos3FlowPolicy,
    _Cosmos3FlowCondition,
    validate_cosmos3_cold_lora_inventory,
)


def _flow_condition(
    *,
    tokens: list[int] | None = None,
    is_image: bool = False,
    condition_frames: list[int] | None = None,
    fps: float | None = 24.0,
    resolutions: list[str] | None = None,
) -> _Cosmos3FlowCondition:
    return _Cosmos3FlowCondition(
        gen_data_clean=SimpleNamespace(
            fps_vision=None if fps is None else torch.tensor([fps])  # [B] | None
        ),
        sequence_plans=[SimpleNamespace(condition_frame_indexes_vision=list(condition_frames or []))],
        text_tokens=[tokens or [1, 2, 3]],
        is_image_batch=is_image,
        resolutions=resolutions,
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_cosmos3_flow_policy_requires_matching_preference_context() -> None:
    policy = object.__new__(Cosmos3FlowPolicy)
    winner = _flow_condition()
    policy.assert_same_condition_context(winner, _flow_condition())

    mismatches = [
        (_flow_condition(tokens=[1, 2, 9]), "text_tokens differ"),
        (_flow_condition(is_image=True), "modality mismatch"),
        (_flow_condition(resolutions=["256"]), "resolutions mismatch"),
        (_flow_condition(condition_frames=[0]), "condition_frame_indexes_vision differ"),
        (_flow_condition(fps=30.0), "fps_vision mismatch"),
        (_flow_condition(fps=None), "fps_vision mismatch"),
    ]
    for loser, message in mismatches:
        with pytest.raises(ValueError, match=message):
            policy.assert_same_condition_context(winner, loser)


@pytest.mark.L0
@pytest.mark.CPU
def test_cosmos3_flow_policy_rejects_foreign_condition_type() -> None:
    class _ForeignCondition:
        pass

    policy = object.__new__(Cosmos3FlowPolicy)
    with pytest.raises(TypeError, match="requires _Cosmos3FlowCondition"):
        policy.assert_same_condition_context(_flow_condition(), _ForeignCondition())


@pytest.mark.L0
@pytest.mark.CPU
def test_cosmos3_cold_lora_inventory_accepts_complete_target_only_pairs() -> None:
    validate_cosmos3_cold_lora_inventory(
        source_keys=frozenset({"proj.weight"}),
        target_only_keys=frozenset({"proj.lora_A.weight", "proj.lora_B.weight"}),
    )


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("source_keys", "target_only_keys", "match"),
    [
        pytest.param(
            frozenset({"proj.weight", "proj.lora_A.weight"}),
            frozenset({"proj.lora_B.weight"}),
            "source already contains LoRA",
            id="source-lora",
        ),
        pytest.param(
            frozenset({"proj.weight"}),
            frozenset({"proj.extra.weight"}),
            "non-LoRA target-only",
            id="non-lora",
        ),
        pytest.param(
            frozenset({"proj.weight"}),
            frozenset({"proj.lora_A.weight"}),
            "incomplete target-only LoRA pairs",
            id="incomplete-pair",
        ),
    ],
)
def test_cosmos3_cold_lora_inventory_rejects_unsupported_schemas(
    source_keys: frozenset[str],
    target_only_keys: frozenset[str],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        validate_cosmos3_cold_lora_inventory(
            source_keys=source_keys,
            target_only_keys=target_only_keys,
        )


class _DiffusionForcingOmniMot:
    """Minimal OmniMoT surface whose scheduler returns per-frame noise levels."""

    def __init__(self, latent_frames: int) -> None:
        self.net: torch.nn.Module = torch.nn.Linear(1, 1, bias=False)  # weight: [1,1]
        self.net_ema: None = None
        self._latent_frames: int = latent_frames

    def _get_train_noise_level_vision(
        self,
        *,
        batch_size: int,
        is_image_batch: bool,
        num_vision_latent_frames: list[int],
        resolutions: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:  # returns timesteps: [B,T], sigmas: [B,T]
        del is_image_batch, num_vision_latent_frames, resolutions
        timesteps = torch.zeros(batch_size, self._latent_frames)  # [B,T]
        sigmas = torch.full((batch_size, self._latent_frames), 0.5)  # [B,T]
        return timesteps, sigmas


class _StochasticVisionOmniMot:
    """Tiny OmniMoT surface with a stochastic vision-tokenizer posterior."""

    input_image_key: str = "image"
    input_video_key: str = "video"

    def __init__(self) -> None:
        self.net: torch.nn.Module = torch.nn.Linear(1, 1, bias=False)  # weight: [1,1]
        self.fail_on_value: float | None = None

    def is_image_batch(self, batch: dict[str, Any]) -> bool:
        return self.input_image_key in batch

    def _encode_vision_item(
        self,
        state: torch.Tensor,
        *,
        num_views: int,
    ) -> torch.Tensor:  # state: [1,C,T,H,W], returns [1,C,T,H,W]
        del num_views
        posterior_sample = torch.randn_like(state.float())  # [1,C,T,H,W]
        if self.fail_on_value is not None and float(state.item()) == self.fail_on_value:
            raise RuntimeError("synthetic tokenizer failure")
        return state.float() + posterior_sample  # [1,C,T,H,W]

    def get_data_and_condition(self, batch: dict[str, Any]) -> list[torch.Tensor]:
        media_key = self.input_image_key if self.is_image_batch(batch) else self.input_video_key
        encoded: list[torch.Tensor] = []
        for sample in batch[media_key]:
            items = sample if isinstance(sample, (list, tuple)) else [sample]
            for item in items:
                encoded_item = self._encode_vision_item(
                    item,
                    num_views=1,
                )  # [1,C,T,H,W]
                encoded.append(encoded_item)
        return encoded


class _StatefulValidationOmniMot:
    """Minimal OmniMoT surface containing mutable model/tokenizer validation state."""

    def __init__(self) -> None:
        self.net: torch.nn.Module = torch.nn.Linear(1, 1)
        self.tokenizer_vision_gen: Any = SimpleNamespace(keep_encoder_cache=False)


class _MoeCallbackModule(torch.nn.Module):
    """Minimal inherited Qwen MoE callback surface."""

    def __init__(self) -> None:
        super().__init__()
        self.callback_calls: int = 0

    def _update_moe_callback_stats(self) -> None:
        self.callback_calls += 1


class _TelemetryValidationOmniMot:
    """OmniMoT surface whose encode path drives training-only MoE callback stats."""

    def __init__(self) -> None:
        self.net: _MoeCallbackModule = _MoeCallbackModule()
        self.net_ema: _MoeCallbackModule = _MoeCallbackModule()
        self.net_ema_worker: Any = SimpleNamespace(update_average=lambda *_args, **_kwargs: None)
        self.ema_beta: Any = lambda _iteration: 0.999
        self.config: Any = SimpleNamespace(ema=SimpleNamespace(enabled=True))
        self.tokenizer_vision_gen: Any = SimpleNamespace(keep_encoder_cache=False)
        self.fail: bool = False

    def get_data_and_condition(self, _batch: dict[str, Any]) -> str:
        self.net._update_moe_callback_stats()
        if self.fail:
            raise RuntimeError("synthetic encode failure")
        return "encoded"


class _ReferenceSemanticsConfig:
    """Minimal config surface that varies OmniMoT's non-tensor timestep scale."""

    def __init__(self, num_train_timesteps: int) -> None:
        self.rectified_flow_inference_config: dict[str, int] = {
            "num_train_timesteps": num_train_timesteps,
        }

    def __getattr__(self, _name: str) -> None:
        return None


class _ReferenceSemanticsOmniMot:
    """Minimal policy target for reference-semantics identity tests."""

    def __init__(self, num_train_timesteps: int) -> None:
        self.net: torch.nn.Module = torch.nn.Linear(1, 1, bias=False)  # weight: [1,1]
        self.net_ema: None = None
        self.config: _ReferenceSemanticsConfig = _ReferenceSemanticsConfig(num_train_timesteps)


def _group_vision_items(items: list[torch.Tensor], counts: list[int]) -> list[list[torch.Tensor]]:
    """Undo OmniMoT's sample-major flattened vision-item layout."""
    grouped: list[list[torch.Tensor]] = []
    offset = 0
    for count in counts:
        grouped.append(items[offset : offset + count])
        offset += count
    assert offset == len(items)
    return grouped


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("sample_seeds", [None, [11, 22]], ids=["unseeded", "seeded"])
def test_draw_sigma_epsilon_rejects_diffusion_forcing_per_frame_sigma(
    sample_seeds: list[int] | None,
) -> None:
    policy = Cosmos3FlowPolicy(_DiffusionForcingOmniMot(latent_frames=3), build_ref=False)
    x0_shape = torch.Size((2, 4, 3, 8, 8))  # [B,C,T,H,W]
    condition = _flow_condition()

    with pytest.raises(ValueError, match=r"sigma shape \(.*3\).*diffusion_forcing"):
        policy.draw_sigma_epsilon(x0_shape, condition, sample_seeds=sample_seeds)


@pytest.mark.L0
@pytest.mark.CPU
def test_reference_identity_binds_inference_timestep_scale_config() -> None:
    """Changing RF inference timesteps changes reference forward semantics and identity."""
    policy_1000 = Cosmos3FlowPolicy(_ReferenceSemanticsOmniMot(1000), build_ref=False)
    policy_2000 = Cosmos3FlowPolicy(_ReferenceSemanticsOmniMot(2000), build_ref=False)

    assert policy_1000.reference_semantics_identity() != policy_2000.reference_semantics_identity()


@pytest.mark.L0
@pytest.mark.CPU
def test_seeded_validation_noise_does_not_seed_every_visible_cuda_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = Cosmos3FlowPolicy(_DiffusionForcingOmniMot(latent_frames=1), build_ref=False)
    x0_shape = torch.Size((2, 4, 1, 8, 8))  # [B,C,T,H,W]
    condition = _flow_condition()

    def _forbid_global_seed(_seed: int) -> torch.Generator:
        raise AssertionError("validation must not seed every visible CUDA device")

    monkeypatch.setattr(torch, "manual_seed", _forbid_global_seed)
    sigmas, epsilon = policy.draw_sigma_epsilon(x0_shape, condition, sample_seeds=[11, 22])

    assert tuple(sigmas.shape) == (2, 1)
    assert tuple(epsilon.shape) == tuple(x0_shape)


@pytest.mark.L0
@pytest.mark.CPU
def test_validation_context_restores_dynamic_rope_after_nested_failure() -> None:
    class _RotaryEmbedding(torch.nn.Module):
        def __init__(self, rope_type: str) -> None:
            super().__init__()
            self.rope_type: str = rope_type
            self.max_seq_len_cached: int = 8
            self.attention_scaling: float = 1.0
            self.register_buffer("inv_freq", torch.tensor([1.0, 2.0]), persistent=False)  # [D/2]
            self.original_inv_freq: torch.Tensor = self.inv_freq

    for rope_type in ("dynamic", "longrope"):
        omnimot = _StatefulValidationOmniMot()
        policy_rope = _RotaryEmbedding(rope_type)
        reference_rope = _RotaryEmbedding(rope_type)
        omnimot.net = torch.nn.Sequential(policy_rope)
        non_fsdp_policy = Cosmos3FlowPolicy(omnimot, build_ref=False)
        non_fsdp_policy.net_ref = torch.nn.Sequential(reference_rope)
        rotary_modules = [policy_rope, reference_rope]
        with pytest.raises(RuntimeError, match="validation failed"), non_fsdp_policy.validation_context():
            for rotary in rotary_modules:
                rotary.register_buffer("inv_freq", torch.tensor([9.0]), persistent=True)  # [D/2]
                rotary.max_seq_len_cached = torch.tensor(32)  # []
                rotary.original_inv_freq = torch.tensor([8.0])  # [D/2]
                rotary.long_inv_freq = torch.tensor([7.0])  # [D/2]
                rotary.attention_scaling = torch.tensor(0.5)  # []
            with non_fsdp_policy.validation_context():
                raise RuntimeError("validation failed")
        assert non_fsdp_policy._validation_depth == 0
        for rotary in rotary_modules:
            assert torch.equal(rotary.inv_freq, torch.tensor([1.0, 2.0]))
            assert rotary.original_inv_freq is rotary.inv_freq
            assert rotary.max_seq_len_cached == 8
            assert rotary.attention_scaling == 1.0
            assert "inv_freq" in rotary._non_persistent_buffers_set
            assert "long_inv_freq" not in vars(rotary)


@pytest.mark.L0
@pytest.mark.CPU
def test_validation_context_suppresses_and_restores_moe_callback_stats() -> None:
    omnimot = _TelemetryValidationOmniMot()
    policy = Cosmos3FlowPolicy(omnimot, build_ref=False)
    reference = _MoeCallbackModule()
    policy.net_ref = reference

    with policy.validation_context():
        with policy.validation_context():
            assert policy._get_data_and_condition({}) == "encoded"
            assert policy._validation_depth == 2
            assert omnimot.net.callback_calls == 0
            assert all(
                "_update_moe_callback_stats" in vars(module) for module in (omnimot.net, reference, omnimot.net_ema)
            )
    assert policy._validation_depth == 0
    assert all("_update_moe_callback_stats" not in vars(module) for module in (omnimot.net, reference, omnimot.net_ema))

    omnimot.fail = True
    with pytest.raises(RuntimeError, match="synthetic encode failure"), policy.validation_context():
        policy._get_data_and_condition({})
    assert policy._validation_depth == 0
    omnimot.net._update_moe_callback_stats()
    assert omnimot.net.callback_calls == 1


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("incompatibility", "message"),
    [
        ("retained_encoder_cache", "keep_encoder_cache=False"),
        ("cuda_graph_sequence_packing", "pad_for_cuda_graphs=False"),
    ],
)
def test_validation_context_rejects_incompatible_runtime_state(
    incompatibility: str,
    message: str,
) -> None:
    omnimot = _StatefulValidationOmniMot()
    if incompatibility == "retained_encoder_cache":
        omnimot.tokenizer_vision_gen.keep_encoder_cache = True
    else:
        omnimot.net.pad_for_cuda_graphs = True
    policy = Cosmos3FlowPolicy(omnimot, build_ref=False)

    with pytest.raises(ValueError, match=message):
        with policy.validation_context():
            pass
    assert policy._validation_depth == 0


@pytest.mark.L0
@pytest.mark.CPU
def test_seeded_stochastic_encoder_is_reorder_and_split_invariant() -> None:
    """Each logical sample/item owns its posterior RNG stream, independent of batching."""
    omnimot = _StochasticVisionOmniMot()
    policy = Cosmos3FlowPolicy(omnimot, build_ref=False)
    samples = [
        [
            torch.full((1, 1, 1, 1, 1), 10.0),  # [1,C,T,H,W]
            torch.full((1, 1, 1, 1, 1), 11.0),  # [1,C,T,H,W]
        ],
        [torch.full((1, 1, 1, 1, 1), 20.0)],  # 1 x [1,C,T,H,W]
        [
            torch.full((1, 1, 1, 1, 1), 30.0),  # [1,C,T,H,W]
            torch.full((1, 1, 1, 1, 1), 31.0),  # [1,C,T,H,W]
            torch.full((1, 1, 1, 1, 1), 32.0),  # [1,C,T,H,W]
        ],
    ]
    seeds = [101, 202, 303]
    counts = [2, 1, 3]

    torch.random.default_generator.manual_seed(999)
    rng_before = torch.random.get_rng_state().clone()  # [RNG]
    full_flat = policy._get_data_and_condition(
        {"video": samples},
        sample_encode_seeds=seeds,
    )
    rng_after = torch.random.get_rng_state()  # [RNG]
    assert torch.equal(rng_after, rng_before)
    full = _group_vision_items(full_flat, counts)

    order = [2, 0, 1]
    reordered_flat = policy._get_data_and_condition(
        {"video": [samples[index] for index in order]},
        sample_encode_seeds=[seeds[index] for index in order],
    )
    reordered = _group_vision_items(reordered_flat, [counts[index] for index in order])
    for reordered_index, original_index in enumerate(order):
        assert all(
            torch.equal(actual, expected)
            for actual, expected in zip(reordered[reordered_index], full[original_index], strict=True)
        )

    split = [
        policy._get_data_and_condition(
            {"video": [sample]},
            sample_encode_seeds=[seed],
        )
        for sample, seed in zip(samples, seeds, strict=True)
    ]
    for actual_items, expected_items in zip(split, full, strict=True):
        assert all(torch.equal(actual, expected) for actual, expected in zip(actual_items, expected_items, strict=True))


@pytest.mark.L0
@pytest.mark.CPU
def test_seeded_stochastic_encoder_restores_rng_and_hook_after_failure() -> None:
    """A tokenizer exception cannot leak its seeded RNG or temporary item hook."""
    omnimot = _StochasticVisionOmniMot()
    omnimot.fail_on_value = 2.0
    policy = Cosmos3FlowPolicy(omnimot, build_ref=False)
    samples = [
        [torch.full((1, 1, 1, 1, 1), 1.0)],  # 1 x [1,C,T,H,W]
        [torch.full((1, 1, 1, 1, 1), 2.0)],  # 1 x [1,C,T,H,W]
    ]

    torch.random.default_generator.manual_seed(1234)
    rng_before = torch.random.get_rng_state().clone()  # [RNG]
    with pytest.raises(RuntimeError, match="synthetic tokenizer failure"):
        policy._get_data_and_condition({"video": samples}, sample_encode_seeds=[10, 20])
    rng_after = torch.random.get_rng_state()  # [RNG]

    assert torch.equal(rng_after, rng_before)
    assert not policy._seeded_vision_active
    assert "_encode_vision_item" not in vars(omnimot)

    omnimot.fail_on_value = None
    unseeded = policy._get_data_and_condition({"video": samples})
    assert len(unseeded) == 2
