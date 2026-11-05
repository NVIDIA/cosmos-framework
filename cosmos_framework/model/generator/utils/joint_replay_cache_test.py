# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Numerical replay-cache oracles with real storage, metadata and detached clean K/V."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

import attrs
import pytest
import torch

from cosmos_framework.configs.base.defaults.multiview_attention import AttentionScope
from cosmos_framework.configs.base.defaults.replay_attention import TeacherForcingReplayPolicyConfig
from cosmos_framework.model.generator.attention_io_layout import AttentionIOLayout
from cosmos_framework.model.generator.mot.causal_flex_attention import (
    TeacherForcingFlexMetadata,
    build_teacher_forcing_clean_target_token_indexes,
    build_teacher_forcing_multiview_flex_metadata,
)
from cosmos_framework.model.generator.mot.flex_attention import CaptionMaskItem, SensorMaskItem
from cosmos_framework.model.generator.mot.maskless_attention import build_replay_maskless_plan
from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel, OmniMoTCausalModelConfig
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean
from cosmos_framework.model.generator.utils.kv_cache import (
    DualKVCache,
    JointARMemoryValue,
    JointChunkMemory,
    TeacherForcingMemoryState,
    TFNoisyMemoryValue,
)
from cosmos_framework.model.generator.utils.multiview_ar import JointARCache, JointARChunk, joint_ar_chunks
from cosmos_framework.model.generator.utils.multiview_test_utils import _dense_reference, _pass_weights
from cosmos_framework.data.generator.sequence_packing import ModalityData, PackedSequence

_WIDTH = 8
_HEADS = 2
_LAYERS = 3
_TEXT = 4


@dataclass
class _ReplayPack:
    packed: PackedSequence
    items: list[list[SensorMaskItem]]
    identities: list[tuple[int, int, int, int]]  # item, camera, physical frame, spatial token
    values: torch.Tensor  # [G,C]


def _pack(rgb: tuple[int, ...], lidar: tuple[int, ...], conditioned: int) -> _ReplayPack:
    """Build actual packed sensor objects from explicit physical token identities."""
    items: list[SensorMaskItem] = []
    modalities: list[ModalityData] = []
    identities: list[tuple[int, int, int, int]] = []
    positions: list[tuple[int, int, int]] = []
    offset = _TEXT
    for sensor, frames in enumerate((rgb, lidar)):
        if not frames:
            continue
        views = 2 if sensor == 0 else 1
        condition_count = conditioned if sensor == 0 else (0, 1, 3)[conditioned]
        masks: list[torch.Tensor] = []  # each [V*T,1,1]
        start = offset
        for control in (True, False):
            item = 2 * sensor + int(not control)
            condition = torch.tensor(
                [control or frame < condition_count for _ in range(views) for frame in frames]
            )  # [V*T]
            masks.append(condition.reshape(-1, 1, 1))  # [V*T,1,1]
            items.append(
                SensorMaskItem(
                    token_shape=(views * len(frames), 1, 2),
                    condition_mask=condition,
                    num_views=views,
                    view_offset=2 * sensor,
                    is_control=control,
                    seconds_per_frame=4 / 30 if sensor == 0 else 0.1,
                    caption_access="camera" if sensor == 0 else "all_captions",
                )
            )
            for view in range(views):
                for local_frame, frame in enumerate(frames):
                    for spatial in range(2):
                        identities.append((item, view, frame, spatial))
                        # Chunk packs initially use local time, just as a real fresh pack does.
                        positions.append((local_frame, view + 2 * sensor, spatial))
                        offset += 1
        modalities.append(
            ModalityData(
                token_shapes=[(views * len(frames), 1, 2)] * 2,
                condition_mask=masks,
                sequence_indexes=torch.arange(start, offset),  # [G_sensor]
                tokens=[torch.zeros(1, _WIDTH, views * len(frames), 1, 2)] * 2,  # each [1,C,V*T,1,2]
                seconds_per_frame=[4 / 30 if sensor == 0 else 0.1] * 2,
            )
        )
    physical_ids = torch.tensor(
        [item * 101 + view * 31 + frame * 5 + spatial for item, view, frame, spatial in identities]
    )  # [G]
    channels = torch.arange(_WIDTH)  # [C]
    values = torch.sin(physical_ids[:, None] * 0.17 + channels[None] * 0.43) + torch.cos(
        physical_ids[:, None] * 0.07 - channels[None] * 0.29
    )  # [G,C]
    packed = PackedSequence(
        sample_lens=[offset],
        split_lens=[_TEXT, offset - _TEXT],
        attn_modes=["causal", "full"],
        sequence_length=offset,
        text_ids=torch.arange(_TEXT),  # [U]
        text_indexes=torch.arange(_TEXT),  # [U]
        position_ids=torch.tensor([(0, 0, 0)] * _TEXT + positions).T.contiguous(),  # [3,U+G]
        vision=modalities[0],
        lidar=modalities[1] if lidar else None,
        num_views_per_vision_item=[2, 2],
        num_vision_items_per_sample=[2],
        num_lidar_items_per_sample=[2] if lidar else None,
    )
    return _ReplayPack(packed, [items], identities, values)


def _policy(scope: AttentionScope) -> TeacherForcingReplayPolicyConfig:
    policy = TeacherForcingReplayPolicyConfig(
        control_visibility="causal",
        controls_read_strict_past_clean_rgb=True,
        clean_pass_causality="chunk",
        multiview_attention_scope="same_view",
    )
    policy.multiview_attention_scope = scope
    return policy


def _metadata(
    pack: _ReplayPack, policy: TeacherForcingReplayPolicyConfig, *, noisy: bool, capacity: int = 0
) -> TeacherForcingFlexMetadata:
    metadata = build_teacher_forcing_multiview_flex_metadata(
        seq_len=len(pack.identities),
        full_q_offsets=torch.tensor([0, len(pack.identities)]),  # [2]
        sensor_mask_items=pack.items,
        caption_mask_items=[[CaptionMaskItem(view_id=view, num_tokens=2) for view in range(2)]],
        device=torch.device("cpu"),
        num_und=_TEXT,
        causal_offsets=torch.tensor([0, _TEXT]),  # [2]
        frames_per_chunk=3,
        pass_kind="noisy" if noisy else "clean",
        clean_memory_seq_len=capacity,
        teacher_forcing_replay_policy=attrs.evolve(policy, multiview_attention_scope="same_view"),
    )
    return replace(metadata, teacher_forcing_replay_policy=policy)


def _memory(pack: _ReplayPack, policy: TeacherForcingReplayPolicyConfig) -> TeacherForcingMemoryState:
    assert pack.packed.vision is not None
    indexes = build_teacher_forcing_clean_target_token_indexes(
        sensor_mask_items=pack.items, device=torch.device("cpu")
    )  # [M_real]
    return TeacherForcingMemoryState(
        vision_token_shapes=[item.token_shape for item in pack.items[0][:2]],
        num_action_tokens_per_supertoken=0,
        null_action_supertokens=False,
        segment_idx=0,
        dual_kv_cache=[DualKVCache(gen_cache_size=2) for _ in range(_LAYERS)],
        num_kv_heads=_HEADS,
        head_dim=_WIDTH // _HEADS,
        detach_clean_kv=True,
        frames_per_chunk=3,
        teacher_forcing_replay_policy=policy,
        selected_clean_gen_token_indexes=indexes,
        selected_clean_gen_padded_capacity=indexes.numel(),
    )


def _rotate(value: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Apply a small genuine rotary encoding so incorrect cache positions change attention."""
    # value: [1,S,H,D], positions: [3,S], returns [1,S,H,D]
    phase = (positions.float() * torch.tensor([0.19, 0.11, 0.07])[:, None]).sum(0)  # [S]
    phase = phase[:, None] * torch.tensor([1.0, 0.3])[None]  # [S,D/2]
    even, odd = value[..., ::2], value[..., 1::2]  # each [1,S,H,D/2]
    cosine, sine = phase.cos()[None, :, None], phase.sin()[None, :, None]  # each [1,S,1,D/2]
    return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1).flatten(-2)  # [1,S,H,D]


class _DenseStudent(torch.nn.Module):
    """Three attention layers expose history-dependent K/V and trainable noisy controls."""

    qkv: torch.nn.ModuleList
    output: torch.nn.ModuleList
    control_scale: torch.nn.Parameter
    num_hidden_layers: int = _LAYERS
    num_kv_heads: int = _HEADS
    head_dim: int = _WIDTH // _HEADS
    teacher_forcing_maskless: bool = True

    def __init__(self) -> None:
        super().__init__()
        self.qkv = torch.nn.ModuleList(torch.nn.Linear(_WIDTH, 3 * _WIDTH, bias=False) for _ in range(_LAYERS))
        self.output = torch.nn.ModuleList(torch.nn.Linear(_WIDTH, _WIDTH, bias=False) for _ in range(_LAYERS))
        self.control_scale = torch.nn.Parameter(torch.linspace(-0.2, 0.2, _WIDTH))  # [C]

    def forward(
        self,
        pack: _ReplayPack,
        metadata: TeacherForcingFlexMetadata,
        values: torch.Tensor,  # [G,C]
        memory: JointChunkMemory | TeacherForcingMemoryState,
    ) -> torch.Tensor:  # [G,C]
        plan = build_replay_maskless_plan(metadata, pack.items)
        weights = sum(_pass_weights(plan).values())  # [G,U+G+M]
        controls = torch.tensor([item % 2 == 0 for item, _, _, _ in pack.identities])  # [G]
        sensors = values * (1 + controls[:, None] * self.control_scale[None])  # [G,C]
        captions = torch.sin(torch.arange(_TEXT * _WIDTH).reshape(_TEXT, _WIDTH) * 0.23)  # [U,C]
        hidden = torch.cat((captions, sensors))  # [U+G,C]
        memory.init(
            {"causal_seq": captions, "_num_causal_tokens": _TEXT, "_num_full_tokens": len(pack.identities)},
            torch.device("cpu"),
        )
        caption_weights = torch.block_diag(torch.ones(2, 2).tril(), torch.ones(2, 2).tril())  # [U,U]
        for layer, (projection, output) in enumerate(zip(self.qkv, self.output, strict=True)):
            projected = projection(hidden).reshape(1, hidden.shape[0], 3, _HEADS, _WIDTH // _HEADS)
            # projected: [1,U+G,3,H,D]
            q, k, v = projected.unbind(2)  # each [1,U+G,H,D]
            q_ = _rotate(q, pack.packed.position_ids)  # [1,U+G,H,D]
            k_ = _rotate(k, pack.packed.position_ids)  # [1,U+G,H,D]
            state = memory.read_for_layer(layer)
            cached_k: torch.Tensor | None = None  # [1,M,H,D] or None
            cached_v: torch.Tensor | None = None  # [1,M,H,D] or None
            if isinstance(state, JointARMemoryValue):
                cached_k, cached_v = state.cached_gen_k, state.cached_gen_v  # each [1,M,H,D] or None
            elif isinstance(state, TFNoisyMemoryValue):
                cached_k, cached_v = state.cached_clean_gen_k, state.cached_clean_gen_v  # each [1,M,H,D]
            keys = k_ if cached_k is None else torch.cat((k_, cached_k), dim=1)  # [1,U+G+M,H,D]
            vals = v if cached_v is None else torch.cat((v, cached_v), dim=1)  # [1,U+G+M,H,D]
            sensor_output = _dense_reference(
                q_[:, _TEXT:], keys, vals, weights, torch.ones(len(pack.identities), dtype=torch.bool)
            )  # [1,G,H,D]
            caption_output = _dense_reference(
                q_[:, :_TEXT],
                k_[:, :_TEXT],
                v[:, :_TEXT],
                caption_weights,
                torch.ones(_TEXT, dtype=torch.bool),
            )  # [1,U,H,D]
            combined = torch.cat((caption_output, sensor_output), dim=1).reshape(-1, _WIDTH)  # [U+G,C]
            hidden = hidden + 0.3 * torch.tanh(output(combined))  # [U+G,C]
            memory.write_for_layer(layer, (k_[:, _TEXT:], v[:, _TEXT:], k_[:, :_TEXT], v[:, :_TEXT]))
        return hidden[_TEXT:]  # [G,C]


def _chunks(has_lidar: bool, *, singleton: bool = False) -> list[JointARChunk]:
    chunks = joint_ar_chunks(
        vision_frames=1 if singleton else 7,
        lidar_frames=1 if singleton else 9,
        vision_seconds_per_frame=4 / 30,
        lidar_seconds_per_frame=0.1,
        frames_per_chunk=3,
    )
    return chunks if has_lidar else [replace(chunk, lidar_frames=(), lidar_prefix_end=0) for chunk in chunks]


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("conditioned", [0, 1, 2])
@pytest.mark.parametrize("has_lidar", [False, True])
@pytest.mark.parametrize("scope", ["same_view", "decomposed"])
def test_persistent_clean_kv_matches_recomputed_student_outputs_and_gradients(
    conditioned: int, has_lidar: bool, scope: AttentionScope
) -> None:
    _assert_student_replay_parity(conditioned, has_lidar, scope)


@pytest.mark.L0
@pytest.mark.CPU
def test_first_generated_chunk_backpropagates_without_any_committed_cache() -> None:
    _assert_student_replay_parity(0, True, "decomposed", singleton=True)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "policy",
    [TeacherForcingReplayPolicyConfig(), TeacherForcingReplayPolicyConfig(control_visibility="current")],
    ids=["default_global_controls", "current_controls"],
)
def test_native_clean_cache_builder_rejects_noncausal_control_policy(
    monkeypatch: pytest.MonkeyPatch, policy: TeacherForcingReplayPolicyConfig
) -> None:
    """Enforce the joint rollout contract even when the cache builder is called directly."""
    host = OmniMoTCausalModel.__new__(OmniMoTCausalModel)
    torch.nn.Module.__init__(host)
    host.config = OmniMoTCausalModelConfig(
        causal_training_strategy="teacher_forcing",
        teacher_forcing_kv_implementation="multiview_maskless_kv",
        teacher_forcing_frames_per_chunk=3,
        teacher_forcing_replay_policy=policy,
    )
    host.net = _DenseStudent()
    host.parallel_dims = None
    reference = _pack(tuple(range(7)), tuple(range(9)), 0)
    prefix = _pack(tuple(range(4)), tuple(range(5)), 0)
    cache = JointARCache(reference.packed, num_layers=_LAYERS)

    def reject_forward_or_reuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("Reject unsupported controls before running or loading clean K/V")

    monkeypatch.setattr(host, "denoise", reject_forward_or_reuse)
    monkeypatch.setattr(cache, "populate_replay_memory", reject_forward_or_reuse)
    with pytest.raises(ValueError, match="Persistent clean replay requires causal control visibility"):
        host._build_clean_tf_cache(
            net=host.net,
            packed_sequence=prefix.packed,
            gen_data_clean=GenerationDataClean(batch_size=1, is_image_batch=False),
            memory_info={},
            detach_clean_kv=True,
            persistent_cache=cache,
        )


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "has_lidar,empty,attention_io_layout",
    [
        (True, False, "sequence_sharded"),
        (False, False, "replicated"),
        (True, True, "sequence_sharded"),
        (False, True, "replicated"),
    ],
)
def test_native_clean_cache_builder_reuses_persistent_values_without_clean_denoising(
    monkeypatch: pytest.MonkeyPatch, has_lidar: bool, empty: bool, attention_io_layout: AttentionIOLayout
) -> None:
    """Keep the production metadata and TF-memory builders, including empty-cache CP head sizing."""
    host = OmniMoTCausalModel.__new__(OmniMoTCausalModel)
    torch.nn.Module.__init__(host)
    host.config = OmniMoTCausalModelConfig(
        causal_training_strategy="teacher_forcing",
        teacher_forcing_kv_implementation="multiview_maskless_kv",
        teacher_forcing_frames_per_chunk=3,
        teacher_forcing_replay_policy=_policy("decomposed"),
        attention_io_layout=attention_io_layout,
    )
    host.net = _DenseStudent()
    # This factory only needs the CP dimensions; actual collectives have separate distributed coverage.
    monkeypatch.setattr(host, "parallel_dims", SimpleNamespace(cp_enabled=True, cp_size=2), raising=False)

    def reject_clean_forward(*args: object, **kwargs: object) -> None:
        raise AssertionError("Persistent replay must not invoke the clean denoiser")

    monkeypatch.setattr(host, "denoise", reject_clean_forward)
    conditioned = 0 if empty else 2
    reference = _pack(
        tuple(range(1 if empty else 7)), tuple(range(1 if empty else 9)) if has_lidar else (), conditioned
    )
    prefix = _pack(tuple(range(1 if empty else 4)), tuple(range(1 if empty else 5)) if has_lidar else (), conditioned)
    cache = JointARCache(reference.packed, num_layers=_LAYERS)
    heads = _HEADS if attention_io_layout == "replicated" else _HEADS // 2
    if not empty:
        cache.memory.spans = [(0, 0, len(reference.identities))]
        for layer in range(_LAYERS):
            keys = (
                torch.arange(len(reference.identities) * heads * (_WIDTH // _HEADS))
                .reshape(1, len(reference.identities), heads, _WIDTH // _HEADS)
                .float()
                + layer * 1000
            )  # [1,G_full,H_local,D]
            values = keys + 500  # [1,G_full,H_local,D]
            captions = torch.empty(1, 0, heads, _WIDTH // _HEADS)  # [1,0,H_local,D]
            cache.memory.write_for_layer(layer, (keys, values, captions, captions))
        cache.memory.spans = None
    memory = host._build_clean_tf_cache(
        net=host.net,
        packed_sequence=prefix.packed,
        gen_data_clean=GenerationDataClean(batch_size=1, is_image_batch=False),
        memory_info={},
        detach_clean_kv=True,
        persistent_cache=cache,
    )
    selected_rows = [
        row
        for row, (item, _, frame, _) in enumerate(prefix.identities)
        if item % 2 == 1 and frame >= (conditioned if item == 1 else (0, 1, 3)[conditioned])
    ]
    expected_indexes = torch.tensor(selected_rows)  # [M]
    assert memory.selected_clean_gen_token_indexes is not None
    torch.testing.assert_close(memory.selected_clean_gen_token_indexes, expected_indexes)
    assert memory.pass_number == 2 and memory.detach_clean_kv
    assert memory.num_kv_heads == _HEADS // 2 and memory.context_parallel_size == 2
    assert prefix.packed.teacher_forcing_pass == "noisy"
    assert prefix.packed.teacher_forcing_selected_clean_target_padded_capacity == len(selected_rows)
    for found, item in zip(
        prefix.packed.teacher_forcing_original_condition_masks_sensors, prefix.items[0], strict=True
    ):
        torch.testing.assert_close(found.reshape(-1), item.condition_mask)
    reference_rows = {identity: row for row, identity in enumerate(reference.identities)}
    slots = torch.tensor([reference_rows[prefix.identities[row]] for row in selected_rows])  # [M]
    for layer, cached in enumerate(memory._clean_gen_kv):
        assert cached is not None
        for field, found in enumerate(cached):  # found: [1,M,H_local,D]
            assert found.shape == (1, len(selected_rows), heads, _WIDTH // _HEADS)
            assert not found.requires_grad and found.grad_fn is None
            if empty:
                assert torch.count_nonzero(found) == 0  # []
            else:
                stored = cache.memory.cache[layer]
                assert stored is not None
                torch.testing.assert_close(found, stored[field][:, slots], atol=0, rtol=0)


def _assert_student_replay_parity(
    conditioned: int, has_lidar: bool, scope: AttentionScope, *, singleton: bool = False
) -> None:
    torch.manual_seed(181)
    student = _DenseStudent()
    policy = _policy(scope)
    chunks = _chunks(has_lidar, singleton=singleton)
    reference = _pack(
        tuple(range(chunks[-1].vision_prefix_end)), tuple(range(chunks[-1].lidar_prefix_end)), conditioned
    )
    cache = JointARCache(reference.packed, num_layers=_LAYERS)
    with torch.no_grad():
        for chunk in chunks[:-1]:
            current = _pack(chunk.vision_frames, chunk.lidar_frames, conditioned)
            cache.prepare(current.packed, chunk.vision_frames, chunk.lidar_frames)
            metadata = cache.augment_metadata(_metadata(current, policy, noisy=False), current.items, 3, noisy=False)
            cache.memory.spans = cache.spans
            student(current, metadata, current.values, cache.memory)  # [G_current,C]
            cache.commit_metadata()
    snapshots = [None if kv is None else (kv[0].clone(), kv[1].clone()) for kv in cache.memory.cache]
    # snapshots: list[tuple[[1,M,H,D],[1,M,H,D]] | None]

    for chunk in chunks:
        pack = _pack(tuple(range(chunk.vision_prefix_end)), tuple(range(chunk.lidar_prefix_end)), conditioned)
        # Independent identities determine active/history rows, without the cache's span or slot mapper.
        active = torch.tensor(
            [
                item % 2 == 1
                and frame in (chunk.vision_frames if item == 1 else chunk.lidar_frames)
                and frame >= (conditioned if item == 1 else (0, 1, 3)[conditioned])
                for item, _, frame, _ in pack.identities
            ]
        )  # [G]
        if not active.any():
            continue
        clean_values = pack.values.clone()  # [G,C]
        clean_values[active] = 0  # [G_current_target,C]
        expected_memory = _memory(pack, policy)
        actual_memory = _memory(pack, policy)
        with torch.no_grad():
            student(pack, _metadata(pack, policy, noisy=False), clean_values, expected_memory)  # [G,C]
            expected_memory.pass_number = 2
            cache.populate_replay_memory(pack.packed, actual_memory)
        assert actual_memory.pass_number == 2
        selected = expected_memory.selected_clean_gen_token_indexes  # [M_real]
        assert selected is not None
        history = ~active[selected]  # [M_real]
        for expected, actual in zip(expected_memory._clean_gen_kv, actual_memory._clean_gen_kv, strict=True):
            assert expected is not None and actual is not None
            for wanted, found in zip(expected, actual, strict=True):  # each [1,M,H,D]
                assert not found.requires_grad and found.grad_fn is None
                torch.testing.assert_close(
                    found[:, : selected.numel()][:, history], wanted[:, : selected.numel()][:, history]
                )
                # Finalized current/future K/V differs from clean replay's zero target placeholders.
                # Make that difference extreme: the strict-past mask must completely exclude it.
                found[:, : selected.numel()][:, ~history] = 1000

        noisy_values = pack.values.clone()  # [G,C]
        noisy_values[active] = 0.4 * torch.cos(pack.values[active] * 1.7)  # [G_current_target,C]
        metadata = _metadata(pack, policy, noisy=True, capacity=expected_memory.selected_clean_gen_padded_capacity)
        parameter_grads: list[dict[str, torch.Tensor]] = []
        outputs: list[torch.Tensor] = []  # each [G_current_target,C]
        losses: list[torch.Tensor] = []  # each []
        input_gradients: list[torch.Tensor] = []  # each [G,C]
        for memory in (expected_memory, actual_memory):
            student.zero_grad(set_to_none=True)
            values = noisy_values.detach().clone().requires_grad_()  # [G,C]
            output = student(pack, metadata, values, memory)  # [G,C]
            endpoint = values[active] - 0.625 * output[active]  # [G_current_target,C]
            target = torch.sin(pack.values[active] * 0.9)  # [G_current_target,C]
            loss = 0.5 * (endpoint - target).square().mean()  # []
            loss.backward()
            outputs.append(endpoint.detach())  # [G_current_target,C]
            losses.append(loss.detach())  # []
            assert values.grad is not None
            input_gradients.append(values.grad.detach().clone())  # [G,C]
            gradients: dict[str, torch.Tensor] = {}
            for name, parameter in student.named_parameters():
                assert parameter.grad is not None
                assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
                gradients[name] = parameter.grad.detach().clone()  # parameter shape
            parameter_grads.append(gradients)
        torch.testing.assert_close(outputs[0], outputs[1])
        torch.testing.assert_close(losses[0], losses[1])
        torch.testing.assert_close(input_gradients[0], input_gradients[1])
        for name in parameter_grads[0]:
            torch.testing.assert_close(parameter_grads[0][name], parameter_grads[1][name], atol=2e-6, rtol=2e-5)
        if chunk.step:
            historical_controls = torch.tensor(
                [
                    item % 2 == 0 and frame < (chunk.vision_frames[0] if item == 0 else chunk.lidar_frames[0])
                    for item, _, frame, _ in pack.identities
                ]
            )  # [G]
            assert input_gradients[1][historical_controls].abs().sum() > 0
        for before, after in zip(snapshots, cache.memory.cache, strict=True):
            if before is None:
                assert after is None
            else:
                assert after is not None
                for old, new in zip(before, after, strict=True):  # each [1,M,H,D]
                    torch.testing.assert_close(old, new, atol=0, rtol=0)
