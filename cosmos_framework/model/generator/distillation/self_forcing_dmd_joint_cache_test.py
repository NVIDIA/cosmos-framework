# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Critic rollout parity through real persistent storage and three dense attention layers."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import attrs
import pytest
import torch

from cosmos_framework.configs.base.defaults.replay_attention import TeacherForcingReplayPolicyConfig
from cosmos_framework.data.generator.sequence_packing import ModalityData, PackedSequence, SequencePlan
from cosmos_framework.model.generator.distillation.self_forcing_dmd_joint_test import AnalyticJointModel
from cosmos_framework.model.generator.mot.causal_cosmos3_vfm_network import build_interactive_multiview_mask_items
from cosmos_framework.model.generator.mot.causal_flex_attention import build_teacher_forcing_multiview_flex_metadata
from cosmos_framework.model.generator.mot.flex_attention import CaptionMaskItem
from cosmos_framework.model.generator.mot.maskless_attention import build_replay_maskless_plan
from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean
from cosmos_framework.model.generator.utils.kv_cache import (
    JointARMemoryValue,
    JointChunkMemory,
    TeacherForcingMemoryState,
    TFNoisyMemoryValue,
)
from cosmos_framework.model.generator.utils.multiview_ar import JointARCache, _build_replay_branch
from cosmos_framework.model.generator.utils.multiview_test_utils import _dense_reference, _pass_weights

_WIDTH = 8
_HEADS = 2
_LAYERS = 3
_TEXT = 4


def _rotate(value: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Make rotary-position errors observable in the dense attention output."""
    # value: [1,S,H,D], positions: [3,S], returns [1,S,H,D]
    phase = (positions.float() * torch.tensor([0.19, 0.11, 0.07])[:, None]).sum(0)  # [S]
    phase = phase[:, None] * torch.tensor([1.0, 0.3])[None]  # [S,D/2]
    even, odd = value[..., ::2], value[..., 1::2]  # each [1,S,H,D/2]
    cosine, sine = phase.cos()[None, :, None], phase.sin()[None, :, None]  # each [1,S,1,D/2]
    return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1).flatten(-2)  # [1,S,H,D]


class _DenseStudent(torch.nn.Module):
    qkv: torch.nn.ModuleList
    output: torch.nn.ModuleList
    num_hidden_layers: int = _LAYERS
    num_kv_heads: int = _HEADS
    head_dim: int = _WIDTH // _HEADS
    teacher_forcing_maskless: bool = True

    def __init__(self) -> None:
        super().__init__()
        self.qkv = torch.nn.ModuleList(torch.nn.Linear(_WIDTH, 3 * _WIDTH, bias=False) for _ in range(_LAYERS))
        self.output = torch.nn.ModuleList(torch.nn.Linear(_WIDTH, _WIDTH, bias=False) for _ in range(_LAYERS))


class _DenseJointModel(AnalyticJointModel):
    """Keep rollout, branch construction, clean replay, cache commits and sampling native."""

    cache: JointARCache | None
    clean_frames: list[tuple[int, ...]]
    history_read_layers: set[int]
    last_clean_kv: list[tuple[torch.Tensor, torch.Tensor]]  # each [1,G,H,D]

    # The analytic fixture only supplies lightweight configuration and CPU placement.
    _build_clean_tf_cache = OmniMoTCausalModel._build_clean_tf_cache

    def __init__(self) -> None:
        super().__init__()
        self.net = _DenseStudent()
        self.config.kv_cache_inference_size = None
        self.config.attention_sink_size = 0
        self.config.rolling_kv_cache_chunks = None
        self.config.teacher_forcing_target_only_no_text_pass2 = False
        self.config.clamp_empty_varlen_kv = True
        self.config.causal_training_strategy = "teacher_forcing"
        self.cache = None
        self.clean_frames = []
        self.history_read_layers = set()
        self.last_clean_kv = []

    def _get_teacher_forcing_kv_implementation(self) -> str:
        return "multiview_maskless_kv"

    def _get_teacher_forcing_replay_policy(self) -> TeacherForcingReplayPolicyConfig:
        return TeacherForcingReplayPolicyConfig(
            control_visibility="causal",
            controls_read_strict_past_clean_rgb=True,
            clean_pass_causality="chunk",
            multiview_attention_scope="decomposed",
        )

    def _pack_input_sequence(
        self,
        plans: list[SequencePlan],
        text: list[list[int]],
        data: GenerationDataClean,
        timestep: torch.Tensor,  # [1] or [1,1]
        **kwargs: Any,
    ) -> PackedSequence:
        modalities: dict[str, ModalityData] = {}
        positions = [(0, 0, 0)] * _TEXT
        offset = _TEXT
        for sensor, name in enumerate(self._joint_modalities(data)):
            values = getattr(data, f"x0_tokens_{name}")
            views = 2 if name == "vision" else 1
            frames, height, width = values[0].shape[2:]
            masks: list[torch.Tensor] = []  # each [V*T,1,1]
            start = offset
            for control in (True, False):
                conditioned = getattr(plans[0], f"condition_frame_indexes_{name}")
                mask = torch.tensor(
                    [control or frame in conditioned for _ in range(views) for frame in range(frames // views)],
                    dtype=torch.float32,
                ).reshape(frames, 1, 1)  # [V*T,1,1]
                masks.append(mask)
                positions.extend(
                    (frame, view + 2 * sensor, spatial)
                    for view in range(views)
                    for frame in range(frames // views)
                    for spatial in range(height * width)
                )
                offset += frames * height * width
            modalities[name] = ModalityData(
                tokens=values,
                token_shapes=[(frames, height, width)] * 2,
                condition_mask=masks,
                sequence_indexes=torch.arange(start, offset),  # [G_sensor]
                timesteps=timestep.flatten().clone(),  # [1]
                seconds_per_frame=[4 / 30 if name == "vision" else 0.1] * 2,
            )
        return PackedSequence(
            sample_lens=[offset],
            split_lens=[_TEXT, offset - _TEXT],
            attn_modes=["causal", "full"],
            sequence_length=offset,
            text_ids=torch.arange(_TEXT),  # [U]
            text_indexes=torch.arange(_TEXT),  # [U]
            position_ids=torch.tensor(positions).T.contiguous(),  # [3,U+G]
            vision=modalities["vision"],
            lidar=modalities.get("lidar"),
            num_views_per_vision_item=[2, 2],
            num_vision_items_per_sample=[2],
            num_lidar_items_per_sample=[2] if "lidar" in modalities else None,
        )

    def denoise(
        self,
        *,
        data_batch_packed: PackedSequence,
        memory: JointChunkMemory | TeacherForcingMemoryState,
        net: torch.nn.Module | None = None,
        video_temporal_causal: bool | None = None,
    ) -> dict[str, list[torch.Tensor]]:
        """Substitute CPU projections/attention while consuming real per-layer memory values."""
        pack = data_batch_packed
        cache: JointARCache | None = getattr(pack, "joint_ar_cache", None)
        clean = pack.teacher_forcing_pass == "clean"
        items = build_interactive_multiview_mask_items(
            pack, condition_masks=pack.teacher_forcing_original_condition_masks_sensors
        )
        count = pack.sequence_length - _TEXT
        policy = self._get_teacher_forcing_replay_policy()
        metadata = build_teacher_forcing_multiview_flex_metadata(
            seq_len=count,
            full_q_offsets=torch.tensor([0, count]),  # [2]
            sensor_mask_items=items,
            caption_mask_items=[[CaptionMaskItem(view_id=view, num_tokens=2) for view in range(2)]],
            device=torch.device("cpu"),
            num_und=_TEXT,
            causal_offsets=torch.tensor([0, _TEXT]),  # [2]
            frames_per_chunk=3,
            pass_kind="clean" if cache is not None else pack.teacher_forcing_pass,
            clean_memory_seq_len=getattr(pack, "teacher_forcing_selected_clean_target_padded_capacity", 0)
            if not clean and cache is None
            else 0,
            teacher_forcing_replay_policy=attrs.evolve(policy, multiview_attention_scope="same_view"),
        )
        if cache is not None:
            self.cache = cache
            assert (memory.spans is not None) == clean
            metadata = cache.augment_metadata(metadata, items, 3, noisy=not clean)
            if clean:
                self.clean_frames.append(cache.rgb_frames)
        metadata = replace(metadata, teacher_forcing_replay_policy=policy)
        weights = sum(_pass_weights(build_replay_maskless_plan(metadata, items)).values())  # [G,U+G+M]
        modalities = [modality for modality in (pack.vision, pack.lidar) if modality is not None]
        tensors = [token for modality in modalities for token in modality.tokens]  # each [1,C,V*T,H,W]
        sensors = torch.cat([token.flatten(2).transpose(1, 2)[0] for token in tensors])  # [G,C]
        captions = torch.sin(torch.arange(_TEXT * _WIDTH).reshape(_TEXT, _WIDTH) * 0.23)  # [U,C]
        hidden = torch.cat((captions, sensors))  # [U+G,C]
        memory.init(
            {"causal_seq": captions, "_num_causal_tokens": _TEXT, "_num_full_tokens": count}, torch.device("cpu")
        )
        caption_weights = torch.block_diag(torch.ones(2, 2).tril(), torch.ones(2, 2).tril())  # [U,U]
        if clean:
            self.last_clean_kv = []
        for layer, (projection, output) in enumerate(zip(self.net.qkv, self.net.output, strict=True)):
            projected = projection(hidden).reshape(1, hidden.shape[0], 3, _HEADS, _WIDTH // _HEADS)
            # projected: [1,U+G,3,H,D]
            q, k, v = projected.unbind(2)  # each [1,U+G,H,D]
            q_ = _rotate(q, pack.position_ids)  # [1,U+G,H,D]
            k_ = _rotate(k, pack.position_ids)  # [1,U+G,H,D]
            state = memory.read_for_layer(layer)
            cached_k: torch.Tensor | None = None  # [1,M,H,D] or None
            cached_v: torch.Tensor | None = None  # [1,M,H,D] or None
            if isinstance(state, JointARMemoryValue):
                cached_k, cached_v = state.cached_gen_k, state.cached_gen_v  # each [1,M,H,D] or None
                if cached_k is not None:
                    self.history_read_layers.add(layer)
            elif isinstance(state, TFNoisyMemoryValue):
                cached_k, cached_v = state.cached_clean_gen_k, state.cached_clean_gen_v  # each [1,M,H,D]
            keys = k_ if cached_k is None else torch.cat((k_, cached_k), dim=1)  # [1,U+G+M,H,D]
            values = v if cached_v is None else torch.cat((v, cached_v), dim=1)  # [1,U+G+M,H,D]
            sensor_output = _dense_reference(
                q_[:, _TEXT:], keys, values, weights, torch.ones(count, dtype=torch.bool)
            )  # [1,G,H,D]
            caption_output = _dense_reference(
                q_[:, :_TEXT], k_[:, :_TEXT], v[:, :_TEXT], caption_weights, torch.ones(_TEXT, dtype=torch.bool)
            )  # [1,U,H,D]
            combined = torch.cat((caption_output, sensor_output), dim=1).reshape(-1, _WIDTH)  # [U+G,C]
            hidden = hidden + 0.3 * torch.tanh(output(combined))  # [U+G,C]
            before = (cached_k.clone(), cached_v.clone()) if cached_k is not None and cached_v is not None else None
            # before: tuple[[1,M,H,D],[1,M,H,D]] or None
            memory.write_for_layer(layer, (k_[:, _TEXT:], v[:, _TEXT:], k_[:, :_TEXT], v[:, :_TEXT]))
            if clean:
                self.last_clean_kv.append((k_[:, _TEXT:].clone(), v[:, _TEXT:].clone()))  # each [1,G,H,D]
            elif cache is not None:
                if before is None:
                    assert cache.memory.cache[layer] is None
                else:
                    assert cache.memory.cache[layer] is not None
                    for old, new in zip(before, cache.memory.cache[layer], strict=True):  # each [1,M,H,D]
                        torch.testing.assert_close(old, new, rtol=0, atol=0)
        predictions: dict[str, list[torch.Tensor]] = {}
        offset = _TEXT
        for name, modality in (("vision", pack.vision), ("lidar", pack.lidar)):
            predictions[f"preds_{name}"] = []
            if modality is None:
                predictions[f"preds_{name}"] = [hidden.sum().reshape(1) * 0]  # list[[1]]
                continue
            for tensor in modality.tokens:  # [1,C,V*T,H,W]
                length = tensor[0, 0].numel()
                prediction = hidden[offset : offset + length].T.reshape_as(tensor)  # [1,C,V*T,H,W]
                predictions[f"preds_{name}"].append(0.2 * prediction)  # [1,C,V*T,H,W]
                offset += length
        return predictions


@pytest.mark.level(0)
@pytest.mark.gpus(0)
@pytest.mark.parametrize(
    "has_lidar,conditioned", [(False, 1), (True, 0), (True, 2)], ids=["lidar_dropout", "joint", "two_frame"]
)
def test_persistent_critic_rollout_matches_real_multilayer_replay(
    monkeypatch: pytest.MonkeyPatch, has_lidar: bool, conditioned: int
) -> None:
    # Keep real packing objects on CPU; all branch/cache/commit functions remain unmodified.
    monkeypatch.setattr(PackedSequence, "to_cuda", lambda self: None)
    torch.manual_seed(181)
    model = _DenseJointModel()
    vision = torch.sin(torch.arange(_WIDTH * 14 * 2).reshape(1, _WIDTH, 14, 1, 2) * 0.17)  # [1,C,2*7,1,2]
    lidar = torch.cos(torch.arange(_WIDTH * 9 * 2).reshape(1, _WIDTH, 9, 1, 2) * 0.13)  # [1,C,9,1,2]
    data = GenerationDataClean(
        batch_size=1,
        is_image_batch=False,
        x0_tokens_vision=[vision, vision + 0.4],  # list[[1,C,2*7,1,2]]
        x0_tokens_lidar=[lidar, lidar - 0.3] if has_lidar else None,  # list[[1,C,9,1,2]] or None
        num_vision_items_per_sample=[2],
        num_lidar_items_per_sample=[2] if has_lidar else None,
        num_views_per_vision_item=[2, 2],
    )
    plans = [
        SequencePlan(
            has_text=True,
            has_vision=True,
            has_lidar=has_lidar,
            condition_frame_indexes_vision=list(range(conditioned)),
            condition_frame_indexes_lidar=list(range((0, 1, 3)[conditioned])) if has_lidar else [],
        )
    ]
    with torch.no_grad():
        torch.manual_seed(379)
        replay = model._joint_rollout(data, plans, [[1]])
        replay_rng = torch.get_rng_state()  # [R]
        torch.manual_seed(379)
        persistent = model._joint_rollout(data, plans, [[1]], use_persistent_kv=True)
        persistent_rng = torch.get_rng_state()  # [R]
        assert replay.frames and not persistent.frames
        torch.testing.assert_close(persistent_rng, replay_rng, rtol=0, atol=0)
        for name in model._joint_modalities(data):
            torch.testing.assert_close(
                getattr(persistent.data, f"x0_tokens_{name}")[1], getattr(replay.data, f"x0_tokens_{name}")[1]
            )
        assert model.clean_frames == [(0,), (1, 2, 3)]
        assert model.history_read_layers == set(range(_LAYERS))
        cache = model.cache
        assert cache is not None and cache.metadata is not None and cache.memory.spans is None
        # Independently enumerate physical slots; future slots must stay unoccupied/zero.
        physical_frames = [frame for _ in range(4) for frame in range(7) for _ in range(2)]
        committed = [frame < 4 for frame in physical_frames]
        if has_lidar:
            physical_frames += [frame for _ in range(2) for frame in range(9) for _ in range(2)]
            committed += [frame < 5 for _ in range(2) for frame in range(9) for _ in range(2)]
        occupied = torch.tensor(committed)  # [M]
        expected_frames = torch.tensor(physical_frames).masked_fill(~occupied, -1)  # [M]
        torch.testing.assert_close(cache.metadata["frame_id"], expected_frames, rtol=0, atol=0)
        assert not cache.metadata["is_noisy"].any()
        # Recompute the finalized full clip independently. Every occupied control,
        # condition and target slot must match its clean K/V at every layer, with RoPE.
        _build_replay_branch(model, plans, persistent.data, [[1]])
        for stored, expected in zip(cache.memory.cache, model.last_clean_kv, strict=True):
            assert stored is not None
            for found, wanted in zip(stored, expected, strict=True):  # each [1,M,H,D]
                assert not found.requires_grad and found.grad_fn is None
                torch.testing.assert_close(found[:, occupied], wanted[:, occupied])
                assert torch.count_nonzero(found[:, ~occupied]) == 0  # []
