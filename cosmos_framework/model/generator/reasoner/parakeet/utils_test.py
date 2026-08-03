# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU tests for audio placeholder merge helpers."""

import inspect
from dataclasses import dataclass
from typing import Any

import pytest
import torch
from torch import nn
from transformers.utils import ModelOutput

from cosmos_framework.model.generator.reasoner.parakeet.utils import (
    merge_audio_embeddings,
    merge_projected_audio_embeddings,
    patch_reasoner_audio_forward,
)

_AUDIO_TOKEN_ID = 9
_AUDIO_START_TOKEN_ID = 6
_AUDIO_END_TOKEN_ID = 7
_IMAGE_TOKEN_ID = 8
_HIDDEN_SIZE = 3


@dataclass
class _FakeReasonerOutput(ModelOutput):
    logits: torch.Tensor | None = None


@dataclass(frozen=True)
class _FakeReasonerConfig:
    tie_word_embeddings: bool = False


class _RecordingEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(_HIDDEN_SIZE), requires_grad=False)
        self.forward_calls = 0

    def forward(
        self,
        audio_features: torch.Tensor,
        audio_feature_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.forward_calls += 1
        return audio_features * self.scale, audio_feature_lengths


class _RecordingProjector(nn.Module):
    input_hidden_size = _HIDDEN_SIZE

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(_HIDDEN_SIZE, _HIDDEN_SIZE, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(_HIDDEN_SIZE))
        self.forward_calls = 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        return self.linear(hidden_states)


class _RecordingAudioModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _RecordingEncoder()
        self.projector = _RecordingProjector()

    def forward(
        self,
        audio_features: torch.Tensor,
        audio_feature_lengths: torch.Tensor,
        audio_token_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded, _ = self.encoder(audio_features, audio_feature_lengths)
        return self.projector(encoded), audio_token_lengths.to(torch.int32)


class _RecordingInnerModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rope_calls: list[dict[str, Any]] = []
        self.rope_deltas: torch.Tensor | None = None

    def get_rope_index(
        self,
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.rope_calls.append(
            {
                "input_ids": input_ids,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "attention_mask": attention_mask,
                "cu_seqlens": cu_seqlens,
            }
        )
        position_ids = torch.full(
            (3, input_ids.shape[0], input_ids.shape[1]),
            17,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        rope_deltas = torch.full(
            (input_ids.shape[0], 1),
            5,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        return position_ids, rope_deltas


class _FakeRawReasoner(nn.Module):
    """Tiny conditional model whose forward performs native-style vision scatter."""

    def __init__(self, *, tie_word_embeddings: bool = False) -> None:
        super().__init__()
        self.config = _FakeReasonerConfig(tie_word_embeddings=tie_word_embeddings)
        self.embedding = nn.Embedding(32, _HIDDEN_SIZE)
        with torch.no_grad():
            values = torch.arange(32 * _HIDDEN_SIZE, dtype=torch.float32).reshape(32, _HIDDEN_SIZE)
            self.embedding.weight.copy_(values)
        self.lm_head: nn.Linear | None = None
        if tie_word_embeddings:
            self.lm_head = nn.Linear(_HIDDEN_SIZE, 32, bias=False)
            self.lm_head.weight = self.embedding.weight
        self.model = _RecordingInnerModel()
        self.sound_und_model = _RecordingAudioModel()
        self.last_forward: dict[str, Any] = {}

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | dict[str, torch.Tensor] | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        return_dict: bool = True,
        **kwargs,
    ) -> _FakeReasonerOutput | tuple:
        self.last_forward = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "inputs_embeds": inputs_embeds,
            "labels": labels,
            "pixel_values": pixel_values,
            "pixel_values_videos": pixel_values_videos,
            "image_grid_thw": image_grid_thw,
            "video_grid_thw": video_grid_thw,
            "cu_seqlens": cu_seqlens,
            **kwargs,
        }
        hidden_states = self.embedding(input_ids) if inputs_embeds is None else inputs_embeds
        if pixel_values is not None:
            image_token_embedding = self.embedding.weight[_IMAGE_TOKEN_ID]
            image_mask = (hidden_states == image_token_embedding).all(dim=-1).unsqueeze(-1).expand_as(hidden_states)
            hidden_states = hidden_states.masked_scatter(image_mask, pixel_values)
        logits = self.lm_head(hidden_states) if self.lm_head is not None else hidden_states  # [B,N,V] or [B,N,D]
        if return_dict:
            return _FakeReasonerOutput(logits=logits)
        if labels is not None:
            return logits.mean(), logits, "past"
        return logits, "past"


def _audio_forward_kwargs(audio_features: torch.Tensor, sample_lengths: list[int]) -> dict[str, torch.Tensor]:
    num_clips, audio_time = audio_features.shape[:2]
    clip_lengths = torch.full((num_clips,), audio_time, dtype=torch.long)
    return {
        "audio_features": audio_features,
        "audio_feature_lengths": clip_lengths,
        "audio_token_lengths": clip_lengths,
        "audio_sample_token_lengths": torch.tensor(sample_lengths, dtype=torch.long),
    }


@pytest.mark.L0
@pytest.mark.CPU
def test_merge_supports_batched_embeddings() -> None:
    input_ids = torch.tensor([[1, 9, 2, 9], [9, 3, 4, 5]], dtype=torch.long)
    inputs_embeds = torch.zeros(2, 4, 3)
    audio_embeddings = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )

    merged = merge_audio_embeddings(input_ids, inputs_embeds, audio_embeddings, audio_token_id=9)

    assert torch.equal(merged[0, 1], audio_embeddings[0])
    assert torch.equal(merged[0, 3], audio_embeddings[1])
    assert torch.equal(merged[1, 0], audio_embeddings[2])
    assert torch.count_nonzero(inputs_embeds) == 0


@pytest.mark.L0
@pytest.mark.CPU
def test_merge_supports_packed_embeddings() -> None:
    input_ids = torch.tensor([9, 1, 9, 2], dtype=torch.long)
    inputs_embeds = torch.zeros(4, 2)
    audio_embeddings = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    merged = merge_audio_embeddings(input_ids, inputs_embeds, audio_embeddings, audio_token_id=9)

    assert torch.equal(merged, torch.tensor([[1.0, 2.0], [0.0, 0.0], [3.0, 4.0], [0.0, 0.0]]))


@pytest.mark.L0
@pytest.mark.CPU
def test_merge_rejects_placeholder_feature_count_mismatch() -> None:
    input_ids = torch.tensor([9, 1, 9], dtype=torch.long)
    inputs_embeds = torch.zeros(3, 2)
    audio_embeddings = torch.ones(1, 2)

    with pytest.raises(ValueError, match="do not match"):
        merge_audio_embeddings(input_ids, inputs_embeds, audio_embeddings, audio_token_id=9)


@pytest.mark.L0
@pytest.mark.CPU
def test_projected_merge_rejects_cross_row_placeholder_mismatch_with_equal_global_total() -> None:
    input_ids = torch.tensor([[9, 9, 1], [2, 9, 3]], dtype=torch.long)

    with pytest.raises(ValueError, match="every text row"):
        merge_projected_audio_embeddings(
            input_ids,
            torch.zeros(2, 3, 2),
            torch.ones(2, 2, 2),
            audio_embedding_lengths=torch.tensor([1, 2]),
            audio_sample_token_lengths=torch.tensor([1, 2]),
            audio_token_id=9,
        )


@pytest.mark.L0
@pytest.mark.CPU
def test_raw_forward_patch_supports_vision_audio_and_audio_visual_rows() -> None:
    model = _FakeRawReasoner()
    patch_reasoner_audio_forward(model, audio_token_id=_AUDIO_TOKEN_ID, model_type="qwen3_vl")
    input_ids = torch.tensor(
        [
            [1, _IMAGE_TOKEN_ID, 2, 3],
            [1, _AUDIO_TOKEN_ID, _AUDIO_TOKEN_ID, 2],
            [_IMAGE_TOKEN_ID, _AUDIO_TOKEN_ID, 1, 2],
        ]
    )
    attention_mask = torch.ones_like(input_ids)
    image_grid_thw = torch.tensor([[1, 1, 1], [1, 1, 1]])
    image_embeddings = torch.tensor([[101.0, 102.0, 103.0], [201.0, 202.0, 203.0]])
    audio_features = torch.tensor(
        [
            [[11.0, 12.0, 13.0]],
            [[14.0, 15.0, 16.0]],
            [[21.0, 22.0, 23.0]],
        ]
    )

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=image_embeddings,
        image_grid_thw=image_grid_thw,
        audio_features=audio_features,
        audio_feature_lengths=torch.tensor([1, 1, 1]),
        audio_token_lengths=torch.tensor([1, 1, 1]),
        audio_sample_token_lengths=torch.tensor([0, 2, 1]),
    )

    assert isinstance(output, _FakeReasonerOutput)
    assert torch.equal(output.logits[0, 1], image_embeddings[0])
    assert torch.equal(output.logits[1, 1:3], audio_features[:2, 0])
    assert torch.equal(output.logits[2, 0], image_embeddings[1])
    assert torch.equal(output.logits[2, 1], audio_features[2, 0])
    assert torch.equal(output.logits[0, 0], model.embedding.weight[1])
    assert torch.equal(output.logits[1, 0], model.embedding.weight[1])
    assert torch.equal(output.logits[2, 2], model.embedding.weight[1])
    assert model.sound_und_model.encoder.forward_calls == 1
    assert model.sound_und_model.projector.forward_calls == 1

    assert model.last_forward["input_ids"] is None
    received_embeddings = model.last_forward["inputs_embeds"]
    assert torch.equal(received_embeddings[0, 1], model.embedding.weight[_IMAGE_TOKEN_ID])
    assert torch.equal(received_embeddings[1, 1:3], audio_features[:2, 0])
    assert torch.equal(received_embeddings[2, 0], model.embedding.weight[_IMAGE_TOKEN_ID])
    assert torch.equal(received_embeddings[2, 1], audio_features[2, 0])
    assert torch.equal(model.last_forward["position_ids"], torch.full((3, 3, 4), 17))
    assert len(model.model.rope_calls) == 1
    rope_call = model.model.rope_calls[0]
    assert rope_call["input_ids"] is input_ids
    assert rope_call["image_grid_thw"] is image_grid_thw
    assert rope_call["video_grid_thw"] is None
    assert rope_call["attention_mask"] is attention_mask
    assert torch.equal(model.model.rope_deltas, torch.full((3, 1), 5))

    output.logits.sum().backward()
    projector_grad = model.sound_und_model.projector.linear.weight.grad
    assert projector_grad is not None
    assert torch.count_nonzero(projector_grad) > 0
    assert model.sound_und_model.encoder.scale.grad is None


@pytest.mark.L0
@pytest.mark.CPU
def test_raw_forward_patch_restricts_input_embedding_gradients_to_boundary_tokens() -> None:
    model = _FakeRawReasoner()
    model.requires_grad_(False)
    model.embedding.weight.requires_grad_(True)
    model.sound_und_model.projector.requires_grad_(True)
    patch_reasoner_audio_forward(
        model,
        audio_token_id=_AUDIO_TOKEN_ID,
        model_type="qwen3_vl",
        trainable_token_ids=(_AUDIO_START_TOKEN_ID, _AUDIO_END_TOKEN_ID),
    )
    input_ids = torch.tensor(  # [1,5]
        [[1, _AUDIO_START_TOKEN_ID, _AUDIO_TOKEN_ID, _AUDIO_END_TOKEN_ID, 2]],
        dtype=torch.long,
    )
    audio_features = torch.ones(1, 1, _HIDDEN_SIZE)  # [1,1,D]

    output = model(
        input_ids=input_ids,
        **_audio_forward_kwargs(audio_features, [1]),
    )
    output.logits.sum().backward()  # scalar

    embedding_grad = model.embedding.weight.grad  # [V,D]
    assert embedding_grad is not None
    rows_with_grad = torch.nonzero(embedding_grad.abs().sum(dim=1), as_tuple=False).flatten()  # [2]
    expected_rows = torch.tensor([_AUDIO_START_TOKEN_ID, _AUDIO_END_TOKEN_ID])  # [2]
    assert torch.equal(rows_with_grad, expected_rows)
    projector_grad = model.sound_und_model.projector.linear.weight.grad  # [D,D]
    assert projector_grad is not None
    assert torch.count_nonzero(projector_grad) > 0
    assert model.sound_und_model.encoder.scale.grad is None


@pytest.mark.L0
@pytest.mark.CPU
def test_boundary_token_training_rejects_no_audio_batch() -> None:
    model = _FakeRawReasoner()
    patch_reasoner_audio_forward(
        model,
        audio_token_id=_AUDIO_TOKEN_ID,
        model_type="qwen3_vl",
        trainable_token_ids=(_AUDIO_START_TOKEN_ID, _AUDIO_END_TOKEN_ID),
    )
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)  # [B,N]

    with pytest.raises(ValueError, match="requires audio inputs on every training batch"):
        model(input_ids=input_ids)

    model.eval()
    with torch.no_grad():
        output = model(input_ids=input_ids)
        expected_logits = model.embedding(input_ids)  # [B,N,D]
    assert isinstance(output, _FakeReasonerOutput)
    assert torch.equal(output.logits, expected_logits)


@pytest.mark.L0
@pytest.mark.CPU
def test_tied_output_embedding_does_not_update_shared_weight() -> None:
    model = _FakeRawReasoner(tie_word_embeddings=True)
    model.requires_grad_(False)
    model.embedding.weight.requires_grad_(True)
    model.sound_und_model.projector.requires_grad_(True)
    assert model.lm_head is not None
    assert model.lm_head.weight is model.embedding.weight
    probe_hidden_states = torch.randn(1, 2, _HIDDEN_SIZE)  # [B,N,D]
    expected_logits = model.lm_head(probe_hidden_states).detach()  # [B,N,V]

    patch_reasoner_audio_forward(
        model,
        audio_token_id=_AUDIO_TOKEN_ID,
        model_type="qwen3_vl",
        trainable_token_ids=(_AUDIO_START_TOKEN_ID, _AUDIO_END_TOKEN_ID),
    )

    actual_logits = model.lm_head(probe_hidden_states)  # [B,N,V]
    assert torch.equal(actual_logits, expected_logits)
    assert tuple(inspect.signature(model.lm_head.forward).parameters) == ("hidden_states",)
    input_ids = torch.tensor(  # [B,N]
        [[1, _AUDIO_START_TOKEN_ID, _AUDIO_TOKEN_ID, _AUDIO_END_TOKEN_ID, 2]],
        dtype=torch.long,
    )
    audio_features = torch.ones(1, 1, _HIDDEN_SIZE)  # [C,T,D]

    output = model(
        input_ids=input_ids,
        **_audio_forward_kwargs(audio_features, [1]),
    )
    output.logits.sum().backward()  # scalar

    embedding_grad = model.embedding.weight.grad  # [V,D]
    assert embedding_grad is not None
    rows_with_grad = torch.nonzero(embedding_grad.abs().sum(dim=1), as_tuple=False).flatten()  # [2]
    expected_rows = torch.tensor([_AUDIO_START_TOKEN_ID, _AUDIO_END_TOKEN_ID])  # [2]
    assert torch.equal(rows_with_grad, expected_rows)
    projector_grad = model.sound_und_model.projector.linear.weight.grad  # [D,D]
    assert projector_grad is not None
    assert torch.count_nonzero(projector_grad) > 0


@pytest.mark.L0
@pytest.mark.CPU
def test_raw_forward_patch_uses_edge_native_cu_seqlens() -> None:
    model = _FakeRawReasoner()
    patch_reasoner_audio_forward(model, audio_token_id=_AUDIO_TOKEN_ID, model_type="nemotron_siglip2")
    input_ids = torch.tensor([[_AUDIO_TOKEN_ID, 1]])
    cu_seqlens = torch.tensor([0, 2], dtype=torch.int32)

    model(
        input_ids=input_ids,
        cu_seqlens=cu_seqlens,
        **_audio_forward_kwargs(torch.ones(1, 1, _HIDDEN_SIZE), [1]),
    )

    assert model.model.rope_calls[0]["input_ids"] is input_ids
    assert model.model.rope_calls[0]["cu_seqlens"] is cu_seqlens
    assert model.last_forward["cu_seqlens"] is cu_seqlens


@pytest.mark.L0
@pytest.mark.CPU
def test_raw_forward_patch_preserves_explicit_position_ids() -> None:
    model = _FakeRawReasoner()
    patch_reasoner_audio_forward(model, audio_token_id=_AUDIO_TOKEN_ID, model_type="qwen3_vl")
    position_ids = torch.full((3, 1, 2), 29)

    model(
        input_ids=torch.tensor([[_AUDIO_TOKEN_ID, 1]]),
        position_ids=position_ids,
        **_audio_forward_kwargs(torch.ones(1, 1, _HIDDEN_SIZE), [1]),
    )

    assert model.last_forward["position_ids"] is position_ids
    assert model.model.rope_calls == []


@pytest.mark.L0
@pytest.mark.CPU
def test_no_audio_forward_is_unchanged_and_connects_only_trainable_projector() -> None:
    model = _FakeRawReasoner()
    patch_reasoner_audio_forward(model, audio_token_id=_AUDIO_TOKEN_ID, model_type="qwen3_vl")
    input_ids = torch.tensor([[1, 2, 3]])
    expected_logits = model.embedding(input_ids).detach()

    output = model(input_ids)

    assert isinstance(output, _FakeReasonerOutput)
    assert torch.equal(output.logits, expected_logits)
    assert model.last_forward["input_ids"] is input_ids
    assert model.last_forward["inputs_embeds"] is None
    assert model.last_forward["position_ids"] is None
    assert model.model.rope_calls == []
    assert model.sound_und_model.encoder.forward_calls == 0
    assert model.sound_und_model.projector.forward_calls == 1

    output.logits.sum().backward()
    projector_grad = model.sound_und_model.projector.linear.weight.grad
    assert projector_grad is not None
    assert torch.count_nonzero(projector_grad) == 0
    assert model.sound_und_model.encoder.scale.grad is None
