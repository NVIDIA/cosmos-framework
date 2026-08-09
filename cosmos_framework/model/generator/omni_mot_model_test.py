# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


def test_reasoner_only_setup_skips_vision_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    from cosmos_framework.model.generator import omni_mot_model

    vlm_tokenizer = SimpleNamespace(eos_token_id=42)
    vlm_processor = SimpleNamespace(tokenizer=vlm_tokenizer)
    vlm_config = SimpleNamespace(tokenizer="vlm-tokenizer-config")
    vision_config = SimpleNamespace(temporal_compression_factor=4)
    config = SimpleNamespace(
        load_vision_tokenizer=False,
        sound_gen=False,
        tokenizer=vision_config,
        vlm_config=vlm_config,
    )
    instantiated = []

    def _instantiate(candidate):
        instantiated.append(candidate)
        return vlm_processor

    monkeypatch.setattr(omni_mot_model, "lazy_instantiate", _instantiate)
    monkeypatch.setattr(omni_mot_model, "add_special_tokens", lambda tokenizer: (tokenizer, {}))

    model = SimpleNamespace(config=config)
    omni_mot_model.OmniMoTModel.set_up_tokenizers(model)

    assert instantiated == [vlm_config.tokenizer]
    assert model.tokenizer_vision_gen is None
    assert model.tokenizer_sound_gen is None


def test_default_setup_loads_vision_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    from cosmos_framework.model.generator import omni_mot_model

    vlm_tokenizer = SimpleNamespace(eos_token_id=42)
    vlm_processor = SimpleNamespace(tokenizer=vlm_tokenizer)
    vision_tokenizer = SimpleNamespace(latent_ch=48, reset_dtype=Mock())
    vlm_config = SimpleNamespace(tokenizer="vlm-tokenizer-config")
    vision_config = SimpleNamespace(temporal_compression_factor=4)
    config = SimpleNamespace(
        load_vision_tokenizer=True,
        sound_gen=False,
        state_ch=48,
        tokenizer=vision_config,
        vlm_config=vlm_config,
    )

    def _instantiate(candidate):
        if candidate == vlm_config.tokenizer:
            return vlm_processor
        assert candidate is vision_config
        return vision_tokenizer

    monkeypatch.setattr(omni_mot_model, "lazy_instantiate", _instantiate)
    monkeypatch.setattr(omni_mot_model, "add_special_tokens", lambda tokenizer: (tokenizer, {}))

    model = SimpleNamespace(config=config)
    omni_mot_model.OmniMoTModel.set_up_tokenizers(model)

    assert model.tokenizer_vision_gen is vision_tokenizer
    vision_tokenizer.reset_dtype.assert_called_once_with()


def test_transfer_prefix_encode_keeps_controls_full_and_truncates_target() -> None:
    from cosmos_framework.model.generator import omni_mot_model

    class CausalTokenizer:
        is_causal = True

        @staticmethod
        def get_latent_num_frames(pixel_frames: int) -> int:
            return (pixel_frames - 1) // 4 + 1

        @staticmethod
        def get_pixel_num_frames(latent_frames: int) -> int:
            return (latent_frames - 1) * 4 + 1

    encoded_pixel_lengths: list[int] = []

    def encode_item(item: torch.Tensor, *, num_views: int, frames_per_view: int | None) -> torch.Tensor:
        assert num_views == 1
        assert frames_per_view is None
        pixel_frames = item.shape[2]
        encoded_pixel_lengths.append(pixel_frames)
        latent_frames = CausalTokenizer.get_latent_num_frames(pixel_frames)
        return torch.ones(1, 2, latent_frames, 1, 1)

    model = SimpleNamespace(
        tokenizer_vision_gen=CausalTokenizer(),
        _encode_vision_item=encode_item,
    )
    controls_and_target = [torch.zeros(1, 3, 97, 2, 2) for _ in range(3)]

    with torch.no_grad():
        latents = omni_mot_model.OmniMoTModel._encode_vision_x0_tokens(
            model,
            controls_and_target,
            num_vision_items_per_sample=[3],
            vision_condition_indexes=[[0, 1, 2, 3]],
            prefix_encode_last_vision_item_per_sample=True,
        )

    assert encoded_pixel_lengths == [97, 97, 13]
    assert [latent.shape[2] for latent in latents] == [25, 25, 25]
    assert torch.count_nonzero(latents[-1][:, :, 4:]) == 0
