# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Offline CPU qualification of the Transformers 5.8.1 dense Qwen3.5 SFT path."""

from __future__ import annotations

import copy
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load_file, save_file

transformers = pytest.importorskip("transformers", minversion="5.8.1")
from transformers import AutoConfig, AutoModelForImageTextToText, Qwen3_5Config

from cosmos_framework.posttrain.model.hf import HFModel
from cosmos_framework.posttrain.model.qwen3_5 import patch_qwen3_5_forward
from cosmos_framework.posttrain.model.weight_loader import load_weights

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        yield
    torch.set_num_threads(threads)


@pytest.fixture
def config() -> Qwen3_5Config:
    return Qwen3_5Config(
        text_config=dict(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            linear_num_key_heads=1,
            linear_num_value_heads=2,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            layer_types=["linear_attention", "full_attention"],
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0,
                "mrope_section": [2, 3, 3],
            },
        ),
        vision_config=dict(
            depth=2,
            hidden_size=16,
            intermediate_size=32,
            num_heads=2,
            in_channels=3,
            patch_size=2,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=32,
            num_position_embeddings=16,
        ),
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=62,
        vision_end_token_id=63,
    )


def _batch(config: Qwen3_5Config, modality: str, dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
    ids, types = [1, 2], [0, 0]
    batch = {}
    vision = config.vision_config
    width = vision.in_channels * vision.temporal_patch_size * vision.patch_size**2
    for kind, token, type_id, pixels_key, grid_key in (
        ("image", config.image_token_id, 1, "pixel_values", "image_grid_thw"),
        ("video", config.video_token_id, 2, "pixel_values_videos", "video_grid_thw"),
    ):
        if modality in (kind, "both"):
            # Two temporally distinct video frames, one image; each yields one merged token.
            frames = 2 if kind == "video" else 1
            for _ in range(frames):
                ids.extend([config.vision_start_token_id, token, config.vision_end_token_id])
                types.extend([0, type_id, 0])
            batch[pixels_key] = torch.randn(frames * vision.spatial_merge_size**2, width, dtype=dtype)
            batch[grid_key] = torch.tensor([[frames, vision.spatial_merge_size, vision.spatial_merge_size]])
    ids.extend([3, 4, 0])
    types.extend([0, 0, 0])
    batch["input_ids"] = torch.tensor([ids])  # [B,L]
    batch["mm_token_type_ids"] = torch.tensor([types])  # [B,L]
    batch["attention_mask"] = torch.tensor([[1] * (len(ids) - 1) + [0]])  # right padding
    return batch


@pytest.mark.parametrize("modality", ["text", "image", "video", "both"])
@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_ordered_vision_matches_upstream(
    config: Qwen3_5Config, modality: str, checkpointing: bool, dtype: torch.dtype
) -> None:
    source = AutoModelForImageTextToText.from_config(config, attn_implementation="eager", dtype=dtype).train()
    adapted = copy.deepcopy(source)
    patch_qwen3_5_forward(adapted.model)
    if checkpointing:
        for model in (source, adapted):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    calls: list[int] = []
    backward: list[tuple[int, int]] = []
    block_calls: list[int] = []
    blocks = list(adapted.model.visual.blocks)

    def block_pre(module: torch.nn.Module, args: tuple[Any, ...]) -> None:
        block_calls.append(blocks.index(module))

    def vision_pre(module: torch.nn.Module, args: tuple[Any, ...]) -> None:
        calls.append(len(calls))

    def block_output(index: int) -> Any:
        def record(module: torch.nn.Module, args: tuple[Any, ...], output: torch.Tensor) -> None:
            phase = calls[-1]

            def grad_hook(grad: torch.Tensor) -> None:
                backward.append((phase, index))

            output.register_hook(grad_hook)

        return record

    handles = [adapted.model.visual.register_forward_pre_hook(vision_pre)]
    handles.extend(block.register_forward_hook(block_output(i)) for i, block in enumerate(blocks))
    handles.extend(block.register_forward_pre_hook(block_pre) for block in blocks)
    batch = _batch(config, modality, dtype)
    batch = {key: value.requires_grad_(value.is_floating_point()) for key, value in batch.items()}
    source_batch = {
        key: value.detach().clone().requires_grad_(value.is_floating_point()) for key, value in batch.items()
    }
    expected = source(**source_batch, use_cache=False)
    actual = adapted(**batch, use_cache=False)
    assert actual.past_key_values is None and expected.past_key_values is None
    torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
    actual.logits.float().square().mean().backward()
    expected.logits.float().square().mean().backward()
    assert calls == [0, 1]
    assert backward == [(1, 1), (1, 0), (0, 1), (0, 0)]
    assert block_calls == [0, 1, 0, 1] + ([1, 0, 1, 0] if checkpointing else [])
    for key, value in batch.items():
        if value.is_floating_point():
            assert value.grad is not None and source_batch[key].grad is not None
            torch.testing.assert_close(value.grad, source_batch[key].grad, rtol=1e-5, atol=1e-6)
    for handle in handles:
        handle.remove()
    for (name, param), (expected_name, expected_param) in zip(adapted.named_parameters(), source.named_parameters()):
        assert name == expected_name
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
        if expected_param.grad is None:
            assert name.startswith("model.visual.") and modality == "text"
            assert torch.count_nonzero(param.grad) == 0, name
        else:
            tolerance = 1e-6 if dtype == torch.float32 else 2e-3
            torch.testing.assert_close(param.grad, expected_param.grad, rtol=tolerance, atol=tolerance, msg=name)
    # No recurrent state persists between no-cache training forwards.
    repeated = adapted(**batch, use_cache=False)
    torch.testing.assert_close(repeated.logits, actual.logits, rtol=0, atol=0)


@pytest.mark.parametrize("tied", [False, True])
def test_hf_meta_load_and_mtp_exclusion(config: Qwen3_5Config, tmp_path: Path, tied: bool) -> None:
    config.tie_word_embeddings = tied
    config.text_config.tie_word_embeddings = tied
    source = AutoModelForImageTextToText.from_config(config, attn_implementation="eager", dtype=torch.bfloat16)
    source.save_pretrained(tmp_path)
    checkpoint = tmp_path / "model.safetensors"
    weights = load_file(checkpoint)
    # Dense HF does not construct the optional MTP head. Extra checkpoint tensors must not
    # weaken required-parameter completeness or become trainable parameters.
    extras = {f"mtp.layers.0.extra_{i}": torch.randn(1) for i in range(15)}
    assert all(any(re.search(p, key) for p in source._keys_to_ignore_on_load_unexpected) for key in extras)
    weights.update(extras)
    save_file(weights, checkpoint)

    loaded_config = AutoConfig.from_pretrained(tmp_path, local_files_only=True)
    assert loaded_config.model_type == "qwen3_5"
    model = HFModel(str(tmp_path), dtype=torch.float32, attn_implementation="eager", grad_ckpt=True)
    assert all(param.is_meta for param in model.net.parameters())
    assert {block.__name__ for block in model.shard_plan.block_types} == {"Qwen3_5DecoderLayer", "Qwen3_5VisionBlock"}
    model.setup(None)
    assert all(not param.is_meta and param.dtype == torch.float32 for param in model.net.parameters())
    assert model.net.is_gradient_checkpointing
    assert not any(name.startswith("mtp.") for name, _ in model.net.named_parameters())
    assert set(model.net.state_dict()) == set(source.state_dict())
    for name, expected in source.state_dict().items():
        torch.testing.assert_close(model.net.state_dict()[name], expected.float(), rtol=0, atol=0)
    if tied:
        assert model.net.get_output_embeddings().weight is model.net.get_input_embeddings().weight
    batch = _batch(config, "both")
    torch.testing.assert_close(model(**batch), model(**batch, use_cache=True), rtol=0, atol=0)
    model(**batch).square().mean().backward()
    assert all(param.grad is not None and torch.isfinite(param.grad).all() for param in model.net.parameters())

    required = next(name for name in weights if name.startswith("model.visual.blocks."))
    del weights[required]
    save_file(weights, checkpoint)
    with pytest.raises(ValueError, match=re.escape(required)):
        load_weights(model.net, str(tmp_path), model.load_plan)
