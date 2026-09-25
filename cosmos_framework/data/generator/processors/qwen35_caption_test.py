# -----------------------------------------------------------------------------
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# -----------------------------------------------------------------------------

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import cosmos_framework.data.generator.augmentors.reasoner.filter_seq_length as filter_seq_length_module
import cosmos_framework.data.generator.processors as processor_factory
from cosmos_framework.configs.base.reasoner.defaults.config import DataSetting
from cosmos_framework.data.generator.augmentors.reasoner.bytes_to_media import BytesToMedia
from cosmos_framework.data.generator.augmentors.reasoner.filter_output_key import FilterOutputKey
from cosmos_framework.data.generator.augmentors.reasoner.filter_seq_length import FilterSeqLength
from cosmos_framework.data.generator.augmentors.reasoner.tokenize_data import (
    TokenizeData,
    expand_video_frames_for_framewise,
)
from cosmos_framework.data.generator.reasoner.collate_fn import custom_collate
from cosmos_framework.data.generator.reasoner.video_decoder_qwen import get_effective_temporal_patch_size
from cosmos_framework.data.generator.processors.qwen3vl_processor import Qwen3VLProcessor, maybe_parse_video_content
from cosmos_framework.utils.generator.reasoner.constant import PROCESSOR_KEYS_TO_ADD

pytestmark = [pytest.mark.L1, pytest.mark.CPU]


def _sample(length: int, modality: int = 0) -> dict:
    return {
        "input_ids": torch.arange(length),
        "token_mask": torch.ones(length, dtype=torch.bool),
        "attention_mask": torch.ones(length, dtype=torch.bool),
        "labels": torch.arange(length),
        "mm_token_type_ids": torch.full((length,), modality, dtype=torch.long),
        "pad_token_id": 0,
        "ignore_index": -100,
        "__key__": "sample",
        "__url__": SimpleNamespace(root="root", path="path"),
    }


def test_implicit_native_matches_explicit_native() -> None:
    processor = SimpleNamespace(temporal_patch_size=2)
    implicit = TokenizeData(processor=processor)
    explicit = TokenizeData(processor=processor, video_temporal_mode="native")
    assert implicit.video_temporal_mode == explicit.video_temporal_mode == "native"
    assert implicit.effective_temporal_patch_size == explicit.effective_temporal_patch_size == 2

    implicit_bytes = BytesToMedia(processor=processor)
    explicit_bytes = BytesToMedia(processor=processor, video_temporal_mode="native")
    assert implicit_bytes.video_decoder_params == explicit_bytes.video_decoder_params


def test_framewise_expands_frames_and_preserves_source_indices() -> None:
    frames = [object(), object(), object()]
    expanded_frames, expanded_indices = expand_video_frames_for_framewise(frames, [0, 5, 11], 2)
    assert expanded_frames == [frames[0], frames[0], frames[1], frames[1], frames[2], frames[2]]
    assert expanded_indices == [0, 0, 5, 5, 11, 11]
    assert maybe_parse_video_content(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": expanded_frames,
                        "fps": 24.0,
                        "frames_indices": expanded_indices,
                        "total_num_frames": 12,
                    }
                ],
            }
        ]
    ) == (1, [24.0], [12], [expanded_indices])


def test_unknown_temporal_modes_are_rejected() -> None:
    processor = SimpleNamespace(temporal_patch_size=2)
    with pytest.raises(ValueError, match="Unsupported video_temporal_mode"):
        TokenizeData(processor=processor, video_temporal_mode="invalid")
    with pytest.raises(ValueError, match="Unsupported video_temporal_mode"):
        BytesToMedia(processor=processor, video_temporal_mode="invalid")
    with pytest.raises(ValueError, match="Unsupported video_temporal_mode"):
        get_effective_temporal_patch_size(2, "invalid")


def test_dense_qwen35_paths_use_existing_qwen_processor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        processor_factory,
        "Qwen3VLProcessor",
        lambda name, **kwargs: (name, kwargs),
    )
    for model_path in ("Qwen/Qwen3.5-27B", "/models/qwen3.5-27b"):
        processor, _ = processor_factory.build_processor(model_path)
        assert processor == model_path


def test_processor_retains_and_rebuilds_modality_ids() -> None:
    class _Processor:
        def apply_chat_template(self, *args: object, **kwargs: object) -> dict:
            return {
                "input_ids": torch.tensor([[3, 21, 22, 4]]),
                "attention_mask": torch.ones((1, 4), dtype=torch.long),
                "mm_token_type_ids": torch.zeros((1, 5), dtype=torch.long),
            }

    processor = Qwen3VLProcessor.__new__(Qwen3VLProcessor)
    processor.processor = _Processor()
    processor.retain_mm_token_type_ids = True
    processor.image_token_id = 21
    processor.video_token_id = 22
    inputs = processor.apply_chat_template([{"role": "user", "content": "hello"}])
    assert torch.equal(inputs["mm_token_type_ids"], torch.tensor([0, 1, 2, 0]))
    assert "mm_token_type_ids" in PROCESSOR_KEYS_TO_ADD


def test_modality_ids_survive_filtering_and_padding() -> None:
    filtered = FilterOutputKey(text_only=True)(_sample(3, modality=2))
    assert torch.equal(filtered["mm_token_type_ids"], torch.full((3,), 2, dtype=torch.long))

    samples = [_sample(3, modality=1), _sample(5, modality=2)]
    for sample in samples:
        sample.pop("__key__")
        sample.pop("__url__")
    batch = custom_collate(samples)
    assert batch["mm_token_type_ids"].shape == (2, 16)
    assert torch.equal(batch["mm_token_type_ids"][0, :3], torch.ones(3, dtype=torch.long))
    assert torch.count_nonzero(batch["mm_token_type_ids"][0, 3:]) == 0


def test_length_filter_truncates_by_default_and_strict_drop_is_opt_in() -> None:
    processor = SimpleNamespace(image_token_id=21, video_token_id=22)
    default_sample = _sample(5)
    filtered = FilterSeqLength(max_token_length=4, processor=processor)(default_sample)
    assert filtered is not None
    for key in ("input_ids", "token_mask", "attention_mask", "labels", "mm_token_type_ids"):
        assert filtered[key].shape == (4,)

    strict_sample = _sample(5)
    strict_filter = FilterSeqLength(max_token_length=4, processor=processor, drop_over_max_length=True)
    assert strict_filter(strict_sample) is None
    assert strict_sample["input_ids"].shape == (5,)
    assert DataSetting().qwen_drop_over_max_length is False


def test_new_data_defaults_preserve_native_behavior() -> None:
    settings = DataSetting()
    assert settings.qwen_video_temporal_mode == "native"
    assert settings.qwen_drop_over_max_length is False


def test_legacy_processor_drops_new_modality_metadata() -> None:
    class _Processor:
        def apply_chat_template(self, *args: object, **kwargs: object) -> dict:
            return {
                "input_ids": torch.tensor([[3, 21, 4]]),
                "attention_mask": torch.ones((1, 3), dtype=torch.long),
                "mm_token_type_ids": torch.tensor([[0, 1, 0]]),
            }

    processor = Qwen3VLProcessor.__new__(Qwen3VLProcessor)
    processor.processor = _Processor()
    processor.retain_mm_token_type_ids = False
    processor.image_token_id = 21
    processor.video_token_id = 22

    inputs = processor.apply_chat_template([{"role": "user", "content": "hello"}])

    assert "mm_token_type_ids" not in inputs


def test_strict_drop_logging_is_warning_throttled(monkeypatch: pytest.MonkeyPatch) -> None:
    warnings: list[str] = []
    criticals: list[str] = []
    monkeypatch.setattr(
        filter_seq_length_module.log,
        "warning",
        lambda message, **kwargs: warnings.append(message),
    )
    monkeypatch.setattr(
        filter_seq_length_module.log,
        "critical",
        lambda message, **kwargs: criticals.append(message),
    )
    processor = SimpleNamespace(image_token_id=21, video_token_id=22)
    strict_filter = FilterSeqLength(
        max_token_length=4,
        processor=processor,
        drop_over_max_length=True,
    )

    for _ in range(6):
        assert strict_filter(_sample(5)) is None
    strict_filter._strict_drop_count = 999
    assert strict_filter(_sample(5)) is None

    assert len(warnings) == 6
    assert "drop_count=1000" in warnings[-1]
    assert criticals == []


@pytest.mark.parametrize("omit", [False, True])
def test_qwen35_rebuilds_absent_or_stale_modality_ids(omit: bool) -> None:
    class Processor:
        def apply_chat_template(self, *args: object, **kwargs: object) -> dict:
            output = {"input_ids": torch.tensor([[1, 21, 22]]), "attention_mask": torch.ones((1, 3))}
            if not omit:
                output["mm_token_type_ids"] = torch.zeros((1, 1))
            return output

    processor = Qwen3VLProcessor.__new__(Qwen3VLProcessor)
    processor.processor = Processor()
    processor.retain_mm_token_type_ids = True
    processor.image_token_id = 21
    processor.video_token_id = 22
    output = processor.apply_chat_template([{"role": "user", "content": "caption"}])
    assert output["mm_token_type_ids"].tolist() == [0, 1, 2]


def test_framewise_requires_aligned_source_indices() -> None:
    with pytest.raises(ValueError, match="one source index"):
        expand_video_frames_for_framewise([object(), object()], [4], 2)
    with pytest.raises(ValueError, match="positive"):
        expand_video_frames_for_framewise([object()], [4], 0)
    with pytest.raises(ValueError, match="one index per provided frame"):
        maybe_parse_video_content(
            [{"role": "user", "content": [{"type": "video", "video": [object()], "fps": 1, "frames_indices": []}]}]
        )


def test_collate_rejects_mixed_or_misaligned_modality_ids() -> None:
    samples = [_sample(3), _sample(5)]
    samples[1].pop("mm_token_type_ids")
    with pytest.raises(ValueError, match="must align"):
        custom_collate(samples)
    samples[1]["mm_token_type_ids"] = torch.zeros(6, dtype=torch.long)
    with pytest.raises(ValueError, match="must align"):
        custom_collate(samples)


def test_recurrent_packing_is_rejected_before_mutating_samples() -> None:
    samples = [_sample(3)]
    samples[0]["true_packing"] = True
    with pytest.raises(ValueError, match="recurrent true packing"):
        custom_collate(samples)
    assert samples[0]["attention_mask"].shape == (3,)


def test_framewise_audio_requires_an_explicit_shared_clock() -> None:
    with pytest.raises(ValueError, match="shared audio/video timestamp clock"):
        TokenizeData(processor=SimpleNamespace(temporal_patch_size=2), sound_und=True, video_temporal_mode="framewise")


def test_local_revision_snapshot_routes_qwen35(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "config.json").write_text('{"model_type": "qwen3_5"}')
    monkeypatch.setattr(processor_factory, "Qwen3VLProcessor", lambda name, **kwargs: name)
    assert processor_factory.build_processor(str(tmp_path)) == str(tmp_path)
