# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import io
import re
import wave
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from cosmos_framework.data.generator.augmentors.reasoner.bytes_to_media import BytesToMedia
from cosmos_framework.data.generator.augmentors.reasoner.filter_output_key import FilterOutputKey
from cosmos_framework.data.generator.augmentors.reasoner.filter_seq_length import FilterSeqLength
from cosmos_framework.data.generator.augmentors.reasoner.tokenize_data import TokenizeData
from cosmos_framework.data.generator.processors.parakeet_audio_processor import (
    AUDIO_END_TOKEN,
    AUDIO_PAD_TOKEN,
    AUDIO_START_TOKEN,
)


class _FakeTokenizer:
    unk_token_id = 0

    def __init__(self, extra_vocabulary: dict[str, int] | None = None) -> None:
        self.vocabulary = {
            "before": 11,
            "middle": 12,
            "after": 13,
            "answer": 14,
            "<image>": 15,
            "<video>": 16,
            "<timestamp>": 17,
            "<|vision_start|>": 18,
            "<|vision_end|>": 19,
        }
        self.vocabulary.update(extra_vocabulary or {})

    def get_vocab(self) -> dict[str, int]:
        return self.vocabulary.copy()

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.vocabulary.get(token, self.unk_token_id)

    def add_tokens(self, new_tokens: list[str]) -> int:
        next_id = max(self.vocabulary.values()) + 1
        for token in new_tokens:
            self.vocabulary[token] = next_id
            next_id += 1
        return len(new_tokens)


class _FakeVLMProcessor:
    pad_id = 0
    patch_size = 14
    merge_size = 2
    temporal_patch_size = 2
    use_smart_resize = False
    image_token_id = 15
    video_token_id = 16

    def __init__(self, *, name: str = "fake-vlm", extra_vocabulary: dict[str, int] | None = None) -> None:
        self.name = name
        self.tokenizer = _FakeTokenizer(extra_vocabulary)
        self.last_conversation = None

    def apply_chat_template(self, conversation, **kwargs) -> dict[str, torch.Tensor]:
        self.last_conversation = deepcopy(conversation)
        input_ids: list[int] = []
        video_grid_thw: list[list[int]] = []
        special_tokens = (AUDIO_START_TOKEN, AUDIO_PAD_TOKEN, AUDIO_END_TOKEN)
        for message in conversation:
            content_items = message["content"] if isinstance(message["content"], list) else []
            for content in content_items:
                if content["type"] == "image":
                    input_ids.append(self.tokenizer.convert_tokens_to_ids("<image>"))
                    continue
                if content["type"] == "video":
                    num_chunks = (len(content["video"]) + self.temporal_patch_size - 1) // self.temporal_patch_size
                    video_grid_thw.append([num_chunks, 1, 1])
                    for _ in range(num_chunks):
                        input_ids.extend(
                            [
                                self.tokenizer.convert_tokens_to_ids("<timestamp>"),
                                self.tokenizer.convert_tokens_to_ids("<|vision_start|>"),
                                self.tokenizer.convert_tokens_to_ids("<video>"),
                                self.tokenizer.convert_tokens_to_ids("<|vision_end|>"),
                            ]
                        )
                    continue
                if content["type"] != "text":
                    continue
                text = content["text"]
                if text in self.tokenizer.get_vocab():
                    input_ids.append(self.tokenizer.convert_tokens_to_ids(text))
                    continue
                while text:
                    matched_token = next((token for token in special_tokens if text.startswith(token)), None)
                    if matched_token is None:
                        timestamp_match = re.match(r"<\d+\.\d seconds>", text)
                        if timestamp_match is None:
                            raise AssertionError(f"Unexpected fake-tokenizer input: {text!r}")
                        input_ids.append(self.tokenizer.convert_tokens_to_ids("<timestamp>"))
                        text = text[timestamp_match.end() :]
                        continue
                    input_ids.append(self.tokenizer.convert_tokens_to_ids(matched_token))
                    text = text[len(matched_token) :]
        token_tensor = torch.tensor(input_ids, dtype=torch.long)
        outputs = {
            "input_ids": token_tensor,
            "attention_mask": torch.ones_like(token_tensor, dtype=torch.bool),
            "pixel_values": torch.ones(1, 3),
        }
        if video_grid_thw:
            outputs["video_grid_thw"] = torch.tensor(video_grid_thw)
            outputs["pixel_values_videos"] = torch.ones(sum(row[0] for row in video_grid_thw), 3)
        return outputs

    def add_assistant_tokens_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(input_ids, dtype=torch.bool)

    def decode(self, input_ids: torch.Tensor) -> str:
        return "decoded"


class _FakeAudioProcessor:
    sampling_rate = 16_000

    def __init__(self, token_lengths: tuple[int, ...] = (1, 2)) -> None:
        self.last_audios = None
        self.token_lengths = token_lengths

    def __call__(self, audios, *, sampling_rate: int) -> dict[str, torch.Tensor]:
        assert sampling_rate == self.sampling_rate
        self.last_audios = list(audios)
        features = torch.stack(
            [torch.full((5, 128), float(np.asarray(audio)[0])) for audio in self.last_audios],
        )
        num_clips = len(self.last_audios)
        return {
            "audio_features": features,
            "audio_feature_lengths": torch.tensor([3, 5][:num_clips], dtype=torch.long),
            "audio_token_lengths": torch.tensor(self.token_lengths[:num_clips], dtype=torch.long),
        }


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "media_key,waveform",
    [
        ("audio_0", np.zeros(320, dtype=np.float32)),
        ("sample.wav", torch.zeros(320)),
        ("input_sound", torch.ones(320)),
    ],
)
def test_bytes_to_media_preserves_decoded_audio_waveforms(
    media_key: str,
    waveform: np.ndarray | torch.Tensor,
) -> None:
    augmentor = BytesToMedia(is_input_pickle_byptes=False)
    data = {"media": {media_key: waveform}}

    output = augmentor(data)

    assert output["media"][media_key] is waveform


@pytest.mark.L0
@pytest.mark.CPU
def test_bytes_to_media_decodes_audio_bytes_to_mono_16khz() -> None:
    encoded_audio = io.BytesIO()
    with wave.open(encoded_audio, "wb") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(8_000)
        wav_file.writeframes(np.zeros((320, 2), dtype=np.int16).tobytes())

    output = BytesToMedia(is_input_pickle_byptes=False)(
        {"media": {"clip.wav": encoded_audio.getvalue()}},
    )

    waveform = output["media"]["clip.wav"]
    assert waveform.dtype == torch.float32
    assert waveform.shape == (640,)


@pytest.mark.L0
@pytest.mark.CPU
def test_bytes_to_media_extracts_video_audio_only_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waveform = torch.tensor([0.25, -0.5], dtype=torch.float32)
    monkeypatch.setattr(
        BytesToMedia,
        "_bytes_to_video_frames",
        lambda *args, **kwargs: {"videos": [], "fps": 4.0},
    )
    monkeypatch.setattr(
        BytesToMedia,
        "_bytes_to_audio_waveform",
        lambda *args, **kwargs: waveform,
    )

    video_only = BytesToMedia(is_input_pickle_byptes=False)(
        {"media": {"clip.mp4": b"encoded-video"}},
    )
    audio_visual = BytesToMedia(is_input_pickle_byptes=False, extract_audio=True)(
        {"media": {"clip.mp4": b"encoded-video"}},
    )

    assert "audio" not in video_only["media"]["clip.mp4"]
    assert audio_visual["media"]["clip.mp4"]["audio"] is waveform


@pytest.mark.L0
@pytest.mark.CPU
def test_tokenize_data_preserves_interleaved_audio_order_and_processor_outputs() -> None:
    vlm_processor = _FakeVLMProcessor()
    audio_processor = _FakeAudioProcessor()
    tokenize = TokenizeData(
        processor=vlm_processor,
        sound_und=True,
        audio_processor=audio_processor,
        max_image_token_length=16,
    )
    clip_first = np.full(320, 2.0, dtype=np.float32)
    clip_second = torch.full((480,), 7.0)
    data = {
        "__key__": "sample",
        "__url__": SimpleNamespace(root="root", path="path"),
        "conversation": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "before"},
                    {"type": "audio", "audio": "audio_first"},
                    {"type": "image", "image": "image"},
                    {"type": "text", "text": "middle"},
                    {"type": "audio", "audio": "audio_second"},
                    {"type": "text", "text": "after"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
        ],
        "media": {
            "audio_first": clip_first,
            "image": Image.new("RGB", (4, 4)),
            "audio_second": clip_second,
        },
    }

    output = tokenize(data)

    assert output is not None
    assert audio_processor.last_audios is not None
    assert audio_processor.last_audios[0] is clip_first
    assert audio_processor.last_audios[1] is clip_second
    assert vlm_processor.last_conversation is not None
    user_content = vlm_processor.last_conversation[0]["content"]
    assert [content["type"] for content in user_content] == ["text", "text", "image", "text", "text", "text"]
    assert user_content[1]["text"] == AUDIO_START_TOKEN + "<0.1 seconds>" + AUDIO_PAD_TOKEN + AUDIO_END_TOKEN
    assert user_content[4]["text"] == AUDIO_START_TOKEN + "<0.1 seconds>" + AUDIO_PAD_TOKEN * 2 + AUDIO_END_TOKEN

    token_ids = vlm_processor.tokenizer.vocabulary
    assert output["input_ids"].tolist() == [
        token_ids["before"],
        token_ids[AUDIO_START_TOKEN],
        token_ids["<timestamp>"],
        token_ids[AUDIO_PAD_TOKEN],
        token_ids[AUDIO_END_TOKEN],
        token_ids["<image>"],
        token_ids["middle"],
        token_ids[AUDIO_START_TOKEN],
        token_ids["<timestamp>"],
        token_ids[AUDIO_PAD_TOKEN],
        token_ids[AUDIO_PAD_TOKEN],
        token_ids[AUDIO_END_TOKEN],
        token_ids["after"],
        token_ids["answer"],
    ]
    assert output["audio_features"][:, 0, 0].tolist() == [2.0, 7.0]
    assert torch.equal(output["audio_feature_lengths"], torch.tensor([3, 5]))
    assert torch.equal(output["audio_token_lengths"], torch.tensor([1, 2]))

    filtered = FilterOutputKey()(output)
    assert filtered is not None
    assert set(("audio_features", "audio_feature_lengths", "audio_token_lengths")) <= filtered.keys()
    assert "media" not in filtered


@pytest.mark.L0
@pytest.mark.CPU
def test_tokenize_data_reuses_video_timestamps_for_audio_and_keeps_tail() -> None:
    vlm_processor = _FakeVLMProcessor()
    audio_processor = _FakeAudioProcessor(token_lengths=(10,))
    tokenize = TokenizeData(
        processor=vlm_processor,
        sound_und=True,
        audio_processor=audio_processor,
        max_video_token_length=16,
    )
    frames = [Image.new("RGB", (4, 4)) for _ in range(4)]
    waveform = np.zeros(320, dtype=np.float32)
    data = {
        "__key__": "sample",
        "__url__": SimpleNamespace(root="root", path="path"),
        "conversation": [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": "audio_visual"},
                    {"type": "audio", "audio": "audio_visual"},
                    {"type": "text", "text": "after"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
        ],
        "media": {
            "audio_visual": {"videos": frames, "fps": 4.0, "audio": waveform},
        },
    }
    reversed_data = deepcopy(data)
    reversed_data["__key__"] = "reversed-pair"
    reversed_data["conversation"][0]["content"][:2] = [
        {"type": "audio", "audio": "audio_visual"},
        {"type": "video", "video": "audio_visual"},
    ]

    output = tokenize(data)

    assert output is not None
    assert audio_processor.last_audios is not None
    assert audio_processor.last_audios[0] is waveform
    assert vlm_processor.last_conversation is not None
    audio_text = vlm_processor.last_conversation[0]["content"][1]["text"]
    assert audio_text == (
        f"{AUDIO_START_TOKEN}<0.1 seconds>{AUDIO_PAD_TOKEN * 5}<0.6 seconds>{AUDIO_PAD_TOKEN * 5}{AUDIO_END_TOKEN}"
    )
    assert tokenize(reversed_data) is None


@pytest.mark.L0
@pytest.mark.CPU
def test_audio_only_layouts_keep_mode_1_timestamp_free_and_mode_3_falls_back_to_mode_2() -> None:
    data = {
        "__key__": "audio-only-layouts",
        "__url__": SimpleNamespace(root="root", path="path"),
        "conversation": [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": "audio"},
                    {"type": "text", "text": "after"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
        ],
        "media": {"audio": np.zeros(320, dtype=np.float32)},
    }
    outputs: dict[str, dict[str, torch.Tensor]] = {}
    audio_texts: dict[str, str] = {}
    mode_1_timestamp_id = 0
    for audio_layout in (
        "separate_no_timestamps",
        "separate_with_timestamps",
        "interleaved_av",
    ):
        vlm_processor = _FakeVLMProcessor()
        tokenize = TokenizeData(
            processor=vlm_processor,
            sound_und=True,
            audio_processor=_FakeAudioProcessor(token_lengths=(3,)),
            audio_layout=audio_layout,
        )

        output = tokenize(deepcopy(data))

        assert output is not None
        assert vlm_processor.last_conversation is not None
        outputs[audio_layout] = output
        audio_texts[audio_layout] = vlm_processor.last_conversation[0]["content"][0]["text"]
        if audio_layout == "separate_no_timestamps":
            mode_1_timestamp_id = vlm_processor.tokenizer.convert_tokens_to_ids("<timestamp>")

    assert audio_texts["separate_no_timestamps"] == (f"{AUDIO_START_TOKEN}{AUDIO_PAD_TOKEN * 3}{AUDIO_END_TOKEN}")
    assert int((outputs["separate_no_timestamps"]["input_ids"] == mode_1_timestamp_id).sum()) == 0
    assert audio_texts["interleaved_av"] == audio_texts["separate_with_timestamps"]
    for key in (
        "input_ids",
        "attention_mask",
        "token_mask",
        "labels",
    ):
        assert torch.equal(outputs["interleaved_av"][key], outputs["separate_with_timestamps"][key])


@pytest.mark.L0
@pytest.mark.CPU
def test_interleaved_av_supports_vision_only_video_and_multiple_pairs() -> None:
    vlm_processor = _FakeVLMProcessor()
    audio_processor = _FakeAudioProcessor(token_lengths=(10, 4))
    tokenize = TokenizeData(
        processor=vlm_processor,
        sound_und=True,
        audio_processor=audio_processor,
        audio_layout="interleaved_av",
        max_video_token_length=16,
    )
    frames_one_chunk = [Image.new("RGB", (4, 4)) for _ in range(2)]
    frames_two_chunks = [Image.new("RGB", (4, 4)) for _ in range(4)]
    first_audio = np.full(320, 2.0, dtype=np.float32)
    second_audio = np.full(320, 7.0, dtype=np.float32)
    data = {
        "__key__": "multi-pair",
        "__url__": SimpleNamespace(root="root", path="path"),
        "conversation": [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": "vision_only"},
                    {"type": "video", "video": "video_first"},
                    {"type": "audio", "audio": "audio_first"},
                    {"type": "video", "video": "video_second"},
                    {"type": "audio", "audio": "audio_second"},
                    {"type": "text", "text": "after"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
        ],
        "media": {
            "vision_only": {"videos": frames_one_chunk, "fps": 4.0},
            "video_first": {"videos": frames_two_chunks, "fps": 4.0},
            "audio_first": first_audio,
            "video_second": {"videos": frames_two_chunks, "fps": 4.0},
            "audio_second": second_audio,
        },
    }

    output = tokenize(data)

    assert output is not None
    assert vlm_processor.last_conversation is not None
    assert [item["type"] for item in vlm_processor.last_conversation[0]["content"]] == [
        "video",
        "video",
        "video",
        "text",
    ]
    token_ids = vlm_processor.tokenizer.vocabulary
    native_chunk = [
        token_ids["<timestamp>"],
        token_ids["<|vision_start|>"],
        token_ids["<video>"],
        token_ids["<|vision_end|>"],
    ]
    audio_start_id = token_ids[AUDIO_START_TOKEN]
    audio_pad_id = token_ids[AUDIO_PAD_TOKEN]
    audio_end_id = token_ids[AUDIO_END_TOKEN]
    expected_ids = [*native_chunk]
    for segment_length in (5, 5, 4, 0):
        expected_ids.extend(native_chunk)
        expected_ids.extend([audio_start_id, *([audio_pad_id] * segment_length), audio_end_id])
    expected_ids.extend([token_ids["after"], token_ids["answer"]])
    assert output["input_ids"].tolist() == expected_ids
    assert output["input_ids"].tolist().count(token_ids["<timestamp>"]) == 5
    assert output["input_ids"].tolist().count(audio_pad_id) == 14
    assert output["audio_features"][:, 0, 0].tolist() == [2.0, 7.0]
    assert output["input_ids"].shape == output["attention_mask"].shape
    assert output["input_ids"].shape == output["token_mask"].shape
    assert output["input_ids"].shape == output["labels"].shape


@pytest.mark.L0
@pytest.mark.CPU
def test_interleaved_av_rejects_multiple_audios_for_one_video() -> None:
    frames = [Image.new("RGB", (4, 4)) for _ in range(2)]
    data = {
        "__key__": "multiple-audios-for-one-video",
        "__url__": SimpleNamespace(root="root", path="path"),
        "conversation": [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": "video"},
                    {"type": "audio", "audio": "audio_first"},
                    {"type": "audio", "audio": "audio_second"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
        ],
        "media": {
            "video": {"videos": frames, "fps": 4.0},
            "audio_first": np.zeros(320, dtype=np.float32),
            "audio_second": np.ones(320, dtype=np.float32),
        },
    }
    vlm_processor = _FakeVLMProcessor()
    tokenize = TokenizeData(
        processor=vlm_processor,
        sound_und=True,
        audio_processor=_FakeAudioProcessor(token_lengths=(2, 2)),
        audio_layout="interleaved_av",
        max_video_token_length=16,
    )

    assert tokenize(data) is None
    assert vlm_processor.last_conversation is None


@pytest.mark.L0
@pytest.mark.CPU
def test_filter_seq_length_drops_audio_crossing_the_cut_and_truncates_text_tail() -> None:
    processor = _FakeVLMProcessor()
    filter_seq_length = FilterSeqLength(
        max_token_length=4,
        processor=processor,
        sound_und=True,
    )
    audio_pad_id = filter_seq_length.audio_pad_token_id
    assert audio_pad_id is not None
    audio_crosses_cut = {
        "__key__": "audio-crosses-cut",
        "__url__": SimpleNamespace(root="root", path="path"),
        "input_ids": torch.tensor([11, audio_pad_id, audio_pad_id, audio_pad_id, audio_pad_id, 13]),
        "token_mask": torch.ones(6, dtype=torch.bool),
        "attention_mask": torch.ones(6, dtype=torch.bool),
        "labels": torch.arange(6),
    }
    text_tail = {
        "__key__": "text-tail",
        "__url__": SimpleNamespace(root="root", path="path"),
        "input_ids": torch.tensor([11, audio_pad_id, 13, 14, 11, 13]),
        "token_mask": torch.ones(6, dtype=torch.bool),
        "attention_mask": torch.ones(6, dtype=torch.bool),
        "labels": torch.arange(6),
    }

    assert filter_seq_length(audio_crosses_cut) is None
    truncated = filter_seq_length(text_tail)
    assert truncated is not None
    assert truncated["input_ids"].tolist() == [11, audio_pad_id, 13, 14]


@pytest.mark.L0
@pytest.mark.CPU
def test_tokenize_data_disabled_does_not_construct_audio_processor_or_mutate_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vlm_processor = _FakeVLMProcessor()
    vocabulary_before = vlm_processor.tokenizer.get_vocab()

    def _unexpected_audio_processor():
        raise AssertionError("Disabled sound understanding constructed an audio processor")

    monkeypatch.setattr(
        "cosmos_framework.data.generator.augmentors.reasoner.tokenize_data.ParakeetAudioProcessor",
        _unexpected_audio_processor,
    )
    tokenize = TokenizeData(processor=vlm_processor, sound_und=False)
    data = {
        "__key__": "sample",
        "__url__": SimpleNamespace(root="root", path="path"),
        "conversation": [
            {"role": "user", "content": [{"type": "audio", "audio": "clip"}]},
            {"role": "assistant", "content": "answer"},
        ],
        "media": {"clip": torch.zeros(320)},
    }

    assert vlm_processor.tokenizer.get_vocab() == vocabulary_before
    assert tokenize(data) is None
