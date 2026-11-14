# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Decoded evaluation videos retain source timing and processor-owned resizing."""

from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

from cosmos_framework.inference.reasoner.framework_torchcodec_video import FrameworkTorchCodecVideoPreprocessor
from cosmos_framework.inference.reasoner.runtime import CosmosFrameworkRuntime


def _runtime():
    runtime = CosmosFrameworkRuntime.__new__(CosmosFrameworkRuntime)
    runtime.model_type = "qwen3_vl"
    runtime.processor = SimpleNamespace(video_processor=Qwen3VLVideoProcessor())
    runtime._video_feature_cache_active_key = None
    runtime._video_processor_cache_capacity = 0
    return runtime


@pytest.mark.parametrize("modern", [False, True])
@pytest.mark.parametrize("temporal", [False, True])
def test_video_prefix_rope_call_preserves_both_pinned_signatures(modern, temporal):
    class ReachedRope(Exception):
        pass

    ids = torch.tensor([[30, 118, 121, 119, 31]])
    grid = torch.tensor([[1, 2, 2]])
    mask = torch.ones_like(ids)
    types = torch.tensor([[0, 0, 2, 0, 0]])

    def old(input_ids, image_grid_thw=None, video_grid_thw=None, attention_mask=None):
        assert input_ids is ids and image_grid_thw is None and video_grid_thw is grid
        assert attention_mask is mask
        raise ReachedRope

    def new(input_ids, mm_token_type_ids, image_grid_thw=None, video_grid_thw=None, attention_mask=None):
        assert mm_token_type_ids is types
        return old(input_ids, image_grid_thw, video_grid_thw, attention_mask)

    def timed(
        input_ids,
        mm_token_type_ids,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=None,
        second_per_grid_ts=None,
    ):
        assert torch.equal(second_per_grid_ts, torch.tensor([0.5]))
        return new(input_ids, mm_token_type_ids, image_grid_thw, video_grid_thw, attention_mask)

    runtime = _runtime()
    runtime.model = SimpleNamespace(
        config=SimpleNamespace(video_token_id=121, vision_end_token_id=119),
        model=SimpleNamespace(get_rope_index=timed if temporal else new if modern else old),
    )
    inputs = dict(input_ids=ids, video_grid_thw=grid, attention_mask=mask, mm_token_type_ids=types)
    if temporal:
        inputs["second_per_grid_ts"] = torch.tensor([0.5])
    with pytest.raises(ReachedRope):
        runtime._generate_with_video_prefix_cache(inputs, active_key=("video",), capacity=1, generation_kwargs={})
    if temporal:
        inputs.pop("second_per_grid_ts")
        with pytest.raises(ValueError, match="second_per_grid_ts"):
            runtime._generate_with_video_prefix_cache(inputs, active_key=("video",), capacity=1, generation_kwargs={})
        inputs["second_per_grid_ts"] = torch.tensor([0.5])
    if modern or temporal:
        inputs.pop("mm_token_type_ids")
        with pytest.raises(ValueError, match="processor mm_token_type_ids"):
            runtime._generate_with_video_prefix_cache(inputs, active_key=("video",), capacity=1, generation_kwargs={})


def _conversation():
    frames = [Image.fromarray(np.full((720, 1280, 3), n * 40, dtype=np.uint8)) for n in range(4)]
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": frames,
                    "video_metadata": {"fps": 30.0, "total_num_frames": 2211, "frames_indices": [0, 737, 1473, 2210]},
                    "min_pixels": 50176,
                    "max_pixels": 81920,
                },
                {"type": "text", "text": "What happens?"},
            ],
        }
    ]


@pytest.mark.parametrize("total,indices", [(21, [0, 7, 13, 20]), (2, [0, 0, 1, 1]), (1, [0, 0, 0, 0])])
def test_decoder_retains_actual_metadata_and_fixed_count(tmp_path, monkeypatch, total, indices):
    from cosmos_framework.utils.generator import torchcodec_video

    seen = []

    class Reader:
        last_output_device = "cpu"

        def __init__(self, *args, **kwargs):
            pass

        def __len__(self):
            return total

        def get_avg_fps(self):
            return 30.0

        def get_frames_nhwc_uint8(self, selected):
            seen.append(selected)
            return np.stack([np.full((8, 8, 3), n, dtype=np.uint8) for n in selected])

    monkeypatch.setattr(torchcodec_video, "TorchCodecVideoReader", Reader)
    source = tmp_path / "video.mp4"
    source.touch()
    preprocessor = FrameworkTorchCodecVideoPreprocessor(num_frames=4, cache_size=1, process_threads=1, device="cpu")
    task = {"media_mode": "video", "media_paths": [str(source)]}
    first = preprocessor.prepare_task(task)["_framework_decoded_media"][0]
    expected = {"fps": 30.0, "total_num_frames": total, "frames_indices": indices}
    assert first["video_metadata"] == expected
    assert len(first["frames"]) == 4
    first["video_metadata"]["frames_indices"].append(999)
    assert preprocessor.prepare_task(task)["_framework_decoded_media"][0]["video_metadata"] == expected
    assert seen == [indices]


def test_predecoded_video_keeps_raw_pixels_source_time_and_sft_resize_budget():
    runtime = _runtime()
    conversation = _conversation()
    images, videos, kwargs = runtime._process_vision_info(conversation)
    assert images is None
    raw, metadata = videos[0]
    item = conversation[0]["content"][0]
    assert metadata == item["video_metadata"]
    assert metadata is not item["video_metadata"]
    assert raw.shape == (4, 3, 720, 1280)
    assert raw.dtype == torch.uint8
    for n in range(4):
        assert (raw[n] == n * 40).all()
    assert kwargs["videos_kwargs"] == {
        "do_resize": True,
        "do_sample_frames": False,
        "size": {"shortest_edge": 200704, "longest_edge": 327680},
    }
    actual = runtime.processor.video_processor(
        videos=[raw], video_metadata=[metadata], return_tensors="pt", **kwargs["videos_kwargs"]
    )
    assert actual["video_grid_thw"].tolist() == [[2, 12, 22]]


@pytest.mark.parametrize("batched", [False, True])
def test_actual_processor_accepts_separate_image_and_video_resize_kwargs(batched):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast, Qwen2VLImageProcessor, Qwen3VLProcessor

    special = ["[UNK]", "[PAD]", "<|video_pad|>", "<|image_pad|>", "<|vision_start|>", "<|vision_end|>"]
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({s: i for i, s in enumerate(special)}, unk_token="[UNK]")),
        unk_token="[UNK]",
        pad_token="[PAD]",
        additional_special_tokens=special[2:],
    )
    runtime = _runtime()
    runtime.processor = Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessor(patch_size=16),
        video_processor=runtime.processor.video_processor,
        tokenizer=tokenizer,
        chat_template="<|vision_start|><|video_pad|><|vision_end|>",
    )
    conversation = _conversation()
    if batched:
        actual = runtime._prepare_transformers_batch([conversation, conversation])
        assert actual["video_grid_thw"].tolist() == [[2, 12, 22], [2, 12, 22]]
    else:
        actual = runtime._prepare_transformers_inputs(conversation)
        assert actual["video_grid_thw"].tolist() == [[2, 12, 22]]


@pytest.mark.parametrize(
    "mutation,message",
    [
        (lambda c: c[0]["content"][0]["video_metadata"].update(frames_indices=[0]), "one source frame"),
        (lambda c: c[0]["content"][0].update(min_pixels=-1), "positive integers"),
        (lambda c: c[0]["content"][0].update(total_pixels=123), "omit total_pixels"),
        (lambda c: c[0]["content"].append({"type": "video", "video": "untracked.mp4"}), "cannot mix"),
    ],
)
def test_incompatible_predecoded_inputs_fail_before_processing(mutation, message):
    conversation = _conversation()
    mutation(conversation)
    with pytest.raises(ValueError, match=message):
        _runtime()._process_vision_info(conversation)


def test_video_cache_preserves_metadata_and_batch_preparation_copies_it():
    runtime = _runtime()
    conversation = _conversation()
    runtime._video_feature_cache_active_key = ("same-video",)
    runtime._video_processor_cache_capacity = 1
    runtime._video_input_cache = OrderedDict()
    runtime._video_input_cache_hits = runtime._video_input_cache_misses = 0
    first = runtime._process_vision_info_cached([conversation])
    second = runtime._process_vision_info_cached([conversation])
    assert first is second

    class Processor:
        def apply_chat_template(self, *args, **kwargs):
            return "<|video_pad|>"

        def __call__(self, **kwargs):
            # Emulate a processor padding its private metadata in place.
            kwargs["videos_kwargs"]["video_metadata"][0]["frames_indices"].append(2210)
            kwargs["videos_kwargs"].pop("size")
            return kwargs

    runtime.processor = Processor()
    result = runtime._prepare_transformers_batch([conversation])
    assert result["videos_kwargs"]["do_resize"] is True
    assert len(first[1][0][1]["frames_indices"]) == 4
    assert first[2]["videos_kwargs"]["size"] == {"shortest_edge": 200704, "longest_edge": 327680}
    assert runtime._video_input_cache_misses == 1 and runtime._video_input_cache_hits == 2


def test_source_metadata_accepts_the_existing_video_processor_cache_wrapper():
    from cosmos_framework.inference.reasoner.runtime import _RankLocalVideoProcessorCache

    runtime = _runtime()
    runtime.processor.video_processor = _RankLocalVideoProcessorCache(runtime.processor.video_processor, runtime)
    _, videos, kwargs = runtime._process_vision_info(_conversation())
    assert videos[0][1]["frames_indices"] == [0, 737, 1473, 2210]
    assert kwargs["videos_kwargs"]["size"] == {"shortest_edge": 200704, "longest_edge": 327680}


def _qwen2_runtime(family):
    from importlib.metadata import version

    if version("transformers") != "5.8.1":
        pytest.skip("Qwen2/2.5 source-metadata inference pins Transformers 5.8.1")
    pytest.importorskip("transformers.models.qwen2_vl.video_processing_qwen2_vl")
    from transformers import Qwen2_5_VLProcessor, Qwen2VLImageProcessor, Qwen2VLProcessor, Qwen2VLVideoProcessor

    from cosmos_framework.posttrain.tests.helpers.sft_packer import CHAT_TEMPLATE, local_tokenizer

    runtime = _runtime()
    runtime.model_type = family
    runtime.processor = (Qwen2VLProcessor if family == "qwen2_vl" else Qwen2_5_VLProcessor)(
        image_processor=Qwen2VLImageProcessor(patch_size=14),
        video_processor=Qwen2VLVideoProcessor(patch_size=14, temporal_patch_size=2, merge_size=2),
        tokenizer=local_tokenizer(),
        chat_template=CHAT_TEMPLATE,
    )
    return runtime


@pytest.mark.parametrize("family", ["qwen2_vl", "qwen2_5_vl"])
@pytest.mark.parametrize("batched", [False, True])
def test_qwen2_predecoded_native_processor_pixels_tokens_grid_and_time(family, batched):
    import copy

    runtime = _qwen2_runtime(family)
    conversation = _conversation()
    before = copy.deepcopy(conversation[0]["content"][0]["video_metadata"])
    _, videos, kwargs = runtime._process_vision_info(conversation)
    raw, metadata = videos[0]
    assert metadata == before
    assert kwargs["videos_kwargs"] == {
        "do_resize": True,
        "do_sample_frames": False,
        "size": {"shortest_edge": 50176, "longest_edge": 81920},
    }
    for index in range(4):
        assert torch.equal(
            raw[index], torch.from_numpy(np.array(conversation[0]["content"][0]["video"][index])).permute(2, 0, 1)
        )
    count = 2 if batched else 1
    text = runtime.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    expected = runtime.processor(
        text=[text] * count,
        videos=[raw] * count,
        videos_kwargs={**kwargs["videos_kwargs"], "video_metadata": [copy.deepcopy(metadata) for _ in range(count)]},
        common_kwargs={"return_tensors": "pt"},
    )
    actual = (
        runtime._prepare_transformers_batch([conversation] * count)
        if batched
        else runtime._prepare_transformers_inputs(conversation)
    )
    assert actual.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(torch.as_tensor(actual[name]), torch.as_tensor(expected[name]), rtol=0, atol=0)
    assert actual["video_grid_thw"].shape == (count, 3)
    assert actual["video_grid_thw"][:, 0].tolist() == [2] * count
    assert conversation[0]["content"][0]["video_metadata"] == before
    if family == "qwen2_5_vl":
        # Use the native sampled_fps formula; no replacement with qwen-vl-utils'
        # fabricated 2 FPS or the decoder's requested nominal sampling rate.
        torch.testing.assert_close(
            torch.as_tensor(actual["second_per_grid_ts"]), torch.full((count,), 2 / (4 / 2211 * 30))
        )


@pytest.mark.parametrize("family", ["qwen2_vl", "qwen2_5_vl"])
def test_qwen2_images_use_recorded_patch_geometry(family):
    from qwen_vl_utils import process_vision_info

    runtime = _qwen2_runtime(family)
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": Image.fromarray(np.arange(71 * 99 * 3, dtype=np.uint8).reshape(71, 99, 3)),
                    "min_pixels": 56**2,
                    "max_pixels": 112**2,
                },
                {"type": "text", "text": "Describe."},
            ],
        }
    ]
    actual, videos, kwargs = runtime._process_vision_info(conversation)
    expected, _, _ = process_vision_info(
        conversation, image_patch_size=14, return_video_kwargs=True, return_video_metadata=True
    )
    assert videos is None
    assert np.array_equal(np.asarray(actual[0]), np.asarray(expected[0]))
    inputs = runtime._prepare_transformers_inputs(conversation)
    assert inputs["image_grid_thw"].tolist() == [[1, actual[0].height // 14, actual[0].width // 14]]


def test_qwen2_foreign_video_processor_rejected():
    runtime = _qwen2_runtime("qwen2_vl")
    runtime.processor.video_processor = Qwen3VLVideoProcessor()
    with pytest.raises(ValueError, match="requires Qwen2VLVideoProcessor"):
        runtime._process_vision_info(_conversation())


def test_qwen2_source_timing_rejects_older_processor_semantics(monkeypatch):
    import importlib.metadata

    runtime = _qwen2_runtime("qwen2_5_vl")
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "4.57.6")
    with pytest.raises(ValueError, match="pinned Transformers 5.8.1"):
        runtime._process_vision_info(_conversation())
    # The existing Qwen3 metadata pipeline keeps its prior admission behavior.
    assert _runtime()._process_vision_info(_conversation())[1][0][1]["total_num_frames"] == 2211


def test_qwen2_video_cache_keeps_source_metadata():
    runtime = _qwen2_runtime("qwen2_5_vl")
    runtime._video_feature_cache_active_key = ("same-qwen2-video",)
    runtime._video_processor_cache_capacity = 1
    runtime._video_input_cache = OrderedDict()
    runtime._video_input_cache_hits = runtime._video_input_cache_misses = 0
    conversation = _conversation()
    first = runtime._process_vision_info_cached([conversation])
    assert runtime._process_vision_info_cached([conversation]) is first
    prepared = runtime._prepare_transformers_batch([conversation])
    assert prepared["second_per_grid_ts"]
    assert first[1][0][1]["frames_indices"] == [0, 737, 1473, 2210]
    assert first[2]["videos_kwargs"]["size"] == {"shortest_edge": 50176, "longest_edge": 81920}


def test_feature_cache_preserves_explicit_and_legacy_return_forms():
    calls = []

    class Features:
        def get_video_features(self, pixels, grid, *, return_dict=None):
            calls.append(return_dict)
            return (pixels.sum(), return_dict)

    runtime = _runtime()
    owner = Features()
    runtime.model = SimpleNamespace(model=owner)
    runtime._video_feature_cache_wrapped = False
    runtime._video_feature_cache = OrderedDict()
    runtime._video_feature_cache_hits = runtime._video_feature_cache_misses = 0
    runtime._install_video_feature_cache(4)
    runtime._video_feature_cache_active_key = ("fixed-video",)
    pixels, grid = torch.ones(4, 24), torch.tensor([[1, 2, 2]])
    for value in (None, True, False, None, True, False):
        kwargs = {} if value is None else {"return_dict": value}
        actual = owner.get_video_features(pixels, grid, **kwargs)
        assert actual[1] is value
    assert calls == [None, True, False]
    assert runtime._video_feature_cache_hits == 3


def test_task_conversation_transfers_source_metadata_without_mutating_task():
    task = {
        "media_mode": "video",
        "media_paths": ["source.mp4"],
        "question": "What happens?",
        "_framework_decoded_media": [
            {
                "frames": _conversation()[0]["content"][0]["video"],
                "fps": 0.5,
                "video_metadata": {"fps": 30.0, "total_num_frames": 21, "frames_indices": [0, 7, 13, 20]},
            }
        ],
    }
    conversation = CosmosFrameworkRuntime._task_conversation(task, {"num_frames": 4})
    item = conversation[0]["content"][0]
    assert item["sample_fps"] == 0.5
    assert item["video_metadata"] == task["_framework_decoded_media"][0]["video_metadata"]
    item["video_metadata"]["frames_indices"].append(20)
    assert len(task["_framework_decoded_media"][0]["video_metadata"]["frames_indices"]) == 4
