# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import json
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor

import pytest

# These unit tests exercise dataset/config construction, not video decoding.
# Keep collection deterministic on CPU hosts that do not have FFmpeg/CUDA
# libraries; decoder availability is covered by the image/compute preflight.
torchcodec_video = types.ModuleType("cosmos_framework.utils.generator.torchcodec_video")
torchcodec_video.TorchCodecVideoReader = object
sys.modules.setdefault("cosmos_framework.utils.generator.torchcodec_video", torchcodec_video)

from cosmos_framework.configs.base.reasoner.experiment.tao_video_sft import (
    VideoConversationDataset,
    VideoSFTProcessor,
    tao_task_aware_video_reasoning,
    tao_task_aware_video_reasoning_edge,
    tao_video_conversation_edge,
)
from cosmos_framework.data.generator.local_datasets.tao_vl_reason import (
    TaoVlReasonDaftDataset,
    apply_daft_chat_template,
    parse_path_list,
)


def _install_fake_daft(monkeypatch) -> list[object]:
    calls: list[object] = []

    class FakeDataset:
        def __init__(self, **kwargs) -> None:
            calls.append(kwargs)
            self._raw_length = 2

        def __getitem__(self, index: int) -> list[dict]:
            return [{"role": "assistant", "content": f"daft-{index}"}]

    def fake_template(processor) -> None:
        calls.append(processor)

    package = types.ModuleType("nvidia_tao_daft")
    datasets = types.ModuleType("nvidia_tao_daft.datasets")
    module = types.ModuleType("nvidia_tao_daft.datasets.tao_vl_reason_v1_0")
    module.TaoVlReasonV1_0CosmosRLConversationDataset = FakeDataset
    module.apply_chat_template_override = fake_template
    monkeypatch.setitem(sys.modules, "nvidia_tao_daft", package)
    monkeypatch.setitem(sys.modules, "nvidia_tao_daft.datasets", datasets)
    monkeypatch.setitem(sys.modules, "nvidia_tao_daft.datasets.tao_vl_reason_v1_0", module)
    return calls


def test_video_conversation_dataset_resolves_media_paths_and_limit(tmp_path) -> None:
    media = tmp_path / "videos"
    media.mkdir()
    (media / "clip.mp4").write_bytes(b"video")
    annotations = tmp_path / "annotations.json"
    annotations.write_text(
        json.dumps(
            [
                {
                    "video": "clip.mp4",
                    "conversations": [
                        {"from": "human", "value": "<video> what happens?"},
                        {"from": "gpt", "value": "A car stops."},
                    ],
                },
                {
                    "video": "clip.mp4",
                    "conversations": [
                        {"from": "human", "value": "<video> what happens next?"},
                        {"from": "gpt", "value": "Traffic moves."},
                    ],
                },
            ]
        ),
        encoding="utf-8",
    )

    dataset = VideoConversationDataset(str(annotations), str(media), limit="1")
    assert len(dataset) == 1
    assert dataset[0]["video"] == str(media / "clip.mp4")


def test_video_conversation_dataset_rejects_invalid_record(tmp_path) -> None:
    annotations = tmp_path / "annotations.json"
    annotations.write_text(json.dumps([{"video": "clip.mp4"}]), encoding="utf-8")

    with pytest.raises(ValueError, match="conversation"):
        VideoConversationDataset(str(annotations), str(tmp_path))


def test_video_conversation_dataset_accepts_generic_media_and_messages(tmp_path) -> None:
    media = tmp_path / "media"
    media.mkdir()
    (media / "clip.mp4").write_bytes(b"video")
    annotations = tmp_path / "annotations.json"
    annotations.write_text(json.dumps([{
        "media_path": "clip.mp4",
        "messages": [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}],
    }]))
    dataset = VideoConversationDataset(str(annotations), str(media))
    assert dataset[0]["video"] == str(media / "clip.mp4")


def test_generic_edge_recipe_uses_native_edge_policy() -> None:
    assert tao_video_conversation_edge["defaults"][4] == {"override /vlm_policy": "cosmos3_edge_reasoner"}
    assert "lr_multipliers" not in tao_video_conversation_edge["optimizer"]
    assert tao_video_conversation_edge["model"]["config"]["policy"]["model_max_length"] == 16000


def test_video_max_pixels_is_a_runtime_processor_setting() -> None:
    tokenizer = types.SimpleNamespace(pad_token_id=0)
    video_processor = types.SimpleNamespace(size={"shortest_edge": 4096, "longest_edge": 25165824})
    hf_processor = types.SimpleNamespace(tokenizer=tokenizer, video_processor=video_processor)
    wrapped_processor = types.SimpleNamespace(tokenizer=tokenizer, processor=hf_processor)

    processor = VideoSFTProcessor(wrapped_processor, video_max_pixels="16384")

    assert processor.video_max_pixels == 16384
    assert video_processor.size == {"shortest_edge": 4096, "longest_edge": 16384}


def test_video_max_pixels_rejects_incompatible_processor_budget() -> None:
    tokenizer = types.SimpleNamespace(pad_token_id=0)
    video_processor = types.SimpleNamespace(size={"shortest_edge": 4096, "longest_edge": 25165824})
    hf_processor = types.SimpleNamespace(tokenizer=tokenizer, video_processor=video_processor)
    wrapped_processor = types.SimpleNamespace(tokenizer=tokenizer, processor=hf_processor)

    with pytest.raises(ValueError, match="shortest_edge"):
        VideoSFTProcessor(wrapped_processor, video_max_pixels=1024)


def test_video_override_map_is_validated_and_applied(tmp_path, monkeypatch) -> None:
    tokenizer = types.SimpleNamespace(pad_token_id=0)
    wrapped_processor = types.SimpleNamespace(tokenizer=tokenizer)
    source = tmp_path / "source.mp4"
    replacement = tmp_path / "replacement.mp4"
    source.write_bytes(b"source")
    replacement.write_bytes(b"replacement")
    override = tmp_path / "overrides.json"
    override.write_text(json.dumps({str(source): str(replacement)}), encoding="utf-8")
    decoded_paths: list[str] = []

    class FakeReader:
        def __init__(self, path, **_kwargs):
            decoded_paths.append(path)
            self.last_output_device = "cuda:0"

        def __len__(self):
            return 1

        def get_frames_nhwc_uint8(self, _indices):
            import numpy as np
            return np.zeros((1, 2, 2, 3), dtype=np.uint8)

        def get_avg_fps(self):
            return 30.0

    monkeypatch.setattr(
        "cosmos_framework.configs.base.reasoner.experiment.wts_vlm.TorchCodecVideoReader",
        FakeReader,
    )
    processor = VideoSFTProcessor(wrapped_processor, video_override_map=str(override))
    processor._decode_video(str(source))
    assert decoded_paths == [str(replacement)]


def test_video_cache_single_flight_avoids_duplicate_concurrent_decodes(tmp_path, monkeypatch) -> None:
    tokenizer = types.SimpleNamespace(pad_token_id=0)
    wrapped_processor = types.SimpleNamespace(tokenizer=tokenizer)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    calls = 0
    calls_lock = threading.Lock()

    class FakeReader:
        def __init__(self, _path, **_kwargs):
            nonlocal calls
            self.last_output_device = "cuda:0"
            with calls_lock:
                calls += 1
            time.sleep(0.05)

        def __len__(self):
            return 2

        def get_frames_nhwc_uint8(self, _indices):
            import numpy as np

            return np.zeros((2, 2, 2, 3), dtype=np.uint8)

        def get_avg_fps(self):
            return 30.0

    monkeypatch.setattr(
        "cosmos_framework.configs.base.reasoner.experiment.wts_vlm.TorchCodecVideoReader",
        FakeReader,
    )
    processor = VideoSFTProcessor(wrapped_processor, video_cache_size=2)

    with ThreadPoolExecutor(max_workers=4) as executor:
        decoded = list(executor.map(processor._decode_video, [str(source)] * 4))

    assert calls == 1
    assert all(result is decoded[0] for result in decoded)


def test_video_decoder_rejects_cuda_to_cpu_fallback(tmp_path, monkeypatch) -> None:
    tokenizer = types.SimpleNamespace(pad_token_id=0)
    wrapped_processor = types.SimpleNamespace(tokenizer=tokenizer)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")

    class FakeReader:
        last_output_device = "cpu"

        def __init__(self, _path, **_kwargs):
            pass

        def __len__(self):
            return 1

        def get_frames_nhwc_uint8(self, _indices):
            import numpy as np

            return np.zeros((1, 2, 2, 3), dtype=np.uint8)

        def get_avg_fps(self):
            return 30.0

    monkeypatch.setattr(
        "cosmos_framework.configs.base.reasoner.experiment.wts_vlm.TorchCodecVideoReader",
        FakeReader,
    )
    processor = VideoSFTProcessor(wrapped_processor, video_device="cuda")

    with pytest.raises(RuntimeError, match="did not decode on the requested CUDA device"):
        processor._decode_video(str(source))


def test_video_decoder_binds_generic_cuda_to_local_rank(tmp_path, monkeypatch) -> None:
    tokenizer = types.SimpleNamespace(pad_token_id=0)
    wrapped_processor = types.SimpleNamespace(tokenizer=tokenizer)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    decoder_devices: list[str] = []

    class FakeReader:
        def __init__(self, _path, **kwargs):
            decoder_devices.append(kwargs["device"])
            self.last_output_device = kwargs["device"]

        def __len__(self):
            return 1

        def get_frames_nhwc_uint8(self, _indices):
            import numpy as np

            return np.zeros((1, 2, 2, 3), dtype=np.uint8)

        def get_avg_fps(self):
            return 30.0

    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setattr(
        "cosmos_framework.configs.base.reasoner.experiment.wts_vlm.TorchCodecVideoReader",
        FakeReader,
    )
    processor = VideoSFTProcessor(wrapped_processor, video_device="cuda")

    processor._decode_video(str(source))

    assert processor.requested_video_device == "cuda"
    assert processor.video_device == "cuda:3"
    assert decoder_devices == ["cuda:3"]


def test_video_decoder_rejects_wrong_rank_cuda_device(tmp_path, monkeypatch) -> None:
    tokenizer = types.SimpleNamespace(pad_token_id=0)
    wrapped_processor = types.SimpleNamespace(tokenizer=tokenizer)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")

    class FakeReader:
        last_output_device = "cuda:0"

        def __init__(self, _path, **_kwargs):
            pass

        def __len__(self):
            return 1

        def get_frames_nhwc_uint8(self, _indices):
            import numpy as np

            return np.zeros((1, 2, 2, 3), dtype=np.uint8)

        def get_avg_fps(self):
            return 30.0

    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setattr(
        "cosmos_framework.configs.base.reasoner.experiment.wts_vlm.TorchCodecVideoReader",
        FakeReader,
    )
    processor = VideoSFTProcessor(wrapped_processor, video_device="cuda")

    with pytest.raises(RuntimeError, match="requested=cuda:3 actual=cuda:0"):
        processor._decode_video(str(source))


def test_task_aware_edge_recipe_uses_runtime_video_profile() -> None:
    assert tao_task_aware_video_reasoning_edge["defaults"][4] == {"override /vlm_policy": "cosmos3_edge_reasoner"}
    assert "video_max_pixels" in tao_task_aware_video_reasoning_edge["dataloader_train"]["processor"]
    assert "video_max_pixels" in tao_video_conversation_edge["dataloader_train"]["processor"]
    source = __import__("inspect").getsource(sys.modules[VideoSFTProcessor.__module__])
    assert "TAO_VIDEO_MAX_PIXELS" in source
    assert "TAO_FRAMEWORK_SFT_PROCESS_THREADS" in source


def test_daft_dataset_matches_internal_hybrid_index_order(monkeypatch) -> None:
    calls = _install_fake_daft(monkeypatch)
    dataset = TaoVlReasonDaftDataset(
        annotation_paths='["bcq.json", "mcq.json"]',
        media_root="/data/customer-media",
        response_mode="hybrid",
        system_prompt="system",
    )

    assert len(dataset) == 4
    assert [dataset[index]["messages"][0]["content"] for index in range(4)] == [
        "daft-0",
        "daft-2",
        "daft-1",
        "daft-3",
    ]
    assert calls[0]["annotation_paths"] == ["bcq.json", "mcq.json"]
    assert calls[0]["media_roots"] == "/data/customer-media"


def test_daft_chat_template_targets_wrapped_hf_processor(monkeypatch) -> None:
    calls = _install_fake_daft(monkeypatch)
    wrapped = types.SimpleNamespace(processor=object())
    apply_daft_chat_template(wrapped)
    assert calls == [wrapped.processor]


def test_task_aware_recipe_and_json_path_parser() -> None:
    assert parse_path_list('["a.json", "b.json"]') == ["a.json", "b.json"]
    assert tao_task_aware_video_reasoning["job"]["group"] == "tao_task_aware_video_reasoning_sft"
    assert tao_task_aware_video_reasoning["dataloader_train"]["processor"]["use_daft_chat_template"]
