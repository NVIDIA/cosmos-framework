# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import json
import sys
import types

import pytest

# These unit tests exercise dataset/config construction, not video decoding.
# Keep collection deterministic on CPU hosts that do not have FFmpeg/CUDA
# libraries; decoder availability is covered by the image/compute preflight.
torchcodec_video = types.ModuleType("cosmos_framework.utils.generator.torchcodec_video")
torchcodec_video.TorchCodecVideoReader = object
sys.modules.setdefault("cosmos_framework.utils.generator.torchcodec_video", torchcodec_video)

from cosmos_framework.configs.base.reasoner.experiment.wts_vlm import (
    WTSProcessor,
    WTSLlavaDataset,
    aetc_daft_vlm,
    aetc_daft_vlm_edge,
    wts_vlm_edge,
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


def test_wts_dataset_resolves_video_paths_and_limit(tmp_path) -> None:
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

    dataset = WTSLlavaDataset(str(annotations), str(media), limit="1")
    assert len(dataset) == 1
    assert dataset[0]["video"] == str(media / "clip.mp4")


def test_wts_dataset_rejects_invalid_record(tmp_path) -> None:
    annotations = tmp_path / "annotations.json"
    annotations.write_text(json.dumps([{"video": "clip.mp4"}]), encoding="utf-8")

    with pytest.raises(ValueError, match="conversation"):
        WTSLlavaDataset(str(annotations), str(tmp_path))


def test_wts_edge_uses_native_edge_policy() -> None:
    assert wts_vlm_edge["defaults"][4] == {"override /vlm_policy": "cosmos3_edge_reasoner"}
    assert "lr_multipliers" not in wts_vlm_edge["optimizer"]
    assert wts_vlm_edge["model"]["config"]["policy"]["model_max_length"] == 16000


def test_video_max_pixels_is_a_runtime_processor_setting() -> None:
    tokenizer = types.SimpleNamespace(pad_token_id=0)
    video_processor = types.SimpleNamespace(size={"shortest_edge": 4096, "longest_edge": 25165824})
    hf_processor = types.SimpleNamespace(tokenizer=tokenizer, video_processor=video_processor)
    wrapped_processor = types.SimpleNamespace(tokenizer=tokenizer, processor=hf_processor)

    processor = WTSProcessor(wrapped_processor, video_max_pixels="16384")

    assert processor.video_max_pixels == 16384
    assert video_processor.size == {"shortest_edge": 4096, "longest_edge": 16384}


def test_video_max_pixels_rejects_incompatible_processor_budget() -> None:
    tokenizer = types.SimpleNamespace(pad_token_id=0)
    video_processor = types.SimpleNamespace(size={"shortest_edge": 4096, "longest_edge": 25165824})
    hf_processor = types.SimpleNamespace(tokenizer=tokenizer, video_processor=video_processor)
    wrapped_processor = types.SimpleNamespace(tokenizer=tokenizer, processor=hf_processor)

    with pytest.raises(ValueError, match="shortest_edge"):
        WTSProcessor(wrapped_processor, video_max_pixels=1024)


def test_edge_aetc_recipe_uses_edge_policy_and_runtime_video_profile() -> None:
    assert aetc_daft_vlm_edge["defaults"][4] == {"override /vlm_policy": "cosmos3_edge_reasoner"}
    assert "video_max_pixels" in aetc_daft_vlm_edge["dataloader_train"]["processor"]
    assert "video_max_pixels" in wts_vlm_edge["dataloader_train"]["processor"]
    source = __import__("inspect").getsource(sys.modules[WTSProcessor.__module__])
    assert "AETC_VIDEO_MAX_PIXELS" in source
    assert "WTS_VIDEO_MAX_PIXELS" in source


def test_daft_dataset_matches_internal_hybrid_index_order(monkeypatch) -> None:
    calls = _install_fake_daft(monkeypatch)
    dataset = TaoVlReasonDaftDataset(
        annotation_paths='["bcq.json", "mcq.json"]',
        media_root="/data/aetc",
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
    assert calls[0]["media_roots"] == "/data/aetc"


def test_daft_chat_template_targets_wrapped_hf_processor(monkeypatch) -> None:
    calls = _install_fake_daft(monkeypatch)
    wrapped = types.SimpleNamespace(processor=object())
    apply_daft_chat_template(wrapped)
    assert calls == [wrapped.processor]


def test_daft_recipe_and_json_path_parser() -> None:
    assert parse_path_list('["a.json", "b.json"]') == ["a.json", "b.json"]
    assert aetc_daft_vlm["job"]["group"] == "aetc_daft_sft"
    assert aetc_daft_vlm["dataloader_train"]["processor"]["use_daft_chat_template"]
