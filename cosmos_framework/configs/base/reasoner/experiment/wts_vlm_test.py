# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import json
import sys
import types

import pytest

from cosmos_framework.configs.base.reasoner.experiment.wts_vlm import (
    WTSLlavaDataset,
    aetc_daft_vlm,
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
