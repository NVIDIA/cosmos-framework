# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import base64
import json
import subprocess
import sys
from pathlib import Path

import pytest

from cosmos_framework.data.reasoner.qa_dataset import ReasoningConversationDataset, build_response
from cosmos_framework.inference.reasoner.service import prepare_messages, resolve_media
from cosmos_framework.utils.workflow_status import log_workflow_status, monitor_status


def test_dataset_resolves_declared_root_and_hybrid(tmp_path):
    path = tmp_path / "questions.json"
    path.write_text(
        json.dumps(
            {
                "media_root": "videos",
                "items": [{"video_id": "clip.mp4", "question": "What?", "answer": "Yes", "reasoning": "Because"}],
            }
        )
    )
    dataset = ReasoningConversationDataset([str(path)], response_mode="hybrid")
    assert len(dataset) == 2
    assert dataset[0][-1]["content"] == "Yes"
    assert "<think>" in dataset[1][-1]["content"]
    assert dataset[0][0]["content"][0]["video"] == str(tmp_path / "videos" / "clip.mp4")
    with pytest.raises(IndexError):
        dataset[2]


def test_invalid_response_mode_always_rejected():
    with pytest.raises(ValueError):
        build_response("answer", "", "invalid")


def test_inline_media_and_message_order(tmp_path):
    url = "data:image/png;base64," + base64.b64encode(b"sample").decode()
    messages = [
        {"role": "system", "content": "Rules"},
        {
            "role": "user",
            "content": [{"type": "text", "text": "Describe"}, {"type": "image_url", "image_url": {"url": url}}],
        },
    ]
    prepared = prepare_messages(messages, tmp_path)
    assert prepared[0] == messages[0]
    assert Path(prepared[1]["content"][1]["image"]).read_bytes() == b"sample"
    assert messages[1]["content"][1]["type"] == "image_url"


def test_media_rejects_unmounted_files_and_cloud_uris(tmp_path):
    for value in ("/etc/passwd", "s3://bucket/video.mp4", "http://127.0.0.1/private", "data:text/plain;base64,YQ=="):
        with pytest.raises(ValueError):
            resolve_media(value, tmp_path, [])
    media = tmp_path / "test.mp4"
    media.write_bytes(b"video")
    assert resolve_media(str(media), tmp_path, [tmp_path]) == str(media)


@pytest.mark.parametrize("fail", [False, True])
def test_action_terminal_status(tmp_path, monkeypatch, fail):
    monkeypatch.delenv("COSMOS_STATUS_FILE", raising=False)
    monkeypatch.setenv("RANK", "0")

    @monitor_status(results_dir=str(tmp_path))
    def action():
        log_workflow_status({"loss": 1})
        if fail:
            raise RuntimeError("failed")

    if fail:
        with pytest.raises(RuntimeError):
            action()
    else:
        action()
    records = [json.loads(line) for line in (tmp_path / "status.json").read_text().splitlines()]
    assert records[-1]["status"] == ("FAILURE" if fail else "SUCCESS")
    assert records[1]["kpi"] == {"loss": 1}


def test_lightweight_imports_do_not_need_training_or_source_packages():
    code = """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.startswith(("nvidia_tao", "cosmos_rl", "torch")):
            raise RuntimeError("Forbidden runtime dependency: " + fullname)
sys.meta_path.insert(0, Block())
from cosmos_framework.data.reasoner.qa_dataset import ReasoningConversationDataset
from cosmos_framework.inference.reasoner.service import create_app
from cosmos_framework.inference.reasoner.runtime import CosmosFrameworkRuntime
from cosmos_framework.evaluation.reasoner.evaluator import Evaluator
from cosmos_framework.scripts.quantize_reasoner import main
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_runtime_provenance_destinations_and_local_import_closure():
    import ast

    root = Path(__file__).parents[2]
    manifest = json.loads((root / "cosmos_framework/licenses/workflows/provenance.json").read_text())
    for item in manifest["runtime_implementations"]:
        if "repository" in item:
            assert item["repository"] in manifest["runtime_sources"]
        assert len(item["source_sha256"]) == 64
        assert (root / item["destination"]).is_file()
    for subtree in (
        "inference/reasoner",
        "evaluation/reasoner",
        "data/reasoner/formats",
    ):
        for path in (root / "cosmos_framework" / subtree).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                modules = (
                    [item.name for item in node.names]
                    if isinstance(node, ast.Import)
                    else ([node.module or ""] if isinstance(node, ast.ImportFrom) else [])
                )
                for module in modules:
                    assert not module.startswith("nvidia_tao"), (path, module)
                    if module.startswith("cosmos_framework"):
                        relative = Path(*module.split("."))
                        assert (root / relative).is_dir() or (root / relative.with_suffix(".py")).is_file(), (
                            path,
                            module,
                        )


def test_status_barrier_rejects_stale_publishers(monkeypatch, tmp_path):
    from cosmos_framework.evaluation.reasoner.status_barrier import wait_for_status_publishers

    monkeypatch.setenv("COSMOS_RUN_ID", "attempt-one")
    wait_for_status_publishers(tmp_path, 1, 2, 1)
    wait_for_status_publishers(tmp_path, 0, 2, 1)
    with pytest.raises(FileExistsError):
        wait_for_status_publishers(tmp_path, 0, 2, 1)
    monkeypatch.setenv("COSMOS_RUN_ID", "attempt-two")
    with pytest.raises(TimeoutError):
        wait_for_status_publishers(tmp_path, 0, 2, 0)
