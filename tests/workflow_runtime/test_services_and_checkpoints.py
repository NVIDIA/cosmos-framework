# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from cosmos_framework.checkpoint.reasoner import (
    _check_output_path,
    ensure_evaluation_checkpoint,
    ensure_hf_checkpoint,
    is_hf_checkpoint,
)


def test_hf_index_requires_all_shards_and_rejects_traversal(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"a": "one.safetensors", "b": "two.safetensors"}}))
    (tmp_path / "one.safetensors").write_bytes(b"weights")
    assert not is_hf_checkpoint(tmp_path)
    (tmp_path / "two.safetensors").write_bytes(b"weights")
    assert is_hf_checkpoint(tmp_path)
    index.write_text(json.dumps({"weight_map": {"a": "../secret.safetensors"}}))
    assert not is_hf_checkpoint(tmp_path)


@pytest.mark.parametrize("destination", ["source", "source/child", "."])
def test_checkpoint_output_cannot_overlap_input(tmp_path, destination):
    with pytest.raises(ValueError, match="overlaps"):
        _check_output_path((tmp_path / destination).resolve(), tmp_path / "source")


def test_stale_dcp_export_is_preserved_and_rejected(tmp_path):
    checkpoint = tmp_path / "dcp"
    checkpoint.mkdir()
    (checkpoint / ".metadata").write_bytes(b"new metadata")
    config = tmp_path / "config.yaml"
    config.write_text("model: {}")
    output = tmp_path / "export"
    output.mkdir()
    (output / "config.json").write_text("{}")
    (output / "model.safetensors").write_bytes(b"previous export")
    (output / "checkpoint.json").write_text("{}")
    with pytest.raises(ValueError, match="matching provenance"):
        ensure_hf_checkpoint(str(checkpoint), config_file=str(config), export_dir=str(output))
    assert (output / "model.safetensors").read_bytes() == b"previous export"


def test_stale_lora_merge_is_preserved_and_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "0")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"new adapter")
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}")
    (base / "model.safetensors").write_bytes(b"base")
    output = tmp_path / "merged"
    output.mkdir()
    (output / "config.json").write_text("{}")
    (output / "model.safetensors").write_bytes(b"old merge")
    (output / ".cosmos_lora_merge_complete.json").write_text("{}")
    with pytest.raises(ValueError, match="matching provenance"):
        ensure_evaluation_checkpoint(str(adapter), base_model_path=str(base), enable_lora=True, export_dir=str(output))
    assert (output / "model.safetensors").read_bytes() == b"old merge"


def test_lora_merge_delegates_cache_validation_without_loading_models(tmp_path, monkeypatch):
    from cosmos_framework.checkpoint import reasoner
    from cosmos_framework.checkpoint.lora import merge_lora_model

    calls = []

    def prepare(path, **kwargs):
        calls.append((path, kwargs))
        return str(tmp_path / "merged")

    monkeypatch.setattr(reasoner, "ensure_evaluation_checkpoint", prepare)
    assert merge_lora_model(str(tmp_path / "adapter"), str(tmp_path / "base")) == str(tmp_path / "merged")
    assert calls == [(str(tmp_path / "adapter"), {"base_model_path": str(tmp_path / "base"), "enable_lora": True})]


@pytest.mark.parametrize("module", ["inference_reasoner", "quantize_reasoner", "serve_reasoner"])
def test_action_help_needs_no_models_or_runtime_packages(module, tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "cosmos_framework.scripts." + module, "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "usage:" in result.stdout
    assert not list(tmp_path.iterdir())


def test_reasoner_http_contract_and_rejected_paths(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from cosmos_framework.inference.reasoner.service import create_app

    calls = []

    def generate(tasks, **kwargs):
        calls.append((tasks, kwargs))
        return ["answer"] * len(tasks), {}

    client = TestClient(create_app(SimpleNamespace(generate_tasks=generate), "cosmos", allowed_roots=[tmp_path]))
    assert client.get("/health").json()["model_loaded"] is True
    assert client.get("/v1/models").json()["data"][0]["id"] == "cosmos"
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "cosmos",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGk="}},
                    ],
                }
            ],
            "seed": 42,
            "max_tokens": 13,
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == "answer"
    assert calls[-1][1]["generation_config"] == {"seed": 42, "max_tokens": 13}
    path = calls[-1][0][0]["prompt"][0]["content"][1]["image"]
    assert not Path(path).exists()
    assert client.post("/infer", json={"media": "/etc/passwd"}).status_code == 400
    assert client.post("/v1/chat/completions", json={"stream": True}).status_code == 400
