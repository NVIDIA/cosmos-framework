# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest


def _run_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    samples: list[dict[str, Any]],
    *,
    build_sample: Any = None,
) -> tuple[Mock, list[str], Path]:
    """Run ``scripts/inference.main`` with downloads, model construction and collectives stubbed out."""
    from cosmos_framework.inference.common import init

    monkeypatch.setattr(init, "init_script", Mock())
    from cosmos_framework.inference.args import OmniSampleOverrides, OmniSetupOverrides
    from cosmos_framework.inference.common import inference as common_inference
    from cosmos_framework.scripts import inference

    input_path = tmp_path / "samples.jsonl"
    input_path.write_text("".join(json.dumps(sample) + "\n" for sample in samples))
    output_dir = tmp_path / "outputs"
    monkeypatch.setattr(
        sys,
        "argv",
        ["inference", "-i", str(input_path), "--output-dir", str(output_dir), "--checkpoint-path", "Cosmos3-Nano"],
    )

    events: list[str] = []
    setup = Mock(
        output_dir=output_dir,
        sample_overrides=None,
        benchmark=False,
        skip_invalid_samples=True,
        experiment_overrides=[],
    )
    setup.get_sample_overrides_cls.return_value = OmniSampleOverrides
    pipe = setup.get_inference_cls.return_value.create.return_value
    setup.get_inference_cls.return_value.create.side_effect = lambda _setup: events.append("create") or pipe

    def build_setup(overrides: OmniSetupOverrides) -> Mock:
        setup.sample_overrides = overrides.sample_overrides
        setup.model_dump_json.return_value = overrides.model_dump_json()
        return setup

    def download(self: OmniSampleOverrides, output_dir: Path) -> None:
        events.append(f"download {self.name}")

    monkeypatch.setattr(OmniSetupOverrides, "build_setup", build_setup)
    monkeypatch.setattr(OmniSampleOverrides, "download", download)
    monkeypatch.setattr(OmniSampleOverrides, "build_sample", build_sample or (lambda self, **kwargs: self))
    monkeypatch.setattr(inference, "init_output_dir", lambda path: path.mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr(inference, "is_rank0", lambda: True)
    monkeypatch.setattr(inference, "barrier", lambda: events.append("barrier"))
    monkeypatch.setattr(common_inference, "sync_distributed_errors", nullcontext)

    inference.main()
    return pipe, events, output_dir


@pytest.mark.L0
@pytest.mark.CPU
def test_ranks_wait_for_rank0_downloads_before_building_the_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = [{"name": "first", "prompt": "a"}, {"name": "second", "prompt": "b"}]

    pipe, events, output_dir = _run_cli(tmp_path, monkeypatch, samples)

    assert events == ["download first", "download second", "barrier", "create"]
    assert [sample.name for sample in pipe.generate.call_args.args[0]] == ["first", "second"]
    receipt = json.loads((output_dir / "setup_args.json").read_text())
    assert receipt["checkpoint_path"] == "Cosmos3-Nano"


@pytest.mark.L0
@pytest.mark.CPU
def test_skip_receipt_keeps_the_requested_sample_args(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def build_sample(self: Any, **kwargs: Any) -> Any:
        # Sample preparation mutates the overrides before it can reject them.
        self.prompt = "mutated during build"
        raise ValueError("unsupported modality")

    pipe, _, output_dir = _run_cli(
        tmp_path, monkeypatch, [{"name": "skipped", "prompt": "as requested"}], build_sample=build_sample
    )

    assert pipe.generate.call_args.args[0] == []
    receipt = json.loads((output_dir / "skipped" / "sample_outputs.json").read_text())
    assert receipt["status"] == "skip"
    assert receipt["args"]["prompt"] == "as requested"


@pytest.mark.L0
@pytest.mark.CPU
def test_nested_sample_names_stay_under_the_output_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipe, _, output_dir = _run_cli(tmp_path, monkeypatch, [{"name": "inference_prompt/episode", "prompt": "a"}])

    (sample,) = pipe.generate.call_args.args[0]
    assert sample.output_dir == output_dir / "inference_prompt" / "episode"


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("name", ["../escape", "/abs/path", "a/../../escape"])
def test_sample_names_cannot_escape_the_output_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    with pytest.raises(ValueError, match="Sample name"):
        _run_cli(tmp_path, monkeypatch, [{"name": name, "prompt": "a"}])
