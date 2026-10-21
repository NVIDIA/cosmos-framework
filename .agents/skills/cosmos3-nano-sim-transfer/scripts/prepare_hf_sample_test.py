# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Check that the sample CLI loads packaged defaults independently of the cwd."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.level(0), pytest.mark.gpus(0)]
SCRIPT = Path(__file__).resolve().with_name("prepare_hf_sample.py")


@pytest.mark.parametrize("example", ["example1", "example2"])
def test_prepare_sample_from_another_working_directory(tmp_path: Path, example: str) -> None:
    samples = tmp_path / "sample data"
    samples.mkdir()
    prompt_file = samples / f"{example}_upsampled_prompt.json"
    caption = {"prompt": "A robot moves."} if example == "example1" else {"temporal_caption": "A robot moves."}
    prompt_file.write_text(json.dumps(caption))
    output = tmp_path / "prepared"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--example",
            example,
            "--sample-dir",
            str(samples),
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    defaults = json.loads((SCRIPT.parents[1] / "inference/assets/sample_args.json").read_text())
    modalities = ("depth", "seg") if example == "example1" else ("edge", "blur", "depth", "seg")
    assert {path.stem for path in (output / "requests").glob("*.json")} == set(modalities)
    for modality in modalities:
        request = json.loads((output / "requests" / f"{modality}.json").read_text())
        assert request["name"] == f"{example}_{modality}"
        assert request["prompt"] == (caption["prompt"] if example == "example1" else prompt_file.read_text())
        for key in ("resolution", "num_steps", "guidance", "control_guidance", "autoregressive"):
            assert request[key] == defaults[key]
        assert request["num_frames"] == (601 if example == "example1" else 121)
        assert request["max_frames"] == (600 if example == "example1" else 121)
        assert request["fps"] == 30.0
        if modality in ("depth", "seg"):
            suffix = "depth" if modality == "depth" else "segmentation"
            assert request[modality]["control_path"] == str(samples / f"{example}_{suffix}.mp4")
        else:
            assert request["vision_path"] == str(samples / f"{example}.mp4")

    if example == "example2":
        row = json.loads((output / "training.jsonl").read_text())
        assert row["identity"] == example
        assert row["caption_path"] == str(prompt_file)
        assert row["caption_end"] == 121
    else:
        assert not (output / "training.jsonl").exists()
