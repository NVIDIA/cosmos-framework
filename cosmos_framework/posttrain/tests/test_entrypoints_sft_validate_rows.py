# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The SFT entrypoint's ``--validate-rows`` flag, imported lazily so this module collects on any tree.

Without the flag in ``utils/entry.py``, argparse drops ``--validate-rows`` as an unknown argument and ``main()``
enters ``run()``: this node then fails for that BEHAVIOURAL reason (``run()`` raises here), not at import. With
the flag, the entry exits 1 naming the over-long second row before ``run()`` or any trainer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = [pytest.mark.L0, pytest.mark.CPU]

_RECIPE = "cosmos_framework/posttrain/configs/sft_qwen3p8_27b_multimodal.py"


def _tiny_vlm_model(path: Path) -> None:
    from cosmos_framework.posttrain.tests.helpers.sft_packer import local_processor, local_tokenizer

    local_processor(local_tokenizer()).save_pretrained(path)
    (path / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))


def _two_rows(path: Path) -> None:
    rows = [
        {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]},
        {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "word " * 200}]},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _three_rows(path: Path) -> None:
    rows = [
        {"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": "ok"}]} for i in range(3)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_validate_rows_checks_dp_divisibility_from_the_launch_world_or_says_it_did_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """FIXTURE: on the parent ``--validate-rows`` called ``validate_from_config`` without ``dp_world``, so three rows
    passed under ``WORLD_SIZE=2`` and nothing said the divisibility was unchecked. Under torchrun's ``WORLD_SIZE=2``
    three valid rows exit 1 naming the refusal the setup applies; without ``WORLD_SIZE`` the same rows exit 0 and the
    train gate line says ``dp_world not checked``."""
    pytest.importorskip("transformers", minversion="5.8.1")
    from cosmos_framework.posttrain.entrypoints import sft as entry

    model, rows = tmp_path / "model", tmp_path / "rows.jsonl"
    model.mkdir()
    _tiny_vlm_model(model)
    _three_rows(rows)
    monkeypatch.setenv("IMAGINAIRE_OUTPUT_ROOT", str(tmp_path / "out"))
    monkeypatch.setattr(entry, "run", lambda *args, **kwargs: pytest.fail("run() was entered"))
    argv = [
        "sft",
        "--config",
        _RECIPE,
        "--validate-rows",
        "--",
        f"model_hf.name_or_path={model}",
        f"rl.data.path={rows}",
        "rl.validation.enabled=false",
        "rl.processor_kwargs={images_kwargs: {size: {shortest_edge: 16, longest_edge: 64}}}",
        "rl.max_length=256",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(SystemExit) as under_torchrun:
        entry.main()
    out = capsys.readouterr().out
    assert under_torchrun.value.code == 1, out
    assert "FAIL train: 3 rows are not divisible by dp_world=2" in out and "rows % dp_world(2) = 1" in out, out
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as outside:
        entry.main()
    out = capsys.readouterr().out
    assert outside.value.code == 0, out
    assert "train: dp_world not checked" in out and "--dp-world" in out, out


def test_validate_rows_flag_stops_before_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("transformers", minversion="5.8.1")
    from cosmos_framework.posttrain.entrypoints import sft as entry

    model, rows = tmp_path / "model", tmp_path / "rows.jsonl"
    model.mkdir()
    _tiny_vlm_model(model)
    _two_rows(rows)
    monkeypatch.setenv("IMAGINAIRE_OUTPUT_ROOT", str(tmp_path / "out"))

    def never(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("run() was entered: --validate-rows was dropped by the CLI")

    monkeypatch.setattr(entry, "run", never)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sft",
            "--config",
            _RECIPE,
            "--validate-rows",
            "--",
            f"model_hf.name_or_path={model}",
            f"rl.data.path={rows}",
            "rl.validation.enabled=false",
            "rl.processor_kwargs={images_kwargs: {size: {shortest_edge: 16, longest_edge: 64}}}",
            "rl.max_length=256",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        entry.main()
    out = capsys.readouterr().out
    assert exc.value.code == 1, out
    assert "FAIL train row 2: " in out and "exceed rl.max_length=256" in out, out
