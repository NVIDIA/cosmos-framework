# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Training bundle preflight and native-loader handoff, without model weights."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from cosmos_framework.scripts.validate_bimanual_data import materialize_batch, validate_bundle

pytestmark = [pytest.mark.level(0), pytest.mark.gpus(0)]


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    root = tmp_path / "bundle"
    for name in (
        "global_asinh.json",
        "tokenizer/vocab.json",
        "tokenizer/merges.txt",
        "tokenizer/tokenizer_config.json",
    ):
        _write(root / "resources" / name, {})
    _write(root / "assets/payload", {})
    _write(
        root / "manifest.json",
        {
            "format_version": 1,
            "model": "bimanual",
            "complete": True,
            "sample_count": 1,
            "transforms": {"robot": {"stats": "bundle://resources/global_asinh.json"}},
        },
    )
    _write(
        root / "samples.json",
        [
            {
                "identity": "sample",
                "stratum": "robot",
                "request": {"frame_count": 901},
                "source_rows": {"row": {}},
                "assets": {"payload": "assets/payload"},
            }
        ],
    )
    return root


def test_static_bundle_without_authoring_receipts(bundle: Path) -> None:
    assert validate_bundle(bundle) == {"samples": 1, "assets": 1, "strata": {"robot": 1}, "materialized": False}
    assert not (bundle / "receipts").exists()


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("model", "auto"),
        ("format_version", 2),
        ("complete", False),
        ("sample_count", 2),
        ("sample_count", True),
        ("transforms", []),
        ("transforms", {"robot": None}),
    ],
)
def test_invalid_manifest(bundle: Path, key: str, value: Any) -> None:
    path = bundle / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest[key] = value
    _write(path, manifest)
    with pytest.raises(ValueError):
        validate_bundle(bundle)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("identity", ""),
        ("stratum", "unknown"),
        ("request", []),
        ("request", {}),
        ("source_rows", [{}]),
        ("source_rows", {}),
        ("source_rows", {"row": 1}),
        ("assets", {}),
    ],
)
def test_invalid_sample(bundle: Path, key: str, value: Any) -> None:
    path = bundle / "samples.json"
    records = json.loads(path.read_text())
    records[0][key] = value
    _write(path, records)
    with pytest.raises(ValueError):
        validate_bundle(bundle)


def test_duplicate_identity(bundle: Path) -> None:
    records = json.loads((bundle / "samples.json").read_text())
    _write(bundle / "samples.json", records * 2)
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["sample_count"] = 2
    _write(bundle / "manifest.json", manifest)
    with pytest.raises(ValueError, match="unique"):
        validate_bundle(bundle)


@pytest.mark.parametrize(
    "relative", ["assets/payload", "resources/tokenizer/merges.txt", "resources/global_asinh.json"]
)
def test_missing_file(bundle: Path, relative: str) -> None:
    (bundle / relative).unlink()
    with pytest.raises(FileNotFoundError):
        validate_bundle(bundle)


@pytest.mark.parametrize("escape", ["../outside", "absolute", "symlink"])
def test_asset_cannot_escape_bundle(bundle: Path, escape: str) -> None:
    outside = bundle.parent / "outside"
    outside.write_text("not a bundle asset")
    if escape == "symlink":
        (bundle / "assets/link").symlink_to(outside)
        relative = "assets/link"
    else:
        relative = str(outside) if escape == "absolute" else escape
    records = json.loads((bundle / "samples.json").read_text())
    records[0]["assets"]["payload"] = relative
    _write(bundle / "samples.json", records)
    with pytest.raises(ValueError, match="escapes"):
        validate_bundle(bundle)


def test_transform_resource_is_checked(bundle: Path) -> None:
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["transforms"]["robot"]["nested"] = [{"mesh": "bundle://resources/missing.xml"}]
    _write(bundle / "manifest.json", manifest)
    with pytest.raises(FileNotFoundError, match="missing.xml"):
        validate_bundle(bundle)


def test_static_cli_does_not_import_torch_or_training(bundle: Path) -> None:
    # A fresh process fails if static validation starts importing the training runtime.
    probe = """
import importlib.abc, runpy, sys
class BlockTraining(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('cosmos_framework.data'):
            raise AssertionError('Static validation imported ' + fullname)
sys.meta_path.insert(0, BlockTraining())
sys.argv = ['validate_bimanual_data', '--bundle', sys.argv[1]]
runpy.run_module('cosmos_framework.scripts.validate_bimanual_data', run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-c", probe, str(bundle)], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["materialized"] is False


@pytest.mark.parametrize("args", [["--materialize"], ["--stage", "0"], ["--allow-missing-camera-resolution"]])
def test_cli_requires_explicit_materialization_stage(bundle: Path, args: list[str]) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "cosmos_framework.scripts.validate_bimanual_data", "--bundle", str(bundle), *args],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert '"passed": true' not in result.stdout


@pytest.mark.parametrize("stage", range(4))
def test_materialization_reports_native_batch(monkeypatch: pytest.MonkeyPatch, bundle: Path, stage: int) -> None:
    import torch

    from cosmos_framework.data.generator.local_simulation import bimanual_config

    def factory(**kwargs: Any) -> list[dict[str, Any]]:
        assert kwargs["stage"] == stage and kwargs["bundle_path"] == str(bundle)
        assert kwargs["max_samples_per_batch"] == 1 and kwargs["num_workers"] == 0
        assert "allow_missing_camera_resolution" not in kwargs
        return [
            {
                "_num_samples": 1,
                "_num_tokens": 12,
                "video": [torch.zeros(3, 5, 16, 16, dtype=torch.uint8)],
                "action": [torch.zeros(4, 64)],
            }
        ]

    monkeypatch.setattr(bimanual_config, "make_bimanual_local_dataloader", factory)
    report = materialize_batch(bundle, stage)
    assert report["tensors"]["batch.action[0]"] == {"shape": [4, 64], "dtype": "torch.float32"}
    assert report["samples"] == 1 and report["tokens"] == 12


def test_nonfinite_batch_fails(monkeypatch: pytest.MonkeyPatch, bundle: Path) -> None:
    import torch

    from cosmos_framework.data.generator.local_simulation import bimanual_config

    batch = {
        "_num_samples": 1,
        "_num_tokens": 12,
        "video": [torch.zeros(3, 5, 16, 16)],
        "action": [torch.full((4, 64), float("nan"))],
    }
    monkeypatch.setattr(bimanual_config, "make_bimanual_local_dataloader", lambda **kwargs: [batch])
    with pytest.raises(ValueError, match="Non-finite tensor"):
        materialize_batch(bundle, 0)
