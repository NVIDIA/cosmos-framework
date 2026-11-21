# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
import sys
from types import SimpleNamespace

import pytest
from failuregen.fd_training import completed_checkpoint, prepare_dataset_view
from failuregen.yam_training.contract import file_sha256


@pytest.fixture
def bundle(tmp_path):
    source, bundle, work = (tmp_path / name for name in ("source", "bundle", "work"))
    for name in ("meta", "data", "videos"):
        (source / name).mkdir(parents=True)
    for name in ("meta/info.json", "cosmos_annotations.json", "data/episode.parquet", "videos/episode.mp4"):
        (source / name).write_text(name)
    (bundle / "eef").mkdir(parents=True)
    (bundle / "eef/manifest.json").write_text("{}")
    (bundle / "splits.json").write_text("{}")
    (bundle / "failuregen_annotations.json").write_bytes((source / "cosmos_annotations.json").read_bytes())
    (bundle / "source_identity.json").write_text(
        json.dumps(
            {
                "info_sha256": file_sha256(source / "meta/info.json"),
                "annotations_sha256": file_sha256(source / "cosmos_annotations.json"),
                "parquet_sha256": {"data/episode.parquet": file_sha256(source / "data/episode.parquet")},
                "video_sizes": {"videos/episode.mp4": (source / "videos/episode.mp4").stat().st_size},
            }
        )
    )
    return source, bundle, work


def test_view_preserves_source_and_supports_resume(bundle):
    source, export, work = bundle
    view = prepare_dataset_view(*bundle)
    assert (view / "meta").resolve() == source / "meta"
    assert (view / "failuregen_annotations.json").resolve() == export / "failuregen_annotations.json"
    assert not (source / "failuregen_annotations.json").exists()
    assert prepare_dataset_view(*bundle) == view


@pytest.mark.parametrize(
    "name", ["meta/info.json", "cosmos_annotations.json", "data/episode.parquet", "videos/episode.mp4"]
)
def test_changed_source_rejected_before_output_creation(bundle, name):
    source, _, work = bundle
    (source / name).write_text("modified")
    with pytest.raises(ValueError, match="differs"):
        prepare_dataset_view(*bundle)
    assert not work.exists()


def test_existing_view_not_overwritten(bundle):
    _, _, work = bundle
    view = prepare_dataset_view(*bundle)
    (view / "meta").unlink()
    (view / "meta").mkdir()
    with pytest.raises(ValueError, match="Refusing to replace"):
        prepare_dataset_view(*bundle)


def test_handoff_requires_final_checkpoint_and_matching_contract(tmp_path, monkeypatch):
    run = tmp_path / "run"
    saved = run / "checkpoints/iter_000000010/model"
    saved.mkdir(parents=True)
    (saved / ".metadata").touch()
    (run / "yam_training_contract.json").write_text(json.dumps({"max_iter": 10}))
    latest = run / "checkpoints/latest_checkpoint.txt"
    latest.write_text("iter_000000005")
    checks = []
    module = SimpleNamespace(load_context=lambda *args: checks.append(args))
    monkeypatch.setitem(sys.modules, "failuregen.yam_training.failure_test", module)
    with pytest.raises(ValueError, match="incomplete"):
        completed_checkpoint(run, tmp_path / "prepared", tmp_path / "dataset")
    latest.write_text("iter_000000010")
    assert completed_checkpoint(run, tmp_path / "prepared", tmp_path / "dataset") == saved.parent
    assert checks[-1] == (tmp_path / "prepared", tmp_path / "dataset", run)

    def mismatch(*args):
        raise ValueError("mismatched contract")

    module.load_context = mismatch
    with pytest.raises(ValueError, match="mismatched contract"):
        completed_checkpoint(run, tmp_path / "prepared", tmp_path / "dataset")
