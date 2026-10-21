# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU tests for the tar-backed DPO preference dataset (cosmos-rl tar_dpo parity).

Builds a tiny real tar (2 JPG members) + an offset jsonl exactly like
`build_tar_index.py`/`resolve_pairs.py` produce, then checks the roundtrip: offset read,
JPG decode, bucket snap, and the parent's `(prompt, winner, loser)` contract."""

from __future__ import annotations

import io
import json
import os
import shutil
import tarfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")

from cosmos_framework.posttrain.data.datasets.tar_dpo_preference import TarDPOPreferenceDataset


def _jpg_bytes(color, size=(640, 360)):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


def _build_tar_and_jsonl(tmp_path, cfg_dropout=0.0):
    """One plain tar with two JPGs + an offset jsonl row referencing them."""
    tar_path = tmp_path / "images.tar"
    blobs = {"win.jpg": _jpg_bytes((255, 0, 0)), "lose.jpg": _jpg_bytes((0, 0, 255))}
    with tarfile.open(tar_path, "w") as tf:  # plain tar (no compression) — offsets are stable
        for name, b in blobs.items():
            info = tarfile.TarInfo(name)
            info.size = len(b)
            tf.addfile(info, io.BytesIO(b))
    with tarfile.open(tar_path) as tf:  # recover the byte offsets like build_tar_index.py
        offs = {m.name: (m.offset_data, m.size) for m in tf.getmembers()}
    row = {
        "prompt": "a test pair",
        "win_offset": offs["win.jpg"][0],
        "win_length": offs["win.jpg"][1],
        "lose_offset": offs["lose.jpg"][0],
        "lose_length": offs["lose.jpg"][1],
        "score_gap": 1.0,
        "fps": 1,
    }
    jsonl = tmp_path / "pairs.jsonl"
    jsonl.write_text(json.dumps(row) + "\n")
    return str(jsonl), str(tar_path)


@pytest.mark.L0
@pytest.mark.CPU
def test_tar_offset_roundtrip_video_t1(tmp_path):
    """Wallace-parity defaults: T=1 video samples, VIDEO res table, snap to bucket."""
    jsonl, tar = _build_tar_and_jsonl(tmp_path)
    ds = TarDPOPreferenceDataset(jsonl, tar_path=tar, resolution="720", cfg_dropout_rate=0.0)
    prompt, winner, loser = ds[0]
    assert prompt == "a test pair"
    w = winner["video"][0]
    assert w.dtype == torch.uint8 and w.dim() == 4 and w.shape[1] == 1  # [C, T=1, H, W]
    assert w.shape == loser["video"][0].shape
    assert winner["ai_caption"] == loser["ai_caption"] == prompt  # symmetric caption
    # Snapped to a 720-tier bucket: both dims match the recorded target size.
    th, tw = int(winner["image_size"][0]), int(winner["image_size"][1])
    assert (w.shape[2], w.shape[3]) == (th, tw)


@pytest.mark.L0
@pytest.mark.CPU
def test_tar_wrong_schema_fails_loud(tmp_path):
    """A path-based jsonl (no offsets) must be rejected up front, not at step N."""
    jsonl = tmp_path / "paths.jsonl"
    jsonl.write_text(json.dumps({"prompt": "x", "winner_video": "a.mp4", "loser_video": "b.mp4"}) + "\n")
    with pytest.raises(ValueError, match="missing 'win_offset'"):
        TarDPOPreferenceDataset(str(jsonl), tar_path=str(tmp_path / "none.tar"))


@pytest.mark.L0
@pytest.mark.CPU
def test_tar_gap_filter(tmp_path):
    jsonl, tar = _build_tar_and_jsonl(tmp_path)
    with pytest.raises(RuntimeError, match="no entries passed"):
        TarDPOPreferenceDataset(jsonl, tar_path=tar, min_score_gap=2.0)  # row has score_gap=1.0


@pytest.mark.L0
@pytest.mark.CPU
def test_tar_cursor_fingerprint_is_mtime_stable_and_size_sensitive(
    tmp_path: Path,
) -> None:
    jsonl, tar = _build_tar_and_jsonl(tmp_path)
    before = TarDPOPreferenceDataset(jsonl, tar_path=tar).cursor_fingerprint()
    stat = os.stat(tar)
    os.utime(tar, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    restaged = TarDPOPreferenceDataset(jsonl, tar_path=tar).cursor_fingerprint()
    assert restaged == before

    with open(tar, "ab") as archive:
        archive.write(b"changed-size")
    changed = TarDPOPreferenceDataset(jsonl, tar_path=tar).cursor_fingerprint()
    assert changed != before


@pytest.mark.L0
@pytest.mark.CPU
def test_relative_tar_path_resolves_against_jsonl_not_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jsonl, tar = _build_tar_and_jsonl(tmp_path)
    launch_dir = tmp_path / "launch"
    launch_dir.mkdir()
    expected = TarDPOPreferenceDataset(
        jsonl,
        tar_path=os.path.basename(tar),
        resolution=None,
    ).cursor_fingerprint()
    absolute = TarDPOPreferenceDataset(
        jsonl,
        tar_path=tar,
        resolution=None,
    ).cursor_fingerprint()
    dotted = TarDPOPreferenceDataset(
        jsonl,
        tar_path=f"./{os.path.basename(tar)}",
        resolution=None,
    ).cursor_fingerprint()
    assert absolute == dotted == expected

    monkeypatch.chdir(launch_dir)
    dataset = TarDPOPreferenceDataset(
        jsonl,
        tar_path=os.path.basename(tar),
        resolution=None,
    )
    assert dataset.cursor_fingerprint() == expected
    prompt, winner, loser = dataset[0]
    assert prompt == "a test pair"
    assert winner["video"][0].shape == loser["video"][0].shape


@pytest.mark.L0
@pytest.mark.CPU
def test_row_tar_path_spellings_share_effective_identity_and_default(
    tmp_path: Path,
) -> None:
    jsonl, tar = _build_tar_and_jsonl(tmp_path)
    row = json.loads(Path(jsonl).read_text())
    fingerprints: list[str] = []
    for tar_spec in (None, "", os.path.basename(tar), f"./{os.path.basename(tar)}", tar):
        manifest_row = dict(row)
        if tar_spec is not None:
            manifest_row["tar_path"] = tar_spec
        Path(jsonl).write_text(json.dumps(manifest_row) + "\n")
        dataset = TarDPOPreferenceDataset(
            jsonl,
            tar_path=os.path.basename(tar),
            resolution=None,
        )
        fingerprints.append(dataset.cursor_fingerprint())
        prompt, winner, loser = dataset[0]
        assert prompt == "a test pair"
        assert winner["video"][0].shape == loser["video"][0].shape
    assert len(set(fingerprints)) == 1

    base_ids = set(TarDPOPreferenceDataset(jsonl, tar_path=tar, resolution=None).membership_ids())
    row["winner_sha256"] = "A" * 64
    row["loser_sha256"] = "sha256:" + "B" * 64
    declared_jsonl = tmp_path / "tar-declared-digests.jsonl"
    declared_jsonl.write_text(json.dumps(row) + "\n")
    declared_ids = set(TarDPOPreferenceDataset(str(declared_jsonl), tar_path=tar).membership_ids())
    assert base_ids.intersection(declared_ids)

    copied_tar = tmp_path / "copied-images.tar"
    shutil.copyfile(tar, copied_tar)
    copied_row = {**row, "winner_sha256": "sha256:" + "a" * 64, "loser_sha256": "b" * 64}
    copied_jsonl = tmp_path / "copied-tar-digests.jsonl"
    copied_jsonl.write_text(json.dumps(copied_row) + "\n")
    copied_ids = set(TarDPOPreferenceDataset(str(copied_jsonl), tar_path=str(copied_tar)).membership_ids())
    assert declared_ids.intersection(copied_ids)
