# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU tests for the ported DPO preference dataset (no model required).

Covers the behavior that matters for correctness of the offline DPO data path:
canonical `VIDEO_RES_SIZE_INFO` bucket-snap (shape + aspect-ratio bin), score-gap
filtering, symmetric per-pair cfg-dropout, winner/loser shape parity, and that the
`DatasetPreferenceEnv` → `unpack_preferences` collate yields the list-of-vision-items
batch shape `Cosmos3FlowPolicy.encode_and_condition` expects.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.data.generator.utils import VIDEO_RES_SIZE_INFO
from cosmos_framework.posttrain.data.datasets.dpo_preference import DPOPreferenceDataset, _aspect_ratio_bin
from cosmos_framework.posttrain.data.packer.preference import unpack_preferences
from cosmos_framework.posttrain.roles.env.preference import (
    DatasetPreferenceEnv,
    PreferenceSourceIdentity,
    validate_preference_sources,
)


def _save_video(path, C=3, T=4, H=64, W=64):
    torch.save({"frames": torch.randint(0, 256, (C, T, H, W), dtype=torch.uint8)}, path)


def _write_jsonl(tmp_path, entries):
    p = tmp_path / "pairs.jsonl"
    with open(p, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    return str(p)


def _pair(tmp_path, name, h, w, gap, fps=24):
    win, los = tmp_path / f"{name}_w.pt", tmp_path / f"{name}_l.pt"
    _save_video(win, H=h, W=w)
    _save_video(los, H=h, W=w)
    return {"prompt": f"a {name}", "winner_video": str(win), "loser_video": str(los), "score_gap": gap, "fps": fps}


@pytest.mark.L0
@pytest.mark.CPU
def test_score_gap_filter(tmp_path):
    jsonl = _write_jsonl(
        tmp_path, [_pair(tmp_path, "keep", 64, 64, gap=0.5), _pair(tmp_path, "drop", 64, 64, gap=0.05)]
    )
    ds = DPOPreferenceDataset(jsonl, resolution="256", min_score_gap=0.1)
    assert len(ds) == 1
    prompt, _, _ = ds[0]
    assert prompt == "a keep"


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "h,w,bin_key",
    [(64, 64, "1,1"), (90, 160, "16,9"), (160, 90, "9,16")],
    ids=["square", "wide", "tall"],
)
def test_bucket_snap_shape(tmp_path, h, w, bin_key):
    assert _aspect_ratio_bin(w, h) == bin_key
    jsonl = _write_jsonl(tmp_path, [_pair(tmp_path, "v", h, w, gap=1.0)])
    ds = DPOPreferenceDataset(jsonl, resolution="256", cfg_dropout_rate=0.0)
    _prompt, win, los = ds[0]
    tw, th = VIDEO_RES_SIZE_INFO["256"][bin_key]
    assert win["video"][0].shape == (3, 4, th, tw)
    assert win["video"][0].dtype == torch.uint8
    # winner/loser identical (C,T,H,W) — required for shared (sigma, epsilon).
    assert win["video"][0].shape == los["video"][0].shape
    # image_size = [target_h, target_w, new_h, new_w]
    assert win["image_size"][0].item() == th and win["image_size"][1].item() == tw
    assert win["conditioning_fps"].dtype == torch.bfloat16 and win["conditioning_fps"].shape == (1,)


@pytest.mark.L0
@pytest.mark.CPU
def test_cfg_dropout_symmetric(tmp_path):
    jsonl = _write_jsonl(tmp_path, [_pair(tmp_path, "v", 64, 64, gap=1.0)])
    # Always-drop: both sides see the empty caption (DPO symmetry preserved).
    ds1 = DPOPreferenceDataset(jsonl, resolution="256", cfg_dropout_rate=1.0)
    prompt, win, los = ds1[0]
    assert prompt == "" and win["ai_caption"] == "" and los["ai_caption"] == ""
    # Never-drop: both sides see the real caption.
    ds0 = DPOPreferenceDataset(jsonl, resolution="256", cfg_dropout_rate=0.0)
    prompt, win, los = ds0[0]
    assert prompt == "a v" and win["ai_caption"] == "a v" and los["ai_caption"] == "a v"


@pytest.mark.L0
@pytest.mark.CPU
def test_cfg_dropout_rng_cursor_round_trip(tmp_path):
    jsonl = _write_jsonl(tmp_path, [_pair(tmp_path, "v", 16, 16, gap=1.0)])
    source = DPOPreferenceDataset(jsonl, resolution=None, cfg_dropout_rate=0.5, seed=17)
    state = source.cursor_state_dict()
    expected = [source[0][0] for _ in range(12)]

    restored = DPOPreferenceDataset(jsonl, resolution=None, cfg_dropout_rate=0.5, seed=999)
    restored.load_cursor_state_dict(state)
    assert [restored[0][0] for _ in range(12)] == expected
    assert restored.cursor_fingerprint() == source.cursor_fingerprint()

    coerced_version = dict(state)
    coerced_version["version"] = 1.0
    with pytest.raises(ValueError, match="version must be an integer"):
        restored.load_cursor_state_dict(coerced_version)


@pytest.mark.L0
@pytest.mark.CPU
def test_cursor_fingerprint_ignores_media_mtime_but_detects_row_changes(
    tmp_path: Path,
) -> None:
    pair = _pair(tmp_path, "v", 16, 16, gap=1.0)
    jsonl = _write_jsonl(tmp_path, [pair])
    before = DPOPreferenceDataset(jsonl, resolution=None).cursor_fingerprint()
    stat = os.stat(pair["winner_video"])
    os.utime(
        pair["winner_video"],
        ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000),
    )
    restaged = DPOPreferenceDataset(jsonl, resolution=None).cursor_fingerprint()
    assert restaged == before

    changed = dict(pair)
    changed["prompt"] = "a semantically different pair"
    changed_jsonl = tmp_path / "changed.jsonl"
    changed_jsonl.write_text(json.dumps(changed) + "\n")
    after = DPOPreferenceDataset(str(changed_jsonl), resolution=None).cursor_fingerprint()
    assert after != before


@pytest.mark.L0
@pytest.mark.CPU
def test_membership_identity_ignores_presentation_knobs_and_media_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leakage identity describes the pair, not FPS/CFG or decoded media bytes."""
    pair = _pair(tmp_path, "v", 16, 16, gap=1.0, fps=12)
    jsonl = _write_jsonl(tmp_path, [pair])
    base = DPOPreferenceDataset(jsonl, resolution=None, cfg_dropout_rate=0.0)
    expected = base.membership_ids()

    changed = dict(pair)
    changed["fps"] = 48
    changed_jsonl = tmp_path / "changed-fps.jsonl"
    changed_jsonl.write_text(json.dumps(changed) + "\n")
    assert (
        DPOPreferenceDataset(
            str(changed_jsonl),
            resolution="256",
            cfg_dropout_rate=1.0,
        ).membership_ids()
        == expected
    )

    real_open = open
    media_paths = {os.path.realpath(pair["winner_video"]), os.path.realpath(pair["loser_video"])}

    def _guarded_open(path: str | os.PathLike[str], *args: object, **kwargs: object) -> object:
        if os.path.realpath(os.fspath(path)) in media_paths:
            raise AssertionError("membership audit must not open media payloads")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", _guarded_open)
    assert base.membership_ids() == expected


@pytest.mark.L0
@pytest.mark.CPU
def test_membership_identity_matches_locator_and_digest_aliases(tmp_path: Path) -> None:
    """Digest metadata cannot hide an obvious same-path pair and can identify a copied pair."""
    base_pair = _pair(tmp_path, "base", 16, 16, gap=1.0)
    base_pair["prompt"] = "the same held-out prompt"
    base_jsonl = _write_jsonl(tmp_path, [base_pair])
    base_ids = set(DPOPreferenceDataset(base_jsonl, resolution=None).membership_ids())

    declared = dict(base_pair)
    declared["winner_sha256"] = "A" * 64
    declared["loser_sha256"] = "sha256:" + "B" * 64
    declared_jsonl = tmp_path / "declared-digests.jsonl"
    declared_jsonl.write_text(json.dumps(declared) + "\n")
    declared_dataset = DPOPreferenceDataset(str(declared_jsonl), resolution=None)
    declared_ids = set(declared_dataset.membership_ids())
    assert base_ids.intersection(declared_ids)
    base_env = DatasetPreferenceEnv(
        DPOPreferenceDataset(base_jsonl, resolution=None),
        shuffle=False,
        cursor_resume="off",
        raw_source_fingerprint="raw-base",
    )
    declared_env = DatasetPreferenceEnv(
        declared_dataset,
        shuffle=False,
        cursor_resume="off",
        raw_source_fingerprint="raw-declared",
    )
    assert set(declared_env.membership_ids()) == declared_ids
    with pytest.raises(ValueError, match="overlap semantically"):
        validate_preference_sources(
            base_env,
            SimpleNamespace(validation_source_identity=declared_env.preference_source_identity),
        )
    with pytest.raises(ValueError, match="overlap semantically"):
        base_env.reject_preference_source_overlap(
            PreferenceSourceIdentity(
                raw_fingerprint="raw-base",
                semantic_fingerprint="different-semantic-source",
                membership_ids=("different-pair",),
            )
        )
    base_env.reject_preference_source_overlap(
        PreferenceSourceIdentity(
            raw_fingerprint="different-raw-source",
            semantic_fingerprint="different-semantic-source",
            membership_ids=("different-pair",),
        )
    )

    copied = _pair(tmp_path, "copy", 16, 16, gap=1.0)
    copied["prompt"] = declared["prompt"]
    copied["winner_sha256"] = "sha256:" + "a" * 64
    copied["loser_sha256"] = "b" * 64
    copied_jsonl = tmp_path / "copied-with-digests.jsonl"
    copied_jsonl.write_text(json.dumps(copied) + "\n")
    copied_ids = set(DPOPreferenceDataset(str(copied_jsonl), resolution=None).membership_ids())
    assert declared_ids.intersection(copied_ids)

    invalid = dict(base_pair)
    invalid["winner_sha256"] = "not-a-sha256"
    invalid_jsonl = tmp_path / "invalid-digest.jsonl"
    invalid_jsonl.write_text(json.dumps(invalid) + "\n")
    with pytest.raises(ValueError, match="64 hexadecimal SHA-256"):
        DPOPreferenceDataset(str(invalid_jsonl), resolution=None).membership_ids()


@pytest.mark.L0
@pytest.mark.CPU
def test_relative_media_paths_resolve_against_jsonl_not_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    launch_dir = tmp_path / "launch"
    data_dir.mkdir()
    launch_dir.mkdir()
    win, los = data_dir / "winner.pt", data_dir / "loser.pt"
    _save_video(win, H=16, W=16)
    _save_video(los, H=16, W=16)
    jsonl = data_dir / "pairs.jsonl"
    jsonl.write_text(
        json.dumps(
            {
                "prompt": "relative paths",
                "winner_video": win.name,
                "loser_video": los.name,
            }
        )
        + "\n"
    )

    expected = DPOPreferenceDataset(str(jsonl), resolution=None).cursor_fingerprint()
    monkeypatch.chdir(launch_dir)
    dataset = DPOPreferenceDataset(str(jsonl), resolution=None)

    assert dataset.cursor_fingerprint() == expected
    prompt, winner, loser = dataset[0]
    assert prompt == "relative paths"
    assert winner["video"][0].shape == loser["video"][0].shape == (3, 4, 16, 16)


@pytest.mark.L0
@pytest.mark.CPU
def test_resolution_none_keeps_native_shape(tmp_path):
    jsonl = _write_jsonl(tmp_path, [_pair(tmp_path, "v", 48, 80, gap=1.0)])
    ds = DPOPreferenceDataset(jsonl, resolution=None, cfg_dropout_rate=0.0)
    _prompt, win, _los = ds[0]
    assert win["video"][0].shape == (3, 4, 48, 80)  # unresized
    assert win["image_size"].tolist() == [48, 80, 48, 80]


@pytest.mark.L0
@pytest.mark.CPU
def test_env_unpack_batch_shape(tmp_path):
    jsonl = _write_jsonl(tmp_path, [_pair(tmp_path, "a", 64, 64, gap=1.0), _pair(tmp_path, "b", 64, 64, gap=1.0)])
    ds = DPOPreferenceDataset(jsonl, resolution="256", cfg_dropout_rate=0.0)
    env = DatasetPreferenceEnv(ds)
    trajs = env.get_samples(2)
    winner_batch, loser_batch = unpack_preferences(trajs)
    # encode_and_condition expects "video" as a list (B) of one-item vision lists.
    assert len(winner_batch["video"]) == 2
    assert isinstance(winner_batch["video"][0], list) and winner_batch["video"][0][0].shape == (3, 4, 256, 256)
    assert len(winner_batch["ai_caption"]) == 2 and len(loser_batch["video"]) == 2
    assert len(winner_batch["image_size"]) == 2 and len(winner_batch["conditioning_fps"]) == 2


@pytest.mark.L0
@pytest.mark.CPU
def test_env_rank_sharding_is_disjoint_and_complete(tmp_path):
    # 8 pairs; two ranks (world=2) must see disjoint, complete, DistributedSampler-style
    # strided shards — aligning with cosmos-rl's DistributedSampler-based DPO loader.
    entries = [_pair(tmp_path, f"p{i}", 64, 64, gap=1.0) for i in range(8)]
    jsonl = _write_jsonl(tmp_path, entries)
    ds = DPOPreferenceDataset(jsonl, resolution="256", cfg_dropout_rate=0.0)

    e0 = DatasetPreferenceEnv(ds, rank=0, world_size=2, shuffle=False)
    e1 = DatasetPreferenceEnv(ds, rank=1, world_size=2, shuffle=False)
    p0 = [t.transitions[0].obs.representations["prompt"] for t in e0.get_samples(4)]
    p1 = [t.transitions[0].obs.representations["prompt"] for t in e1.get_samples(4)]
    assert p0 == ["a p0", "a p2", "a p4", "a p6"]  # strided indices[0::2]
    assert p1 == ["a p1", "a p3", "a p5", "a p7"]  # strided indices[1::2]
    assert set(p0).isdisjoint(p1) and set(p0) | set(p1) == {f"a p{i}" for i in range(8)}


@pytest.mark.L0
@pytest.mark.CPU
def test_env_reshuffles_each_epoch(tmp_path):
    entries = [_pair(tmp_path, f"p{i}", 64, 64, gap=1.0) for i in range(6)]
    jsonl = _write_jsonl(tmp_path, entries)
    ds = DPOPreferenceDataset(jsonl, resolution="256", cfg_dropout_rate=0.0)
    env = DatasetPreferenceEnv(ds, rank=0, world_size=1, shuffle=True, seed=0)
    epoch1 = [t.transitions[0].obs.representations["prompt"] for t in env.get_samples(6)]
    epoch2 = [t.transitions[0].obs.representations["prompt"] for t in env.get_samples(6)]
    assert sorted(epoch1) == sorted(epoch2) == [f"a p{i}" for i in range(6)]  # same set
    assert epoch1 != epoch2  # different order per epoch (set_epoch-style reshuffle)


@pytest.mark.L0
@pytest.mark.CPU
def test_image_media_yields_images_key(tmp_path):
    # media="image": single-frame [C,H,W] under the `images` key (not `video`), snapped
    # via IMAGE_RES_SIZE_INFO — the T2I (T=1) data path.
    def _save_img(path, h, w):
        torch.save({"frames": torch.randint(0, 256, (3, h, w), dtype=torch.uint8)}, path)

    win, los = tmp_path / "iw.pt", tmp_path / "il.pt"
    _save_img(win, 64, 64)
    _save_img(los, 64, 64)
    jsonl = _write_jsonl(
        tmp_path, [{"prompt": "an image", "winner_video": str(win), "loser_video": str(los), "score_gap": 1.0}]
    )
    ds = DPOPreferenceDataset(jsonl, media="image", resolution="256", cfg_dropout_rate=0.0)
    _prompt, w, _l = ds[0]
    assert "images" in w and "video" not in w
    assert w["images"][0].shape == (3, 256, 256)  # [C,H,W], single frame (no T dim)
    assert w["images"][0].dtype == torch.uint8


@pytest.mark.L0
@pytest.mark.CPU
def test_env_empty_shard_fails_loud(tmp_path):
    # world_size > #pairs: a strided shard would be empty → fail loud at rebuild with a
    # descriptive message, not a bare IndexError mid-training.
    jsonl = _write_jsonl(tmp_path, [_pair(tmp_path, "only", 64, 64, gap=1.0)])  # 1 pair
    ds = DPOPreferenceDataset(jsonl, resolution="256", cfg_dropout_rate=0.0)
    env = DatasetPreferenceEnv(ds, rank=0, world_size=4)  # 1 pair, world 4 → unservable
    with pytest.raises(ValueError, match="world_size"):
        env.get_samples(1)


@pytest.mark.L0
@pytest.mark.CPU
def test_unpack_empty_batch_fails_loud():
    with pytest.raises(ValueError, match="at least one sample"):
        unpack_preferences([])


@pytest.mark.L0
@pytest.mark.CPU
def test_winner_loser_shape_mismatch_raises(tmp_path):
    # Cross-bin pair (square winner, wide loser) → different snapped (H,W) → ValueError.
    win, los = tmp_path / "w.pt", tmp_path / "l.pt"
    _save_video(win, H=64, W=64)  # 1,1 -> (256,256)
    _save_video(los, H=90, W=160)  # 16,9 -> (192,320)
    jsonl = _write_jsonl(
        tmp_path, [{"prompt": "x", "winner_video": str(win), "loser_video": str(los), "score_gap": 1.0}]
    )
    ds = DPOPreferenceDataset(jsonl, resolution="256", cfg_dropout_rate=0.0)
    with pytest.raises(ValueError, match="winner/loser shape mismatch"):
        _ = ds[0]
