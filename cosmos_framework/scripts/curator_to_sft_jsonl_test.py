# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Unit tests for the curator -> SFT JSONL converter."""

import json
from pathlib import Path
from typing import Any

import pytest

from cosmos_framework.scripts.curator_to_sft_jsonl import (
    DEFAULT_TEMPORAL_INTERVAL,
    MAX_VIDEO_DURATION_S,
    MIN_WINDOW_FRAMES,
    _build_sft_row,
    _relativize_vision_path,
    _resolve_window_caption,
    main,
)

# Pick window-frame bounds well above the loader's 61-frame floor so test
# fixtures can pass / fail the filter for the reason a test actually targets.
_PASSING_WINDOW = (0, MIN_WINDOW_FRAMES + 10)
_SHORT_WINDOW = (0, MIN_WINDOW_FRAMES - 10)


def _make_record(
    *,
    span_uuid: str = "clip-uuid-0",
    clip_location: str = "/out/clips/clip-uuid-0.mp4",
    width: int = 256,
    height: int = 256,
    num_frames: int = 120,
    framerate: float = 24.0,
    windows: list[dict[str, Any]] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a minimal curator metas_jsonl row used by the test fixtures."""
    record: dict[str, Any] = {
        "span_uuid": span_uuid,
        "source_video": "/in/video.mp4",
        "duration_span": [0.0, 5.0],
        "clip_location": clip_location,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "framerate": framerate,
        "windows": windows
        if windows is not None
        else [
            {
                "start_frame": _PASSING_WINDOW[0],
                "end_frame": _PASSING_WINDOW[1],
                "qwen_caption": "A robotic arm grasping a cube.",
            }
        ],
    }
    if extra:
        record.update(extra)
    return record


def _default_kwargs() -> dict[str, Any]:
    return {
        "caption_model": None,
        "enhanced_caption_model": None,
        "min_short_edge": 0,
        "min_window_frames": MIN_WINDOW_FRAMES,
        "max_duration_s": MAX_VIDEO_DURATION_S,
        "temporal_interval": DEFAULT_TEMPORAL_INTERVAL,
    }


@pytest.mark.L0
def test_resolve_window_caption_prefers_enhanced_then_base() -> None:
    window = {
        "qwen_caption": "base text",
        "qwen_lm_enhanced_caption": "enhanced text",
    }
    assert _resolve_window_caption(window, caption_model="qwen", enhanced_caption_model="qwen_lm") == "enhanced text"


@pytest.mark.L0
def test_resolve_window_caption_falls_back_to_base_when_enhanced_missing() -> None:
    window = {"qwen_caption": "base text"}
    assert _resolve_window_caption(window, caption_model="qwen", enhanced_caption_model="qwen_lm") == "base text"


@pytest.mark.L0
def test_resolve_window_caption_alphabetical_fallback_when_unspecified() -> None:
    window = {
        "nemotron_caption": "nemo text",
        "qwen_caption": "qwen text",
    }
    assert _resolve_window_caption(window, caption_model=None, enhanced_caption_model=None) == "nemo text"


@pytest.mark.L0
def test_resolve_window_caption_returns_none_when_empty() -> None:
    window = {"qwen_caption": "   "}
    assert _resolve_window_caption(window, caption_model="qwen", enhanced_caption_model=None) is None


@pytest.mark.L0
def test_build_sft_row_happy_path_emits_loader_schema() -> None:
    record = _make_record()
    row, reason = _build_sft_row(record, **_default_kwargs())
    assert reason is None
    assert row == {
        "uuid": "clip-uuid-0",
        "duration": pytest.approx(120 / 24.0),
        "width": 256,
        "height": 256,
        "nb_frames": 120,
        "framerate": 24.0,
        "vision_path": "/out/clips/clip-uuid-0.mp4",
        "t2w_windows": [
            {
                "start_frame": _PASSING_WINDOW[0],
                "end_frame": _PASSING_WINDOW[1],
                "temporal_interval": DEFAULT_TEMPORAL_INTERVAL,
                "caption": "A robotic arm grasping a cube.",
            }
        ],
    }


@pytest.mark.L0
def test_build_sft_row_drops_clip_longer_than_max_duration() -> None:
    # 120 frames at 1 fps = 120 s > 61 s.
    record = _make_record(num_frames=120, framerate=1.0)
    row, reason = _build_sft_row(record, **_default_kwargs())
    assert row is None
    assert reason == "duration_too_long"


@pytest.mark.L0
def test_build_sft_row_keeps_clip_at_exactly_max_duration() -> None:
    # 61 frames at 1 fps = 61.0 s, must pass because loader uses strict >.
    record = _make_record(
        num_frames=61,
        framerate=1.0,
        windows=[
            {
                "start_frame": 0,
                "end_frame": MIN_WINDOW_FRAMES - 1,
                "qwen_caption": "boundary",
            }
        ],
    )
    row, reason = _build_sft_row(record, **_default_kwargs())
    assert reason is None
    assert row is not None
    assert row["duration"] == pytest.approx(MAX_VIDEO_DURATION_S)


@pytest.mark.L0
def test_build_sft_row_drops_when_short_edge_below_threshold() -> None:
    record = _make_record(width=128, height=512)
    kwargs = _default_kwargs() | {"min_short_edge": 256}
    row, reason = _build_sft_row(record, **kwargs)
    assert row is None
    assert reason == "short_edge_too_small"


@pytest.mark.L0
def test_build_sft_row_drops_when_all_windows_too_short() -> None:
    record = _make_record(
        windows=[
            {
                "start_frame": _SHORT_WINDOW[0],
                "end_frame": _SHORT_WINDOW[1],
                "qwen_caption": "short window",
            }
        ],
    )
    row, reason = _build_sft_row(record, **_default_kwargs())
    assert row is None
    assert reason == "no_valid_window"


@pytest.mark.L0
def test_build_sft_row_drops_when_no_window_has_caption() -> None:
    record = _make_record(
        windows=[
            {
                "start_frame": _PASSING_WINDOW[0],
                "end_frame": _PASSING_WINDOW[1],
                # no caption keys at all
            }
        ],
    )
    row, reason = _build_sft_row(record, **_default_kwargs())
    assert row is None
    assert reason == "no_valid_window"


@pytest.mark.L0
def test_build_sft_row_keeps_only_windows_with_captions() -> None:
    record = _make_record(
        windows=[
            {
                "start_frame": _PASSING_WINDOW[0],
                "end_frame": _PASSING_WINDOW[1],
                "qwen_caption": "kept",
            },
            {
                "start_frame": _PASSING_WINDOW[1] + 1,
                "end_frame": _PASSING_WINDOW[1] + 1 + (_PASSING_WINDOW[1] - _PASSING_WINDOW[0]),
                # no caption -> dropped window, but row should survive.
            },
        ],
    )
    row, reason = _build_sft_row(record, **_default_kwargs())
    assert reason is None
    assert row is not None
    assert len(row["t2w_windows"]) == 1
    assert row["t2w_windows"][0]["caption"] == "kept"


@pytest.mark.L0
def test_build_sft_row_duration_falls_back_to_span_when_clip_metadata_missing() -> None:
    record = _make_record(num_frames=None, framerate=None, extra={"duration_span": [0.0, 4.5]})
    # Without num_frames we can't emit a valid SFT row (loader expects nb_frames),
    # so the row should be dropped for missing_clip_metadata, not duration math.
    row, reason = _build_sft_row(record, **_default_kwargs())
    assert row is None
    assert reason == "missing_clip_metadata"


@pytest.mark.L0
def test_relativize_vision_path_rewrites_filesystem_path_relative_to_jsonl_dir(tmp_path: Path) -> None:
    """Filesystem paths get rewritten relative to the output JSONL's parent dir."""
    output_jsonl = tmp_path / "out" / "cosmos3_sft.jsonl"
    output_jsonl.parent.mkdir(parents=True)
    clip = str(tmp_path / "out" / "clips" / "abc.mp4")
    assert _relativize_vision_path(clip, output_jsonl) == "clips/abc.mp4"


@pytest.mark.L0
def test_relativize_vision_path_emits_parent_segments_when_clip_outside_jsonl_tree(tmp_path: Path) -> None:
    """Clip in a sibling tree is still expressible as a relative path."""
    output_jsonl = tmp_path / "outputs" / "cosmos3_sft.jsonl"
    output_jsonl.parent.mkdir(parents=True)
    clip = str(tmp_path / "curator_out" / "clips" / "abc.mp4")
    assert _relativize_vision_path(clip, output_jsonl) == "../curator_out/clips/abc.mp4"


@pytest.mark.L0
def test_relativize_vision_path_passes_through_uris_unchanged() -> None:
    """s3://, gs://, https:// — anything with a scheme — must not be rewritten."""
    output_jsonl = Path("/anywhere/out.jsonl")
    assert _relativize_vision_path("s3://bucket/clips/abc.mp4", output_jsonl) == "s3://bucket/clips/abc.mp4"
    assert _relativize_vision_path("gs://bucket/clips/abc.mp4", output_jsonl) == "gs://bucket/clips/abc.mp4"
    assert (
        _relativize_vision_path("https://cdn.example.com/clips/abc.mp4", output_jsonl)
        == "https://cdn.example.com/clips/abc.mp4"
    )


@pytest.mark.L0
def test_main_emits_relative_vision_paths(tmp_path: Path) -> None:
    """End-to-end: written JSONL rows must carry relative paths the loader can resolve.

    The cosmos team requested relative paths so the loader's
    relative-to-JSONL branch (sft_dataset.py:548-550) fires and datasets stay
    portable across mount points.
    """
    curator_output = tmp_path / "curator_out"
    metas_dir = curator_output / "metas_jsonl" / "v0"
    metas_dir.mkdir(parents=True)
    record = _make_record(clip_location=str(curator_output / "clips" / "abc.mp4"))
    (metas_dir / "shard_0.jsonl").write_text(json.dumps(record) + "\n")

    output = curator_output / "cosmos3_sft.jsonl"
    main(
        curator_output=curator_output,
        output=output,
        caption_model="qwen",
        enhanced_caption_model=None,
        min_short_edge=0,
        min_window_frames=MIN_WINDOW_FRAMES,
        max_duration_s=MAX_VIDEO_DURATION_S,
        temporal_interval=DEFAULT_TEMPORAL_INTERVAL,
    )

    rows = [json.loads(line) for line in output.read_text().splitlines() if line]
    assert len(rows) == 1
    # JSONL lives at curator_out/cosmos3_sft.jsonl; clip lives at curator_out/clips/abc.mp4.
    # Relative form must be just "clips/abc.mp4" — not absolute, not starting with "./".
    assert rows[0]["vision_path"] == "clips/abc.mp4"


@pytest.mark.L0
def test_main_end_to_end_writes_jsonl_and_summary(tmp_path: Path) -> None:
    """End-to-end: drop a curator-style metas_jsonl fixture, run main(), assert outputs."""
    curator_output = tmp_path / "curator_out"
    metas_dir = curator_output / "metas_jsonl" / "v0"
    metas_dir.mkdir(parents=True)

    kept = _make_record(span_uuid="kept-1")
    too_long = _make_record(span_uuid="too-long", num_frames=120, framerate=1.0)
    no_caption = _make_record(
        span_uuid="no-caption",
        windows=[
            {
                "start_frame": _PASSING_WINDOW[0],
                "end_frame": _PASSING_WINDOW[1],
            }
        ],
    )
    shard_path = metas_dir / "video-uuid_0.jsonl"
    shard_path.write_text(
        "\n".join(json.dumps(r) for r in [kept, too_long, no_caption]) + "\n",
    )

    output = curator_output / "cosmos3_sft.jsonl"
    main(
        curator_output=curator_output,
        output=output,
        caption_model="qwen",
        enhanced_caption_model=None,
        min_short_edge=0,
        min_window_frames=MIN_WINDOW_FRAMES,
        max_duration_s=MAX_VIDEO_DURATION_S,
        temporal_interval=DEFAULT_TEMPORAL_INTERVAL,
    )

    assert output.exists()
    rows = [json.loads(line) for line in output.read_text().splitlines() if line]
    assert len(rows) == 1
    assert rows[0]["uuid"] == "kept-1"
    assert rows[0]["t2w_windows"][0]["caption"] == "A robotic arm grasping a cube."

    summary_path = output.with_suffix(output.suffix + ".summary.json")
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary["records_seen"] == 3
    assert summary["records_kept"] == 1
    assert summary["drops_by_reason"]["duration_too_long"] == 1
    assert summary["drops_by_reason"]["no_valid_window"] == 1


@pytest.mark.L0
def test_main_partial_split_keeps_exactly_the_passing_rows(tmp_path: Path) -> None:
    """A duration filter must split the batch — not drop everything, not pass everything.

    Guards against accidental drop-all / pass-all regressions in the filter logic.
    Mirrors the bridge-sample dry-run shape: 5 rows, 3 short enough to pass at
    --max-duration-s 5.0 and 2 too long; verify the exact uuids in each bucket.
    """
    curator_output = tmp_path / "curator_out"
    metas_dir = curator_output / "metas_jsonl" / "v0"
    metas_dir.mkdir(parents=True)

    # Build a mix where exactly 3 of 5 pass --max-duration-s 5.0.
    # All 5 must have windows long enough to clear the min_window_frames filter
    # so the only reason for drops is duration.
    long_window = [
        {
            "start_frame": 0,
            "end_frame": MIN_WINDOW_FRAMES + 5,
            "qwen_caption": "kept",
        }
    ]
    rows = [
        _make_record(span_uuid="short-1", num_frames=80, framerate=24.0, windows=long_window),  # 3.3s — keep
        _make_record(span_uuid="long-1", num_frames=240, framerate=24.0, windows=long_window),  # 10.0s — drop
        _make_record(span_uuid="short-2", num_frames=96, framerate=24.0, windows=long_window),  # 4.0s — keep
        _make_record(span_uuid="long-2", num_frames=300, framerate=24.0, windows=long_window),  # 12.5s — drop
        _make_record(span_uuid="short-3", num_frames=72, framerate=24.0, windows=long_window),  # 3.0s — keep
    ]
    (metas_dir / "shard_0.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    output = curator_output / "cosmos3_sft.jsonl"
    main(
        curator_output=curator_output,
        output=output,
        caption_model="qwen",
        enhanced_caption_model=None,
        min_short_edge=0,
        min_window_frames=MIN_WINDOW_FRAMES,
        max_duration_s=5.0,
        temporal_interval=DEFAULT_TEMPORAL_INTERVAL,
    )

    written = [json.loads(line) for line in output.read_text().splitlines() if line]
    kept_uuids = {r["uuid"] for r in written}
    assert kept_uuids == {"short-1", "short-2", "short-3"}, (
        f"Expected exactly the short uuids to pass; got {kept_uuids}"
    )

    summary = json.loads(output.with_suffix(output.suffix + ".summary.json").read_text())
    assert summary["records_seen"] == 5
    assert summary["records_kept"] == 3
    assert summary["records_dropped"] == 2
    assert summary["drops_by_reason"] == {"duration_too_long": 2}


@pytest.mark.L0
def test_main_partial_split_on_min_window_frames(tmp_path: Path) -> None:
    """min_window_frames must split the batch like duration does — guard against drop-all bugs.

    Build 4 rows where 2 have windows long enough to clear --min-window-frames 80 and
    2 don't. The kept set must match exactly by uuid.
    """
    curator_output = tmp_path / "curator_out"
    metas_dir = curator_output / "metas_jsonl" / "v0"
    metas_dir.mkdir(parents=True)
    threshold = 80

    def _row_with_window_len(uuid: str, length: int) -> dict[str, Any]:
        return _make_record(
            span_uuid=uuid,
            windows=[
                {
                    "start_frame": 0,
                    "end_frame": length - 1,
                    "qwen_caption": "ok",
                }
            ],
        )

    rows = [
        _row_with_window_len("long-1", threshold + 5),  # 85 frames — keep
        _row_with_window_len("short-1", threshold - 5),  # 75 frames — drop (no_valid_window)
        _row_with_window_len("long-2", threshold + 20),  # 100 frames — keep
        _row_with_window_len("short-2", threshold - 1),  # 79 frames — drop
    ]
    (metas_dir / "shard_0.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    output = curator_output / "cosmos3_sft.jsonl"
    main(
        curator_output=curator_output,
        output=output,
        caption_model="qwen",
        enhanced_caption_model=None,
        min_short_edge=0,
        min_window_frames=threshold,
        max_duration_s=MAX_VIDEO_DURATION_S,
        temporal_interval=DEFAULT_TEMPORAL_INTERVAL,
    )

    written = [json.loads(line) for line in output.read_text().splitlines() if line]
    assert {r["uuid"] for r in written} == {"long-1", "long-2"}
    summary = json.loads(output.with_suffix(output.suffix + ".summary.json").read_text())
    assert summary["records_kept"] == 2
    assert summary["drops_by_reason"] == {"no_valid_window": 2}


@pytest.mark.L0
def test_main_glob_reads_all_shards_in_metas_jsonl_dir(tmp_path: Path) -> None:
    """Curator emits one .jsonl per (video_uuid, chunk_index); the converter must read them all."""
    curator_output = tmp_path / "curator_out"
    metas_dir = curator_output / "metas_jsonl" / "v0"
    metas_dir.mkdir(parents=True)

    # Two shards, two rows each, distinct uuids so we can confirm every row arrives.
    shard_a = [_make_record(span_uuid="a-0"), _make_record(span_uuid="a-1")]
    shard_b = [_make_record(span_uuid="b-0"), _make_record(span_uuid="b-1")]
    (metas_dir / "video-a_0.jsonl").write_text("\n".join(json.dumps(r) for r in shard_a) + "\n")
    (metas_dir / "video-b_0.jsonl").write_text("\n".join(json.dumps(r) for r in shard_b) + "\n")

    output = curator_output / "cosmos3_sft.jsonl"
    main(
        curator_output=curator_output,
        output=output,
        caption_model="qwen",
        enhanced_caption_model=None,
        min_short_edge=0,
        min_window_frames=MIN_WINDOW_FRAMES,
        max_duration_s=MAX_VIDEO_DURATION_S,
        temporal_interval=DEFAULT_TEMPORAL_INTERVAL,
    )

    written = [json.loads(line) for line in output.read_text().splitlines() if line]
    assert {r["uuid"] for r in written} == {"a-0", "a-1", "b-0", "b-1"}
    summary = json.loads(output.with_suffix(output.suffix + ".summary.json").read_text())
    assert summary["shards_read"] == 2
    assert summary["records_seen"] == 4
    assert summary["records_kept"] == 4


@pytest.mark.L0
def test_main_caption_resolution_end_to_end_prefers_configured_enhanced_model(tmp_path: Path) -> None:
    """End-to-end: confirm --enhanced-caption-model picks the right field even when other captions are present."""
    curator_output = tmp_path / "curator_out"
    metas_dir = curator_output / "metas_jsonl" / "v0"
    metas_dir.mkdir(parents=True)

    record = _make_record(
        windows=[
            {
                "start_frame": _PASSING_WINDOW[0],
                "end_frame": _PASSING_WINDOW[1],
                "qwen_caption": "wrong-1: base qwen caption",
                "nemotron_caption": "wrong-2: alphabetically-first base caption",
                "gpt_oss_20b_enhanced_caption": "wrong-3: non-configured enhanced caption",
                "qwen_lm_enhanced_caption": "correct: configured enhanced caption",
            }
        ],
    )
    (metas_dir / "shard_0.jsonl").write_text(json.dumps(record) + "\n")

    output = curator_output / "cosmos3_sft.jsonl"
    main(
        curator_output=curator_output,
        output=output,
        caption_model="qwen",
        enhanced_caption_model="qwen_lm",
        min_short_edge=0,
        min_window_frames=MIN_WINDOW_FRAMES,
        max_duration_s=MAX_VIDEO_DURATION_S,
        temporal_interval=DEFAULT_TEMPORAL_INTERVAL,
    )

    rows = [json.loads(line) for line in output.read_text().splitlines() if line]
    assert len(rows) == 1
    assert rows[0]["t2w_windows"][0]["caption"] == "correct: configured enhanced caption"


@pytest.mark.L0
def test_main_accepts_metas_jsonl_dir_directly(tmp_path: Path) -> None:
    """User may pass the metas_jsonl/v0 dir itself instead of the curator root."""
    metas_dir = tmp_path / "metas_jsonl_v0"
    metas_dir.mkdir()
    shard_path = metas_dir / "video-uuid_0.jsonl"
    shard_path.write_text(json.dumps(_make_record()) + "\n")

    output = tmp_path / "cosmos3_sft.jsonl"
    main(
        curator_output=metas_dir,
        output=output,
        caption_model=None,
        enhanced_caption_model=None,
        min_short_edge=0,
        min_window_frames=MIN_WINDOW_FRAMES,
        max_duration_s=MAX_VIDEO_DURATION_S,
        temporal_interval=DEFAULT_TEMPORAL_INTERVAL,
    )
    assert output.exists()
    assert len(output.read_text().splitlines()) == 1


@pytest.mark.L0
def test_main_exits_when_no_records_kept(tmp_path: Path) -> None:
    curator_output = tmp_path / "curator_out"
    metas_dir = curator_output / "metas_jsonl" / "v0"
    metas_dir.mkdir(parents=True)
    # Only emit a row that will fail filters.
    bad = _make_record(span_uuid="bad", num_frames=120, framerate=1.0)
    (metas_dir / "video-uuid_0.jsonl").write_text(json.dumps(bad) + "\n")

    output = curator_output / "cosmos3_sft.jsonl"
    with pytest.raises(SystemExit) as exc_info:
        main(
            curator_output=curator_output,
            output=output,
            caption_model=None,
            enhanced_caption_model=None,
            min_short_edge=0,
            min_window_frames=MIN_WINDOW_FRAMES,
            max_duration_s=MAX_VIDEO_DURATION_S,
            temporal_interval=DEFAULT_TEMPORAL_INTERVAL,
        )
    assert exc_info.value.code == 1


@pytest.mark.L0
def test_main_exits_when_no_metas_jsonl_found(tmp_path: Path) -> None:
    output = tmp_path / "cosmos3_sft.jsonl"
    with pytest.raises(SystemExit) as exc_info:
        main(
            curator_output=tmp_path,
            output=output,
            caption_model=None,
            enhanced_caption_model=None,
            min_short_edge=0,
            min_window_frames=MIN_WINDOW_FRAMES,
            max_duration_s=MAX_VIDEO_DURATION_S,
            temporal_interval=DEFAULT_TEMPORAL_INTERVAL,
        )
    assert exc_info.value.code == 1
