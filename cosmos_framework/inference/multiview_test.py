# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Tests for the multiview caption-chunk frontend and the per-camera media loaders.

The parser itself is released from the upstream codebase and tested there against both on-disk schemas.
What is pinned here is the part this repo owns: that the frontend reads the schema the multiview eval
bundles actually carry, and that the chunk boundaries it hands to generation are the ones the
parser produced.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.data.generator.multiview.caption_chunks import MultiviewCaptionChunk
from cosmos_framework.inference import multiview
from cosmos_framework.inference.args import ConditionVisionMode
from cosmos_framework.inference.multiview import (
    generate_multiview_caption_chunks,
    load_multiview_caption_chunks,
    load_multiview_caption_chunks_per_view,
)
from cosmos_framework.model.generator.mot.replica_partition import WorkUnit


def _write(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "captions.json"
    path.write_text(json.dumps(payload))
    return path


def _structured(*spans: tuple[int, int]) -> dict:
    return {
        "caption_structured": {
            f"chunk_{start}_{end}": {"caption": f"caption {start}", "start_frame": start, "end_frame": end}
            for start, end in spans
        }
    }


def test_reads_the_schema_the_mads_bundles_carry(tmp_path: Path) -> None:
    """The regression this frontend existed with: every bundle caption file is an object.

    The previous local parser accepted only the legacy list and raised
    "Expected multiview caption JSON to contain a list, got dict." on all 1100 caption files in
    the current eval bundle, so no caption-driven multiview run against those bundles was possible.
    """
    chunks = load_multiview_caption_chunks(_write(tmp_path, _structured((0, 300))))

    assert len(chunks) == 1
    assert (chunks[0].frame_start, chunks[0].frame_end) == (0, 300)
    assert chunks[0].prompt == "caption 0"


def test_structured_end_frame_is_already_exclusive(tmp_path: Path) -> None:
    """``frame_end`` is used directly as a slice bound, so the convention must survive the read.

    caption_structured's end_frame is exclusive and the legacy schema's frame_range is inclusive.
    Confusing the two shifts every control slice by a frame, with no error.
    """
    structured = load_multiview_caption_chunks(_write(tmp_path, _structured((0, 300))))
    legacy = load_multiview_caption_chunks(
        _write(tmp_path, [{"frame_range": [0, 299], "captions": {"long": "caption 0"}}])
    )

    assert (structured[0].frame_start, structured[0].frame_end) == (0, 300)
    assert (legacy[0].frame_start, legacy[0].frame_end) == (0, 300)


def test_chunks_are_ordered_by_frame_span(tmp_path: Path) -> None:
    """Chunk index drives the per-chunk seed and the conditioning carry-over, so order matters."""
    chunks = load_multiview_caption_chunks(_write(tmp_path, _structured((300, 600), (0, 300), (600, 900))))

    assert [c.index for c in chunks] == [0, 1, 2]
    assert [c.frame_start for c in chunks] == [0, 300, 600]


def test_the_legacy_list_schema_still_reads(tmp_path: Path) -> None:
    """Older bundles are still pointed at; dropping them would be a silent capability loss."""
    chunks = load_multiview_caption_chunks(
        _write(tmp_path, [{"frame_range": [0, 9], "captions": {"short": "s", "long": "l"}}])
    )

    assert chunks[0].prompt == "l"


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (_structured(), "non-empty caption_structured"),
        ({"caption_structured": {"c": {"caption": None, "start_frame": 0, "end_frame": 1}}}, "caption string"),
        ({"caption_structured": {"c": {"caption": "x", "start_frame": 9, "end_frame": 9}}}, "start_frame < end_frame"),
        ([], "empty"),
        ("not a caption file", "caption_structured object or a list of chunks"),
    ],
)
def test_malformed_caption_files_raise(tmp_path: Path, payload: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_multiview_caption_chunks(_write(tmp_path, payload))


@pytest.mark.parametrize("caption", ["", "  "])
def test_empty_caption_preserves_the_camera_interval(tmp_path: Path, caption: str) -> None:
    payload = {"caption_structured": {"chunk_0_300": {"caption": caption, "start_frame": 0, "end_frame": 300}}}
    (chunk,) = load_multiview_caption_chunks(_write(tmp_path, payload))
    assert chunk.prompt == ""
    assert (chunk.frame_start, chunk.frame_end) == (0, 300)


@pytest.mark.L0
@pytest.mark.CPU
def test_single_caption_chunks_leave_view_prompts_unset(tmp_path: Path) -> None:
    """Single-caption parsing leaves per-view prompts to the per-view loader."""
    (chunk,) = load_multiview_caption_chunks(_write(tmp_path, _structured((0, 300))))

    assert isinstance(chunk, MultiviewCaptionChunk)
    assert chunk.view_prompts is None


# Set COSMOS_MADS_EVAL_BUNDLE to a bundle's `samples/` directory to run the real-data check
# below. Deliberately not defaulted to a path: the bundle lives at a different location on every
# cluster, and hardcoding one ties this repo's tests to one site.
_EVAL_BUNDLE_ENV = "COSMOS_MADS_EVAL_BUNDLE"


def test_parses_a_real_eval_bundle_when_one_is_configured() -> None:
    """The check a fixture cannot make: it reads files this repo did not write.

    Skipped unless COSMOS_MADS_EVAL_BUNDLE points at a bundle's `samples/` directory.
    """
    bundle_root = os.environ.get(_EVAL_BUNDLE_ENV)
    if not bundle_root or not Path(bundle_root).is_dir():
        pytest.skip(f"set {_EVAL_BUNDLE_ENV} to a MADS eval bundle's samples/ directory")

    caption_files = sorted(Path(bundle_root).glob("*/captions/raw/c*.json"))[:11]
    if not caption_files:
        pytest.skip("MADS eval bundle carries no per-view caption files")

    for caption_file in caption_files:
        chunks = load_multiview_caption_chunks(caption_file)
        assert chunks, caption_file
        assert all(c.frame_end > c.frame_start for c in chunks), caption_file
        assert all(c.prompt.strip() for c in chunks), caption_file


class _StubMultiview:
    caption_path = "/nonexistent/captions.json"
    use_first_chunk_only = True
    first_caption_paragraph_only = False


class _StubSampleArgs:
    """Only what the guard reads, so the check is reached before any model work."""

    multiview = _StubMultiview()

    def __init__(self, control_guidance: float) -> None:
        self.control_guidance = control_guidance


@pytest.mark.parametrize("control_guidance", [1.5, 3.0])
def test_control_guidance_is_refused_on_the_caption_chunk_path(control_guidance: float) -> None:
    """Silently ignoring the flag, or silently mis-applying it, are both worse than refusing.

    This loop calls generate_samples_from_batch directly, so it never reaches the
    velocity_postprocess_builder that generate_batch installs -- control_guidance was accepted,
    validated, then had no effect. Wiring the existing builder in instead would be silently wrong:
    _build_no_control_inference_state drops num_views_per_vision_item, so the extra forward pass
    would read an 11-camera item as one view.
    """
    with pytest.raises(NotImplementedError, match="not view-aware"):
        generate_multiview_caption_chunks(_StubSampleArgs(control_guidance), model=None)


def test_the_parity_value_is_not_refused() -> None:
    """1.0 is what both runtimes resolve for a multiview WSM sample, and is a no-op by construction.

    The guard must let it through untouched -- it fails later on the stub's missing caption file,
    which is past the point under test.
    """
    with pytest.raises(Exception) as excinfo:
        generate_multiview_caption_chunks(_StubSampleArgs(1.0), model=None)

    assert not isinstance(excinfo.value, NotImplementedError)


def _bundle_views(tmp_path: Path, captions: dict[str, dict]) -> list:
    """Camera entries pointing at per-camera caption files, as the arg schema produces them."""
    from types import SimpleNamespace

    views = []
    for index, (camera, payload) in enumerate(captions.items()):
        caption = tmp_path / f"c{index}.json"
        caption.write_text(json.dumps(payload))
        views.append(SimpleNamespace(camera_key=camera, caption_path=str(caption)))
    return views


def _span_payload(start: int, end: int, caption: str) -> dict:
    return {
        "caption_structured": {f"chunk_{start}_{end}": {"caption": caption, "start_frame": start, "end_frame": end}}
    }


def test_every_camera_contributes_its_own_caption(tmp_path: Path) -> None:
    """One prompt per camera, in view order -- not one camera's caption repeated for the rig."""
    views = _bundle_views(
        tmp_path,
        {
            "camera_front_wide_120fov": _span_payload(0, 300, "the road ahead"),
            "camera_cross_right_120fov": _span_payload(0, 300, "a parked van"),
        },
    )

    (chunk,) = load_multiview_caption_chunks_per_view(views)

    assert chunk.view_prompts is not None
    assert len(chunk.view_prompts) == 2
    assert "the road ahead" in chunk.view_prompts[0]
    assert "a parked van" in chunk.view_prompts[1]
    assert "the road ahead" not in chunk.view_prompts[1]


def test_each_view_prompt_carries_the_rig_header_and_its_own_camera(tmp_path: Path) -> None:
    """The header is what the checkpoint was trained on; it names the rig, then this camera."""
    views = _bundle_views(
        tmp_path,
        {
            "camera_front_wide_120fov": _span_payload(0, 300, "a"),
            "camera_cross_right_120fov": _span_payload(0, 300, "b"),
        },
    )

    (chunk,) = load_multiview_caption_chunks_per_view(views)

    assert chunk.view_prompts is not None
    for prompt in chunk.view_prompts:
        assert "2 vehicle-mounted cameras" in prompt
    assert "front wide-angle camera mounted on the vehicle" in chunk.view_prompts[0]
    assert "right-side wide-angle camera mounted on the vehicle" in chunk.view_prompts[1]


def test_the_merged_prompt_is_still_built_alongside(tmp_path: Path) -> None:
    """Both forms travel on the chunk; generation picks one from the checkpoint's recorded layout."""
    views = _bundle_views(
        tmp_path,
        {
            "camera_front_wide_120fov": _span_payload(0, 300, "the road ahead"),
            "camera_cross_right_120fov": _span_payload(0, 300, "a parked van"),
        },
    )

    (chunk,) = load_multiview_caption_chunks_per_view(views)

    assert "the road ahead" in chunk.prompt
    assert "a parked van" in chunk.prompt


def test_first_caption_paragraph_only_trims_each_camera_before_headers(tmp_path: Path) -> None:
    """Training recipes that trim captions (Transfer-Auto stage 3, LiDAR) must see the same text at inference."""
    views = _bundle_views(
        tmp_path,
        {
            "camera_front_wide_120fov": _span_payload(0, 300, "the road ahead\n\nlater the light changes"),
            "camera_cross_right_120fov": _span_payload(0, 300, "a parked van\n\nthen it drives off"),
        },
    )

    (full,) = load_multiview_caption_chunks_per_view(views)
    (trimmed,) = load_multiview_caption_chunks_per_view(views, first_caption_paragraph_only=True)

    assert "later the light changes" in full.prompt
    assert "the road ahead" in trimmed.prompt
    assert "a parked van" in trimmed.prompt
    assert "later the light changes" not in trimmed.prompt
    assert "then it drives off" not in trimmed.prompt
    assert trimmed.view_prompts is not None
    assert "2 vehicle-mounted cameras" in trimmed.view_prompts[0]
    assert "later the light changes" not in trimmed.view_prompts[0]


def test_cameras_disagreeing_on_their_chunk_count_raise(tmp_path: Path) -> None:
    """Rollout slices one shared frame count, so a per-camera mismatch cannot be reconciled."""
    views = _bundle_views(
        tmp_path,
        {
            "camera_front_wide_120fov": {
                "caption_structured": {
                    "chunk_0_300": {"caption": "a", "start_frame": 0, "end_frame": 300},
                    "chunk_300_600": {"caption": "b", "start_frame": 300, "end_frame": 600},
                }
            },
            "camera_cross_right_120fov": _span_payload(0, 300, "c"),
        },
    )

    with pytest.raises(ValueError, match="disagree on their caption chunk count"):
        load_multiview_caption_chunks_per_view(views)


def test_cameras_disagreeing_on_a_frame_span_raise(tmp_path: Path) -> None:
    """A caption written against other frames describes footage that was never generated."""
    views = _bundle_views(
        tmp_path,
        {
            "camera_front_wide_120fov": _span_payload(0, 300, "a"),
            "camera_cross_right_120fov": _span_payload(0, 200, "b"),
        },
    )

    with pytest.raises(ValueError, match="different frame span per camera"):
        load_multiview_caption_chunks_per_view(views)


def test_an_uncaptioned_camera_raises_rather_than_being_skipped(tmp_path: Path) -> None:
    from types import SimpleNamespace

    views = _bundle_views(tmp_path, {"camera_front_wide_120fov": _span_payload(0, 300, "a")})
    views.append(SimpleNamespace(camera_key="camera_cross_right_120fov", caption_path=None))

    with pytest.raises(ValueError, match="needs a caption for camera"):
        load_multiview_caption_chunks_per_view(views)


def test_a_camera_with_no_known_prefix_raises(tmp_path: Path) -> None:
    """The merged form labels each caption by camera; an unknown camera has no label to use."""
    views = _bundle_views(tmp_path, {"camera_made_up": _span_payload(0, 300, "a")})

    with pytest.raises(ValueError, match="No caption prefix is defined"):
        load_multiview_caption_chunks_per_view(views)


def test_the_system_prompt_follows_the_caption_layout() -> None:
    """Not cosmetic: an explicit batch["system_prompt"] overrides use_system_prompt inside
    tokenize_caption, so this string is always part of what the model reads. Getting it wrong
    changes the text token ids deterministically, for every sample, with no other symptom.
    """
    from cosmos_framework.data.generator.augmentors.text_tokenizer import (
        _SYSTEM_PROMPT_AV_MULTIVIEW_TRANSFER,
        _SYSTEM_PROMPT_TRANSFER,
    )

    assert _SYSTEM_PROMPT_AV_MULTIVIEW_TRANSFER != _SYSTEM_PROMPT_TRANSFER


def _decode_to_pixels(latent_value: float) -> float:
    """The normalisation the caption-chunk path applies to the decoder's output."""
    import torch

    decoded = torch.full((1, 3, 2, 4, 4), latent_value)
    return float((((1.0 + decoded) / 2).clamp(0, 1))[0, 0, 0, 0, 0])


def test_decoder_output_is_mapped_from_minus_one_one_to_zero_one() -> None:
    """The Wan decoder emits [-1, 1]; every consumer downstream of here works in [0, 1].

    Handing a [-1, 1] tensor to a [0, 1] consumer crushes everything the model put below 0.5 to
    black. Measured on an 11-camera run before this was fixed: 70.7% of all pixels were exactly
    zero, and the saved video cf related to the upstream run's output u by cf = clamp(2*u - 255, 0, 255) to within
    0.61/255.
    """
    assert _decode_to_pixels(-1.0) == pytest.approx(0.0)
    assert _decode_to_pixels(0.0) == pytest.approx(0.5)
    assert _decode_to_pixels(1.0) == pytest.approx(1.0)


def test_the_normalisation_is_the_one_the_rest_of_the_repo_uses() -> None:
    """inference.py applies ((1 + v) / 2).clamp(0, 1) on its other decode paths.

    Pinned as source, because the multiview path diverging from it silently is exactly the bug
    this test exists for: nothing about the output says the range is wrong, it just looks dark.
    """
    from pathlib import Path

    inference_py = (Path(__file__).parent / "inference.py").read_text()
    assert inference_py.count("((1.0 + vision) / 2).clamp(0, 1)") >= 2

    multiview_py = (Path(__file__).parent / "multiview.py").read_text()
    assert ".clamp(0, 1)" in multiview_py
    assert ".clamp(-1, 1)" not in multiview_py


def _emphasis_batch(prompt_entry: list, *, hint: str = "wsm") -> tuple[list, str]:
    """Run get_multiview_sample_data's emphasis block over a prepared caption entry."""

    hint_names = hint
    emphasis = (
        f" Follow the {hint_names} control videos precisely for every camera view: shape, contour,"
        f" position, and motion must align with the {hint_names} signal at every frame."
    )
    return [
        [caption.rstrip() + emphasis for caption in prompt] if isinstance(prompt, list) else prompt.rstrip() + emphasis
        for prompt in prompt_entry
    ], emphasis


def test_control_emphasis_reaches_every_camera_caption() -> None:
    """Per-view captions nest one level; the merged path's flat comprehension raised on them.

    `AttributeError: 'list' object has no attribute 'rstrip'`, after the per-view builder had
    already run -- so the failure landed in the middle of a real generation, not at startup. Each
    camera's caption has to carry the emphasis into the tokens that view attends to.
    """
    out, emphasis = _emphasis_batch([["the road ahead.", "a parked van."]])

    assert out == [["the road ahead." + emphasis, "a parked van." + emphasis]]


def test_control_emphasis_still_works_on_a_merged_prompt() -> None:
    out, emphasis = _emphasis_batch(["a merged prompt."])

    assert out == ["a merged prompt." + emphasis]


def test_control_emphasis_matches_the_imaginaire4_shape() -> None:
    """Both forms in one batch entry is what num_outputs > 1 produces; neither may be skipped."""
    out, emphasis = _emphasis_batch([["a.", "b."], ["c.", "d."]])

    assert out == [["a." + emphasis, "b." + emphasis], ["c." + emphasis, "d." + emphasis]]


# --- camera media loaders: one load per camera across the inference replica ---------------------

_NUM_VIEWS = 7
_LOADER_H, _LOADER_W = 4, 6


class _ReplicaRankOne:
    """Rank 1 of a 4-rank replica, as ``ReplicaPartitioner.run`` behaves on it.

    It runs only the units it owns (round-robin: indices 1 and 5 of 7) and "receives" a sentinel
    clip for every other unit, delivered in unit order through ``on_collected``.
    """

    group_size = 4
    group_rank = 1

    def __init__(self) -> None:
        self.labels: list[str] = []
        self.received = {
            index: torch.full((3, 1, _LOADER_H, _LOADER_W), 200 + index, dtype=torch.uint8)
            for index in range(_NUM_VIEWS)
            if index % self.group_size != self.group_rank
        }

    def run(
        self,
        units: Sequence[WorkUnit],
        *,
        collect_on: int | None = None,
        on_collected: Callable[[int, torch.Tensor], None],
        label: str = "",
    ) -> list[None]:
        assert collect_on is None  # every rank needs every camera
        self.labels.append(label)
        for index, unit in enumerate(units):
            on_collected(index, unit.run() if index % self.group_size == self.group_rank else self.received[index])
        return [None] * len(units)


def _loader_sample_args(*, control_paths: list[str | None] | None = None) -> SimpleNamespace:
    views = [
        SimpleNamespace(
            camera_key=f"camera_{index}",
            vision_path=f"/clips/rgb/c{index}.mp4",
            control_path=f"/clips/controls/c{index}.mp4" if control_paths is None else control_paths[index],
        )
        for index in range(_NUM_VIEWS)
    ]
    return SimpleNamespace(
        multiview=SimpleNamespace(views=views, condition_vision_mode=ConditionVisionMode.IMAGE),
        condition_frame_indexes_vision=[0],
        condition_video_keep=None,
        num_frames=9,
    )


@pytest.fixture
def recorded_video_loads(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, int]]:
    """Patch the clip decoder to a recorder whose frames encode the view index as their pixel value."""
    loads: list[tuple[str, int]] = []

    def load_conditioning_video(
        path: Path, *, target_h: int, target_w: int, max_frames: int, keep: str
    ) -> torch.Tensor:
        loads.append((path.as_posix(), max_frames))
        view_index = int(path.stem.removeprefix("c"))
        return torch.full((3, max_frames, target_h, target_w), view_index * 10 / 127.5 - 1.0)  # [3,T,H,W] in [-1,1]

    monkeypatch.setattr(multiview, "load_conditioning_video", load_conditioning_video)
    return loads


def _load(kind: str, sample_args: SimpleNamespace, **partitioner: object) -> list:
    if kind == "condition":
        return multiview._get_multiview_condition_pixel_frames(
            sample_args, None, height=_LOADER_H, width=_LOADER_W, **partitioner
        )
    return multiview._get_multiview_control_pixel_frames(sample_args, height=_LOADER_H, width=_LOADER_W, **partitioner)


@pytest.mark.parametrize(
    ("kind", "folder", "max_frames", "label"),
    [("condition", "rgb", 1, "camera condition media load"), ("control", "controls", 9, "camera control media load")],
)
def test_a_replica_rank_loads_only_its_own_cameras_and_returns_every_view_in_order(
    recorded_video_loads: list[tuple[str, int]], kind: str, folder: str, max_frames: int, label: str
) -> None:
    """At CP every rank used to decode every camera; now each decodes its share and receives the rest."""
    partitioner = _ReplicaRankOne()

    frames = _load(kind, _loader_sample_args(), partitioner=partitioner)

    assert recorded_video_loads == [(f"/clips/{folder}/c{index}.mp4", max_frames) for index in (1, 5)]
    assert partitioner.labels == [label]
    assert len(frames) == _NUM_VIEWS
    for index, view_frames in enumerate(frames):
        if index in partitioner.received:
            assert view_frames is partitioner.received[index]
        else:
            assert view_frames.dtype == torch.uint8 and view_frames.device.type == "cpu"
            assert torch.equal(
                view_frames, torch.full((3, max_frames, _LOADER_H, _LOADER_W), index * 10, dtype=torch.uint8)
            )


@pytest.mark.parametrize(("kind", "folder", "max_frames"), [("condition", "rgb", 1), ("control", "controls", 9)])
def test_without_a_partitioner_every_view_loads_here_in_order(
    recorded_video_loads: list[tuple[str, int]], kind: str, folder: str, max_frames: int
) -> None:
    """The single-rank path is the loop it replaced: every camera, on this rank, in view order."""
    frames = _load(kind, _loader_sample_args())  # the default, which every caller used before partitioning

    assert recorded_video_loads == [(f"/clips/{folder}/c{index}.mp4", max_frames) for index in range(_NUM_VIEWS)]
    assert len(frames) == _NUM_VIEWS
    for index, view_frames in enumerate(frames):
        assert view_frames.device.type == "cpu"
        assert torch.equal(
            view_frames, torch.full((3, max_frames, _LOADER_H, _LOADER_W), index * 10, dtype=torch.uint8)
        )


def test_a_missing_control_path_raises_before_the_replica_load_starts(
    recorded_video_loads: list[tuple[str, int]],
) -> None:
    """Every replica rank holds the same input, so each raises this itself before decoding anything
    or entering the load's collectives, instead of only the camera's owner raising it mid-load."""
    partitioner = _ReplicaRankOne()
    control_paths: list[str | None] = [f"/clips/controls/c{index}.mp4" for index in range(_NUM_VIEWS)]
    control_paths[6] = None

    with pytest.raises(ValueError, match="requires control_path for every view"):
        _load("control", _loader_sample_args(control_paths=control_paths), partitioner=partitioner)

    assert partitioner.labels == []
    assert recorded_video_loads == []
