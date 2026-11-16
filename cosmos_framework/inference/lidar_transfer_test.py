# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Tests for the LiDAR inference seam: sweep arithmetic, control shapes, args validation."""

import io
import json
import tarfile
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pydantic
import pytest
import torch

from cosmos_framework.data.generator.sequence_packing import pack_input_sequence
from cosmos_framework.inference.args import LidarTransferArgs, ModelMode, OmniSampleOverrides, TransferHintKey
from cosmos_framework.inference.lidar_transfer import (
    JOINT_CAM_LIDAR_DATASET_NAME,
    _coerce_lidar_clip,
    build_joint_cam_lidar_batch,
    joint_lidar_num_sweeps,
    load_joint_lidar_condition,
)
from cosmos_framework.model.generator.algorithm.loss.flow_matching import compute_flow_matching_loss
from cosmos_framework.model.generator.diffusion.rectified_flow import RectifiedFlow
from cosmos_framework.model.generator.tokenizers.lidar.range_projection import LidarRangeProjectionConfig
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _touch(tmp_path: Path, name: str) -> str:
    path = tmp_path / name
    path.touch()
    return str(path)


def _views(tmp_path: Path, count: int = 2, *, control: bool = True, vision: bool = False) -> list[dict]:
    views = []
    for index in range(count):
        view: dict = {"camera_key": f"cam{index}"}
        if control:
            view["control_path"] = _touch(tmp_path, f"ctl{index}.mp4")
        if vision:
            view["vision_path"] = _touch(tmp_path, f"vis{index}.mp4")
        views.append(view)
    return views


def test_joint_sweep_count_follows_the_camera_clock() -> None:
    """Training is 200 camera frames at 30 fps against 67 sweeps; a wrong ratio is invisible to shape checks."""
    assert joint_lidar_num_sweeps(200, camera_fps=30, lidar_fps=10) == 67
    assert joint_lidar_num_sweeps(93, camera_fps=30, lidar_fps=10) == 31
    # Never degenerate to zero sweeps for a short clip.
    assert joint_lidar_num_sweeps(1, camera_fps=30, lidar_fps=10) == 1


def test_lidar_clip_coercion_accepts_the_layouts_controls_ship_in() -> None:
    """[C,T,H,W], [T,C,H,W] and a batched [1,...] all normalise to [C,T,H,W]."""
    chw = torch.zeros(3, 8, 64, 1024)
    assert _coerce_lidar_clip(chw, max_frames=None).shape == (3, 8, 64, 1024)
    tchw = torch.zeros(8, 3, 64, 1024)
    assert _coerce_lidar_clip(tchw, max_frames=None).shape == (3, 8, 64, 1024)
    batched = torch.zeros(1, 3, 8, 64, 1024)
    assert _coerce_lidar_clip(batched, max_frames=None).shape == (3, 8, 64, 1024)
    single_channel = torch.zeros(1, 8, 64, 1024)
    assert _coerce_lidar_clip(single_channel, max_frames=None).shape == (1, 8, 64, 1024)
    assert _coerce_lidar_clip(chw, max_frames=4).shape == (3, 4, 64, 1024)


def test_joint_camera_lidar_is_recognised(tmp_path: Path) -> None:
    sample = OmniSampleOverrides(
        lidar={"control_path": _touch(tmp_path, "lidar.pt")},
        wsm={},
        multiview={"views": _views(tmp_path)},
    )
    sample._validate_lidar_transfer()
    assert sample.is_joint_cam_lidar_transfer
    assert set(sample.transfer_hints) == {TransferHintKey.WSM, TransferHintKey.LIDAR}


def test_lidar_only_is_not_joint(tmp_path: Path) -> None:
    sample = OmniSampleOverrides(lidar={"control_path": _touch(tmp_path, "lidar.pt")})
    sample._validate_lidar_transfer()
    assert not sample.is_joint_cam_lidar_transfer


def test_wsm_multiview_without_lidar_is_untouched(tmp_path: Path) -> None:
    """The joint path must not change plain per-view WSM transfer."""
    sample = OmniSampleOverrides(wsm={}, multiview={"views": _views(tmp_path)})
    sample._validate_lidar_transfer()
    assert not sample.is_joint_cam_lidar_transfer


@pytest.mark.parametrize(
    ("kwargs_factory", "match"),
    [
        (lambda p: {"lidar": {"control_path": _touch(p, "l.pt")}, "edge": {}}, "other control hints"),
        (lambda p: {"lidar": {}}, "requires lidar.control_path"),
        (
            lambda p: {"lidar": {"control_path": _touch(p, "l.pt")}, "vision_path": _touch(p, "v.mp4")},
            "vision_path is not supported",
        ),
        (lambda p: {"lidar": {"control_path": _touch(p, "l.pt")}, "wsm": {}}, "requires multiview"),
        (
            lambda p: {
                "lidar": {"control_path": _touch(p, "l.pt")},
                "multiview": {"views": _views(p)},
            },
            "does not support multiview",
        ),
        (
            lambda p: {
                "lidar": {"control_path": _touch(p, "l.pt")},
                "wsm": {"control_path": _touch(p, "w.mp4")},
                "multiview": {"views": _views(p)},
            },
            "do not also set wsm.control_path",
        ),
    ],
)
def test_illegal_lidar_combinations_are_rejected(tmp_path: Path, kwargs_factory, match: str) -> None:
    sample = OmniSampleOverrides(**kwargs_factory(tmp_path))
    with pytest.raises(ValueError, match=match):
        sample._validate_lidar_transfer()


def test_joint_partial_view_conditioning_is_rejected(tmp_path: Path) -> None:
    """A partly-conditioned batch would generate silently."""
    views = _views(tmp_path)
    views[0]["vision_path"] = _touch(tmp_path, "only_one.mp4")
    with pytest.raises(pydantic.ValidationError, match="vision_path"):
        OmniSampleOverrides(
            lidar={"control_path": _touch(tmp_path, "lidar.pt")},
            wsm={},
            multiview={"views": views},
        )


_V, _F, _H, _W = 3, 8, 64, 128
_SWEEPS = joint_lidar_num_sweeps(_F, camera_fps=30, lidar_fps=10)


def _joint_batch(**overrides):
    kwargs = dict(
        camera_control=[torch.rand(3, _F, _H, _W) for _ in range(_V)],
        lidar_control=torch.rand(3, _SWEEPS, 64, 1808),
        num_camera_frames=_F,
        num_lidar_sweeps=_SWEEPS,
        camera_height=_H,
        camera_width=_W,
        camera_fps=30.0,
        prompt_key="ai_caption",
        prompt=["view"] * _V,
        negative_prompt=None,
        device="cpu",
    )
    kwargs.update(overrides)
    return build_joint_cam_lidar_batch(**kwargs)


def test_joint_batch_sets_the_flag_the_sampler_guards_on() -> None:
    """``has_lidar`` is what the joint sampler guards on, and nothing else in the package sets it."""
    plan = _joint_batch()["sequence_plan"][0]
    assert plan.has_lidar is True
    assert plan.has_vision is True
    assert plan.has_text is True


def test_joint_batch_packs_camera_major_and_pairs_control_with_target() -> None:
    """Two vision items and two LiDAR items: [control, target], camera-major over V*F frames."""
    batch = _joint_batch()
    assert batch["dataset_name"] == JOINT_CAM_LIDAR_DATASET_NAME
    control, target = batch["video"]
    assert control.shape == (1, 3, _V * _F, _H, _W)
    assert target.shape == control.shape
    lidar_control, lidar_target = batch["lidar"]
    assert lidar_control.shape == (3, _SWEEPS, 64, 1808)
    assert lidar_target.shape == lidar_control.shape
    # The target stream is what gets denoised, so it must start empty rather than echo the control.
    assert torch.count_nonzero(lidar_target) == 0
    assert batch["sample_n_views"].tolist() == [_V]
    assert batch["num_frames"].tolist() == [_V * _F]
    assert batch["num_video_frames_per_view"].tolist() == [_F]
    assert batch["num_vision_items_per_sample"] == [2]
    assert batch["num_lidar_items_per_sample"] == [2]


def test_joint_batch_system_prompt_follows_the_prompt_shape() -> None:
    """A per-view caption list is the joint recipe; a single string is plain transfer."""
    per_view = _joint_batch(prompt=["a"] * _V)["system_prompt"]
    single = _joint_batch(prompt="one caption")["system_prompt"]
    assert per_view != single
    assert "jointly" in per_view


def test_joint_batch_rejects_malformed_inputs() -> None:
    with pytest.raises(ValueError, match="at least one camera control"):
        _joint_batch(camera_control=[])
    with pytest.raises(ValueError, match=r"\[3,T,H,W\]"):
        _joint_batch(camera_control=[torch.rand(_F, _H, _W)])
    with pytest.raises(ValueError, match=r"\[C,T,H,W\]"):
        _joint_batch(lidar_control=torch.rand(3, _SWEEPS, 64))
    with pytest.raises(ValueError, match="non-empty condition frame indexes"):
        _joint_batch(camera_conditions=[torch.rand(3, _F, _H, _W) for _ in range(_V)])
    with pytest.raises(ValueError, match="condition tensors"):
        _joint_batch(
            camera_conditions=[torch.rand(3, _F, _H, _W)],
            condition_frame_indexes_vision=[0],
        )


def test_joint_sample_does_not_reach_the_multiview_batch_builder(tmp_path: Path, monkeypatch) -> None:
    """The real call path is create_batches -> get_sample_data, not generate_batch directly.

    get_sample_data dispatches on ``multiview is not None``, which a joint sample satisfies, and
    get_multiview_sample_data then refuses two transfer hints. Testing generate_batch alone missed
    this: the joint branch there is never reached, because assembly happens earlier.
    """
    from cosmos_framework.inference import inference as inf

    called = []
    monkeypatch.setattr(
        "cosmos_framework.inference.multiview.get_multiview_sample_data",
        lambda *a, **k: called.append("multiview") or {},
    )
    # model_mode is required: get_sample_data reads sample_args.model_mode.is_reasoner before it
    # reaches the joint branch, and an overrides object leaves it None.
    sample = OmniSampleOverrides(
        model_mode=ModelMode.IMAGE2VIDEO,
        lidar={"control_path": _touch(tmp_path, "lidar.pt")},
        wsm={},
        multiview={"views": _views(tmp_path)},
    )
    assert sample.is_joint_cam_lidar_transfer
    assert inf.get_sample_data(sample, model=None, device="cpu") == {}
    assert called == [], "joint must not be routed to the multiview batch builder"


def test_joint_sample_args_file_records_the_resolved_frame_count(tmp_path: Path, monkeypatch) -> None:
    """sample_args.json must describe the media written beside it, not the request.

    Per-view captions take num_frames from the caption's own span, so the caller's copy is stale
    once generation has run. Writing it before the call produced a file claiming 93 frames next to
    297-frame videos.
    """
    from cosmos_framework.inference import inference as inf
    from cosmos_framework.inference import lidar_transfer as lt

    requested, resolved = 93, 297
    sample = OmniSampleOverrides(
        model_mode=ModelMode.IMAGE2VIDEO,
        num_frames=requested,
        output_dir=tmp_path,
        lidar={"control_path": _touch(tmp_path, "lidar.pt")},
        wsm={},
        multiview={"views": _views(tmp_path)},
    )

    def fake_generate(*, sample_args, model, per_view_captions):
        del model, per_view_captions
        return lt.JointCamLidarGenerationOutput(
            output_video=torch.zeros(1, 3, 1, 8, 8),
            output_lidar=torch.zeros(1, 3, 1, 8, 8),
            control_video=torch.zeros(3, 1, 8, 8),
            control_lidar=torch.zeros(1, 3, 1, 8, 8),
            fps=30.0,
            lidar_fps=10.0,
            original_hw=(8, 8),
            sample_n_views=1,
            num_video_frames_per_view=resolved,
            lidar_frame_ids=None,
            resolved_sample_args=sample_args.model_copy(update={"num_frames": resolved}),
        )

    # _generate_joint_cam_lidar_batch imports these inside the function, so patch the source module.
    monkeypatch.setattr(lt, "generate_joint_cam_lidar_sample", fake_generate)
    monkeypatch.setattr(lt, "save_lidar_range_view", lambda *a, **k: tmp_path / "lidar.safetensors")
    monkeypatch.setattr(lt, "lidar_range_projection", lambda _model: None)
    # This CPU test checks saved metadata; distributed CUDA error propagation is separate.
    monkeypatch.setattr(inf, "sync_distributed_errors", nullcontext)

    pipe = object.__new__(inf.OmniInference)
    pipe.model = None
    pipe.per_view_captions = True
    pipe.should_process_sample = lambda _a: True
    pipe._save_multiview_view_videos = lambda *a, **k: ([], {})

    inf.OmniInference._generate_joint_cam_lidar_batch(pipe, sample, save_outputs=True)

    written = json.loads((tmp_path / "sample_args.json").read_text())
    assert written["num_frames"] == resolved, (
        f"sample_args.json says {written['num_frames']} frames, generation produced {resolved}"
    )


# --------------------------------------------------------------------------------------------------
# Measured LiDAR prefix (lidar.condition_path)
# --------------------------------------------------------------------------------------------------


def _causal_lidar_model(projection: LidarRangeProjectionConfig) -> SimpleNamespace:
    return SimpleNamespace(
        tokenizer_lidar_gen=SimpleNamespace(range_projection=projection, temporal_compression_factor=1, is_causal=True)
    )


def _write_rangemap_archive(path: Path, frame_ids: list[int]) -> None:
    """A multiview ``lidar_rangemap`` tar with one occupied ray per sweep and measured intensity 0.25."""
    with tarfile.open(path, "w") as archive:
        for frame_id in frame_ids:
            arrays = {"row": [0], "col": [0], "range": [20.0 + frame_id], "intensity": [0.25]}
            for kind, values in arrays.items():
                payload = io.BytesIO()
                np.savez(payload, np.asarray(values))
                member = tarfile.TarInfo(f"clip.{frame_id:06d}.lidar_{kind}.npz")
                member.size = len(payload.getvalue())
                archive.addfile(member, io.BytesIO(payload.getvalue()))


@pytest.mark.parametrize("rgb_prefix", [False, True])
def test_joint_lidar_prefix_is_clean_and_excluded_from_loss(rgb_prefix: bool) -> None:
    control = torch.zeros(3, 3, 4, 8)  # [C,T,H,W]
    measured = torch.full((3, 1, 4, 8), 7.0)  # [C,T_prefix,H,W]
    batch = build_joint_cam_lidar_batch(
        camera_control=torch.zeros(3, 9, 4, 8),  # [C,T_camera,H,W]
        camera_conditions=[torch.ones(3, 1, 4, 8)] if rgb_prefix else None,  # list[[C,1,H,W]]
        condition_frame_indexes_vision=[0] if rgb_prefix else [],
        lidar_control=control,
        lidar_condition=measured,
        num_camera_frames=9,
        num_lidar_sweeps=3,
        camera_height=4,
        camera_width=8,
        camera_fps=30.0,
        prompt_key="ai_caption",
        prompt="street",
        negative_prompt=None,
        device="cpu",
    )
    torch.testing.assert_close(batch["lidar"][0], control)
    torch.testing.assert_close(batch["lidar"][1][:, :1], measured)
    assert torch.count_nonzero(batch["lidar"][1][:, 1:]) == 0
    assert batch["sequence_plan"][0].condition_frame_indexes_lidar == [0]
    data = GenerationDataClean(
        batch_size=1,
        is_image_batch=False,
        x0_tokens_vision=[torch.zeros(1, 1, 3, 1, 1) for _ in range(2)],  # list[[B,C,T,H,W]]
        num_vision_items_per_sample=[2],
        x0_tokens_lidar=[torch.ones(1, 1, 3, 1, 1) for _ in range(2)],  # list[[B,C,T,H,W]]
        num_lidar_items_per_sample=[2],
        fps_lidar=torch.tensor([10.0]),  # [B]
    )
    packed = pack_input_sequence(
        sequence_plans=batch["sequence_plan"],
        input_text_indexes=[[10]],
        gen_data_clean=data,
        input_timesteps=torch.tensor([0.5]),  # [B]
        special_tokens={"eos_token_id": 151645, "start_of_generation": 151652, "end_of_generation": 151653},
        temporal_compression_factor=1,
        lidar_temporal_compression_factor=1,
    )
    assert packed.lidar is not None
    assert packed.lidar.condition_mask[0].flatten().tolist() == [1.0, 1.0, 1.0]
    assert packed.lidar.condition_mask[1].flatten().tolist() == [1.0, 0.0, 0.0]
    pred = [torch.ones(1, 3, 1, 1, requires_grad=True) for _ in range(2)]  # list[[C,T,H,W]]
    loss, _ = compute_flow_matching_loss(
        pred=pred,
        target=[torch.zeros_like(item) for item in pred],  # list[[C,T,H,W]]
        condition_mask=packed.lidar.condition_mask,
        timesteps=torch.full((2, 1), 0.5),  # [N_items,1]
        has_valid_tokens=True,
        rectified_flow=RectifiedFlow(velocity_field=torch.nn.Identity()),
        tensor_kwargs_fp32={"device": "cpu", "dtype": torch.float32},
    )  # loss: []
    loss.backward()
    assert torch.count_nonzero(pred[0].grad) == 0
    assert torch.count_nonzero(pred[1].grad[:, :1]) == 0
    assert torch.count_nonzero(pred[1].grad[:, 1:]) == 2


def test_joint_batch_without_a_prefix_keeps_an_empty_lidar_target() -> None:
    """No condition_path: the default joint batch is unchanged."""
    batch = _joint_batch()
    assert torch.count_nonzero(batch["lidar"][1]) == 0
    assert batch["sequence_plan"][0].condition_frame_indexes_lidar == []


@pytest.mark.parametrize(
    "prefix",
    [
        torch.zeros(3, 64, 1808),  # missing the time axis
        torch.zeros(1, 1, 64, 1808),  # channel count differs from the control
        torch.zeros(3, 1, 64, 1800),  # grid differs from the control
        torch.zeros(3, 0, 64, 1808),  # empty
        torch.zeros(3, _SWEEPS, 64, 1808),  # leaves nothing to generate
    ],
    ids=["rank", "channels", "grid", "empty", "whole_clip"],
)
def test_joint_batch_rejects_a_prefix_that_is_not_a_proper_prefix_of_the_control(prefix: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="nonempty, proper prefix"):
        _joint_batch(lidar_condition=prefix)


def test_joint_lidar_prefix_tensor_preserves_metric_values_and_rejects_short_input(tmp_path: Path) -> None:
    model = _causal_lidar_model(LidarRangeProjectionConfig(semantic_width=8, model_width=16))
    path = tmp_path / "measured.pt"
    measured = torch.zeros(3, 2, 128, 8)  # [C,T,H,W_semantic]
    measured[0] = 20.0  # [T,H,W_semantic], meters
    measured[1] = 0.25  # [T,H,W_semantic], unit intensity
    measured[2] = 1.0  # [T,H,W_semantic], validity
    torch.save(measured, path)
    args = LidarTransferArgs(condition_path=str(path), num_conditional_sweeps=2)
    clip = load_joint_lidar_condition(model, args, num_lidar_sweeps=5, control_frame_ids=None)  # [C,T,H,W_model]
    assert clip.shape == (3, 2, 128, 16)
    torch.testing.assert_close(clip[..., 4:-4], measured)
    model.tokenizer_lidar_gen.is_causal = False
    with pytest.raises(ValueError, match="temporally causal"):
        load_joint_lidar_condition(model, args, num_lidar_sweeps=5, control_frame_ids=None)
    model.tokenizer_lidar_gen.is_causal = True
    model.tokenizer_lidar_gen.temporal_compression_factor = 4
    with pytest.raises(ValueError, match="temporal compression factor 1"):
        load_joint_lidar_condition(model, args, num_lidar_sweeps=5, control_frame_ids=None)
    model.tokenizer_lidar_gen.temporal_compression_factor = 1
    args.num_conditional_sweeps = 3
    with pytest.raises(ValueError, match=r"must contain \[3,3,H,W\]"):
        load_joint_lidar_condition(model, args, num_lidar_sweeps=5, control_frame_ids=None)
    args.num_conditional_sweeps = 5
    with pytest.raises(ValueError, match="leave at least one"):
        load_joint_lidar_condition(model, args, num_lidar_sweeps=5, control_frame_ids=None)


def test_joint_lidar_prefix_is_absent_without_a_condition_path() -> None:
    model = _causal_lidar_model(LidarRangeProjectionConfig(semantic_width=8, model_width=16))
    assert load_joint_lidar_condition(model, LidarTransferArgs(), num_lidar_sweeps=5, control_frame_ids=None) is None


def test_joint_lidar_prefix_rejects_unsupported_files(tmp_path: Path) -> None:
    model = _causal_lidar_model(LidarRangeProjectionConfig(semantic_width=8, model_width=16))
    path = tmp_path / "measured.safetensors"
    path.touch()
    args = LidarTransferArgs(condition_path=str(path))
    with pytest.raises(ValueError, match=r"metric 3-channel \.pt clip or a prepared lidar_rangemap \.tar"):
        load_joint_lidar_condition(model, args, num_lidar_sweeps=5, control_frame_ids=None)


def test_joint_lidar_prefix_tensor_is_refused_against_a_control_that_recorded_source_ids(tmp_path: Path) -> None:
    """A tensor prefix carries no IDs, so beside a .tar control the window is unprovable.

    The archive path matches the condition's source IDs against the control prefix and
    refuses a mismatch. A tensor clip has nothing to match, so pairing it with a control
    that does record IDs would condition on an unrelated drive in silence.
    """
    model = _causal_lidar_model(LidarRangeProjectionConfig(semantic_width=8, model_width=16))
    path = tmp_path / "measured.pt"
    torch.save(torch.ones(3, 2, 128, 8), path)  # [C,T,H,W_semantic]
    args = LidarTransferArgs(condition_path=str(path), num_conditional_sweeps=2)
    with pytest.raises(ValueError, match="records no source frame IDs"):
        load_joint_lidar_condition(
            model,
            args,
            num_lidar_sweeps=5,
            control_frame_ids=torch.tensor([0, 3, 6, 9, 12]),  # [T]
        )
    # The same clip is served where the control records no IDs either.
    assert load_joint_lidar_condition(model, args, num_lidar_sweeps=5, control_frame_ids=None).shape == (3, 2, 128, 16)


def test_joint_lidar_prefix_archive_aligns_source_ids_and_keeps_measured_intensity(tmp_path: Path) -> None:
    path = tmp_path / "lidar_rangemap.tar"
    _write_rangemap_archive(path, [0, 3, 6])
    model = _causal_lidar_model(LidarRangeProjectionConfig(semantic_width=1800, model_width=1808))
    args = LidarTransferArgs(condition_path=str(path), num_conditional_sweeps=1)
    clip = load_joint_lidar_condition(
        model,
        args,
        num_lidar_sweeps=3,
        control_frame_ids=torch.tensor([3, 6, 9]),  # [T]
    )  # [C,T_prefix,H,W_model]
    assert clip.shape == (3, 1, 128, 1808)
    assert clip[0, 0, 0, 4].item() == 23.0
    assert clip[1, 0, 0, 4].item() == 0.25
    assert clip[2, 0, 0, 4].item() == 1.0
    args.num_conditional_sweeps = 2
    with pytest.raises(ValueError, match="do not match control prefix"):
        load_joint_lidar_condition(
            model,
            args,
            num_lidar_sweeps=3,
            control_frame_ids=torch.tensor([0, 6, 9]),  # [T]
        )


def test_joint_lidar_prefix_archive_without_control_ids_reads_its_first_sweeps(tmp_path: Path) -> None:
    """A control that recorded no IDs leaves the archive's own first sweeps as the prefix."""
    path = tmp_path / "lidar_rangemap.tar"
    _write_rangemap_archive(path, [6, 9, 12])
    model = _causal_lidar_model(LidarRangeProjectionConfig(semantic_width=1800, model_width=1808))
    args = LidarTransferArgs(condition_path=str(path), num_conditional_sweeps=2)
    clip = load_joint_lidar_condition(model, args, num_lidar_sweeps=3, control_frame_ids=None)  # [C,T_prefix,H,W]
    assert clip.shape == (3, 2, 128, 1808)
    assert clip[0, :, 0, 4].tolist() == [26.0, 29.0]


@pytest.mark.parametrize(
    ("control_name", "recorded_ids"),
    [("hdmap.safetensors", None), ("hdmap.tar", [0, 3, 6])],
    ids=["safetensors_control", "archive_control"],
)
def test_joint_generation_feeds_the_measured_prefix_into_the_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, control_name: str, recorded_ids: list[int] | None
) -> None:
    """The prefix is matched against the IDs the control recorded, never synthesized ones, and reaches the batch."""
    from cosmos_framework.inference import inference as inf
    from cosmos_framework.inference import lidar_transfer as lt

    lidar = LidarTransferArgs(
        control_path=_touch(tmp_path, control_name), condition_path=_touch(tmp_path, "measured.pt")
    )
    sample = SimpleNamespace(
        is_joint_cam_lidar_transfer=True,
        multiview=SimpleNamespace(
            views=[SimpleNamespace(camera_key="camera_front_wide_120fov", control_path=tmp_path / "camera.mp4")],
            load_caption_from_data=False,
            has_condition_vision=False,
            optimize_raw_pixel_memory=False,
        ),
        lidar=lidar,
        wsm=None,
        resolution="480",
        vision_size=(2, 2),
        num_frames=9,
        fps=30,
        emphasize_control_in_prompt=False,
        condition_frame_indexes_vision=[],
    )
    sample.model_copy = lambda **kwargs: SimpleNamespace(**(vars(sample) | kwargs.get("update", {})))
    model = SimpleNamespace(
        tokenizer_vision_gen=object(),
        config=SimpleNamespace(
            tokenizer=SimpleNamespace(temporal_compression_factor=4),
            lidar_fps=10,
            diffusion_expert_config=SimpleNamespace(num_view_embeddings=0),
        ),
        input_caption_key="ai_caption",
    )
    control = torch.zeros(3, 3, 2, 2)  # [C,T_lidar,H,W]
    ids = None if recorded_ids is None else torch.tensor(recorded_ids)  # [T_lidar]
    measured = torch.ones(3, 1, 2, 2)  # [C,T_prefix,H,W]
    monkeypatch.setattr(lt, "load_multiview_media_pixels", Mock(return_value=torch.zeros(3, 9, 2, 2)))
    monkeypatch.setattr(lt, "load_lidar_control", Mock(return_value=(lidar.control_path, control, ids)))
    load_prefix = Mock(return_value=measured)
    monkeypatch.setattr(lt, "load_joint_lidar_condition", load_prefix)
    monkeypatch.setattr(inf, "_get_prompt_sample_data", Mock(return_value={"ai_caption": ["A driving scene."]}))
    build_batch = Mock(side_effect=RuntimeError("batch built"))
    monkeypatch.setattr(lt, "build_joint_cam_lidar_batch", build_batch)

    with pytest.raises(RuntimeError, match="batch built"):
        lt.generate_joint_cam_lidar_sample(sample, model)

    assert load_prefix.call_args.args[1] is lidar
    assert load_prefix.call_args.kwargs["num_lidar_sweeps"] == 3
    passed_ids = load_prefix.call_args.kwargs["control_frame_ids"]
    if recorded_ids is None:
        assert passed_ids is None
    else:
        assert passed_ids.tolist() == recorded_ids
    assert build_batch.call_args.kwargs["lidar_condition"] is measured
