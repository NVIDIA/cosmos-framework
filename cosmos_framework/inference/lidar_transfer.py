# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""LiDAR control loading and range-view output for joint camera+LiDAR generation.

The model side of joint generation already lives in ``cosmos_framework.model.generator``
(``pack_lidar_tokens``, ``encode_lidar``/``decode_lidar``, the ``lidar2llm``/``llm2lidar`` heads
and ``sample_joint_transfer_ar``). What this module adds is the inference seam: turning a
``lidar.control_path`` into the ``[C,T,H,W]`` clip the tokenizer expects, reading an optional
measured ``lidar.condition_path`` into a clean prefix of the target sweeps, and turning a decoded
clip back into metric range and intensity on disk.

A LiDAR control is not an RGB video and must not be read as one. It carries metric range in
channel 0, intensity in channel 1 and an occupancy/validity signal in channel 2, on a range-view
canvas whose width is a property of the tokenizer rather than of the file.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from cosmos_framework.data.generator.augmentors.text_tokenizer import (
    _SYSTEM_PROMPT_AV_JOINT_CAMERA_LIDAR_TRANSFER,
    _SYSTEM_PROMPT_TRANSFER,
)
from cosmos_framework.data.generator.multiview.camera_attributes import CAMERA_VIEW_MAPPING
from cosmos_framework.data.generator.multiview.caption_chunks import caption_chunk_frame_count
from cosmos_framework.data.generator.sequence_packing import SequencePlan
from cosmos_framework.inference.args import ConditionVisionMode, LidarTransferArgs
from cosmos_framework.inference.common.args import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from cosmos_framework.inference.multiview import load_multiview_caption_chunks_per_view
from cosmos_framework.inference.transfer import _resolve_transfer_seed, build_control_cfg_postprocess
from cosmos_framework.inference.vision import (
    load_conditioning_image_pixels,
    load_conditioning_video,
    pad_temporal_frames,
)
from cosmos_framework.model.generator.omni_mot_model import INFERENCE_RAW_VISION_RETAINED_ITEMS_KEY
from cosmos_framework.model.generator.tokenizers.lidar.paths import is_remote_uri
from cosmos_framework.model.generator.tokenizers.lidar.postprocessing import (
    DEFAULT_RANGE_SMOOTHING,
    smooth_metric_range,
    validity_mask,
)
from cosmos_framework.model.generator.tokenizers.lidar.preprocessing import (
    circular_pad_range_video_3ch,
    list_rangemap_sweep_ids,
    load_lidar_rangeview_directory,
    load_prepared_lidar_video_3ch,
    prepare_lidar_video_3ch,
    resize_range_video_width,
    undo_model_width_transform,
)
from cosmos_framework.model.generator.tokenizers.lidar.range_projection import LidarRangeProjectionConfig
from cosmos_framework.utils.generator.multiview import (
    build_camera_major_video,
    decode_multiview_latent_per_view,
    load_multiview_media_pixels,
    pad_multiview_view_video,
)

__all__ = [
    "JointCamLidarGenerationOutput",
    "build_joint_cam_lidar_batch",
    "generate_joint_cam_lidar_sample",
    "joint_lidar_num_sweeps",
    "lidar_range_projection",
    "lidar_temporal_compression_factor",
    "load_joint_lidar_condition",
    "load_lidar_control",
    "load_lidar_transfer_control",
    "save_lidar_range_view",
]

_LIDAR_TENSOR_SUFFIXES = {".pt", ".pth"}
JOINT_CAM_LIDAR_DATASET_NAME = "joint_cam_lidar_transfer"
# LiDAR sweeps are recorded at 10 Hz. Read off the model config when it says so, since a
# checkpoint trained at another rate would need a different sweep count for the same clip.
JOINT_DEFAULT_LIDAR_FPS = 10.0

log = logging.getLogger(__name__)


@dataclasses.dataclass
class JointCamLidarGenerationOutput:
    """Both generated streams plus their controls, kept apart: cameras run at ``fps``, sweeps at ``lidar_fps``."""

    output_video: torch.Tensor  # [1,3,V*F,H,W]
    output_lidar: torch.Tensor  # [1,3,S,H_r,W_model]
    control_video: torch.Tensor  # [3,V*F,H,W]
    control_lidar: torch.Tensor  # [1,C,S,H_r,W_model]
    fps: float
    lidar_fps: float
    original_hw: tuple[int, int]
    sample_n_views: int
    num_video_frames_per_view: int
    lidar_frame_ids: torch.Tensor | None
    resolved_sample_args: Any  # OmniSampleArgs; typed loosely like the argument it comes from
    """The args generation actually ran with.

    Per-view captions replace ``prompt`` and take ``num_frames`` from the caption's own span, so
    the caller's copy is stale by the time the streams exist. Record this one, or sample_args.json
    reports a frame count the media beside it does not have.
    """


def joint_lidar_num_sweeps(num_camera_frames: int, *, camera_fps: float, lidar_fps: float) -> int:
    """Map camera frames on the 30 Hz clock to 10 Hz LiDAR sweeps."""
    if num_camera_frames < 1:
        raise ValueError(f"num_camera_frames must be positive, got {num_camera_frames}")
    if camera_fps <= 0 or lidar_fps <= 0:
        raise ValueError(f"camera_fps and lidar_fps must be positive, got {camera_fps}, {lidar_fps}")
    return max(1, round(num_camera_frames * lidar_fps / camera_fps))


def _as_lidar_control_uri(control_path: str | Path) -> str:
    """Keep object-store URIs as strings.

    ``Path("s3://bucket/key")`` stringifies to ``s3:/bucket/key``, which then fails
    ``is_remote_uri`` and gets opened as a local file.
    """
    return control_path if isinstance(control_path, str) else str(control_path)


def _tensor_lidar_frame_ids(num_sweeps: int, *, camera_fps: float, lidar_fps: float) -> torch.Tensor:
    """Tensor controls start at camera frame zero and have no source sweep IDs."""
    if num_sweeps < 1 or not all(math.isfinite(rate) and rate > 0 for rate in (camera_fps, lidar_fps)):
        raise ValueError("Tensor LiDAR controls require positive sweep count and finite positive frame rates")
    stride = camera_fps / lidar_fps
    if stride < 1 or not math.isclose(stride, round(stride)):
        raise ValueError("Tensor LiDAR frame IDs require an integer camera-to-LiDAR frame-rate ratio")
    return torch.arange(num_sweeps, dtype=torch.int64) * round(stride)


def _lidar_control_suffix(path_str: str) -> str:
    return Path(path_str.rsplit("/", 1)[-1]).suffix.lower()


def lidar_range_projection(model) -> LidarRangeProjectionConfig:
    """The range-view contract from the active LiDAR tokenizer.

    LiDAR-only checkpoints may install it as the primary tokenizer; joint ones keep the camera VAE there.
    """
    tokenizer = getattr(model, "tokenizer_lidar_gen", None) or getattr(model, "tokenizer_vision_gen", None)
    projection = getattr(tokenizer, "range_projection", None)
    if not isinstance(projection, LidarRangeProjectionConfig):
        raise TypeError(
            "LiDAR inference requires a tokenizer with an explicit "
            f"LidarRangeProjectionConfig, got {type(tokenizer).__name__}"
        )
    return projection


def lidar_temporal_compression_factor(model) -> int:
    """The LiDAR VAE's temporal compression, which sets the sweep chunk length.

    Read off the LiDAR VAE, not ``model.config.tokenizer``: a LiDAR-only run keeps the camera one primary.
    """
    tokenizer = getattr(model, "tokenizer_lidar_gen", None)
    if tokenizer is None:
        raise ValueError(
            "LiDAR transfer inference needs a LiDAR tokenizer: set lidar_tokenizer and "
            "lidar_state_ch on the model config."
        )
    return int(tokenizer.temporal_compression_factor)


def _coerce_lidar_clip(tensor: torch.Tensor, *, max_frames: int | None) -> torch.Tensor:
    """Accept ``[C,T,H,W]``, ``[T,C,H,W]``, or batched ``[1,...]`` metric clips."""
    if tensor.ndim == 5:
        if tensor.shape[0] != 1:
            raise ValueError(f"Expected one LiDAR clip, got shape {tuple(tensor.shape)}")
        tensor = tensor[0]
    if tensor.ndim != 4:
        raise ValueError(f"Expected LiDAR clip [C,T,H,W] or [T,C,H,W], got {tuple(tensor.shape)}")
    if tensor.shape[0] in (1, 3):
        clip = tensor
    elif tensor.shape[1] in (1, 3):
        clip = tensor.permute(1, 0, 2, 3)
    else:
        raise ValueError(f"Expected a 1- or 3-channel LiDAR clip, got {tuple(tensor.shape)}")
    if max_frames is not None:
        clip = clip[:, :max_frames]
    return clip.float().contiguous()


def _conform_lidar_control_width(
    clip: torch.Tensor,  # [C,T,H,W]
    *,
    range_projection: LidarRangeProjectionConfig,
    control_path: str,
) -> torch.Tensor:
    """Accept a clip already at model width, or the semantic grid for resize/pad."""
    width = clip.shape[-1]
    if width == range_projection.model_width:
        return clip
    if width == range_projection.semantic_width:
        if range_projection.model_width_transform == "resize":
            return resize_range_video_width(clip, width=range_projection.model_width)
        return circular_pad_range_video_3ch(clip, model_width=range_projection.model_width)
    raise ValueError(
        f"LiDAR control {control_path} is {width} columns wide, but this tokenizer "
        f"reads {range_projection.model_width}. Export it at {range_projection.model_width}, "
        f"or at semantic width {range_projection.semantic_width} to apply "
        f"{range_projection.model_width_transform}."
    )


def load_lidar_transfer_control(
    control_path: str | Path,
    *,
    max_frames: int | None,
    range_projection: LidarRangeProjectionConfig,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Load a tokenizer-ready ``[C,T,H,W]`` clip and optional source frame IDs.

    Accepts physical ``frames[3,T,H,W]`` in ``.safetensors``, a prepared ``.pt`` tensor,
    a prepared rangemap ``.tar``, or a directory of sparse
    ``*.lidar_{row,col,range}.npz`` frames. HD-map controls carry no intensity, so tar and
    directory loads give every occupied ray unit intensity. Object-store URIs are passed through
    as strings so the rangemap helpers can fetch them remotely.
    """
    path_str = _as_lidar_control_uri(control_path)
    remote = is_remote_uri(path_str)
    if not remote and Path(path_str).is_dir():
        raw = load_lidar_rangeview_directory(path_str, max_frames=max_frames)
        video = prepare_lidar_video_3ch(
            raw,
            None,
            prepend_first_frame=False,
            range_projection=range_projection,
        )
        return video, None
    suffix = _lidar_control_suffix(path_str)
    if suffix == ".safetensors":
        if remote:
            raise ValueError("Download the .safetensors LiDAR control locally before inference")
        with safe_open(path_str, framework="pt", device="cpu") as handle:
            if list(handle.keys()) != ["frames"]:
                raise ValueError("LiDAR safetensors controls must contain only physical 'frames'")
            frames = handle.get_slice("frames")
            shape = frames.get_shape()
            if (
                frames.get_dtype() != "F32"
                or len(shape) != 4
                or shape[0] != 3
                or shape[1] < 1
                or shape[2] != range_projection.semantic_height
                or shape[3] not in (range_projection.semantic_width, range_projection.model_width)
            ):
                raise ValueError(f"Expected physical FP32 LiDAR [3,T,H,W] for this tokenizer, got {shape}")
            if max_frames is not None and (max_frames < 1 or shape[1] < max_frames):
                raise ValueError(f"LiDAR control needs {max_frames} sweeps but has {shape[1]}")
            clip = frames[:, :max_frames].contiguous()
        if not bool(torch.isfinite(clip).all()) or bool((clip < 0).any()) or bool((clip[1:] > 1).any()):
            raise ValueError("LiDAR controls require finite metric range, unit intensity, and unit validity")
        return clip, None
    if suffix in _LIDAR_TENSOR_SUFFIXES:
        if remote:
            raise ValueError(
                f"Remote LiDAR tensor controls are unsupported ({path_str}). "
                "Download the .pt locally or use a prepared rangemap .tar URI."
            )
        tensor = torch.load(path_str, map_location="cpu", weights_only=True)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"LiDAR control {path_str} must contain a tensor, got {type(tensor).__name__}")
        return _coerce_lidar_clip(tensor, max_frames=max_frames), None
    if suffix == ".tar":
        sweep_ids = list_rangemap_sweep_ids(path_str)
        if not sweep_ids:
            raise ValueError(f"LiDAR control {path_str} contains no rangemap sweeps")
        if max_frames is not None:
            sweep_ids = sweep_ids[:max_frames]
        video, selected_ids = load_prepared_lidar_video_3ch(
            path_str,
            min_source_frame_id=sweep_ids[0],
            max_source_frame_id=sweep_ids[-1],
            prepend_first_frame=False,
            range_projection=range_projection,
            with_intensity=False,
        )
        frame_ids = torch.tensor(selected_ids, dtype=torch.int64)  # [T]
        if frame_ids.shape[0] != video.shape[1]:
            raise ValueError(
                f"LiDAR control {path_str} returned {video.shape[1]} frames but {frame_ids.shape[0]} source IDs"
            )
        return video, frame_ids
    raise ValueError(
        f"Unsupported LiDAR control {path_str}. Use .safetensors, .pt, a prepared rangemap .tar, or a rangeview directory."
    )


def load_lidar_control(
    model,
    transfer,
    *,
    max_frames: int | None,
) -> tuple[str, torch.Tensor, torch.Tensor | None]:
    """Resolve ``lidar.control_path`` into a clip conformed to this tokenizer's model width."""
    if transfer.control_path is None:
        raise ValueError("lidar transfer requires lidar.control_path")
    control_path = _as_lidar_control_uri(transfer.control_path)
    projection = lidar_range_projection(model)
    clip, frame_ids = load_lidar_transfer_control(
        control_path,
        max_frames=max_frames,
        range_projection=projection,
    )
    clip = _conform_lidar_control_width(clip, range_projection=projection, control_path=control_path)
    return control_path, clip, frame_ids


def load_joint_lidar_condition(
    model,
    lidar: LidarTransferArgs,
    *,
    num_lidar_sweeps: int,
    control_frame_ids: torch.Tensor | None,  # [T]
) -> torch.Tensor | None:  # [3,T_prefix,H,W_model]
    """Read measured target sweeps with training's metric projection and real intensity.

    Returns ``None`` when ``lidar.condition_path`` is unset, or absent as on a plain control hint.
    ``control_frame_ids`` are the source IDs the control was read with, or ``None`` when it recorded
    none; an archive prefix is matched against them sweep by sweep.
    """
    condition_path = getattr(lidar, "condition_path", None)
    if condition_path is None:
        return None
    count = lidar.num_conditional_sweeps
    if not 0 < count < num_lidar_sweeps:
        raise ValueError("lidar.num_conditional_sweeps must leave at least one sweep to generate")
    # The zero-filled future must not change the measured prefix latents relative to
    # encoding a full measured training clip. This requires causal temporal attention
    # (including the 3D bottleneck), independently of the temporal compression factor.
    tokenizer = model.tokenizer_lidar_gen
    if tokenizer is None or not tokenizer.is_causal:
        raise ValueError("Joint LiDAR prefix conditioning requires a temporally causal LiDAR tokenizer")
    if lidar_temporal_compression_factor(model) != 1:
        raise ValueError("Joint LiDAR prefix conditioning requires temporal compression factor 1")
    path = _as_lidar_control_uri(condition_path)
    projection = lidar_range_projection(model)
    suffix = _lidar_control_suffix(path)
    if suffix == ".tar":
        if control_frame_ids is None:
            selected_ids = list_rangemap_sweep_ids(path)[:count]
        else:
            selected_ids = control_frame_ids[:count].tolist()
        if len(selected_ids) != count:
            raise ValueError(f"LiDAR condition {path} needs {count} sweeps, got {len(selected_ids)}")
        clip, frame_ids = load_prepared_lidar_video_3ch(
            path,
            min_source_frame_id=selected_ids[0],
            max_source_frame_id=selected_ids[-1],
            prepend_first_frame=False,
            range_projection=projection,
            with_intensity=True,
        )  # clip: [3,T_prefix,H,W]
        if list(frame_ids) != selected_ids:
            raise ValueError(f"LiDAR condition source IDs {frame_ids} do not match control prefix {selected_ids}")
    elif suffix in _LIDAR_TENSOR_SUFFIXES:
        # A tensor clip records no source IDs, so nothing ties it to the window the control
        # covers. Where the control does carry IDs the mismatch is unprovable rather than
        # absent, and a prefix from an unrelated drive would be conditioned on in silence.
        if control_frame_ids is not None:
            raise ValueError(
                f"LiDAR condition {path} is a tensor clip, which records no source frame IDs, but the "
                "control was read from a .tar that does. Supply the condition as a .tar so its sweeps "
                "can be matched to the control prefix."
            )
        clip, _ = load_lidar_transfer_control(path, max_frames=count, range_projection=projection)  # [3,T_prefix,H,W]
    else:
        raise ValueError("lidar.condition_path must be a metric 3-channel .pt clip or a prepared lidar_rangemap .tar")
    if clip.shape[0] != 3 or clip.shape[1] != count:
        raise ValueError(f"LiDAR condition must contain [3,{count},H,W], got {tuple(clip.shape)}")
    return _conform_lidar_control_width(clip, range_projection=projection, control_path=path)  # [3,T_prefix,H,W_model]


def save_lidar_range_view(
    output_video: torch.Tensor,  # [B,3,T,H,W_model]
    output_file: Path,
    *,
    range_projection: LidarRangeProjectionConfig,
    frame_ids: torch.Tensor | None = None,  # [T]
) -> Path:
    """Save decoded range, intensity and the source frame ID for each sweep.

    The range map is filtered by ``DEFAULT_RANGE_SMOOTHING`` before it is written, because
    everything that reads this file unprojects it, and per-ray error only becomes visible -- as
    scatter along the ray, and as frame-to-frame jitter in a temporal metric -- once the rays are
    points. Filtering here rather than in each reader is what keeps the evaluated geometry the
    same as the previewed one.
    """
    if output_video.ndim != 5 or output_video.shape[0] != 1 or output_video.shape[1] != 3:
        raise ValueError(f"Expected decoded LiDAR [1,3,T,H,W], got {tuple(output_video.shape)}")
    if frame_ids is None:
        raise ValueError("frame_ids are required for generated LiDAR artifacts")
    semantic = undo_model_width_transform(  # [3,T,H,W_semantic]
        output_video[0], range_projection=range_projection
    )
    range_m = semantic[0].detach().float()  # [T,H,W]
    intensity = semantic[1].detach().float()  # [T,H,W]
    validity = validity_mask(semantic[2].detach().float(), threshold=range_projection.validity_threshold)  # [T,H,W]
    validity &= torch.isfinite(range_m)  # [T,H,W]
    validity &= (range_m >= range_projection.min_range_m) & (range_m <= range_projection.max_range_m)  # [T,H,W]
    if DEFAULT_RANGE_SMOOTHING is not None:
        # Run on the semantic canvas, not the model width: the filter wraps in azimuth and that
        # wrap is only a sensor neighbourhood once the width transform is undone.
        range_m, validity = smooth_metric_range(range_m, validity, smoothing=DEFAULT_RANGE_SMOOTHING)
    # Smoothing can move a boundary value one float32 ULP outside the physical range; clamp that
    # numerical drift without overriding the filter's support-based validity.
    range_m = range_m.clamp(min=range_projection.min_range_m, max=range_projection.max_range_m)  # [T,H,W]
    range_m = torch.where(validity, range_m, torch.zeros_like(range_m)).cpu().contiguous()  # [T,H,W]
    intensity = torch.where(  # [T,H,W]
        validity & torch.isfinite(intensity), intensity.clamp(0.0, 1.0), torch.zeros_like(intensity)
    ).cpu()
    intensity = intensity.to(dtype=torch.float16).contiguous()  # [T,H,W]
    frame_ids = frame_ids.detach().to(device="cpu", dtype=torch.int64).contiguous()  # [T]
    if frame_ids.ndim != 1 or frame_ids.shape[0] != range_m.shape[0]:
        raise ValueError(f"frame_ids must have shape [{range_m.shape[0]}], got {tuple(frame_ids.shape)}")
    if frame_ids.numel() > 1 and not bool(torch.all(frame_ids[1:] > frame_ids[:-1])):
        raise ValueError("frame_ids must be strictly increasing")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_file({"range_m": range_m, "intensity": intensity, "frame_ids": frame_ids}, output_file)
    return output_file


def build_joint_cam_lidar_batch(
    *,
    camera_control: torch.Tensor | list[torch.Tensor],
    lidar_control: torch.Tensor,  # [C,T,H,W]
    num_camera_frames: int,
    num_lidar_sweeps: int,
    camera_height: int,
    camera_width: int,
    camera_fps: float,
    prompt_key: str,
    prompt: str | list[str],
    negative_prompt: str | None,
    camera_conditions: list[torch.Tensor] | None = None,
    camera_view_ids: list[int] | None = None,
    condition_frame_indexes_vision: list[int] | None = None,
    lidar_condition: torch.Tensor | None = None,  # [C,T_prefix,H,W]
    control_weight: float = 1.0,
    device: str | torch.device = "cuda",
    optimize_raw_pixel_memory: bool = False,
) -> dict[str, object]:
    """Build the training-shaped joint batch: ``video=[ctrl, target]``, ``lidar=[ctrl, target]``.

    Camera clips are packed camera-major, the same way per-view WSM transfer packs them, so the
    two streams reach the model in the layout it was trained on. This is the only place that emits
    ``data_batch["lidar"]`` together with a ``SequencePlan(has_lidar=True)`` -- the joint sampler
    guards on that flag, so without this the joint path is unreachable no matter what else is set.

    ``lidar_condition`` holds measured target sweeps: they fill the start of the LiDAR target and
    are marked clean in the plan, so the sampler keeps them and generates only the sweeps after.
    """
    camera_controls = [camera_control] if isinstance(camera_control, torch.Tensor) else list(camera_control)
    if not camera_controls:
        raise ValueError("joint camera+LiDAR transfer requires at least one camera control")
    padded_controls: list[torch.Tensor] = []
    for clip in camera_controls:
        if clip.ndim != 4 or int(clip.shape[0]) != 3:
            raise ValueError(f"Expected camera control [3,T,H,W], got {tuple(clip.shape)}")
        padded_controls.append(
            pad_multiview_view_video(clip, num_frames=num_camera_frames, height=camera_height, width=camera_width)
        )
    if lidar_control.ndim != 4:
        raise ValueError(f"Expected LiDAR control [C,T,H,W], got {tuple(lidar_control.shape)}")

    sample_n_views = len(padded_controls)
    condition_frame_indexes_vision = list(condition_frame_indexes_vision or [])
    if camera_conditions is None:
        if condition_frame_indexes_vision:
            raise ValueError("Camera condition indexes require one camera condition per view")
        target_views = [
            pad_multiview_view_video(None, num_frames=num_camera_frames, height=camera_height, width=camera_width)
            for _ in range(sample_n_views)
        ]  # list[[3,F,H,W]]
    else:
        if len(camera_conditions) != sample_n_views:
            raise ValueError(f"Expected {sample_n_views} joint camera condition tensors, got {len(camera_conditions)}.")
        if not condition_frame_indexes_vision:
            raise ValueError("Joint camera conditions require non-empty condition frame indexes")
        target_views = [
            pad_multiview_view_video(frames, num_frames=num_camera_frames, height=camera_height, width=camera_width)
            for frames in camera_conditions
        ]  # list[[3,F,H,W]]
    lidar_control = pad_temporal_frames(lidar_control.float(), num_lidar_sweeps)  # [C,T,H,W]
    lidar_target = torch.zeros_like(lidar_control)  # [C,T,H,W]
    num_lidar_condition_sweeps = 0
    if lidar_condition is not None:
        if (
            lidar_condition.ndim != 4
            or lidar_condition.shape[0] != lidar_control.shape[0]
            or lidar_condition.shape[2:] != lidar_control.shape[2:]
            or not 0 < lidar_condition.shape[1] < num_lidar_sweeps
        ):
            raise ValueError("LiDAR condition must match control channels/grid and be a nonempty, proper prefix")
        num_lidar_condition_sweeps = lidar_condition.shape[1]
        lidar_target[:, :num_lidar_condition_sweeps] = lidar_condition.to(lidar_target)  # [C,T_prefix,H,W]

    raw_pixel_device = "cpu" if optimize_raw_pixel_memory else device
    camera_items = [
        build_camera_major_video(padded_controls, device=raw_pixel_device),
        build_camera_major_video(target_views, device=raw_pixel_device),
    ]  # list[[1,3,V*F,H,W]]
    lidar_items = [lidar_control.to(device=device), lidar_target.to(device=device)]
    size = torch.tensor(
        [[camera_height, camera_width, camera_height, camera_width]], dtype=torch.float32, device=device
    )
    batch: dict[str, object] = {
        "dataset_name": JOINT_CAM_LIDAR_DATASET_NAME,
        # Per-view caption list = joint recipe; a single string = plain transfer conditioning.
        "system_prompt": (
            _SYSTEM_PROMPT_AV_JOINT_CAMERA_LIDAR_TRANSFER if isinstance(prompt, list) else _SYSTEM_PROMPT_TRANSFER
        ),
        "video": camera_items,
        "lidar": lidar_items,
        "image_size": [size, size],
        "num_frames": torch.tensor([sample_n_views * num_camera_frames], device=device),
        "sample_n_views": torch.tensor([sample_n_views], dtype=torch.int64, device=device),
        "num_video_frames_per_view": torch.tensor([num_camera_frames], dtype=torch.int64, device=device),
        "enable_per_camera_vae_encoding": True,
        "num_vision_items_per_sample": [2],
        "num_lidar_items_per_sample": [2],
        "control_weights": [[float(control_weight)]],
        "sequence_plan": [
            SequencePlan(
                has_text=True,
                has_vision=True,
                has_lidar=True,
                condition_frame_indexes_vision=condition_frame_indexes_vision,
                condition_frame_indexes_lidar=list(range(num_lidar_condition_sweeps)),
                vision_temporal_position_groups=[0, 0],
            )
        ],
        "fps": torch.tensor([float(camera_fps)], device=device),
        "conditioning_fps": torch.tensor([float(camera_fps)], device=device),
        prompt_key: [prompt],
    }
    if camera_view_ids is not None:
        if len(camera_view_ids) != sample_n_views:
            raise ValueError(f"Expected {sample_n_views} camera view IDs, got {len(camera_view_ids)}")
        batch["view_indices_selection"] = torch.tensor([camera_view_ids], dtype=torch.int64, device=device)
    if negative_prompt:
        batch[f"neg_{prompt_key}"] = [negative_prompt]
    if optimize_raw_pixel_memory:
        batch[INFERENCE_RAW_VISION_RETAINED_ITEMS_KEY] = [camera_items[0]]
    return batch


def generate_joint_cam_lidar_sample(
    sample_args: Any,
    model: Any,
    *,
    per_view_captions: bool = False,
) -> JointCamLidarGenerationOutput:
    """Generate camera clips and a LiDAR rangemap together from WSM + HD-map controls.

    One denoising pass over both streams; pairing separate runs is what this recipe avoids.
    """
    if not sample_args.is_joint_cam_lidar_transfer:
        raise ValueError("generate_joint_cam_lidar_sample requires wsm, lidar.control_path, and multiview")
    multiview = sample_args.multiview
    lidar = sample_args.lidar
    if sample_args.resolution is None:
        raise ValueError("resolution is required for joint camera+LiDAR inference")
    tokenizer_vision_gen = model.tokenizer_vision_gen
    if tokenizer_vision_gen is None:
        raise ValueError("joint camera+LiDAR inference requires a vision tokenizer")
    views = multiview.views
    if not views:
        raise ValueError("joint camera+LiDAR transfer requires at least one multiview view")

    camera_view_ids: list[int] | None = None
    if getattr(model.config.diffusion_expert_config, "num_view_embeddings", 0):
        camera_keys = [view.camera_key for view in views]
        unknown_keys = [key for key in camera_keys if key not in CAMERA_VIEW_MAPPING]
        if unknown_keys:
            raise ValueError(f"Joint camera+LiDAR rig view embeddings require supported camera keys: {unknown_keys}")
        camera_view_ids = [CAMERA_VIEW_MAPPING[key] for key in camera_keys]

    camera_width, camera_height = sample_args.vision_size
    num_camera_frames = sample_args.num_frames
    view_prompts: list[str] | None = None
    if multiview.load_caption_from_data:
        caption_chunks = load_multiview_caption_chunks_per_view(
            views, first_caption_paragraph_only=multiview.first_caption_paragraph_only
        )
        if not multiview.use_first_chunk_only and len(caption_chunks) > 1:
            raise ValueError(
                "Joint camera+LiDAR inference currently generates one caption chunk; "
                "set multiview.use_first_chunk_only=True."
            )
        caption_chunk = caption_chunks[0]
        num_camera_frames = sample_args.num_video_frames_per_chunk or caption_chunk_frame_count(
            [caption_chunk], tokenizer=tokenizer_vision_gen
        )
        sample_args = sample_args.model_copy(update={"prompt": caption_chunk.prompt, "num_frames": num_camera_frames})
        if per_view_captions:
            if caption_chunk.view_prompts is None:
                raise ValueError("Joint per-view caption inference requires one caption for every camera")
            view_prompts = caption_chunk.view_prompts
    elif per_view_captions:
        raise ValueError(
            "This checkpoint tokenizes one caption per camera (separate_view_text_tokenization). "
            "Run with multiview.load_caption_from_data=True so joint camera+LiDAR inference can "
            "read the per-camera caption files."
        )

    # Snap to a frame count the camera tokenizer can represent; a raw span either fails to encode
    # or is silently padded.
    temporal_compression_factor = model.config.tokenizer.temporal_compression_factor
    num_camera_frames = (
        math.ceil((num_camera_frames - 1) / temporal_compression_factor) * temporal_compression_factor + 1
    )
    camera_fps = float(sample_args.fps)
    lidar_fps = float(getattr(model.config, "lidar_fps", None) or JOINT_DEFAULT_LIDAR_FPS)
    num_lidar_sweeps = joint_lidar_num_sweeps(num_camera_frames, camera_fps=camera_fps, lidar_fps=lidar_fps)

    camera_controls: list[torch.Tensor] = []
    for view_index, view in enumerate(views):
        if view.control_path is None:
            raise ValueError(
                f"joint camera+LiDAR transfer requires control_path for every view (missing at views[{view_index}])"
            )
        camera_controls.append(
            load_multiview_media_pixels(
                Path(view.control_path),
                target_h=camera_height,
                target_w=camera_width,
                max_frames=num_camera_frames,
                video_loader=load_conditioning_video,
                image_loader=load_conditioning_image_pixels,
                video_extensions=VIDEO_EXTENSIONS,
                image_extensions=IMAGE_EXTENSIONS,
            )
        )

    camera_conditions: list[torch.Tensor] | None = None
    if multiview.has_condition_vision:
        if not sample_args.condition_frame_indexes_vision:
            raise ValueError("Joint camera conditioning requires at least one condition frame index")
        if multiview.condition_vision_mode == ConditionVisionMode.VIDEO:
            num_condition_latent_frames = max(sample_args.condition_frame_indexes_vision) + 1
            num_condition_pixel_frames = tokenizer_vision_gen.get_pixel_num_frames(
                num_condition_latent_frames, resolution=getattr(model.config, "resolution", None)
            )
        else:
            num_condition_pixel_frames = 1
        camera_conditions = [
            load_multiview_media_pixels(
                Path(view.vision_path),
                target_h=camera_height,
                target_w=camera_width,
                max_frames=num_condition_pixel_frames,
                keep=sample_args.condition_video_keep or "first",
                video_loader=load_conditioning_video,
                image_loader=load_conditioning_image_pixels,
                video_extensions=VIDEO_EXTENSIONS,
                image_extensions=IMAGE_EXTENSIONS,
            )
            for view in views
        ]

    lidar_control_path, lidar_control, lidar_frame_ids = load_lidar_control(model, lidar, max_frames=num_lidar_sweeps)
    # A measured prefix is matched only against source IDs the control recorded, never against the
    # ones synthesized below for a .safetensors control.
    recorded_lidar_frame_ids = lidar_frame_ids
    # The shared safetensors schema has no source IDs. Keep legacy controls
    # unchanged, including their support for non-integral camera/LiDAR ratios.
    if lidar_frame_ids is None and _lidar_control_suffix(str(lidar_control_path)) == ".safetensors":
        lidar_frame_ids = _tensor_lidar_frame_ids(num_lidar_sweeps, camera_fps=camera_fps, lidar_fps=lidar_fps)
    if lidar_frame_ids is not None and lidar_frame_ids.shape[0] != num_lidar_sweeps:
        raise ValueError(
            f"Joint LiDAR control needs {num_lidar_sweeps} sweeps but selected {lidar_frame_ids.shape[0]} source IDs"
        )
    lidar_condition = load_joint_lidar_condition(
        model, lidar, num_lidar_sweeps=num_lidar_sweeps, control_frame_ids=recorded_lidar_frame_ids
    )  # [3,T_prefix,H,W] or None
    log.info(
        "Loaded joint controls: %d camera views %s and LiDAR %s from %s",
        len(camera_controls),
        [tuple(clip.shape) for clip in camera_controls],
        tuple(lidar_control.shape),
        lidar_control_path,
    )

    # Imported here rather than at module scope: inference.py imports this module for the joint
    # dispatch, so a top-level import would close the cycle. multiview.py defers the same symbol
    # for the same reason.
    from cosmos_framework.inference.inference import _get_prompt_sample_data

    prompt_sample_args = sample_args.model_copy(update={"num_frames": num_camera_frames, "fps": int(round(camera_fps))})
    prompt_data = _get_prompt_sample_data(
        prompt_sample_args, model, h=camera_height, w=camera_width, device="cuda", view_prompts=view_prompts
    )
    prompt = prompt_data[model.input_caption_key][0]
    negative_prompt = prompt_data.get("neg_" + model.input_caption_key, [None])[0]
    if sample_args.emphasize_control_in_prompt:
        control_suffix = (
            " Follow the wsm and lidar control videos precisely: every camera view must"
            " align with its world-scenario map, and the LiDAR rangemap must align with the"
            " HD-map rangemap, at every frame."
        )
        if isinstance(prompt, list):
            prompt = [caption.rstrip() + control_suffix for caption in prompt]
        else:
            prompt = prompt.rstrip() + control_suffix

    wsm_weight = 1.0 if sample_args.wsm is None or sample_args.wsm.weight is None else float(sample_args.wsm.weight)
    data_batch = build_joint_cam_lidar_batch(
        camera_control=camera_controls,
        camera_view_ids=camera_view_ids,
        camera_conditions=camera_conditions,
        lidar_control=lidar_control,
        lidar_condition=lidar_condition,
        num_camera_frames=num_camera_frames,
        num_lidar_sweeps=num_lidar_sweeps,
        camera_height=camera_height,
        camera_width=camera_width,
        camera_fps=camera_fps,
        prompt_key=model.input_caption_key,
        prompt=prompt,
        negative_prompt=negative_prompt,
        condition_frame_indexes_vision=sample_args.condition_frame_indexes_vision,
        control_weight=wsm_weight,
        optimize_raw_pixel_memory=getattr(multiview, "optimize_raw_pixel_memory", False),
    )

    is_distilled = model.config.fixed_step_sampler_config is not None
    sampler = model.fixed_step_sampler if is_distilled else None
    guidance = 1.0 if is_distilled else sample_args.guidance

    model.eval()
    outputs = model.generate_samples_from_batch(
        data_batch,
        sampler=sampler,
        guidance=guidance,
        guidance_interval=sample_args.guidance_interval,
        velocity_postprocess_builder=build_control_cfg_postprocess(
            control_guidance=sample_args.control_guidance,
            control_guidance_interval=sample_args.control_guidance_interval,
        ),
        seed=[_resolve_transfer_seed(sample_args)],
        n_sample=1,
        has_negative_prompt=negative_prompt is not None,
        num_steps=sample_args.num_steps,
        shift=sample_args.shift,
        sigma_max=sample_args.sigma_max,
        normalize_cfg=sample_args.normalize_cfg,
    )
    if not outputs.get("vision"):
        raise RuntimeError("Joint camera+LiDAR generation returned no camera latents")
    if not outputs.get("lidar"):
        raise RuntimeError("Joint camera+LiDAR generation returned no LiDAR latents")

    vision_latent = outputs["vision"][-1]
    if vision_latent.ndim == 4:
        vision_latent = vision_latent.unsqueeze(0)
    sample_n_views = len(camera_controls)
    output_video = decode_multiview_latent_per_view(
        lambda latent: model.decode(latent).clamp(-1, 1),
        vision_latent,
        sample_n_views,
        num_camera_frames,
        assemble_on_cpu=True,
    ).cpu()

    lidar_latent = outputs["lidar"][-1]
    if lidar_latent.ndim == 4:
        lidar_latent = lidar_latent.unsqueeze(0)
    output_lidar = model.decode_lidar(lidar_latent).cpu()

    control_video = data_batch["video"][0]
    return JointCamLidarGenerationOutput(
        output_video=output_video,
        output_lidar=output_lidar,
        control_video=control_video[0].cpu(),
        control_lidar=lidar_control.unsqueeze(0).cpu(),
        fps=camera_fps,
        lidar_fps=lidar_fps,
        original_hw=(camera_height, camera_width),
        sample_n_views=sample_n_views,
        num_video_frames_per_view=num_camera_frames,
        lidar_frame_ids=lidar_frame_ids,
        resolved_sample_args=sample_args,
    )
