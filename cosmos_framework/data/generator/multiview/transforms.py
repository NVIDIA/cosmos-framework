# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""RGB/LiDAR data contracts and transforms; no dataset locations or remote catalog.

Adapted from the corresponding i4 implementation.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any, Optional, TypeAlias

import torch
from einops import rearrange

from cosmos_framework.data.generator.multiview.camera_geometry import SOURCE_FPS_KEY, SOURCE_GEOMETRY_KEY
from cosmos_framework.data.generator.multiview.caption_format import (
    first_caption_paragraph,
    format_multiview_caption,
    format_separate_view_captions,
    format_view_caption,
)
from cosmos_framework.data.generator.multiview.rig_sampler import (
    CameraCountSelection,
    CameraSampler,
    RandomCameraSampler,
    validate_camera_count_selection,
)
from cosmos_framework.data.imaginaire.webdataset.augmentors.augmentor import Augmentor
from cosmos_framework.model.generator.tokenizers.lidar.preprocessing import (
    align_prepared_range_videos,
    load_prepared_lidar_video_3ch,
)
from cosmos_framework.model.generator.tokenizers.lidar.range_projection import (
    LidarRangeProjectionConfig,
    transfer_range_projection,
)
from cosmos_framework.utils import log
from cosmos_framework.utils.generator.data_utils import read_positive_int_metadata
from cosmos_framework.utils.generator.multiview import slice_camera_major_frames
from cosmos_framework.utils.generator.torchcodec_video import TorchCodecVideoReader
from cosmos_framework.utils.generator.video_preprocess import resize_and_center_crop_tensor

CameraKeyType: TypeAlias = str


CameraAttributeMapping: TypeAlias = Mapping[CameraKeyType, Mapping[str, str | int]]


LIDAR_RANGEMAP_KEY = "lidar_rangemap"


HDMAP_RANGEMAP_KEY = "hdmap_rangemap"


LIDAR_RANGEMAP_STORAGE_KEY = f"lidar/{LIDAR_RANGEMAP_KEY}"


HDMAP_RANGEMAP_STORAGE_KEY = f"lidar/{HDMAP_RANGEMAP_KEY}"


RANGEMAP_COMPAT_CLIP_KEY = "_rangemap_compat_clip_key"


DEFAULT_LIDAR_TRANSFER_CAPTION_CAMERA = "camera_front_wide_120fov"


MULTIVIEW_CONTROL_KEY = "control_input"


_MULTIVIEW_TARGET_KEY = "video"


def validate_separate_view_text_tokenization(
    *,
    enabled: bool,
    camera_order: Sequence[CameraKeyType],
    single_caption_camera_name: CameraKeyType | None,
    single_caption_probability: float,
    use_two_pass_caption_format: bool,
    full_video_caption_config: dict[str, dict[str, float]] | None,
    per_view_caption_dropout_rate: float = 0.0,
    add_view_prefix_to_caption: bool = False,
) -> None:
    """Validate that a caption layout has exactly one independently tokenizable caption per camera."""
    if not 0.0 <= per_view_caption_dropout_rate <= 1.0:
        raise ValueError("per_view_caption_dropout_rate must be between 0 and 1")
    if per_view_caption_dropout_rate > 0 and (not enabled or not add_view_prefix_to_caption):
        raise ValueError("Per-view caption dropout requires separate view text tokenization and camera headers")
    if not enabled:
        return
    if not camera_order:
        raise ValueError("Separate view text tokenization requires at least one camera view")
    if single_caption_camera_name is not None and (
        len(camera_order) != 1 or camera_order[0] != single_caption_camera_name
    ):
        raise ValueError("Separate view text tokenization requires one caption per selected camera")
    if single_caption_probability != 1.0:
        raise ValueError("Separate view text tokenization is incompatible with single-caption sampling")
    if use_two_pass_caption_format:
        raise ValueError("Separate view text tokenization is incompatible with two-pass caption formatting")
    if full_video_caption_config is not None:
        raise ValueError("Separate view text tokenization does not support global full-video captions")


def _json_dict_or_none(value: Any) -> dict[str, Any] | None:
    """Return *value* as a dict, parsing it as JSON first when it is a string."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _caption_payload_to_object(value: Any) -> Any:
    """Parse a JSON caption payload, leaving plain-text captions untouched."""
    parsed = _json_dict_or_none(value)
    return value if parsed is None else parsed


def _has_meaningful_caption_text(value: Any) -> bool:
    """Return whether a caption payload contains at least one non-whitespace string."""
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return any(_has_meaningful_caption_text(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return any(_has_meaningful_caption_text(item) for item in value)
    return False


def _normalize_conditioning_config(conditioning_config: dict[int, float] | None) -> dict[int, float] | None:
    """Validate a ``{num_condition_frames: weight}`` map and rescale it to sum to one."""
    if conditioning_config is None:
        return None
    for num_frames, prob in conditioning_config.items():
        if not isinstance(num_frames, int) or num_frames < 0:
            raise ValueError(f"conditioning_config keys must be non-negative integers, got {num_frames}")
        if not isinstance(prob, (int, float)) or prob < 0:
            raise ValueError(f"conditioning_config values must be non-negative numbers, got {prob}")
    total_prob = sum(conditioning_config.values())
    if total_prob <= 0:
        raise ValueError("conditioning_config probabilities must sum to a positive number")
    return {num_frames: prob / total_prob for num_frames, prob in conditioning_config.items()}


class FilterMissingData(Augmentor):
    """Drop or reject samples whose required raw modalities are absent or empty."""

    def __init__(
        self,
        required_keys: Sequence[str],
        *,
        skip_missing: bool = True,
        stats_log_every_n_samples: int = 50,
    ) -> None:
        super().__init__(input_keys=[], output_keys=[])
        if not required_keys:
            raise ValueError("required_keys must not be empty")
        if stats_log_every_n_samples < 0:
            raise ValueError("stats_log_every_n_samples must be nonnegative")
        self.required_keys = tuple(required_keys)
        self.skip_missing = skip_missing
        self.stats_log_every_n_samples = stats_log_every_n_samples
        self.samples_seen = 0
        self.samples_skipped = 0

    def _maybe_log_stats(self) -> None:
        if self.stats_log_every_n_samples == 0 or self.samples_seen % self.stats_log_every_n_samples != 0:
            return
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker is not None else 0
        skip_percentage = 100.0 * self.samples_skipped / self.samples_seen
        log.info(
            f"FilterMissingData stats: worker={worker_id}, samples_seen={self.samples_seen}, "
            f"samples_skipped={self.samples_skipped}, skip_rate={skip_percentage:.2f}%, "
            f"required_keys={self.required_keys}",
            rank0_only=True,
        )

    def __call__(self, data: dict[str, Any]) -> dict[str, Any] | None:
        self.samples_seen += 1
        missing_keys = [
            key
            for key in self.required_keys
            if key not in data
            or data[key] is None
            or (isinstance(data[key], (bytes, bytearray, memoryview)) and len(data[key]) == 0)
        ]
        if not missing_keys:
            self._maybe_log_stats()
            return data

        self.samples_skipped += 1
        self._maybe_log_stats()
        message = (
            f"Missing or empty required data keys {missing_keys}; "
            f"sample_key={data.get('__key__')}, url={data.get('__url__')}"
        )
        if self.skip_missing:
            log.debug(f"FilterMissingData: skipping sample. {message}", rank0_only=False)
            return None
        raise ValueError(message)


class UnpackMetas(Augmentor):
    """Unpack a single ``metas`` JSON list into per-camera meta dicts.

    Used for the **transfer** MADS data where all camera metas are packed into
    a single ``metas`` key (a JSON list indexed by camera position).
    """

    def __init__(
        self,
        position_to_camera_mapping: dict[int, str],
        input_key: str = "metas",
        output_prefix: str = "metas_",
    ) -> None:
        super().__init__([], {})
        self.position_to_camera_mapping = position_to_camera_mapping
        self.input_key = input_key
        self.output_prefix = output_prefix

    def __call__(self, data: dict[str, Any]) -> dict[str, Any] | None:
        if self.input_key not in data:
            log.warning(
                f"UnpackMetas: input key {self.input_key} not found in data, has keys: {data.keys()}", rank0_only=False
            )
            return None

        metas = data.pop(self.input_key)
        for i, meta in enumerate(metas):
            camera_name = self.position_to_camera_mapping[i]
            data[f"{self.output_prefix}{camera_name}"] = meta
        return data


class ExtractFramesAndCaptions(Augmentor):
    """Extract synchronized frames and captions from multiview video samples.

    Works for both "predict" (named keys) and "transfer" (indexed keys)
    dataset layouts — the caller controls the key mappings via config.
    """

    def __init__(
        self,
        camera_order: list[CameraKeyType] | tuple[CameraKeyType, ...],
        num_frames: int,
        resolution_hw: tuple[int, int],
        fps_downsample_factor: int,
        caption_probability: dict[str, float],
        camera_view_mapping: dict[CameraKeyType, int],
        camera_caption_key_mapping: dict[CameraKeyType, str],
        camera_video_key_mapping: dict[CameraKeyType, str],
        camera_control_key_mapping: Optional[dict[CameraKeyType, str]] = None,
        control_output_key: str = MULTIVIEW_CONTROL_KEY,
        control_optional: bool = False,
        add_view_prefix_to_caption: bool = False,
        camera_prefix_mapping: Optional[dict[CameraKeyType, str]] = None,
        use_two_pass_caption_format: bool = False,
        separate_view_text_tokenization: bool = False,
        camera_attribute_mapping: CameraAttributeMapping | None = None,
        single_caption_camera_name: Optional[CameraKeyType] = None,
        single_caption_probability: float = 1.0,
        full_video_caption_config: Optional[dict[str, dict[str, float]]] = None,
        randomize_caption_window_start: bool = False,
        num_cameras_to_keep: CameraCountSelection = None,
        camera_sampler: CameraSampler | None = None,
        passthrough_keys: Optional[list[str]] = None,
        camera_order_from_sample: bool = False,
        first_caption_paragraph_only: bool = False,
        per_view_caption_dropout_rate: float = 0.0,
        require_full_num_frames: bool = False,
        defer_camera_alignment: bool = False,
        minimum_num_frames: int | None = None,
    ) -> None:
        super().__init__([], {})
        self.camera_order = list(camera_order)
        self.num_frames = num_frames
        self.first_caption_paragraph_only: bool = first_caption_paragraph_only
        self.require_full_num_frames: bool = require_full_num_frames
        self.minimum_num_frames: int | None = minimum_num_frames
        if num_frames < 1:
            raise ValueError("num_frames must be positive")
        if minimum_num_frames is not None and not 1 <= minimum_num_frames <= num_frames:
            raise ValueError("minimum_num_frames must be positive and no larger than num_frames")
        # A joint formatter needs the spare decoded frames to align the first
        # camera frame with a LiDAR sweep before rounding to the 1 + 4k grid.
        self.defer_camera_alignment: bool = defer_camera_alignment
        self.expected_num_frames: int = num_frames if defer_camera_alignment else 1 + 4 * ((num_frames - 1) // 4)
        self.resolution_hw = resolution_hw
        self.fps_downsample_factor = fps_downsample_factor
        self.caption_probability = caption_probability
        self.camera_view_mapping = camera_view_mapping
        self.camera_caption_key_mapping = camera_caption_key_mapping
        self.camera_video_key_mapping = camera_video_key_mapping
        self.camera_control_key_mapping = camera_control_key_mapping
        self.control_output_key = control_output_key
        # VT2V rows carry a persisted control stream only for depth/seg; edge and blur
        # are derived from the packed RGB later, so those rows legitimately arrive with
        # no control media at all.
        self.control_optional = control_optional
        self.add_view_prefix_to_caption = add_view_prefix_to_caption
        self.camera_prefix_mapping = camera_prefix_mapping
        self.use_two_pass_caption_format: bool = use_two_pass_caption_format
        self.separate_view_text_tokenization: bool = separate_view_text_tokenization
        self.per_view_caption_dropout_rate: float = per_view_caption_dropout_rate
        self.camera_attribute_mapping = camera_attribute_mapping
        self.single_caption_camera_name = single_caption_camera_name
        self.single_caption_probability: float = single_caption_probability
        self.full_video_caption_config = full_video_caption_config
        self.randomize_caption_window_start = randomize_caption_window_start
        self.num_cameras_to_keep = num_cameras_to_keep
        if camera_sampler is not None and num_cameras_to_keep is not None:
            raise ValueError("Set either camera_sampler or num_cameras_to_keep, not both")
        validate_camera_count_selection(num_cameras_to_keep, max_cameras=len(self.camera_order))
        self.camera_sampler: CameraSampler | None = camera_sampler
        if self.camera_sampler is None and num_cameras_to_keep is not None:
            self.camera_sampler = RandomCameraSampler(
                self.camera_order,
                num_cameras_to_keep=num_cameras_to_keep,
                required_camera_name=(
                    self.single_caption_camera_name
                    if self.single_caption_camera_name is not None and self.single_caption_probability > 0.0
                    else None
                ),
            )
        if self.camera_sampler is not None:
            missing_sampler_cameras = set(self.camera_sampler.camera_names) - set(self.camera_order)
            if missing_sampler_cameras:
                raise ValueError(
                    f"camera_sampler requires cameras not present in camera_order: {sorted(missing_sampler_cameras)}"
                )
        # Keys to copy verbatim from the input sample into the output sample.
        self.passthrough_keys = list(passthrough_keys) if passthrough_keys else []
        # ``camera_order`` normally states what every sample must carry, and a sample
        # short of one of those cameras is a broken sample. A source reading each row's
        # own streams instead binds as many of these cameras as the row can fill, so
        # there the order is an upper bound and a sample legitimately arrives with fewer.
        self.camera_order_from_sample = camera_order_from_sample
        if camera_order_from_sample and single_caption_camera_name is not None:
            raise ValueError(
                "camera_order_from_sample cannot guarantee any particular camera is in a "
                "sample, so it cannot caption every sample from single_caption_camera_name."
            )

        if (
            self.add_view_prefix_to_caption
            and not self.separate_view_text_tokenization
            and self.camera_prefix_mapping is None
        ):
            raise ValueError("camera_prefix_mapping is required when add_view_prefix_to_caption is True")
        if (
            self.separate_view_text_tokenization
            and self.add_view_prefix_to_caption
            and self.camera_attribute_mapping is None
        ):
            raise ValueError("camera_attribute_mapping is required for sampled-rig caption prefixes")

        if set(self.camera_caption_key_mapping.keys()) != set(self.camera_video_key_mapping.keys()):
            raise ValueError(
                f"Mismatching keys {set(self.camera_caption_key_mapping.keys())} "
                f"!= {set(self.camera_video_key_mapping.keys())}"
            )
        if self.camera_control_key_mapping is not None:
            if set(self.camera_control_key_mapping.keys()) != set(self.camera_caption_key_mapping.keys()):
                raise ValueError(
                    f"Mismatching keys {set(self.camera_control_key_mapping.keys())} "
                    f"!= {set(self.camera_caption_key_mapping.keys())}"
                )
        for camera_name in self.camera_caption_key_mapping.keys():
            if camera_name not in self.camera_view_mapping:
                raise ValueError(f"Camera name {camera_name} not found in camera view mapping")
            if self.separate_view_text_tokenization and self.add_view_prefix_to_caption:
                assert self.camera_attribute_mapping is not None
                if camera_name not in self.camera_attribute_mapping:
                    raise ValueError(f"Camera name {camera_name} not found in camera attribute mapping")
        if self.single_caption_camera_name and self.single_caption_camera_name not in self.camera_order:
            raise ValueError(
                f"Single caption camera name {self.single_caption_camera_name} must appear in selected cameras"
            )
        if not 0.0 <= self.single_caption_probability <= 1.0:
            raise ValueError("single_caption_probability must be between 0 and 1")
        if self.single_caption_camera_name is None and self.single_caption_probability != 1.0:
            raise ValueError("single_caption_camera_name is required when single_caption_probability is not 1")
        validate_separate_view_text_tokenization(
            enabled=self.separate_view_text_tokenization,
            camera_order=self.camera_order,
            single_caption_camera_name=self.single_caption_camera_name,
            single_caption_probability=self.single_caption_probability,
            use_two_pass_caption_format=self.use_two_pass_caption_format,
            full_video_caption_config=self.full_video_caption_config,
            per_view_caption_dropout_rate=self.per_view_caption_dropout_rate,
            add_view_prefix_to_caption=self.add_view_prefix_to_caption,
        )

    def _sample_full_video_caption_chunk(self, data: dict[str, Any]) -> dict[str, Any] | None:
        if self.full_video_caption_config is None:
            return None

        available_options: list[tuple[str, float, dict[str, float]]] = []
        for caption_key, option in self.full_video_caption_config.items():
            if caption_key not in data or not isinstance(data[caption_key], dict):
                continue
            ratio = float(option.get("ratio", 1.0))
            if ratio > 0:
                available_options.append((caption_key, ratio, option))

        if not available_options:
            log.warning(
                f"ExtractFramesAndCaptions: none of the configured full-video caption keys "
                f"{list(self.full_video_caption_config.keys())} are present. "
                f"url: {data.get('__url__')}, key: {data.get('__key__')}",
                rank0_only=False,
            )
            return None

        caption_key = random.choices(
            [key for key, _, _ in available_options],
            weights=[ratio for _, ratio, _ in available_options],
            k=1,
        )[0]
        caption_option = next(option for key, _, option in available_options if key == caption_key)
        caption_dict = data[caption_key]

        try:
            chunks = json.loads(caption_dict["caption_structured"])
            stride = int(caption_option.get("min_stride", self.fps_downsample_factor))
            chunk_options: list[tuple[int, str, dict[str, Any], list[int]]] = []
            for index, (chunk_key, chunk) in enumerate(chunks.items()):
                if not isinstance(chunk, dict):
                    continue
                try:
                    start_frame = int(chunk["start_frame"])
                    end_frame = int(chunk["end_frame"])
                except (KeyError, TypeError, ValueError):
                    continue
                frame_indices = self._frame_indices_for_caption_range(
                    start_frame, end_frame, stride, align_to_tokenizer=not self.defer_camera_alignment
                )
                if frame_indices:
                    chunk_options.append((index, chunk_key, chunk, frame_indices))
            if not chunk_options:
                log.warning(
                    f"ExtractFramesAndCaptions: no usable full-video caption chunks for {caption_key}. "
                    f"url: {data.get('__url__')}, key: {data.get('__key__')}",
                    rank0_only=False,
                )
                return None
            chunk_index, sampled_chunk_key, sampled_chunk, frame_indices = random.choice(chunk_options)
            structured = json.loads(sampled_chunk["caption"])
        except Exception as e:
            log.warning(
                f"ExtractFramesAndCaptions: failed to decode full-video caption chunk for {caption_key}. "
                f"url: {data.get('__url__')}, key: {data.get('__key__')}, error: {e}",
                rank0_only=False,
            )
            return None

        return {
            "caption": json.dumps(structured),
            "caption_key": caption_key,
            "chunk_key": sampled_chunk_key,
            "chunk_index": chunk_index,
            "start_frame": int(sampled_chunk["start_frame"]),
            "end_frame": int(sampled_chunk["end_frame"]),
            "stride": stride,
            "frame_indices": frame_indices,
        }

    @staticmethod
    def _frame_indices_for_caption_range(
        start_frame: int,
        end_frame: int,
        stride: int,
        *,
        max_num_frames: int | None = None,
        randomize_start: bool = False,
        align_to_tokenizer: bool = True,
    ) -> list[int]:
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        if max_num_frames is not None and max_num_frames <= 0:
            raise ValueError(f"max_num_frames must be positive, got {max_num_frames}")
        start_frame = max(0, int(start_frame))
        end_frame = max(start_frame, int(end_frame))
        if end_frame <= start_frame:
            return []
        frame_indices = list(range(start_frame, end_frame, stride))
        capped_num_frames = min(len(frame_indices), max_num_frames or len(frame_indices))
        num_video_frames = 1 + 4 * ((capped_num_frames - 1) // 4) if align_to_tokenizer else capped_num_frames
        start_index = random.randint(0, len(frame_indices) - num_video_frames) if randomize_start else 0
        return frame_indices[start_index : start_index + num_video_frames]

    @staticmethod
    def _clamp_frame_indices_to_video_length(
        frame_indices: list[int], video_num_frames: int, *, align_to_tokenizer: bool = True
    ) -> list[int]:
        if video_num_frames <= 0:
            return []
        valid_frame_indices = [frame_index for frame_index in frame_indices if 0 <= frame_index < video_num_frames]
        if not valid_frame_indices:
            return []
        num_video_frames = (
            1 + 4 * ((len(valid_frame_indices) - 1) // 4) if align_to_tokenizer else len(valid_frame_indices)
        )
        return valid_frame_indices[:num_video_frames]

    @staticmethod
    def _fps_matches(lhs: float, rhs: float) -> bool:
        # Decoders can report nominally identical 30 FPS RGB/control videos as
        # 30.0 vs. 30.04 due to container timestamp jitter. Treat small drift
        # as equivalent while still catching genuinely misaligned streams.
        return abs(lhs - rhs) <= max(0.5, 0.02 * max(abs(lhs), abs(rhs)))

    def _sample_structured_caption_chunk(
        self,
        meta_dict: dict[str, Any],
        *,
        stride: int,
        preferred_chunk_key: str | None = None,
        preferred_start_frame: int | None = None,
    ) -> dict[str, Any] | None:
        """Sample a chunk from a per-camera ``caption_structured`` JSON caption.

        Expected schema mirrors ``TextTransformForVideoJsonCaption``:
        ``caption_structured`` is a JSON dict of chunks, each containing
        ``caption`` (itself a JSON string or dict), ``start_frame`` and
        ``end_frame``. A direct structured-caption dict is also accepted as a
        best-effort fallback, using ``t2w_windows`` only for frame boundaries.
        """
        caption_structured = _json_dict_or_none(meta_dict.get("caption_structured"))
        if caption_structured is None:
            return None

        chunk_options: list[tuple[int, str, dict[str, Any], list[int]]] = []
        saw_chunk_map = False
        for index, (chunk_key, chunk) in enumerate(caption_structured.items()):
            if not isinstance(chunk, dict):
                continue
            if "caption" not in chunk and "caption_json" not in chunk:
                continue
            saw_chunk_map = True
            try:
                start_frame = int(chunk["start_frame"])
                end_frame = int(chunk["end_frame"])
            except (KeyError, TypeError, ValueError):
                continue
            frame_indices = self._frame_indices_for_caption_range(
                start_frame,
                end_frame,
                stride,
                max_num_frames=self.num_frames,
                randomize_start=self.randomize_caption_window_start,
                align_to_tokenizer=not self.defer_camera_alignment,
            )
            if frame_indices and (not self.require_full_num_frames or len(frame_indices) == self.expected_num_frames):
                chunk_options.append((index, chunk_key, chunk, frame_indices))

        if chunk_options:
            if preferred_chunk_key is not None:
                preferred = [option for option in chunk_options if option[1] == preferred_chunk_key]
                if not preferred and preferred_start_frame is not None:
                    preferred = [
                        option for option in chunk_options if int(option[2]["start_frame"]) == preferred_start_frame
                    ]
                if not preferred:
                    return None
                index, chunk_key, chunk, frame_indices = preferred[0]
            else:
                index, chunk_key, chunk, frame_indices = random.choice(chunk_options)
            caption_payload = chunk.get("caption_json", chunk.get("caption", {}))
            return {
                "caption": _caption_payload_to_object(caption_payload),
                "chunk_key": chunk_key,
                "chunk_index": index,
                "start_frame": int(chunk["start_frame"]),
                "end_frame": int(chunk["end_frame"]),
                "stride": stride,
                "frame_indices": frame_indices,
            }

        if saw_chunk_map:
            return None

        # Fallback: ``caption_structured`` itself is the structured payload,
        # not a chunk map. Use a legacy t2w window, if present, for frame bounds.
        windows = meta_dict.get("t2w_windows")
        window_index = 0
        window = None
        if isinstance(windows, list) and windows:
            window_index = random.choice(list(range(len(windows))))
            window = windows[window_index]
        start_frame = int(window.get("start_frame", 0)) if isinstance(window, dict) else 0
        end_frame = (
            int(window.get("end_frame", start_frame + self.num_frames * stride))
            if isinstance(window, dict)
            else start_frame + self.num_frames * stride
        )
        frame_indices = self._frame_indices_for_caption_range(
            start_frame,
            end_frame,
            stride,
            max_num_frames=self.num_frames,
            randomize_start=self.randomize_caption_window_start,
            align_to_tokenizer=not self.defer_camera_alignment,
        )
        if not frame_indices or (self.require_full_num_frames and len(frame_indices) != self.expected_num_frames):
            return None
        return {
            "caption": caption_structured,
            "chunk_key": "caption_structured",
            "chunk_index": window_index,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "stride": stride,
            "frame_indices": frame_indices,
        }

    def __call__(self, data: dict[str, Any]) -> dict[str, Any] | None:  # noqa: C901
        """Extract synchronized frames and captions from all camera views."""

        extract_frames = self._extract_frames
        if self.minimum_num_frames is not None:
            extract_frames = partial(extract_frames, minimum_num_frames=self.minimum_num_frames)
        if self.defer_camera_alignment:
            extract_frames = partial(extract_frames, align_to_tokenizer=False)
        chunk_index: int | None = None
        extracted_frame_ids: list[int] | None = None
        video_fps: float | None = None
        camera_source_fps: list[float] = []
        full_video_caption_chunk = self._sample_full_video_caption_chunk(data)
        if self.full_video_caption_config is not None and full_video_caption_chunk is None:
            return None
        frame_stride = (
            int(full_video_caption_chunk["stride"])
            if full_video_caption_chunk is not None
            else self.fps_downsample_factor
        )
        selected_frame_indices = (
            list(full_video_caption_chunk["frame_indices"]) if full_video_caption_chunk is not None else None
        )

        captions: list[Any] = []
        multiview_frames: list[torch.Tensor] = []
        multiview_control: list[torch.Tensor] = []
        view_indices: list[int] = []
        view_indices_selection: list[int] = []
        camera_keys_selection: list[str] = []
        original_sizes: list[list[int]] = []

        structured_caption_key: str | None = None
        structured_caption_start_frame: int | None = None
        use_structured_caption = "caption_structured" in self.caption_probability
        camera_order = list(self.camera_order)
        if self.camera_order_from_sample:
            camera_order = [
                camera_name
                for camera_name in camera_order
                if data.get(self.camera_video_key_mapping[camera_name]) is not None
            ]
        if self.camera_sampler is not None:
            selected_camera_names = set(self.camera_sampler.sample())
            # Preserve canonical order so prompt and video view slots have stable identities.
            camera_order = [camera_name for camera_name in camera_order if camera_name in selected_camera_names]
        if not camera_order:
            # Only reachable under ``camera_order_from_sample``: otherwise the order is
            # fixed and non-empty, and the sampler draws at least one camera from it.
            _drop_sample(self, "no camera in camera_order carries video media", data)
            return None
        use_single_caption = (
            self.single_caption_camera_name is not None
            and self.single_caption_camera_name in camera_order
            and (
                self.single_caption_probability >= 1.0
                or (self.single_caption_probability > 0.0 and random.random() < self.single_caption_probability)
            )
        )

        extract_control = self.camera_control_key_mapping is not None
        if extract_control and self.control_optional:
            cameras_with_control = [
                camera_name
                for camera_name in camera_order
                if data.get(self.camera_control_key_mapping[camera_name]) is not None
            ]
            if cameras_with_control and len(cameras_with_control) != len(camera_order):
                _drop_sample(
                    self,
                    f"control media present for only {len(cameras_with_control)}/{len(camera_order)} cameras",
                    data,
                )
                return None
            extract_control = bool(cameras_with_control)

        for view_index, camera_name in enumerate(camera_order):
            video_key = self.camera_video_key_mapping[camera_name]
            if use_single_caption:
                assert self.single_caption_camera_name is not None
                meta_key = self.camera_caption_key_mapping[self.single_caption_camera_name]
            else:
                meta_key = self.camera_caption_key_mapping[camera_name]
            meta_dict = data.get(meta_key)
            if not isinstance(meta_dict, dict):
                _drop_sample(self, f"missing caption metadata {meta_key!r}", data)
                return None

            structured_caption_chunk = None
            window = None
            window_start_frame = None
            window_end_frame = None
            if full_video_caption_chunk is None and use_structured_caption:
                structured_caption_chunk = self._sample_structured_caption_chunk(
                    meta_dict,
                    stride=frame_stride,
                    preferred_chunk_key=structured_caption_key,
                    preferred_start_frame=structured_caption_start_frame,
                )
                if structured_caption_chunk is None:
                    log.warning(
                        f"ExtractFramesAndCaptions: missing usable caption_structured in {meta_key}. "
                        f"url: {data.get('__url__')}, key: {data.get('__key__')}",
                        rank0_only=False,
                    )
                    return None
                if structured_caption_key is None:
                    structured_caption_key = structured_caption_chunk["chunk_key"]
                    structured_caption_start_frame = int(structured_caption_chunk["start_frame"])
                    chunk_index = int(structured_caption_chunk["chunk_index"])
                    selected_frame_indices = list(structured_caption_chunk["frame_indices"])
            elif full_video_caption_chunk is None:
                t2w_windows = meta_dict.get("t2w_windows")
                if not isinstance(t2w_windows, list) or not t2w_windows:
                    _drop_sample(self, f"missing nonempty t2w_windows in {meta_key!r}", data)
                    return None
                if chunk_index is None:
                    chunk_index = random.choice(list(range(len(t2w_windows))))
                if chunk_index >= len(t2w_windows) or not isinstance(t2w_windows[chunk_index], dict):
                    _drop_sample(self, f"missing t2w window index {chunk_index} in {meta_key!r}", data)
                    return None
                window = t2w_windows[chunk_index]
                try:
                    window_start_frame = int(window["start_frame"])
                    if self.randomize_caption_window_start:
                        window_end_frame = int(window["end_frame"])
                except (KeyError, TypeError, ValueError):
                    _drop_sample(self, f"invalid frame range in t2w window index {chunk_index} in {meta_key!r}", data)
                    return None

            # --- caption ---
            if full_video_caption_chunk is not None:
                caption = ""
                if not use_single_caption or camera_name == self.single_caption_camera_name:
                    caption = full_video_caption_chunk["caption"]
            elif structured_caption_chunk is not None:
                caption = ""
                if not use_single_caption or camera_name == self.single_caption_camera_name:
                    caption = structured_caption_chunk["caption"]
            else:
                assert window is not None
                caption_style = random.choices(
                    list(self.caption_probability),
                    weights=list(self.caption_probability.values()),
                    k=1,
                )[0]
                caption = ""
                if not use_single_caption or camera_name == self.single_caption_camera_name:
                    caption = window.get(caption_style)
                    if not isinstance(caption, (str, dict)):
                        _drop_sample(self, f"missing caption field {caption_style!r} in {meta_key!r}", data)
                        return None

            assert isinstance(caption, (str, dict)), f"Caption is not a string or dict: {caption}"
            if self.first_caption_paragraph_only:
                if not isinstance(caption, str):
                    _drop_sample(self, "opening-paragraph captions require plain-text Tier-1 content", data)
                    return None
                caption = first_caption_paragraph(caption)
            if self.separate_view_text_tokenization and not _has_meaningful_caption_text(caption):
                _drop_sample(self, f"missing meaningful caption for camera={camera_name} in {meta_key!r}", data)
                return None
            captions.append(
                format_view_caption(
                    caption=caption,
                    camera_name=camera_name,
                    view_index=view_index,
                    # Separate-view captions receive one sampled-rig prefix after camera
                    # selection, replacing this legacy one-camera direction sentence.
                    add_view_prefix=self.add_view_prefix_to_caption and not self.separate_view_text_tokenization,
                    camera_prefixes=self.camera_prefix_mapping,
                    camera_attributes=self.camera_attribute_mapping,
                )
            )

            # --- frames ---
            if extracted_frame_ids is not None:
                frame_indices = list(extracted_frame_ids)
            elif selected_frame_indices is not None:
                frame_indices = list(selected_frame_indices)
            else:
                assert window is not None
                assert window_start_frame is not None
                if self.randomize_caption_window_start:
                    assert window_end_frame is not None
                    frame_indices = self._frame_indices_for_caption_range(
                        window_start_frame,
                        window_end_frame,
                        self.fps_downsample_factor,
                        max_num_frames=self.num_frames,
                        randomize_start=True,
                        align_to_tokenizer=not self.defer_camera_alignment,
                    )
                    if not frame_indices:
                        _drop_sample(self, f"empty frame range in t2w window index {chunk_index} in {meta_key!r}", data)
                        return None
                else:
                    num_frames = self.expected_num_frames if self.require_full_num_frames else self.num_frames
                    frame_end = window_start_frame + num_frames * self.fps_downsample_factor
                    frame_indices = list(range(window_start_frame, frame_end, self.fps_downsample_factor))
                # Legacy per-camera windows can have different start frames; anchor all views to the first one.
                selected_frame_indices = list(frame_indices)
            try:
                if self.require_full_num_frames and len(frame_indices) != self.expected_num_frames:
                    _drop_sample(self, f"caption window does not cover {self.expected_num_frames} frames", data)
                    return None
                frames, original_fps, original_hw = extract_frames(
                    data[video_key], frame_indices, self.resolution_hw
                )  # [T,C,H,W]
                # Decoders may shorten frame_indices to the available media.
                # A fixed RGB rollout must still reject incomplete coverage.
                if self.require_full_num_frames and len(frame_indices) != self.expected_num_frames:
                    _drop_sample(self, f"video does not cover {self.expected_num_frames} frames", data)
                    return None
            except Exception as e:
                print(f"Failed to extract frames: {e} | __key__={data['__key__']} | __url__={data['__url__']}")
                return None
            expected_num_frames = len(frame_indices)
            assert len(frames) == expected_num_frames, f"Expected {expected_num_frames} frames, got {len(frames)}"
            multiview_frames.append(frames)

            # consistency checks
            if extracted_frame_ids is None:
                extracted_frame_ids = frame_indices
            elif (
                self.defer_camera_alignment
                and not self.require_full_num_frames
                and frame_indices
                and extracted_frame_ids[: len(frame_indices)] == frame_indices
            ):
                # Decoders may differ at the tail. Keep a common prefix before
                # joint sensor alignment; frame offsets and interior gaps still fail.
                extracted_frame_ids = frame_indices
            elif frame_indices != extracted_frame_ids:
                _drop_sample(
                    self,
                    f"extracted frame IDs do not match across cameras: {extracted_frame_ids} vs {frame_indices}",
                    data,
                )
                return None

            if video_fps is None:
                video_fps = original_fps
            elif not self._fps_matches(video_fps, original_fps):
                _drop_sample(self, f"video FPS does not match across cameras: {video_fps} vs {original_fps}", data)
                return None
            original_sizes.append(list(original_hw))
            if SOURCE_GEOMETRY_KEY in data:
                camera_source_fps.append(original_fps)

            # --- optional control frames ---
            if extract_control:
                control_key = self.camera_control_key_mapping[camera_name]
                control_frame_indices = list(frame_indices)
                try:
                    control_frames, control_fps, _ = extract_frames(
                        data[control_key], control_frame_indices, self.resolution_hw
                    )
                except Exception as e:
                    print(
                        f"Failed to extract control frames: {e} | __key__={data['__key__']} | __url__={data['__url__']}"
                    )
                    return None
                if (
                    self.defer_camera_alignment
                    and not self.require_full_num_frames
                    and control_frame_indices
                    and frame_indices[: len(control_frame_indices)] == control_frame_indices
                ):
                    extracted_frame_ids = control_frame_indices
                    expected_num_frames = len(control_frame_indices)
                elif control_frame_indices != frame_indices:
                    log.warning(
                        f"Control frame range does not cover RGB frame range for camera={camera_name}, "
                        f"video_key={video_key}, control_key={control_key}, sample_key={data.get('__key__')}, "
                        f"url={data.get('__url__')}",
                        rank0_only=False,
                    )
                    return None
                if len(control_frames) != expected_num_frames:
                    _drop_sample(
                        self, f"expected {expected_num_frames} control frames, got {len(control_frames)}", data
                    )
                    return None
                if not self._fps_matches(control_fps, original_fps):
                    _drop_sample(
                        self,
                        f"Control FPS {control_fps} does not match video FPS {original_fps} "
                        f"for camera={camera_name}, video_key={video_key}, control_key={control_key}, "
                        f"sample_key={data.get('__key__')}, url={data.get('__url__')}",
                        data,
                    )
                    return None
                # A gray depth clip decodes to one channel, which single-view VT2V widens to
                # three before the model sees it (``_normalize_depth_frames``). Do the same
                # here: the per-camera VAE encodes 3-channel pixels, and the frame-alignment
                # check downstream compares T,H,W only, so a 1-channel control would
                # otherwise reach the tokenizer unnoticed. Staying in uint8 rather than
                # decoding float32 and scaling is deliberate -- that VAE requires uint8
                # pixels, so a deeper source is quantized to 8 bits either way.
                control_channels = control_frames.shape[1]
                if control_channels == 1:
                    control_frames = control_frames.expand(-1, 3, -1, -1)  # [T,3,H,W]
                elif control_channels != 3:
                    _drop_sample(
                        self,
                        f"control video for camera={camera_name} decoded {control_channels} channels, "
                        "expected one or three",
                        data,
                    )
                    return None
                multiview_control.append(control_frames)

            view_indices.extend([self.camera_view_mapping[camera_name]] * expected_num_frames)
            view_indices_selection.append(self.camera_view_mapping[camera_name])
            camera_keys_selection.append(camera_name)

        if self.defer_camera_alignment:
            assert extracted_frame_ids is not None
            common_frames = len(extracted_frame_ids)
            multiview_frames = [frames[:common_frames] for frames in multiview_frames]  # each [T_common,C,H,W]
            multiview_control = [frames[:common_frames] for frames in multiview_control]  # each [T_common,C,H,W]
            view_indices = [view for view in view_indices_selection for _ in range(common_frames)]

        # Determine front-cam position in the sample
        front_cam_view_idx_sample_position = torch.tensor(
            camera_order.index(self.single_caption_camera_name)
            if self.single_caption_camera_name and self.single_caption_camera_name in camera_order
            else -1,
            dtype=torch.int64,
        )
        if (
            use_single_caption
            and int(front_cam_view_idx_sample_position.item()) >= 0
            and not self.add_view_prefix_to_caption
        ):
            captions = [captions[int(front_cam_view_idx_sample_position.item())]]

        # Drop only validated description bodies, independently for the sampled views.
        # Formatting below still adds the same rig/camera headers. Sample-level CFG
        # dropout remains a separate decision in the downstream text tokenizer.
        if self.per_view_caption_dropout_rate > 0:
            captions = ["" if random.random() < self.per_view_caption_dropout_rate else caption for caption in captions]

        # Legacy training serializes the ordered views into one prompt. Separate-view
        # tokenization intentionally keeps the list so the tokenizer can emit one ragged
        # tensor per local camera view. The list order matches the camera-major video layout.
        ai_caption = (
            format_separate_view_captions(
                captions,
                camera_names=camera_order,
                add_camera_rig_prefix=self.add_view_prefix_to_caption,
                camera_attributes=self.camera_attribute_mapping,
            )
            if self.separate_view_text_tokenization
            else format_multiview_caption(
                captions,
                use_two_pass_format=self.use_two_pass_caption_format,
            )
        )

        assert video_fps is not None
        # Use tolerance-based check to handle nominal FPS jitter (e.g. 29.97 or
        # 30.0857 reported for nominal 30 FPS). The modulo is measured in FPS
        # units, so allow a small absolute tolerance around integer multiples.
        fps_stride_tolerance = 0.1
        _mod = video_fps % frame_stride
        if not (_mod < fps_stride_tolerance or abs(_mod - frame_stride) < fps_stride_tolerance):
            # Sample has FPS genuinely not divisible by factor — drop it.
            log.warning(
                f"Dropping sample {data.get('__key__', '?')}: "
                f"video_fps={video_fps} not divisible by frame_stride={frame_stride}"
            )
            return None
        fps = video_fps / frame_stride

        target_h, target_w = self.resolution_hw
        sample: dict[str, Any] = {
            "__key__": data["__key__"],
            "__url__": data["__url__"],
            "video": rearrange(torch.cat(multiview_frames, dim=0), "t c h w -> c t h w"),
            "ai_caption": ai_caption,
            "view_indices": torch.tensor(view_indices, dtype=torch.int64),
            "fps": torch.tensor(fps, dtype=torch.float64),
            "conditioning_fps": torch.tensor(fps, dtype=torch.float32),
            "chunk_index": torch.tensor(-1 if chunk_index is None else chunk_index, dtype=torch.int64),
            "frame_indices": torch.tensor(extracted_frame_ids, dtype=torch.int64),
            "num_video_frames_per_view": torch.tensor(len(extracted_frame_ids), dtype=torch.int64),
            "view_indices_selection": torch.tensor(view_indices_selection, dtype=torch.int64),
            "camera_keys_selection": camera_keys_selection,
            "sample_n_views": torch.tensor(len(camera_keys_selection), dtype=torch.int64),
            "padding_mask": torch.zeros((1, *self.resolution_hw), dtype=torch.float32),
            "image_size": torch.tensor([target_h, target_w, target_h, target_w], dtype=torch.float),
            "ref_cam_view_idx_sample_position": torch.tensor(-1, dtype=torch.int64),
            "front_cam_view_idx_sample_position": front_cam_view_idx_sample_position,
            "original_hw": torch.tensor(original_sizes, dtype=torch.int64),
        }
        if extract_control:
            sample[self.control_output_key] = rearrange(torch.cat(multiview_control, dim=0), "t c h w -> c t h w")
        if full_video_caption_chunk is not None:
            sample["sampled_caption_style"] = full_video_caption_chunk["caption_key"]
            sample["sampled_chunk_key"] = full_video_caption_chunk["chunk_key"]
        if SOURCE_GEOMETRY_KEY in data:
            sample[SOURCE_FPS_KEY] = camera_source_fps
        # Pass through any configured keys verbatim.
        for pk in self.passthrough_keys:
            if pk in data:
                sample[pk] = data[pk]
        return sample

    @staticmethod
    def _extract_frames(
        video: bytes,
        frame_indices: list[int],
        resolution_hw: tuple[int, int],
        *,
        align_to_tokenizer: bool = True,
        minimum_num_frames: int | None = None,
    ) -> tuple[torch.Tensor, float, tuple[int, int]]:  # frames: [T,C,H,W]
        """Decode *frame_indices* from raw video bytes and resize.

        ``frame_indices`` is updated in place to the valid prefix used for
        decoding. Joint windows defer VAE alignment until sensor alignment.
        """
        video_reader = TorchCodecVideoReader(video)
        clamped_frame_indices = ExtractFramesAndCaptions._clamp_frame_indices_to_video_length(
            frame_indices,
            video_reader.metadata.num_frames,
            align_to_tokenizer=align_to_tokenizer,
        )
        if not clamped_frame_indices:
            raise ValueError(f"No valid frames to extract from video with {video_reader.metadata.num_frames} frames.")
        # Exact joint windows already reject this sample downstream. Check the
        # same necessary bound before decoding hundreds of frames across views.
        # Keep all available slack for the later RGB/LiDAR start alignment.
        if minimum_num_frames is not None and len(clamped_frame_indices) < minimum_num_frames:
            raise ValueError(
                f"Video window has {len(clamped_frame_indices)} frames; at least {minimum_num_frames} are required."
            )
        frame_indices[:] = clamped_frame_indices
        frames = video_reader.get_frames_tchw_uint8(frame_indices)  # [T,C,H,W]
        fps = video_reader.metadata.average_fps
        original_h, original_w = frames.shape[-2:]
        target_h, target_w = resolution_hw
        frames = resize_and_center_crop_tensor(frames, target_h=target_h, target_w=target_w)  # [T,C,target_h,target_w]
        return (
            frames,
            fps,
            (original_h, original_w),
        )


def _drop_sample(augmentor: Augmentor, reason: str, data: dict[str, Any]) -> None:
    """Log why *augmentor* is dropping a sample, including its provenance."""
    log.warning(
        f"{type(augmentor).__name__}: {reason}; sample_key={data.get('__key__')}, url={data.get('__url__')}",
        rank0_only=False,
    )


class _ExtractRangeVideo(Augmentor):
    """Decode one MADS nested rangemap tar over the sampled source-frame interval.

    Reads the interval from ``frame_indices`` and writes ``{prefix}_range`` plus
    the source-frame IDs actually found as ``{prefix}_frame_indices``; a clip
    that ends early yields fewer frames than were requested.

    Both tokenizer versions start from the same 3-channel metric prepare, and
    range and intensity stay metric until the shared ``metric_lidar_to_network``
    step inside encode. Azimuth layout comes from ``range_projection``.
    """

    def __init__(
        self,
        *,
        source_key: str,
        prefix: str,
        with_intensity: bool,
        tokenizer_v1: bool = False,
        range_projection: LidarRangeProjectionConfig | None = None,
    ) -> None:
        super().__init__(input_keys=[], output_keys=[])
        self.source_key = source_key
        self.prefix = prefix
        self.with_intensity = with_intensity
        self.tokenizer_v1 = tokenizer_v1
        self.range_projection = range_projection or transfer_range_projection(tokenizer_v1=tokenizer_v1)

    @staticmethod
    def _expected_clip_key(data: dict[str, Any]) -> str | tuple[str, ...]:
        sample_key = str(data["__key__"])
        compat_clip_key = str(data.get(RANGEMAP_COMPAT_CLIP_KEY) or "")
        if not compat_clip_key or compat_clip_key == sample_key:
            return sample_key
        return (sample_key, compat_clip_key)

    def __call__(self, data: dict[str, Any]) -> dict[str, Any] | None:
        raw_tar = data.pop(self.source_key, None)
        frame_indices = data.get("frame_indices")
        if not isinstance(raw_tar, bytes):
            _drop_sample(self, f"missing bytes key {self.source_key!r}", data)
            return None
        if not isinstance(frame_indices, torch.Tensor) or frame_indices.numel() == 0:
            _drop_sample(self, "missing nonempty frame_indices", data)
            return None
        try:
            range_video, sweep_ids = load_prepared_lidar_video_3ch(
                raw_tar,
                min_source_frame_id=int(frame_indices[0].item()),
                max_source_frame_id=int(frame_indices[-1].item()),
                expected_clip_key=self._expected_clip_key(data),
                range_projection=self.range_projection,
                with_intensity=self.with_intensity,
            )
            intensity = range_video[1:2] if self.with_intensity else None
        except Exception as error:
            _drop_sample(self, f"decode failed ({error})", data)
            return None
        data[f"{self.prefix}_range"] = range_video
        data[f"{self.prefix}_frame_indices"] = torch.tensor(sweep_ids, dtype=torch.int64)
        if intensity is not None:
            data[f"{self.prefix}_intensity"] = intensity
        return data


class ExtractLidarRange(_ExtractRangeVideo):
    """Decode the LiDAR rangemap tar into ``lidar_range`` and ``lidar_intensity``."""

    def __init__(
        self,
        lidar_key: str = LIDAR_RANGEMAP_KEY,
        *,
        tokenizer_v1: bool = False,
        range_projection: LidarRangeProjectionConfig | None = None,
    ) -> None:
        super().__init__(
            source_key=lidar_key,
            prefix="lidar",
            with_intensity=True,
            tokenizer_v1=tokenizer_v1,
            range_projection=range_projection,
        )


class ExtractHdmapRange(_ExtractRangeVideo):
    """Decode the HD-map rangemap tar into ``hdmap_range``.

    HD-map tars have no return intensity. The shared 3-channel prepare gives
    every occupied ray unit intensity and leaves range metric until encode.
    """

    def __init__(
        self,
        hdmap_key: str = HDMAP_RANGEMAP_KEY,
        *,
        tokenizer_v1: bool = False,
        range_projection: LidarRangeProjectionConfig | None = None,
    ) -> None:
        super().__init__(
            source_key=hdmap_key,
            prefix="hdmap",
            with_intensity=False,
            tokenizer_v1=tokenizer_v1,
            range_projection=range_projection,
        )


class NormalizeNestedRangemapKeys(Augmentor):
    """Undo WebDataset's slash-to-underscore flattening of the nested range keys."""

    _FLATTENED_TO_SAMPLE_KEY = {
        LIDAR_RANGEMAP_STORAGE_KEY.replace("/", "_"): LIDAR_RANGEMAP_KEY,
        HDMAP_RANGEMAP_STORAGE_KEY.replace("/", "_"): HDMAP_RANGEMAP_KEY,
    }

    def __init__(self) -> None:
        super().__init__(input_keys=[], output_keys=[])

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        for flattened_key, sample_key in self._FLATTENED_TO_SAMPLE_KEY.items():
            if flattened_key in data:
                data[sample_key] = data.pop(flattened_key)
        return data


class AlignLidarClips(Augmentor):
    """Narrow the control and target rangemaps to the sweeps present in both.

    The two tars are decoded independently and can cover different sweeps. Trim
    here rather than at packing time so the caption augmentors in between
    describe the clip that training actually sees.
    """

    def __init__(self) -> None:
        super().__init__(input_keys=[], output_keys=[])

    def __call__(self, data: dict[str, Any]) -> dict[str, Any] | None:
        hdmap = data.get("hdmap_range")
        lidar = data.get("lidar_range")
        hdmap_ids = data.get("hdmap_frame_indices")
        lidar_ids = data.get("lidar_frame_indices")
        if not all(isinstance(value, torch.Tensor) for value in (hdmap, lidar, hdmap_ids, lidar_ids)):
            _drop_sample(self, "missing hdmap/lidar tensors", data)
            return None
        try:
            hdmap, lidar, shared_ids = align_prepared_range_videos(
                hdmap,
                [int(frame_id) for frame_id in hdmap_ids.tolist()],
                lidar,
                [int(frame_id) for frame_id in lidar_ids.tolist()],
            )
        except Exception as error:
            _drop_sample(self, f"sweep alignment failed ({error})", data)
            return None
        shared_frame_indices = torch.tensor(shared_ids, dtype=torch.int64)
        data.update(
            hdmap_range=hdmap,
            lidar_range=lidar,
            hdmap_frame_indices=shared_frame_indices,
            lidar_frame_indices=shared_frame_indices,
        )
        return data


class DescribeLidarTargetClip(Augmentor):
    """Expose the LiDAR clip as ``video`` so the caption augmentors can read it.

    ``DurationFPSTextTimeStamps`` and ``ResolutionTextInfo`` run before the
    transfer packing builds the real ``video`` list, and both describe whatever
    ``video``/``image_size`` hold; point them at the target clip.
    """

    def __init__(self) -> None:
        super().__init__(input_keys=[], output_keys=[])

    def __call__(self, data: dict[str, Any]) -> dict[str, Any] | None:
        lidar_range = data.get("lidar_range")
        if not isinstance(lidar_range, torch.Tensor):
            _drop_sample(self, "missing lidar_range", data)
            return None
        height, width = int(lidar_range.shape[-2]), int(lidar_range.shape[-1])
        data["video"] = lidar_range
        data["image_size"] = torch.tensor([height, width, height, width], dtype=torch.float)
        data["num_video_frames_per_view"] = torch.tensor(lidar_range.shape[1], dtype=torch.int64)
        return data


def _select_structured_caption_window(meta_dict: dict[str, Any]) -> dict[str, Any] | None:
    """Pick one ``caption_structured`` chunk and report its caption and frame bounds.

    MADS v1 metas hold a chunk map keyed by chunk id, each entry carrying a
    caption plus frame bounds. Chunks without both a caption and a start frame
    are unusable, so this returns ``None`` when none qualify.
    """
    caption_structured = _json_dict_or_none(meta_dict.get("caption_structured"))
    if caption_structured is None:
        return None

    chunk_options: list[tuple[int, int, dict[str, Any]]] = []
    for index, chunk in enumerate(caption_structured.values()):
        if not isinstance(chunk, dict) or ("caption" not in chunk and "caption_json" not in chunk):
            continue
        try:
            start_frame = int(chunk["start_frame"])
        except (KeyError, TypeError, ValueError):
            continue
        chunk_options.append((index, start_frame, chunk))
    if not chunk_options:
        return None

    chunk_index, start_frame, chunk = random.choice(chunk_options)
    caption_payload = chunk.get("caption_json", chunk.get("caption", {}))
    return {
        "caption": _caption_payload_to_object(caption_payload),
        "start_frame": start_frame,
        "end_frame": chunk.get("end_frame"),
        "chunk_index": chunk_index,
    }


class ExtractLidarTransferWindow(Augmentor):
    """Sample front-wide captions/frame windows without decoding camera video.

    Used by lidar-transfer training where control/target are rangemaps and text
    still comes from ``metas_camera_front_wide_120fov``.

    Rangemap tars hold one sweep per ``lidar_fps`` but key their members by
    source-frame ID on the ``source_frame_fps`` camera clock, so a 10Hz sweep
    stream inside a 30FPS clip lands on IDs 0, 3, 6, ... The emitted
    ``frame_indices`` therefore stay in source-frame space. By default, use
    every sweep within the caption's half-open frame bounds, rejecting windows
    with more than ``num_frames`` selected sweeps before allocation. Set
    ``native_caption_window=False`` for a fixed ``num_frames`` sweeps starting
    near the caption's start. ``fps_downsample_factor`` strides over sweeps.
    """

    def __init__(
        self,
        *,
        num_frames: int,
        fps_downsample_factor: int,
        caption_probability: dict[str, float],
        camera_caption_key_mapping: dict[CameraKeyType, str],
        caption_camera_name: CameraKeyType = DEFAULT_LIDAR_TRANSFER_CAPTION_CAMERA,
        passthrough_keys: Optional[list[str]] = None,
        source_frame_fps: float = 30.0,
        lidar_fps: float = 10.0,
        native_caption_window: bool = True,
    ) -> None:
        super().__init__([], {})
        if num_frames < 1 or fps_downsample_factor < 1:
            raise ValueError(
                f"num_frames and fps_downsample_factor must be positive, got {num_frames} and {fps_downsample_factor}"
            )
        if source_frame_fps < lidar_fps or lidar_fps <= 0:
            raise ValueError(f"need source_frame_fps >= lidar_fps > 0, got {source_frame_fps} and {lidar_fps}")
        if caption_camera_name not in camera_caption_key_mapping:
            raise ValueError(f"caption camera {caption_camera_name!r} missing from camera_caption_key_mapping")
        self.num_frames = num_frames
        self.native_caption_window: bool = native_caption_window
        self.fps_downsample_factor = fps_downsample_factor
        self.lidar_fps = lidar_fps
        self.source_frames_per_sweep = round(source_frame_fps / lidar_fps)
        self.caption_probability = caption_probability
        self.camera_caption_key_mapping = camera_caption_key_mapping
        self.caption_camera_name = caption_camera_name
        self.passthrough_keys = list(passthrough_keys) if passthrough_keys else []

    def _select_caption(self, meta_dict: dict[str, Any], data: dict[str, Any]) -> tuple[Any, str, int, int, Any] | None:
        """Return the caption, style, chunk index, and source-camera frame bounds."""
        if "caption_structured" in self.caption_probability:
            window = _select_structured_caption_window(meta_dict)
            if window is None:
                _drop_sample(self, "no usable caption_structured chunk", data)
                return None
            return (
                window["caption"],
                "caption_structured",
                window["chunk_index"],
                window["start_frame"],
                window["end_frame"],
            )

        windows = meta_dict.get("t2w_windows")
        if not windows:
            _drop_sample(self, "empty t2w_windows", data)
            return None
        chunk_index = random.randrange(len(windows))
        window = windows[chunk_index]
        caption_style = random.choices(
            list(self.caption_probability),
            weights=list(self.caption_probability.values()),
        )[0]
        try:
            start_frame = int(window["start_frame"])
        except (KeyError, TypeError, ValueError):
            _drop_sample(self, "t2w window without a usable start_frame", data)
            return None
        return window.get(caption_style, ""), caption_style, chunk_index, start_frame, window.get("end_frame")

    def __call__(self, data: dict[str, Any]) -> dict[str, Any] | None:
        meta_key = self.camera_caption_key_mapping[self.caption_camera_name]
        if meta_key not in data:
            _drop_sample(self, f"missing caption metas {meta_key!r}", data)
            return None
        selected = self._select_caption(data[meta_key], data)
        if selected is None:
            return None
        caption, caption_style, chunk_index, camera_start_frame, camera_end_frame = selected
        if not isinstance(caption, (str, dict)):
            _drop_sample(self, f"unsupported caption type {type(caption)}", data)
            return None

        stride = self.source_frames_per_sweep * self.fps_downsample_factor
        if self.native_caption_window:
            try:
                camera_end_frame = int(camera_end_frame)
            except (TypeError, ValueError):
                _drop_sample(self, "native caption window without a usable end_frame", data)
                return None
            if not 0 <= camera_start_frame < camera_end_frame:
                _drop_sample(self, "native caption window needs 0 <= start_frame < end_frame", data)
                return None
            # Caption end frames are exclusive. Snap forward so tail captions
            # starting at frame 400 begin on sweep 402 rather than sweep 399.
            sweep_index = (camera_start_frame + self.source_frames_per_sweep - 1) // self.source_frames_per_sweep
            first_sweep_frame = sweep_index * self.source_frames_per_sweep
            # Count before allocating: malformed metadata must not build an unbounded
            # list/tensor. Reject oversized windows to keep the full caption aligned.
            num_sweeps = (camera_end_frame - first_sweep_frame + stride - 1) // stride
            if num_sweeps > self.num_frames:
                _drop_sample(self, f"native caption window exceeds {self.num_frames} LiDAR sweeps", data)
                return None
            if num_sweeps == 0:
                _drop_sample(self, "native caption window contains no LiDAR sweep", data)
                return None
            frame_indices = list(range(first_sweep_frame, camera_end_frame, stride))
        else:
            sweep_index = max(round(camera_start_frame / self.source_frames_per_sweep), 0)
            first_sweep_frame = sweep_index * self.source_frames_per_sweep
            frame_indices = list(range(first_sweep_frame, first_sweep_frame + self.num_frames * stride, stride))

        sample: dict[str, Any] = {
            "__key__": data.get("__key__"),
            "__url__": data.get("__url__"),
            "ai_caption": caption,
            "frame_indices": torch.tensor(frame_indices, dtype=torch.int64),
            "chunk_index": chunk_index,
            "fps": torch.tensor(self.lidar_fps / self.fps_downsample_factor, dtype=torch.float32),
            "num_video_frames_per_view": torch.tensor(len(frame_indices), dtype=torch.int64),
            "sample_n_views": torch.tensor(1, dtype=torch.int64),
            "selected_caption_type": caption_style,
        }
        for key in self.passthrough_keys:
            if key in data:
                sample[key] = data[key]
        return sample


class LidarTransferToTrainingFormat(Augmentor):
    """Pack HD-map + LiDAR rangemaps as transfer ``lidar=[control, target]``.

    The sample has no camera stream at all, so it carries no ``video``: the rangemaps are
    the LiDAR modality and go to the LiDAR VAE through ``lidar``, exactly as they do in the
    joint camera + LiDAR recipe. :class:`DescribeLidarTargetClip` parks the target clip
    under ``video`` earlier so the caption augmentors have something to measure, and that
    temporary key is removed here.

    Expects :class:`AlignLidarClips` to have matched up the two clips already.
    """

    def __init__(
        self,
        *,
        conditioning_config: dict[int, float] | None = None,
    ) -> None:
        super().__init__([], {})
        self.normalized_conditioning_config = _normalize_conditioning_config(conditioning_config)

    def __call__(self, data_dict: dict[str, Any]) -> dict[str, Any] | None:
        from cosmos_framework.data.generator.sequence_packing import SequencePlan

        hdmap = data_dict.get("hdmap_range")
        lidar = data_dict.get("lidar_range")
        if not all(isinstance(value, torch.Tensor) for value in (hdmap, lidar)):
            _drop_sample(self, "missing hdmap/lidar tensors", data_dict)
            return None
        if hdmap.shape[1] != lidar.shape[1]:
            _drop_sample(
                self, f"unaligned clips: {hdmap.shape[1]} control vs {lidar.shape[1]} target sweeps", data_dict
            )
            return None

        num_frames = int(lidar.shape[1])
        data_dict.update(
            lidar=[hdmap, lidar],
            num_frames=num_frames,
            dataset_name="lidar_transfer",
        )
        # Camera samples describe a rig here -- view counts, slot ids, pixel sizes. None of
        # that has a meaning for a run whose only stream is one LiDAR, and leaving a ``video``
        # key behind would make the model read the sample as a camera batch.
        for consumed_key in ("video", "image_size", "hdmap_range", "lidar_range"):
            data_dict.pop(consumed_key, None)
        data_dict.setdefault("ai_caption", "")
        data_dict.setdefault("selected_caption_type", "transfer_caption")

        num_condition_frames = 0
        if self.normalized_conditioning_config is not None:
            sampled = random.choices(
                list(self.normalized_conditioning_config),
                weights=list(self.normalized_conditioning_config.values()),
            )[0]
            num_condition_frames = min(sampled, max(num_frames - 1, 0))
        data_dict["sequence_plan"] = SequencePlan(
            has_text=True,
            has_vision=False,
            has_lidar=True,
            # The stream is [control, target], so the packer's positional convention already
            # says which item is generated: every item but the last is clean throughout, and
            # the last carries the clean prefix named here. Both items are rewound to the
            # same mRoPE origin by the packer, so the control stays frame-synced with the
            # target without needing the vision stream's shared-grid flag.
            condition_frame_indexes_lidar=list(range(num_condition_frames)),
        )
        return data_dict


def _num_camera_views(data_dict: dict[str, Any]) -> int:
    sample_n_views = data_dict.get("sample_n_views")
    if sample_n_views is None:
        return 1
    if isinstance(sample_n_views, torch.Tensor):
        return int(sample_n_views.reshape(-1)[0].item())
    return int(sample_n_views)


def _sync_sensor_window_start(data_dict: dict[str, Any], sensor_frame_id_key: str) -> tuple[int, int] | None:
    """Leading camera frames and sensor frames to drop so both streams start on one instant.

    A sensor frame exists only on the camera frame IDs where that sensor fired, so a camera
    window that opens between two of them leaves the streams up to one sensor period apart --
    an offset the shared mRoPE origin would silently absorb. Starting both on a frame ID they
    share makes the whole timeline exact rather than just the first frame. Which IDs the sensor
    carries is read off its own index tensor, so a sensor whose IDs are unevenly spaced syncs
    the same way as one on a fixed stride. Returns ``None`` when the IDs are missing or the two
    streams share no frame at all.
    """
    camera_ids = data_dict.get("frame_indices")
    sensor_ids = data_dict.get(sensor_frame_id_key)
    if not (isinstance(camera_ids, torch.Tensor) and isinstance(sensor_ids, torch.Tensor)):
        return None
    camera_index_of_id = {int(frame_id): index for index, frame_id in enumerate(camera_ids.tolist())}
    for sensor_offset, sensor_id in enumerate(sensor_ids.tolist()):
        camera_offset = camera_index_of_id.get(int(sensor_id))
        if camera_offset is not None:
            return camera_offset, sensor_offset
    return None


def _num_sensor_frames_within_camera_window(
    data_dict: dict[str, Any],
    num_camera_frames_kept: int,
    num_sensor_frames: int,
    sensor_frame_id_key: str,
) -> int:
    """Count the sensor frames that fall inside the kept camera window.

    Both streams are keyed by source-frame ID on the camera clock, so the comparison is exact
    and assumes nothing about how the sensor's IDs are spaced. ``_sync_sensor_window_start``
    has already dropped the sample if either ID tensor is missing.
    """
    camera_ids = data_dict["frame_indices"]
    sensor_ids = data_dict[sensor_frame_id_key]
    if camera_ids.numel() < num_camera_frames_kept or sensor_ids.numel() < num_sensor_frames:
        return num_sensor_frames
    last_camera_id = int(camera_ids[num_camera_frames_kept - 1].item())
    return max(int((sensor_ids[:num_sensor_frames] <= last_camera_id).sum().item()), 1)


class JointCamLidarToTrainingFormat(Augmentor):
    """Pack camera clip(s) and one LiDAR clip as two items on each of two streams.

    Emits ``video = [camera_control, camera_target]`` and ``lidar = [lidar_control,
    lidar_target]``, where the camera control is the RGB world-scenario/HD-map render and the
    LiDAR control is the HD-map rangemap. The two streams are separate modalities: each has
    its own VAE, its own projections into the sequence and its own rate — the camera's from
    ``fps``, the LiDAR's from ``OmniMoTModelConfig.lidar_fps``. They meet on one real-time
    axis, so the camera's 7.5 Hz latents and the LiDAR's 10 Hz sweeps interleave by instant.

    Camera tensors stay camera-major ``[C, V*T, H, W]``. Temporal trim and the source-frame
    sync apply to every view's slice, and ``sample_n_views`` / ``view_indices`` from
    :class:`ExtractFramesAndCaptions` are preserved so multiview FlexAttention still sees
    one view id per camera.

    Both streams are cut from the same source-frame window: ``ExtractFramesAndCaptions`` picks
    the camera frames and the rangemap extractors keep the sweeps inside that span. Since the
    sweeps land only on the frame IDs where the LiDAR fired, the window is then advanced to the
    first ID both streams carry, so item 0 of the camera and item 0 of the LiDAR are the same
    instant and the shared mRoPE origin is exact rather than approximate.

    Expects :class:`AlignLidarClips` to have matched the two rangemap clips already.

    ``lidar_dropout_rate`` withholds both LiDAR items from that fraction of samples, leaving
    them camera-only transfer samples. Absence is per-sample state the rest of the stack
    already carries: the packer skips the stream on a plan whose ``has_lidar`` is False, and a
    batch that drew no LiDAR at all takes the loss's zero-weighted probe branch, which exists
    so FSDP ranks may disagree about having LiDAR without the gradient reduce hanging.

    ``control_dropout_rate`` independently withholds the WSM and HD-map controls together,
    leaving target-only T2V/I2V samples with unchanged captions, rig, synchronized window
    and temporal conditioning. It never withholds either target.
    """

    def __init__(
        self,
        *,
        control_key: str = MULTIVIEW_CONTROL_KEY,
        camera_resolution_hw: tuple[int, int] = (720, 1280),
        camera_temporal_compression_factor: int = 4,
        lidar_fps: float = 10.0,
        max_camera_frames: int | None = None,
        fixed_camera_frames: int | None = None,
        exact_camera_frames: int | None = None,
        conditioning_config: dict[int, float] | None = None,
        rgb_view_completion_probability: float = 0.0,
        lidar_conditioning_probability: float = 0.0,
        enable_per_camera_vae_encoding: bool = False,
        lidar_dropout_rate: float = 0.0,
        lidar_prefix_dropout_rate: float = 0.0,
        control_dropout_rate: float = 0.0,
        caption_augmentors: tuple[Augmentor, ...] = (),
    ) -> None:
        super().__init__([], {})
        if camera_temporal_compression_factor < 1:
            raise ValueError(
                f"camera_temporal_compression_factor must be positive, got {camera_temporal_compression_factor}"
            )
        if lidar_fps <= 0:
            raise ValueError(f"lidar_fps must be positive, got {lidar_fps}")
        if max_camera_frames is not None and max_camera_frames < 1:
            raise ValueError(f"max_camera_frames must be positive when set, got {max_camera_frames}")
        if exact_camera_frames is not None:
            if exact_camera_frames < 1 or (exact_camera_frames - 1) % camera_temporal_compression_factor:
                raise ValueError("exact_camera_frames must lie on the camera tokenizer's 1 + stride*k grid")
            if max_camera_frames is not None and exact_camera_frames > max_camera_frames:
                raise ValueError("exact_camera_frames must not exceed max_camera_frames")
            if fixed_camera_frames is not None and exact_camera_frames != fixed_camera_frames:
                raise ValueError("exact_camera_frames and fixed_camera_frames must agree when both are set")
        # A rate of 1 would leave the joint recipe with no LiDAR to generate at all, which is
        # the camera-only recipe rather than a dropout schedule.
        if not 0.0 <= lidar_dropout_rate < 1.0:
            raise ValueError(f"lidar_dropout_rate must be in [0, 1), got {lidar_dropout_rate}")
        if not 0.0 <= lidar_prefix_dropout_rate <= 1.0:
            raise ValueError(f"lidar_prefix_dropout_rate must be in [0, 1], got {lidar_prefix_dropout_rate}")
        if not 0.0 <= control_dropout_rate <= 1.0:
            raise ValueError(f"control_dropout_rate must be in [0, 1], got {control_dropout_rate}")
        if not 0.0 <= rgb_view_completion_probability <= 1.0:
            raise ValueError(
                f"rgb_view_completion_probability must be in [0, 1], got {rgb_view_completion_probability}."
            )
        if not 0.0 <= lidar_conditioning_probability <= 1.0:
            raise ValueError(f"lidar_conditioning_probability must be in [0, 1], got {lidar_conditioning_probability}.")
        if rgb_view_completion_probability > 0 and not enable_per_camera_vae_encoding:
            raise ValueError("View completion requires enable_per_camera_vae_encoding=True.")
        self.control_key = control_key
        self.camera_resolution_hw = camera_resolution_hw
        self.camera_temporal_compression_factor = camera_temporal_compression_factor
        self.lidar_fps = lidar_fps
        self.max_camera_frames = max_camera_frames
        if fixed_camera_frames is not None:
            if fixed_camera_frames < 1 or (fixed_camera_frames - 1) % camera_temporal_compression_factor:
                raise ValueError("fixed_camera_frames must lie on the camera tokenizer's 1 + 4k grid")
            if max_camera_frames is not None and fixed_camera_frames > max_camera_frames:
                raise ValueError("fixed_camera_frames cannot exceed max_camera_frames")
        self.fixed_camera_frames: int | None = fixed_camera_frames
        self.exact_camera_frames: int | None = exact_camera_frames
        self.enable_per_camera_vae_encoding = enable_per_camera_vae_encoding
        self.rgb_view_completion_probability: float = rgb_view_completion_probability
        self.lidar_conditioning_probability: float = lidar_conditioning_probability
        self.normalized_conditioning_config = _normalize_conditioning_config(conditioning_config)
        self.lidar_dropout_rate = lidar_dropout_rate
        self.lidar_prefix_dropout_rate: float = lidar_prefix_dropout_rate
        self.control_dropout_rate: float = control_dropout_rate
        self.caption_augmentors: tuple[Augmentor, ...] = caption_augmentors

    def _camera_frames_to_keep(self, camera_frames: int) -> int:
        """Trim each camera view to the tokenizer's ``1 + 4k`` grid and the token budget.

        The recipe caps the window here rather than after encoding so every view stays
        on the same WAN-safe length.
        """
        limit = camera_frames if self.max_camera_frames is None else min(camera_frames, self.max_camera_frames)
        stride = self.camera_temporal_compression_factor
        return 1 + stride * ((limit - 1) // stride)

    def _exact_window_error(self, data_dict: dict[str, Any], camera_frames: int, lidar_frames: int) -> str | None:
        """Require the same complete, regular sensor clock for every self-forcing sample."""
        if self.exact_camera_frames is None:
            return None
        if camera_frames != self.exact_camera_frames:
            return f"exact aligned window needs {self.exact_camera_frames} camera frames, got {camera_frames}"
        camera_fps = data_dict.get("fps")
        if camera_fps is None:
            return "exact aligned window requires camera fps"
        fps = float(camera_fps.reshape(-1)[0].item() if isinstance(camera_fps, torch.Tensor) else camera_fps)
        sweep_period = round(fps / self.lidar_fps)
        if sweep_period < 1 or not ExtractFramesAndCaptions._fps_matches(fps, sweep_period * self.lidar_fps):
            return "exact aligned window requires an integral camera/LiDAR frame-rate ratio"
        expected_sweeps = 1 + (camera_frames - 1) // sweep_period
        if lidar_frames != expected_sweeps:
            return f"exact aligned window needs {expected_sweeps} LiDAR sweeps, got {lidar_frames}"
        camera_ids = data_dict["frame_indices"][:camera_frames].tolist()
        if len(camera_ids) != camera_frames:
            return "exact aligned window has incomplete camera frame IDs"
        stride = camera_ids[1] - camera_ids[0] if camera_frames > 1 else 1
        if stride < 1 or camera_ids != [camera_ids[0] + index * stride for index in range(camera_frames)]:
            return "exact aligned window has irregular camera frame IDs"
        expected_ids = [camera_ids[0] + index * stride * sweep_period for index in range(expected_sweeps)]
        for key in ("lidar_frame_indices", "hdmap_frame_indices"):
            ids = data_dict.get(key)
            if not isinstance(ids, torch.Tensor) or ids[:lidar_frames].tolist() != expected_ids:
                return f"exact aligned window has incomplete or irregular {key}"
        return None

    @staticmethod
    def _sync_window_start(data_dict: dict[str, Any]) -> tuple[int, int] | None:
        """Leading camera frames and sweeps to drop so both streams start at the same instant.

        Sweeps exist only on the frame IDs where the LiDAR fired, one every 3 frames on a
        30 fps clock, so a camera window that opens between two sweeps leaves the streams up
        to one sweep period apart — an offset the shared mRoPE origin would silently absorb.
        Starting both on a frame ID they share makes the whole timeline exact, not just the
        first frame, since from there the two periods stay in lockstep. Returns ``None`` when
        the IDs are missing or the two streams share no frame at all.
        """
        return _sync_sensor_window_start(data_dict, "lidar_frame_indices")

    @staticmethod
    def _num_sweeps_within_camera_window(
        data_dict: dict[str, Any],
        num_camera_frames_kept: int,
        num_sweeps: int,
    ) -> int:
        """Count the sweeps that fall inside the kept camera window.

        Both streams are keyed by source-frame ID on the camera clock, so the comparison is
        exact. ``_sync_window_start`` has already dropped the sample if either ID tensor is
        missing.
        """
        return _num_sensor_frames_within_camera_window(
            data_dict, num_camera_frames_kept, num_sweeps, "lidar_frame_indices"
        )

    def _sample_num_camera_condition_latents(self, camera_latent_frames: int) -> int:
        if self.normalized_conditioning_config is None:
            return 0
        sampled = random.choices(
            list(self.normalized_conditioning_config),
            weights=list(self.normalized_conditioning_config.values()),
        )[0]
        return min(sampled, max(camera_latent_frames - 1, 0))

    @staticmethod
    def _num_camera_views(data_dict: dict[str, Any]) -> int:
        return _num_camera_views(data_dict)

    def __call__(self, data_dict: dict[str, Any]) -> dict[str, Any] | None:
        from cosmos_framework.data.generator.sequence_packing import SequencePlan

        camera_target = data_dict.get("video")
        camera_control = data_dict.get(self.control_key)
        lidar_control = data_dict.get("hdmap_range")
        lidar_target = data_dict.get("lidar_range")
        items = (camera_control, camera_target, lidar_control, lidar_target)
        if not all(isinstance(value, torch.Tensor) for value in items):
            _drop_sample(self, f"missing one of video/{self.control_key}/hdmap_range/lidar_range", data_dict)
            return None
        if camera_control.shape[1] != camera_target.shape[1]:
            _drop_sample(
                self,
                f"unaligned camera clips: {camera_control.shape[1]} control vs {camera_target.shape[1]} target frames",
                data_dict,
            )
            return None
        if lidar_control.shape[1] != lidar_target.shape[1]:
            _drop_sample(
                self,
                f"unaligned LiDAR clips: {lidar_control.shape[1]} control vs {lidar_target.shape[1]} target sweeps",
                data_dict,
            )
            return None

        n_views = self._num_camera_views(data_dict)
        if n_views < 1:
            _drop_sample(self, f"nonpositive sample_n_views {n_views}", data_dict)
            return None
        if camera_target.shape[1] % n_views != 0:
            _drop_sample(
                self,
                f"camera-major length {camera_target.shape[1]} is not divisible by sample_n_views={n_views}",
                data_dict,
            )
            return None
        frames_per_view = camera_target.shape[1] // n_views

        window_start = self._sync_window_start(data_dict)
        if window_start is None:
            _drop_sample(self, "camera frames and sweeps share no source-frame ID", data_dict)
            return None
        camera_offset, sweep_offset = window_start
        if camera_offset >= frames_per_view:
            _drop_sample(self, f"camera sync offset {camera_offset} is past {frames_per_view} frames/view", data_dict)
            return None
        remaining = frames_per_view - camera_offset
        camera_control = slice_camera_major_frames(camera_control, n_views, camera_offset, remaining)
        camera_target = slice_camera_major_frames(camera_target, n_views, camera_offset, remaining)
        lidar_control = lidar_control[:, sweep_offset:]
        lidar_target = lidar_target[:, sweep_offset:]
        # The ID tensors define the window the tail trim measures, so they move first.
        data_dict["frame_indices"] = data_dict["frame_indices"][camera_offset:]
        for sweep_id_key in ("lidar_frame_indices", "hdmap_frame_indices"):
            sweep_ids = data_dict.get(sweep_id_key)
            if isinstance(sweep_ids, torch.Tensor):
                data_dict[sweep_id_key] = sweep_ids[sweep_offset:]

        remaining_per_view = frames_per_view - camera_offset
        camera_frames = self._camera_frames_to_keep(remaining_per_view)
        if self.fixed_camera_frames is not None:
            if camera_frames < self.fixed_camera_frames:
                _drop_sample(self, "insufficient synchronized camera coverage for the fixed window", data_dict)
                return None
            camera_frames = self.fixed_camera_frames
        if self.exact_camera_frames is not None:
            camera_frames = min(camera_frames, self.exact_camera_frames)
        lidar_frames = self._num_sweeps_within_camera_window(data_dict, camera_frames, int(lidar_target.shape[1]))
        exact_window_error = self._exact_window_error(data_dict, camera_frames, lidar_frames)
        if exact_window_error is not None:
            _drop_sample(self, exact_window_error, data_dict)
            return None
        camera_control = slice_camera_major_frames(camera_control, n_views, 0, camera_frames)
        camera_target = slice_camera_major_frames(camera_target, n_views, 0, camera_frames)
        lidar_control = lidar_control[:, :lidar_frames]
        lidar_target = lidar_target[:, :lidar_frames]
        # Describe the final aligned camera clip before packing changes ``video``
        # and ``image_size`` into per-modality item lists. Apply metadata once so
        # string captions retain the established duration-then-resolution order.
        caption_data = dict(data_dict)
        caption_data["video"] = camera_target  # [C,V*T,H,W]
        caption_data["num_video_frames_per_view"] = camera_frames
        for augmentor in self.caption_augmentors:
            updated_caption_data = augmentor(caption_data)
            if updated_caption_data is None:
                return None
            caption_data = updated_caption_data
        if self.caption_augmentors:
            data_dict["ai_caption"] = caption_data["ai_caption"]
        camera_fps = data_dict.get("fps")
        if camera_fps is None:
            _drop_sample(self, "missing camera fps", data_dict)
            return None
        camera_fps = float(camera_fps.reshape(-1)[0].item() if isinstance(camera_fps, torch.Tensor) else camera_fps)
        if camera_fps <= 0:
            _drop_sample(self, f"nonpositive camera fps {camera_fps}", data_dict)
            return None

        if self.fixed_camera_frames is not None:
            expected_sweeps = round((camera_frames - 1) * self.lidar_fps / camera_fps) + 1
            sweep_ids = data_dict["lidar_frame_indices"][:lidar_frames]  # [T_lidar]
            source_stride = round(camera_fps / self.lidar_fps)
            if lidar_frames != expected_sweeps or not torch.all(sweep_ids[1:] - sweep_ids[:-1] == source_stride):
                _drop_sample(self, "incomplete LiDAR coverage for the fixed synchronized window", data_dict)
                return None

        camera_height, camera_width = self.camera_resolution_hw
        camera_size = torch.tensor([camera_height, camera_width, camera_height, camera_width], dtype=torch.float)

        camera_latent_frames = 1 + (camera_frames - 1) // self.camera_temporal_compression_factor
        # View completion replaces the temporal prefix rather than adding to it: the known
        # cameras are clean for their whole clip, so there is no partial prefix left to draw.
        # Both the probability and the view count are guarded so a single-camera sample falls
        # back to the temporal mix, and so a run with the feature off never touches the RNG.
        condition_view_indexes: list[int] = []
        use_view_completion = (
            self.rgb_view_completion_probability > 0
            and n_views >= 2
            and random.random() < self.rgb_view_completion_probability
        )
        if use_view_completion:
            num_condition_views = random.randint(1, n_views - 1)
            condition_view_indexes = sorted(random.sample(range(n_views), num_condition_views))
            num_camera_condition_latents = 0
        else:
            num_camera_condition_latents = self._sample_num_camera_condition_latents(camera_latent_frames)
        # Give the two targets the same clean prefix in seconds rather than in frames, since a
        # camera latent spans 4/fps seconds and a sweep spans 1/lidar_fps.
        condition_seconds = num_camera_condition_latents * self.camera_temporal_compression_factor / camera_fps
        num_lidar_condition_sweeps = min(round(condition_seconds * self.lidar_fps), max(lidar_frames - 1, 0))

        camera_target_condition = list(range(num_camera_condition_latents))
        lidar_target_condition = list(range(num_lidar_condition_sweeps))

        # LiDAR is either one more generated stream or one more conditioning signal, drawn per
        # sample so one run can train both. The default of 0.0 short-circuits before the draw,
        # so runs that never asked for conditioning LiDAR keep the old behavior *and* the old
        # RNG stream. The camera target always keeps at least one noisy latent -- view
        # completion conditions a proper subset of views and the temporal prefix is capped at
        # ``camera_latent_frames - 1`` -- so this never yields a sample with nothing to learn.
        lidar_is_conditioning = (
            self.lidar_conditioning_probability > 0 and random.random() < self.lidar_conditioning_probability
        )
        if lidar_is_conditioning:
            # The sweep count is an upper bound on the LiDAR VAE's latent count, and the packer
            # keeps only indexes inside each item's latent range, so this marks every latent of
            # the target clean without needing the LiDAR compression factor here.
            lidar_target_condition = list(range(lidar_frames))

        # Drop only the temporal target prefix, preserving RGB conditioning and the full
        # HD-map control. Whole-clip LiDAR conditioning is a separate task. The default
        # and T2V samples without a prefix consume no additional random draw.
        if (
            not lidar_is_conditioning
            and lidar_target_condition
            and self.lidar_prefix_dropout_rate > 0
            and random.random() < self.lidar_prefix_dropout_rate
        ):
            lidar_target_condition = []

        view_indices_selection = data_dict.get("view_indices_selection")
        if not isinstance(view_indices_selection, torch.Tensor) or view_indices_selection.numel() != n_views:
            view_indices_selection = torch.zeros(n_views, dtype=torch.int64)
        view_indices = data_dict.get("view_indices")
        expected_view_index_len = n_views * frames_per_view
        if isinstance(view_indices, torch.Tensor) and view_indices.numel() == expected_view_index_len:
            view_indices = view_indices.reshape(n_views, frames_per_view)[
                :, camera_offset : camera_offset + camera_frames
            ].reshape(-1)
        else:
            view_indices = view_indices_selection.repeat_interleave(camera_frames)

        # Dropping means this sample carries no LiDAR at all -- neither the HD-map range
        # control nor the sweeps it would have generated -- so it trains as one of the
        # camera-only samples the recipe already handles, rather than as a sample asked to
        # invent a sweep from nothing. ``has_lidar`` and the LiDAR conditioning indexes below
        # follow, and the packer costs it no LiDAR tokens.
        #
        # Decided here rather than before the slicing above so a dropped sample's camera half
        # is exactly the one it would have had: the sweeps still pick the window both sensors
        # share, and a sample the sync would have rejected is still rejected. The two
        # populations then differ by the missing sensor alone.
        #
        # Withheld as an empty item list rather than a missing key, because the collate
        # function builds each key by appending from the rows that carry it, so a key that
        # some rows omit stops lining up with its samples. ``multiview_collate_fn`` drops the
        # key once every row in a batch is empty, which is the form the model reads.
        keep_lidar = self.lidar_dropout_rate == 0.0 or random.random() >= self.lidar_dropout_rate
        # One decision for the whole rig and both modalities, after alignment/conditioning.
        # The default consumes no RNG draw, preserving existing transfer runs exactly.
        keep_control = self.control_dropout_rate == 0.0 or random.random() >= self.control_dropout_rate
        data_dict.update(
            video=[camera_control, camera_target] if keep_control else [camera_target],  # items: [C,V*T,H,W]
            lidar=([lidar_control, lidar_target] if keep_control else [lidar_target]) if keep_lidar else [],
            image_size=[camera_size] * (2 if keep_control else 1),  # items: [4]
            num_frames=camera_frames,
            num_video_frames_per_view=torch.tensor(camera_frames, dtype=torch.int64),
            sample_n_views=torch.tensor(n_views, dtype=torch.int64),
            view_indices=view_indices.to(dtype=torch.int64),
            view_indices_selection=view_indices_selection.to(dtype=torch.int64),
            dataset_name="joint_cam_lidar_transfer",
        )
        if self.enable_per_camera_vae_encoding:
            data_dict["enable_per_camera_vae_encoding"] = True
        # The two streams above are the only copies the packer needs; leaving the per-stream
        # keys in place would send a second copy of every clip through collation.
        for consumed_key in (self.control_key, "hdmap_range", "lidar_range"):
            data_dict.pop(consumed_key, None)
        data_dict["frame_indices"] = data_dict["frame_indices"][:camera_frames]
        for sweep_id_key in ("lidar_frame_indices", "hdmap_frame_indices"):
            sweep_ids = data_dict.get(sweep_id_key)
            if isinstance(sweep_ids, torch.Tensor):
                data_dict[sweep_id_key] = sweep_ids[:lidar_frames]
        data_dict.setdefault("ai_caption", "")
        data_dict.setdefault("selected_caption_type", "transfer_caption")

        data_dict["sequence_plan"] = SequencePlan(
            has_text=True,
            has_vision=True,
            has_lidar=keep_lidar,
            # Each stream is [control, target] or [target], so the packer's positional
            # convention already says which item is generated: every item but the last is
            # clean throughout, and the last carries the clean prefix named here.
            condition_frame_indexes_vision=camera_target_condition,
            # Whole known cameras, unioned with the temporal prefix above at pack time. Empty
            # unless this sample drew view completion.
            condition_view_indexes_vision=condition_view_indexes,
            condition_frame_indexes_lidar=lidar_target_condition if keep_lidar else [],
            # One temporal origin for the whole sample: each control is frame-synced with its
            # target, and both sensors' item 0 is the same source frame, so mRoPE places the
            # camera latents and the sweeps on a single exact real-time axis.
            vision_temporal_position_groups=[0] * (2 if keep_control else 1),
        )
        return data_dict


class EnablePerCameraVAEEncoding(Augmentor):
    """Mark multiview video-only samples for per-camera VAE encoding."""

    def __init__(self, input_keys: list | None = None, args: dict | None = None) -> None:
        super().__init__(input_keys or [], None, args)

    def __call__(self, data_dict: dict[str, Any]) -> dict[str, Any]:
        sample_n_views = read_positive_int_metadata(data_dict, "sample_n_views", expected_count=1)
        frames_per_view = read_positive_int_metadata(data_dict, "num_video_frames_per_view", expected_count=1)
        if sample_n_views is None or frames_per_view is None:
            raise ValueError(
                "enable_per_camera_vae_encoding requires sample_n_views and num_video_frames_per_view metadata."
            )
        video = data_dict.get("video")
        if not isinstance(video, torch.Tensor) or video.dim() != 4:
            raise ValueError(
                "enable_per_camera_vae_encoding expects video to have shape [C,V*T,H,W], "
                f"got {type(video).__name__ if not isinstance(video, torch.Tensor) else tuple(video.shape)}."
            )
        expected_frames = sample_n_views[0] * frames_per_view[0]
        actual_frames = int(video.shape[1])
        if actual_frames != expected_frames:
            raise ValueError(
                "enable_per_camera_vae_encoding expected video length sample_n_views * "
                f"num_video_frames_per_view = {expected_frames}, got {actual_frames}."
            )
        data_dict["enable_per_camera_vae_encoding"] = True
        return data_dict


class MultiviewTransferToTrainingFormat(Augmentor):
    """Convert multiview (control, target) pairs into the joint-dataloader training format.

    Mirrors :class:`TransferToTrainingFormat` in ``transfer_control_transform.py``
    but adapted for the multiview pipeline where:

    * The control tensor lives under a configurable key (default
      ``"control_input"``).
    * Resolution is known a-priori from the augmentation config, so
      ``image_size`` is set directly instead of being read from a prior
      resize augmentor.
    * Multiview metadata (``view_indices``, ``sample_n_views``, etc.) is
      preserved for downstream consumption.
    * ``enable_per_camera_vae_encoding`` is emitted only when explicitly
      enabled, gating matching token estimation and model-side VAE behavior.

    Output fields written:

    * ``video`` — ``[control_tensor, target_tensor]`` (list of two ``[C, T, H, W]``).
    * ``sequence_plan`` — :class:`SequencePlan` with optional conditioning frames.
    * ``num_frames``, ``dataset_name``, ``fps``, ``selected_caption_type``,
      ``image_size``.
    """

    def __init__(
        self,
        input_keys: list | None = None,
        control_key: str = MULTIVIEW_CONTROL_KEY,
        resolution_hw: tuple[int, int] = (720, 1280),
        conditioning_config: dict[int, float] | None = None,
        rgb_view_completion_probability: float = 0.0,
        share_vision_temporal_positions: bool = True,
        enable_per_camera_vae_encoding: bool = False,
        args: dict | None = None,
    ) -> None:
        super().__init__(input_keys or [], None, args)
        self.control_key = control_key
        self.resolution_hw = resolution_hw
        if not 0.0 <= rgb_view_completion_probability <= 1.0:
            raise ValueError(
                f"rgb_view_completion_probability must be in [0, 1], got {rgb_view_completion_probability}."
            )
        if rgb_view_completion_probability > 0 and not enable_per_camera_vae_encoding:
            raise ValueError("View completion requires enable_per_camera_vae_encoding=True.")
        self.share_vision_temporal_positions = share_vision_temporal_positions
        self.enable_per_camera_vae_encoding: bool = enable_per_camera_vae_encoding
        self.rgb_view_completion_probability: float = rgb_view_completion_probability
        self.normalized_conditioning_config = _normalize_conditioning_config(conditioning_config)

    def __call__(self, data_dict: dict) -> dict | None:
        from cosmos_framework.data.generator.sequence_packing import SequencePlan

        control = data_dict.get(self.control_key)
        target = data_dict.get("video")

        if control is None or target is None:
            log.warning(
                f"MultiviewTransferToTrainingFormat: missing {self.control_key} or video: "
                f"{data_dict.get('__key__', 'unknown')}",
                rank0_only=False,
            )
            return None

        try:
            num_frames = control.shape[1]
            data_dict["video"] = [control, target]
            data_dict["num_frames"] = num_frames
            data_dict["dataset_name"] = "video_transfer"
            data_dict.setdefault("ai_caption", "")
            data_dict.setdefault("selected_caption_type", "transfer_caption")
            if self.enable_per_camera_vae_encoding:
                data_dict["enable_per_camera_vae_encoding"] = True

            sample_n_views = data_dict.get("sample_n_views")
            if sample_n_views is not None:
                num_views = (
                    int(sample_n_views.reshape(-1)[0].item())
                    if isinstance(sample_n_views, torch.Tensor)
                    else int(sample_n_views)
                )
            else:
                num_views = 1

            condition_view_indexes: list[int] = []
            use_view_completion = (
                self.rgb_view_completion_probability > 0
                and num_views >= 2
                and random.random() < self.rgb_view_completion_probability
            )
            num_condition_frames = 0
            if use_view_completion:
                num_condition_views = random.randint(1, num_views - 1)
                condition_view_indexes = sorted(random.sample(range(num_views), num_condition_views))
            elif self.normalized_conditioning_config is not None:
                frames_options = list(self.normalized_conditioning_config.keys())
                weights = list(self.normalized_conditioning_config.values())
                num_condition_frames = random.choices(frames_options, weights=weights, k=1)[0]
                frames_per_view = target.shape[1] // num_views if num_views > 1 else target.shape[1]
                max_cond = frames_per_view - 1
                num_condition_frames = min(num_condition_frames, max_cond)

            # Per-view-local latent frame indexes; expanded across cameras at pack time
            # when num_views_per_vision_item > 1 (camera-major latent layout).
            if num_condition_frames > 0 and target.shape[1] > 1:
                condition_frames_indexes = list(range(num_condition_frames))
            else:
                condition_frames_indexes = []

            data_dict["sequence_plan"] = SequencePlan(
                has_text=True,
                has_vision=True,
                condition_frame_indexes_vision=condition_frames_indexes,
                condition_view_indexes_vision=condition_view_indexes,
                # ControlNet-style multiview transfer: control item and target
                # item are spatio-temporally aligned (same source clip,
                # frame-synced). Force shared temporal mRoPE grid across both.
                share_vision_temporal_positions=self.share_vision_temporal_positions,
            )
        except Exception as e:
            log.warning(
                f"MultiviewTransferToTrainingFormat error: {data_dict.get('__key__', 'unknown')}, {e}",
                rank0_only=False,
            )
            return None

        h, w = self.resolution_hw
        image_size = torch.tensor([h, w, h, w], dtype=torch.float)
        data_dict["image_size"] = [image_size] * len(data_dict["video"])

        return data_dict
