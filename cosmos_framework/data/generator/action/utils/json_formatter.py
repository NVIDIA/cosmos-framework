# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import math

import torch

from cosmos_framework.utils import log
from cosmos_framework.data.generator.action.utils.viewpoint_utils import DEFAULT_VIEWPOINT_TEMPLATES
from cosmos_framework.data.generator.utils import VIDEO_RES_SIZE_INFO


def _should_append_idle_frame_info(mode: object) -> bool:
    """Return whether idle-frame prompt metadata should be surfaced."""
    return mode != "inverse_dynamics"


def _is_inverse_dynamics(mode: object) -> bool:
    """Return whether this sample predicts actions from video.

    Deliberately separate from :func:`_should_append_idle_frame_info`: the two
    suppressions happen to coincide today but are independent decisions, and
    extending the idle-frame rule to further modes must not silently change
    which modes see the relabeling.
    """
    return mode == "inverse_dynamics"


class ActionPromptJsonFormatter:
    """Format action prompts into a structured JSON-compatible dictionary.

    JSON fields are emitted in this order: ``cinematography``, ``actions``,
    ``duration``, ``fps``, ``resolution``, then ``aspect_ratio``. Like video JSON
    prompts, ``cinematography`` is a dictionary, duration is truncated to an
    integer-second string such as ``"2s"``, and aspect ratio is stored as a
    comma-separated string such as ``"16,9"``. If ``data_dict["mode"]`` is
    ``"inverse_dynamics"``, idle-frame metadata is omitted from the prompt.

    A dataset may additionally supply ``data_dict["relabel_prompt"]`` -- the
    per-keyframe relabeling active at this chunk, as ``task_state`` and/or
    ``situation`` (see the action-level ``relabel_annotations`` module). Its
    keys are merged into the action entry *after* the existing ones, so prompts
    built without relabeling are unchanged. Like idle-frame metadata it is
    suppressed for ``"inverse_dynamics"``.
    """

    def __init__(
        self,
        caption_key: str = "ai_caption",
        viewpoint_key: str = "viewpoint",
        video_key: str = "video",
        fps_key: str = "conditioning_fps",
        image_size_key: str = "image_size",
        idle_frames_key: str = "idle_frames",
        total_frames_key: str = "idle_frames_total",
        action_key: str = "action",
        viewpoint_templates: dict[str, str] | None = None,
        append_duration_fps_timestamps: bool = True,
        float_seconds: bool = False,
        video_num_frames_key: str = "video_num_frames",
    ) -> None:
        self.caption_key: str = caption_key
        self.viewpoint_key: str = viewpoint_key
        self.video_key: str = video_key
        self.fps_key: str = fps_key
        self.image_size_key: str = image_size_key
        self.idle_frames_key: str = idle_frames_key
        self.total_frames_key: str = total_frames_key
        self.action_key: str = action_key
        # When True, ``duration`` and action ``time`` are sub-second-precise
        # floats (e.g. "0.57s", "0.00-0.57s") instead of truncated/rounded whole
        # seconds. Useful for short high-fps clips (e.g. 16-frame chunks at 30 fps
        # = 0.53 s) where the integer format collapses to "0s".
        self.float_seconds: bool = float_seconds
        self.append_duration_fps_timestamps: bool = append_duration_fps_timestamps
        self.video_num_frames_key: str = video_num_frames_key
        self.viewpoint_templates: dict[str, str] = (
            viewpoint_templates if viewpoint_templates is not None else DEFAULT_VIEWPOINT_TEMPLATES
        )

    def __call__(self, data_dict: dict) -> dict:
        """Replace the caption with the action JSON prompt structure."""
        additional_view_description = data_dict.pop("additional_view_description", None)
        # Popped unconditionally -- including on the early-return paths below --
        # so the optional field never survives into collation.
        relabel_prompt = data_dict.pop("relabel_prompt", None)
        caption = data_dict.get(self.caption_key)
        if not isinstance(caption, str) or caption == "":
            return data_dict

        height, width = self._get_resolution(data_dict)
        time_str = None
        duration_str = None
        fps = None
        if self.append_duration_fps_timestamps:
            fps = self._get_scalar_float(data_dict.get(self.fps_key), self.fps_key)
            if fps <= 0:
                raise ValueError(f"ActionPromptJsonFormatter: '{self.fps_key}' must be positive, got {fps}")

            video = data_dict.get(self.video_key)
            if not isinstance(video, torch.Tensor) or video.ndim < 2:
                raise ValueError(
                    f"ActionPromptJsonFormatter: expected '{self.video_key}' to be a video tensor with shape "
                    f"(C, T, H, W), got {type(video).__name__}"
                )
            duration_seconds = int(data_dict.get(self.video_num_frames_key, video.shape[1])) / fps
            if self.float_seconds:
                time_str = f"0.00-{duration_seconds:.2f}s"
                duration_str = f"{duration_seconds:.2f}s"
            else:
                time_str = f"0:00-{self._format_time_mss(self._round_time_seconds(duration_seconds))}"
                duration_str = f"{self._truncate_seconds(duration_seconds)}s"

        action_entry: dict[str, object] = {}
        if time_str is not None:
            action_entry["time"] = time_str
        action_entry["description"] = self._ensure_sentence(caption)
        action_entry["idle_frame"] = self._get_idle_frame_info(data_dict)
        # Appended after the existing keys so prompts without relabeling are
        # byte-identical to before this field existed.
        action_entry.update(self._get_relabel_prompt(data_dict, relabel_prompt))

        prompt: dict[str, object] = {
            "cinematography": {
                "framing": self._get_viewpoint_caption(data_dict, additional_view_description),
            },
            "actions": [action_entry],
        }
        if duration_str is not None and fps is not None:
            prompt["duration"] = duration_str
            prompt["fps"] = float(fps)
        prompt["resolution"] = {"H": height, "W": width}
        prompt["aspect_ratio"] = self._get_aspect_ratio(width, height)
        cleaned_prompt = self._drop_empty_fields(prompt)
        self._raise_if_empty_fields(cleaned_prompt)
        data_dict[self.caption_key] = cleaned_prompt
        return data_dict

    def _truncate_seconds(self, seconds: float) -> int:
        """Truncate duration to integer seconds, matching video JSON-caption augmentors."""
        if seconds < 0 or not math.isfinite(seconds):
            return 0
        return int(seconds)

    def _round_time_seconds(self, seconds: float) -> int:
        """Round an action timestamp to integer seconds, matching video captioning."""
        if seconds < 0 or not math.isfinite(seconds):
            return 0
        return round(seconds)

    def _format_time_mss(self, seconds: int) -> str:
        """Format integer seconds as M:SS for JSON prompt time ranges."""
        minutes, remaining_seconds = divmod(seconds, 60)
        return f"{minutes}:{remaining_seconds:02d}"

    def _get_aspect_ratio(self, width: int, height: int) -> str:
        """Return the canonical width,height aspect ratio string when known."""
        for aspect_ratio_sizes in VIDEO_RES_SIZE_INFO.values():
            for aspect_ratio, (candidate_w, candidate_h) in aspect_ratio_sizes.items():
                if width == candidate_w and height == candidate_h:
                    return aspect_ratio

        divisor = math.gcd(width, height)
        if divisor == 0:
            raise ValueError(
                f"ActionPromptJsonFormatter: width and height must be non-zero, got width={width}, height={height}."
            )
        return f"{width // divisor},{height // divisor}"

    def _get_viewpoint_caption(self, data_dict: dict, additional_view_description: object | None) -> str | None:
        """Resolve the viewpoint text used in the ``cinematography`` field."""
        viewpoint = data_dict.get(self.viewpoint_key)
        template = self.viewpoint_templates.get(viewpoint) if isinstance(viewpoint, str) else None

        if template is None:
            if viewpoint is not None:
                log.warning(
                    f"ActionPromptJsonFormatter: unrecognized viewpoint {viewpoint!r}. "
                    f"Known viewpoints: {sorted(self.viewpoint_templates.keys())}. "
                    f"Using additional view description when available.",
                    rank0_only=False,
                )
            return self._get_optional_text(additional_view_description)

        if additional_view_description:
            separator = " " if template.endswith(".") else ". "
            template = template + separator + str(additional_view_description).rstrip()
        return template

    def _get_resolution(self, data_dict: dict) -> tuple[int, int]:
        """Resolve ``(height, width)`` from the post-padding image size."""
        image_size = data_dict.get(self.image_size_key)
        if image_size is None:
            raise ValueError(f"ActionPromptJsonFormatter: missing '{self.image_size_key}' in data_dict.")

        if isinstance(image_size, torch.Tensor):
            if image_size.numel() < 2:
                raise ValueError(
                    f"ActionPromptJsonFormatter: expected '{self.image_size_key}' to contain at least "
                    f"height and width, got shape {tuple(image_size.shape)}"
                )
            return int(image_size[0].item()), int(image_size[1].item())

        try:
            return int(image_size[0]), int(image_size[1])
        except (TypeError, ValueError, IndexError) as e:
            raise ValueError(
                f"ActionPromptJsonFormatter: expected '{self.image_size_key}' to contain height and width."
            ) from e

    def _get_scalar_float(self, value: object, key: str) -> float:
        """Parse a required scalar float from a tensor or Python value."""
        if value is None:
            raise ValueError(f"ActionPromptJsonFormatter: missing '{key}' in data_dict.")

        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(
                    f"ActionPromptJsonFormatter: expected scalar tensor at '{key}', got shape {tuple(value.shape)}"
                )
            return float(value.item())

        if isinstance(value, (str, int, float)):
            try:
                return float(value)
            except ValueError as e:
                raise ValueError(
                    f"ActionPromptJsonFormatter: expected scalar float-compatible value at '{key}'."
                ) from e
        raise ValueError(f"ActionPromptJsonFormatter: expected scalar float-compatible value at '{key}'.")

    def _get_optional_scalar_int(self, value: object, key: str) -> int | None:
        """Parse an optional scalar integer metadata value."""
        if value is None:
            return None

        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                log.warning(
                    f"ActionPromptJsonFormatter: expected scalar tensor at '{key}', got shape "
                    f"{tuple(value.shape)}. Skipping.",
                    rank0_only=False,
                )
                return None
            return int(value.item())

        if isinstance(value, (str, int, float)):
            try:
                return int(value)
            except ValueError:
                pass
        log.warning(
            f"ActionPromptJsonFormatter: expected integer-compatible value at "
            f"'{key}', got {type(value).__name__}. Skipping.",
            rank0_only=False,
        )
        return None

    def _get_total_frames(self, data_dict: dict) -> int | None:
        """Resolve the total action-frame count for idle-frame text."""
        total_frames = self._get_optional_scalar_int(data_dict.get(self.total_frames_key), self.total_frames_key)
        if total_frames is not None:
            return total_frames

        action = data_dict.get(self.action_key)
        if isinstance(action, torch.Tensor):
            if action.ndim == 0:
                log.warning(
                    f"ActionPromptJsonFormatter: expected action tensor at "
                    f"'{self.action_key}' to have a frame dimension. Skipping total frames.",
                    rank0_only=False,
                )
                return None
            return int(action.shape[0])

        try:
            return len(action) if action is not None else None
        except TypeError:
            return None

    def _get_relabel_prompt(self, data_dict: dict, relabel_prompt: object | None) -> dict:
        """Return the relabeling fragment to merge into the action entry.

        Suppressed for ``inverse_dynamics`` for the same reason idle-frame
        metadata is: that mode predicts the action from the video, and the
        relabeling's ``grip`` / ``next`` fields describe the action itself, so
        conditioning on them would leak the target.
        """
        if not isinstance(relabel_prompt, dict) or not relabel_prompt:
            return {}
        if _is_inverse_dynamics(data_dict.get("mode")):
            return {}
        return relabel_prompt

    def _get_idle_frame_info(self, data_dict: dict) -> str | None:
        """Build the idle-frame string for the action object."""
        if not _should_append_idle_frame_info(data_dict.get("mode")):
            return None

        idle_frames = self._get_optional_scalar_int(data_dict.get(self.idle_frames_key), self.idle_frames_key)
        total_frames = self._get_total_frames(data_dict)

        if idle_frames is not None and total_frames is not None:
            return f"{idle_frames} out of {total_frames}."
        if idle_frames is not None:
            return f"{idle_frames}."
        return None

    def _ensure_sentence(self, text: str) -> str:
        """Return text with terminal sentence punctuation."""
        text = text.strip()
        if text.endswith((".", "!", "?")):
            return text
        return f"{text}."

    def _get_optional_text(self, value: object) -> str | None:
        """Return stripped text, leaving empty optional text for the final prune pass."""
        if value is None:
            return None
        text = str(value).rstrip()
        return text if text else None

    def _drop_empty_fields(self, value: object) -> object:
        """Recursively remove empty strings, dictionaries, lists, and ``None`` values."""
        if isinstance(value, dict):
            return {
                key: cleaned
                for key, item in value.items()
                if not self._is_empty(cleaned := self._drop_empty_fields(item))
            }
        if isinstance(value, list):
            return [cleaned for item in value if not self._is_empty(cleaned := self._drop_empty_fields(item))]
        return value

    def _is_empty(self, value: object) -> bool:
        """Return whether a JSON field should be dropped."""
        return value is None or value == "" or value == [] or value == {}

    def _raise_if_empty_fields(self, value: object, path: str = "prompt") -> None:
        """Validate that no empty JSON fields remain after pruning."""
        if self._is_empty(value):
            raise ValueError(f"ActionPromptJsonFormatter: empty field remains at {path}.")

        if isinstance(value, dict):
            for key, item in value.items():
                self._raise_if_empty_fields(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                self._raise_if_empty_fields(item, f"{path}[{index}]")
