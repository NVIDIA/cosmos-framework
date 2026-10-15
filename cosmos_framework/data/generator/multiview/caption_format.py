# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Shared pure formatting helpers for multiview training and inference captions."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, Final

Caption = str | dict[str, Any]

TWO_PASS_CAPTION_INSTRUCTION: Final[str] = (
    "The following view captions repeat the same camera views so each view can be interpreted using context from "
    "all cameras."
)

DEFAULT_CAPTION_PREFIXES: Final[dict[str, str]] = {
    "camera_front_wide_120fov": "The video is captured from a camera mounted on a car. The camera is facing forward.",
    "camera_cross_right_120fov": "The video is captured from a camera mounted on a car. The camera is facing to the right.",
    "camera_rear_right_70fov": "The video is captured from a camera mounted on a car. The camera is facing the rear right side.",
    "camera_rear_tele_30fov": "The video is captured from a camera mounted on a car. The camera is facing backwards.",
    "camera_rear_left_70fov": "The video is captured from a camera mounted on a car. The camera is facing the rear left side.",
    "camera_cross_left_120fov": "The video is captured from a camera mounted on a car. The camera is facing to the left.",
    "camera_front_tele_30fov": "The video is captured from a telephoto camera mounted on a car. The camera is facing forward.",
    "camera_front_fisheye_200fov": "The video is captured from a fisheye camera mounted on a car. The camera is facing forward.",
    "camera_left_fisheye_200fov": "The video is captured from a fisheye camera mounted on a car. The camera is facing to the left.",
    "camera_right_fisheye_200fov": "The video is captured from a fisheye camera mounted on a car. The camera is facing to the right.",
    "camera_rear_fisheye_200fov": "The video is captured from a fisheye camera mounted on a car. The camera is facing backwards.",
}


_TIMESTAMP: re.Pattern[str] = re.compile(r"\[\s*\d+(?:\.\d+)?\s*s?\s*[-–—]\s*\d+(?:\.\d+)?\s*s?\s*\]", re.IGNORECASE)


def first_caption_paragraph(caption: str) -> str:
    """Keep the opening timestamped paragraph, or the first blank-line paragraph."""
    caption = caption.strip()
    timestamps = list(_TIMESTAMP.finditer(caption))
    if timestamps:
        end = timestamps[1].start() if len(timestamps) > 1 else len(caption)
        caption = caption[timestamps[0].start() : end].rstrip(" ,;\t\r\n")
    return re.split(r"\r?\n\s*\r?\n", caption, maxsplit=1)[0].strip()


def _camera_identity(camera_name: str, camera_attributes: Mapping[str, str | int]) -> str:
    """Render a camera's role and lens type as a natural-language identity."""
    required_attributes = ("camera_role", "camera_type")
    missing_attributes = [attribute for attribute in required_attributes if attribute not in camera_attributes]
    if missing_attributes:
        raise ValueError(f"Camera {camera_name!r} is missing attributes: {missing_attributes}")

    role = str(camera_attributes["camera_role"]).replace("_", "-")
    camera_type = str(camera_attributes["camera_type"]).replace("_", "-")
    camera_type = {"standard": "", "wide": "wide-angle"}.get(camera_type, camera_type)
    role_and_type = " ".join(part for part in (role, camera_type) if part)
    return f"{role_and_type} camera"


def _camera_description(camera_name: str, camera_attributes: Mapping[str, str | int]) -> str:
    """Render stable camera attributes as a compact natural-language description."""
    required_attributes = ("facing", "fov_degrees")
    missing_attributes = [attribute for attribute in required_attributes if attribute not in camera_attributes]
    if missing_attributes:
        raise ValueError(f"Camera {camera_name!r} is missing attributes: {missing_attributes}")

    facing = str(camera_attributes["facing"]).replace("_", "-")
    camera_identity = _camera_identity(camera_name, camera_attributes)
    return f"{camera_identity} ({facing}-facing, {camera_attributes['fov_degrees']}° FOV)"


def format_separate_view_captions(
    captions: Sequence[Caption],
    *,
    camera_names: Sequence[str],
    add_camera_rig_prefix: bool = False,
    camera_attributes: Mapping[str, Mapping[str, str | int]] | None = None,
) -> list[Caption]:
    """Preserve per-view captions, optionally adding the sampled rig in natural language."""
    if not captions:
        raise ValueError("Separate-view captions must not be empty")
    if len(captions) != len(camera_names):
        raise ValueError(
            f"Separate captions must match sampled cameras: captions={len(captions)}, cameras={len(camera_names)}"
        )
    if not add_camera_rig_prefix:
        return list(captions)
    if camera_attributes is None:
        raise ValueError("camera_attributes are required when adding sampled camera-rig prefixes")

    descriptions = [_camera_description(camera, camera_attributes[camera]) for camera in camera_names]
    identities = [_camera_identity(camera, camera_attributes[camera]) for camera in camera_names]
    camera_word = "camera" if len(camera_names) == 1 else "cameras"
    rig_prefix = (
        f"This multiview driving sequence contains time-aligned recordings from {len(camera_names)} vehicle-mounted "
        f"{camera_word}: "
        f"{'; '.join(descriptions)}."
    )
    return [
        f"{rig_prefix}\n\nThe description below is for the {identity} mounted on the vehicle. "
        f"This camera is facing {str(camera_attributes[camera_name]['facing']).replace('_', '-')} and has a "
        f"{camera_attributes[camera_name]['fov_degrees']}° field of view:\n\n"
        f"{caption if isinstance(caption, str) else json.dumps(caption)}"
        for camera_name, identity, caption in zip(camera_names, identities, captions)
    ]


def format_view_caption(
    *,
    caption: Caption,
    camera_name: str,
    view_index: int,
    add_view_prefix: bool = False,
    use_explicit_view_format: bool = False,
    camera_prefixes: Mapping[str, str] | None = None,
    camera_attributes: Mapping[str, Mapping[str, str | int]] | None = None,
) -> Caption:
    """Format one loaded caption using the shared training/inference contract."""
    if use_explicit_view_format:
        if camera_attributes is None:
            raise ValueError("camera_attributes are required for explicit multiview caption formatting")
        return {
            "view_index": view_index,
            **dict(camera_attributes[camera_name]),
            "caption": caption,
        }

    if add_view_prefix:
        if camera_prefixes is None:
            raise ValueError("camera_prefixes are required when adding multiview caption prefixes")
        camera_view = camera_prefixes[camera_name]
        if isinstance(caption, dict):
            formatted = dict(caption)
            formatted["camera_view"] = camera_view
            return formatted
        return f"{camera_view} {caption}"

    return caption


def format_multiview_caption(
    captions: Sequence[Caption],
    *,
    use_explicit_view_format: bool = False,
    use_two_pass_format: bool = False,
) -> Caption:
    """Combine ordered view captions into the exact model prompt payload."""
    if not captions:
        raise ValueError("multiview captions must not be empty")
    if use_explicit_view_format:
        prompt: Caption = {
            "view_order": "The view captions are listed in the same order as the generated video views.",
            "num_views": len(captions),
            "views": list(captions),
        }
    else:
        prompt = captions[0] if len(captions) == 1 else {"views": list(captions)}
    return add_two_pass_multiview_caption(prompt) if use_two_pass_format else prompt


def add_two_pass_multiview_caption(caption: Caption) -> dict[str, Any]:
    """Append a marked second copy of an ordered multiview caption list.

    The first-pass payload is preserved in insertion order. Under causal text
    attention, every caption in ``repeated_views`` can therefore attend the
    complete first-pass ``views`` list without requiring a new tokenizer token
    or a model attention change.
    """
    if not isinstance(caption, dict) or not isinstance(caption.get("views"), list) or not caption["views"]:
        raise ValueError("two-pass multiview captions require a non-empty dict-valued 'views' list")
    if "repeated_views" in caption or "repeated_view_caption_instruction" in caption:
        raise ValueError("multiview caption is already decorated for two-pass prompting")

    prompt = dict(caption)
    prompt["repeated_view_caption_instruction"] = TWO_PASS_CAPTION_INSTRUCTION
    prompt["repeated_views"] = list(caption["views"])
    return prompt
