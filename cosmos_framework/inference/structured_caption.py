# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Canonical structured-JSON caption: schema, robust parsing, and assembly.

The Cosmos3 model's native text-prompt format is structured JSON (see
``docs/prompt_upsampling.md``).  This module is the single source of truth for
that format on the *captioning / training* side:

* :data:`CAPTION_JSON_KEY` — the JSONL / ``t2w_window`` key under which the
  structured caption object is stored (preferred over the dense ``caption``).
* :class:`StructuredCaption` — a permissive pydantic model mirroring
  ``inference/prompting_templates/external_api/t2v_i2v_video_json_schema.json``.
* :func:`parse_structured_caption` — robustly extract the Phase-1
  ``<scene_draft>`` JSON object from a VLM response.
* :func:`assemble_caption_json` — combine the Phase-1 draft, the polished
  Phase-2 dense narrative (stored as ``temporal_caption``), and the clip's real
  media fields into a single validated caption object.

The model is intentionally permissive (every field optional, ``extra="allow"``)
so that partial or slightly-off VLM output still round-trips instead of being
dropped; the goal is structural validation, not rejection.
"""

import json
import re
from typing import Any

import pydantic
from pydantic import ConfigDict

# Key used in the SFT JSONL ``t2w_windows[]`` entries and recognised by the SFT
# loader (sft_dataset.py) as the highest-priority caption.  Kept here so the
# captioner, the JSONL converter, and the loader cannot drift apart.
CAPTION_JSON_KEY = "caption_json"

_PERMISSIVE = ConfigDict(extra="allow")


class _Base(pydantic.BaseModel):
    model_config = _PERMISSIVE


class Subject(_Base):
    description: str | None = None
    appearance_details: str | None = None
    relationship: str | None = None
    location: str | None = None
    relative_size: str | None = None
    orientation: str | None = None
    pose: str | None = None
    action: str | None = None
    state_changes: str | None = None
    clothing: str | None = None
    expression: str | None = None
    gender: str | None = None
    age: str | None = None
    skin_tone_and_texture: str | None = None
    facial_features: str | None = None
    number_of_subjects: int | None = None
    number_of_arms: int | None = None
    number_of_legs: int | None = None


class Lighting(_Base):
    conditions: str | None = None
    direction: str | None = None
    shadows: str | None = None
    illumination_effect: str | None = None


class Aesthetics(_Base):
    composition: str | None = None
    color_scheme: str | None = None
    mood_atmosphere: str | None = None
    patterns: str | None = None


class Cinematography(_Base):
    camera_motion: str | None = None
    framing: str | None = None
    camera_angle: str | None = None
    depth_of_field: str | None = None
    focus: str | None = None
    lens_focal_length: str | None = None


class Action(_Base):
    time: str | None = None
    description: str | None = None


class TextElement(_Base):
    text: str | None = None
    category: str | None = None
    appearance: str | None = None
    spatial_temporal: str | None = None
    context: str | None = None


class Segment(_Base):
    segment_index: int | None = None
    time_range: str | None = None
    description: str | None = None
    key_changes: str | None = None
    camera: str | None = None


class Resolution(_Base):
    H: int | None = None
    W: int | None = None


class StructuredCaption(_Base):
    """Permissive mirror of the external-API T2V/I2V JSON schema."""

    subjects: list[Subject] | None = None
    background_setting: str | None = None
    lighting: Lighting | None = None
    aesthetics: Aesthetics | None = None
    cinematography: Cinematography | None = None
    style_medium: str | None = None
    artistic_style: str | None = None
    context: str | None = None
    actions: list[Action] | None = None
    text_and_signage_elements: list[TextElement] | None = None
    segments: list[Segment] | None = None
    transitions: list[str] | None = None
    temporal_caption: str | None = None
    audio_description: str | None = None
    resolution: Resolution | None = None
    aspect_ratio: str | None = None
    duration: str | None = None
    fps: int | None = None


def extract_xml_tag(text: str, tag: str) -> str | None:
    """Return the inner text of ``<tag>...</tag>`` (DOTALL), or ``None``."""
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else None


def _strip_code_fences(text: str) -> str:
    """Strip a leading ```json / ``` fence and trailing ``` if present."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    return cleaned.strip()


def _first_json_object(text: str) -> str | None:
    """Return the first balanced ``{...}`` block in ``text``, or ``None``.

    Brace-counting fallback for when the model wraps the JSON in prose without
    fences/tags.  Ignores braces inside double-quoted strings.
    """
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_structured_caption(text: str) -> dict | None:
    """Extract the Phase-1 ``<scene_draft>`` JSON object from a VLM response.

    Resolution order, each tolerant of ```` ```json ```` fences:

    1. The ``<scene_draft>`` XML block.
    2. The whole response (if it is itself a JSON object).
    3. The first balanced ``{...}`` block anywhere in the response.

    Returns the parsed ``dict`` on success, or ``None`` if no valid JSON object
    can be recovered (the caller should retry).
    """
    candidates: list[str] = []
    tagged = extract_xml_tag(text, "scene_draft")
    if tagged is not None:
        candidates.append(tagged)
    candidates.append(text)

    for candidate in candidates:
        cleaned = _strip_code_fences(candidate)
        for blob in (cleaned, _first_json_object(cleaned)):
            if not blob:
                continue
            try:
                parsed = json.loads(blob)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    return None


def aspect_ratio_str(width: int, height: int) -> str:
    """Reduce ``width``/``height`` to a ``"W,H"`` ratio string (e.g. ``"1,1"``)."""
    from math import gcd

    if width <= 0 or height <= 0:
        return ""
    g = gcd(int(width), int(height)) or 1
    return f"{int(width) // g},{int(height) // g}"


def media_fields_from_metadata(meta: dict) -> dict:
    """Build the caption's media fields from :func:`probe_video_metadata` output.

    Uses the clip's *actual* values (not the canonical generation enums): the
    enums constrain the upsampler's generation params, not ground-truth captions.
    """
    width, height = int(meta["width"]), int(meta["height"])
    return {
        "resolution": {"H": height, "W": width},
        "aspect_ratio": aspect_ratio_str(width, height),
        "duration": f"{round(float(meta['duration']))}s",
        "fps": int(round(float(meta["fps"]))),
    }


def assemble_caption_json(scene_draft: dict, final_prompt: str, media: dict) -> dict:
    """Assemble the final caption object and validate it.

    * ``temporal_caption`` is set to the polished Phase-2 ``final_prompt`` (this
      is what keeps the dense narrative available *inside* the JSON and equal to
      ``caption.txt``), overriding any draft value from Phase 1.
    * ``media`` (from :func:`media_fields_from_metadata`) is merged in.

    Returns a normalised ``dict`` (None-valued fields dropped, types coerced).
    Raises ``pydantic.ValidationError`` if the structure is unusable.
    """
    data: dict[str, Any] = dict(scene_draft)
    data["temporal_caption"] = (final_prompt or "").strip()
    data.update(media)
    model = StructuredCaption.model_validate(data)
    return model.model_dump(exclude_none=True, mode="json")


def caption_json_to_prompt(caption_json: dict) -> str:
    """Serialise a caption object to the compact JSON string fed to the model.

    Single source of truth for how a structured caption becomes model text, so
    training (sft_dataset.py) and inference prompts use byte-identical encoding.
    """
    return json.dumps(caption_json, ensure_ascii=False)
