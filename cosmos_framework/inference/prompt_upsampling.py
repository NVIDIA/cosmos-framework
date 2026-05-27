# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Reusable prompt upsampling client for Cosmos3 generation/evaluation scripts."""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import re
import time
from base64 import b64encode
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# In Cosmos3 training, we used `json.dumps(...)` for converting dict-structured caption objects to string.
# This JSON-formatted string was then used as text caption input to the tokenizer.
# This has the side effect of converting non-ASCII characters to their ASCII equivalents.
# Although this is not ideal for languages like Chinese, it's how the model was trained.
# For the prompt upsampling client, we therefore ensure that JSON output is ASCII-only by default.
# If this JSON_ENSURE_ASCII environment variable is set to 0, we use `json.dumps(..., ensure_ascii=False)`
# instead and characters like `中文` will be preserved and get tokenized instead of `\\u4e2d\\u6587`.
JSON_ENSURE_ASCII = bool(int(os.environ.get("JSON_ENSURE_ASCII", "1")))

SYSTEM_MESSAGE: dict[str, Any] = {
    "role": "system",
    "content": [{"type": "text", "text": "You are a helpful assistant."}],
}
DEFAULT_USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
log = logging.getLogger(__name__)
DEFAULT_PROMPT_UPSAMPLER_8B_ENDPOINT_URL = "https://b5k2m9x7-cosmos3-reasoner-8b-private.xenon.lepton.run"
DEFAULT_PROMPT_UPSAMPLER_32B_ENDPOINT_URL = "https://b5k2m9x7-cosmos3-reasoner-32b-private.xenon.lepton.run"

RESOLUTION_RATIO_DICT: dict[str, dict[str, dict[str, int]]] = {
    "256": {
        "1,1": {"W": 256, "H": 256},
        "4,3": {"W": 320, "H": 256},
        "3,4": {"W": 256, "H": 320},
        "16,9": {"W": 320, "H": 192},
        "9,16": {"W": 192, "H": 320},
    },
    "480": {
        "1,1": {"W": 640, "H": 640},
        "4,3": {"W": 736, "H": 544},
        "3,4": {"W": 544, "H": 736},
        "16,9": {"W": 832, "H": 480},
        "9,16": {"W": 480, "H": 832},
    },
    "720": {
        "1,1": {"W": 960, "H": 960},
        "4,3": {"W": 1104, "H": 832},
        "3,4": {"W": 832, "H": 1104},
        "16,9": {"W": 1280, "H": 720},
        "9,16": {"W": 720, "H": 1280},
    },
    "768": {
        "1,1": {"W": 1024, "H": 1024},
        "4,3": {"W": 1184, "H": 880},
        "3,4": {"W": 880, "H": 1184},
        "16,9": {"W": 1360, "H": 768},
        "9,16": {"W": 768, "H": 1360},
    },
}

T2I_JSON_TEMPLATE = """{
  "subjects": [
    {
      "description": "full visual description of the subject",
      "appearance_details": "additional visual details (accessories, texture, distinguishing features)",
      "relationship": "how this subject relates to others or to the scene",
      "location": "where in frame (e.g., 'Center foreground', 'Top right')",
      "relative_size": "size within frame",
      "orientation": "direction subject faces relative to camera",
      "pose": "body position and posture",
      "clothing": "clothing and accessories; '' if non-human or N/A",
      "expression": "facial expression; '' if non-human or N/A",
      "gender": "one of 'Male', 'Female', 'Unknown', 'N/A'",
      "age": "age category",
      "skin_tone_and_texture": "skin tone description; '' if non-human",
      "facial_features": "notable facial features, including eye shape/color, hair color/style, lip shape, wrinkles, moles, scars, freckles, facial hair, and other visible fine-grained facial attributes; '' if non-human or not visible",
      "number_of_subjects": "int; total in this subject group, 0 if N/A",
      "number_of_arms": "int; 2 for humans, 0 if non-human",
      "number_of_legs": "int; 2 for humans, 0 if non-human",
      "number_of_hands": "int; 2 for humans, 0 if non-human",
      "number_of_fingers": "int; 10 for humans, 0 if non-human"
    }
  ],
  "subject_details": {
    "key_name_1": "free-form image-specific attribute (keys vary by image content; {} if N/A)"
  },
  "background_setting": "full prose description of the environment and setting",
  "lighting": {
    "conditions": "type and quality of light",
    "direction": "where light comes from; 'None' for flat digital images",
    "shadows": "shadow description; 'None' for flat digital images",
    "illumination_effect": "overall effect of the lighting"
  },
  "aesthetics": {
    "composition": "framing and compositional choices",
    "color_scheme": "dominant colors and palette",
    "mood_atmosphere": "emotional atmosphere in short phrases",
    "patterns": "notable repeating visual patterns; 'None' if none"
  },
  "cinematography": {
    "framing": "shot type",
    "camera_angle": "angle (e.g., 'Eye-level', 'Low angle', 'High angle')",
    "depth_of_field": "'Shallow', 'Deep', 'Uniform focus', or 'N/A'",
    "focus": "what is in sharp focus",
    "lens_focal_length": "descriptive focal length"
  },
  "style_medium": "visual medium (e.g., 'Photography', 'Digital presentation slide', 'Screenshot')",
  "artistic_style": "genre or approach",
  "context": "scene context or use case (brief)",
  "text_and_signage_elements": [
    {
      "text": "the visible text content",
      "category": "one of 'physical_in_scene', 'ui_text', 'body_text', 'scene_sign', 'logo', 'label'",
      "appearance": "font, color, size, style",
      "spatial": "position in image",
      "context": "purpose or meaning of the text"
    }
  ],
  "quadrant_scan": {
    "top_left": "description of what appears in the top-left region",
    "top_right": "description of what appears in the top-right region",
    "bottom_left": "description of what appears in the bottom-left region",
    "bottom_right": "description of what appears in the bottom-right region",
    "absolute_center": "description of what appears at the center"
  },
  "comprehensive_t2i_caption": "a comprehensive, full-scene natural-language prose description of the image",
  "resolution": {
    "H": "must follow resolution_ratio_dict",
    "W": "must follow resolution_ratio_dict"
  },
  "aspect_ratio": "must be one of: '16,9', '1,1', '9,16', '4,3', '3,4'"
}"""

T2V_JSON_TEMPLATE = """{
  "subjects": [
    {
      "description": "full visual description of the subject (appearance, clothing, identifying features)",
      "appearance_details": "additional visual details (accessories, distinguishing features)",
      "relationship": "how this subject relates to others or to the scene",
      "location": "where in frame (e.g., 'center foreground', 'left background')",
      "relative_size": "size within frame (e.g., 'Small within frame', 'Medium within frame', 'Large within frame')",
      "orientation": "direction subject faces relative to camera",
      "pose": "body position and posture",
      "action": "what the subject is doing (brief)",
      "state_changes": "how pose or action changes; 'No significant change.' if static",
      "clothing": "clothing and accessories; '' if non-human or not visible",
      "expression": "facial expression; '' if non-human or not visible",
      "gender": "one of 'Male', 'Female', 'Unknown'; '' if non-human",
      "age": "age category (e.g., 'Child', 'Young adult', 'Adult', 'Middle-aged', 'Elderly')",
      "skin_tone_and_texture": "skin tone description; '' if non-human",
      "facial_features": "notable facial features, including eye shape/color, hair color/style, lip shape, wrinkles, moles, scars, freckles, facial hair, and other visible fine-grained facial attributes; '' if non-human or not visible",
      "number_of_subjects": "int; total in this subject group, 0 if N/A",
      "number_of_arms": "int; 2 for humans, 0 if non-human",
      "number_of_legs": "int; 2 for humans, 0 if non-human"
    }
  ],
  "background_setting": "full prose description of the environment and setting",
  "lighting": {
    "conditions": "type and quality of light (e.g., 'Bright daylight', 'Overcast', 'Studio lighting')",
    "direction": "where light comes from (e.g., 'top-lit', 'front-lit', 'side-lit from right')",
    "shadows": "description of shadows",
    "illumination_effect": "overall effect of the lighting on the scene"
  },
  "aesthetics": {
    "composition": "framing and compositional choices",
    "color_scheme": "dominant colors and palette description",
    "mood_atmosphere": "emotional atmosphere in short phrases",
    "patterns": "notable repeating visual patterns; '' if none"
  },
  "cinematography": {
    "camera_motion": "camera movement (e.g., 'Static', 'Pan left', 'Tracking shot', 'Handheld')",
    "framing": "shot type (e.g., 'Close-up', 'Medium shot', 'Wide shot')",
    "camera_angle": "angle (e.g., 'Eye-level', 'Low angle', 'High angle', 'Overhead')",
    "depth_of_field": "'Shallow', 'Deep', or 'Uniform'",
    "focus": "what is in sharp focus",
    "lens_focal_length": "descriptive focal length"
  },
  "style_medium": "visual medium (e.g., 'Live-action video', 'Animation', 'CGI')",
  "artistic_style": "genre or approach (e.g., 'realistic', 'cinematic', 'documentary')",
  "context": "scene context or use case (brief)",
  "actions": [
    {
      "time": "'M:SS-M:SS' (e.g., '0:00-0:08')",
      "description": "what happens in this timed interval"
    }
  ],
  "text_and_signage_elements": [
    {
      "text": "the visible text content",
      "category": "one of 'physical_in_scene', 'scene_sign', 'ui_text', 'vehicle_graphic', 'logo', 'label'",
      "appearance": "font, color, size, style",
      "spatial_temporal": "position in scene and when visible",
      "context": "purpose or meaning of the text"
    }
  ],
  "segments": [
    {
      "segment_index": "int; 0-based index",
      "time_range": "'M:SS-M:SS' spanning this segment",
      "description": "what happens in this segment",
      "key_changes": "notable changes within the segment",
      "camera": "camera behavior in this segment"
    }
  ],
  "transitions": [
    "transition description between segments; empty list [] if single continuous shot"
  ],
  "temporal_caption": "a temporally coherent, second-by-second natural-language narrative of the whole video",
  "audio_description": "a natural-language description of the audio: speech, music, ambient, and notable effects",
  "resolution": {
    "H": "must follow resolution_ratio_dict",
    "W": "must follow resolution_ratio_dict"
  },
  "aspect_ratio": "must be one of: '16,9', '1,1', '9,16', '4,3', '3,4'",
  "duration": "must be one of: '2s','3s','4s','5s','6s','7s','8s','9s','10s'",
  "fps": "must be one of: 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30"
}"""


def _resolution_ratio_dict_text() -> str:
    resolution_ratio_dict = {
        resolution: {aspect_ratio: {"H": size["H"], "W": size["W"]} for aspect_ratio, size in aspect_ratio_dict.items()}
        for resolution, aspect_ratio_dict in RESOLUTION_RATIO_DICT.items()
    }
    return json.dumps(resolution_ratio_dict, indent=2)


def build_nl_description(
    prompt: str,
    *,
    resolution: str,
    aspect_ratio: str,
    duration: str | None = None,
    fps: int | None = None,
) -> str:
    """Append literal output parameters so the upsampler can extract them."""
    params = [f"resolution {resolution}", f"aspect_ratio {aspect_ratio}"]
    if duration is not None:
        params.append(f"duration {duration}")
    if fps is not None:
        params.append(f"fps {fps}")
    return f"{prompt.strip()}\n\nOutput parameters: {', '.join(params)}."


def derive_duration_label(num_frames: int, fps: int) -> str:
    """Match inference.py's JSON-prompt duration metadata injection."""
    if fps <= 0:
        raise ValueError("fps must be positive.")
    seconds = int(num_frames / fps)
    return f"{seconds}s"


def build_t2i_prompt_text(prompt: str, *, resolution: str, aspect_ratio: str) -> str:
    nl_description = build_nl_description(prompt, resolution=resolution, aspect_ratio=aspect_ratio)
    return f"""Given the user's natural-language request below, generate a dense structured JSON that fully describes the image to be produced. The JSON must strictly follow the template provided after the request, including every top-level key and every nested sub-field.

The output is always DENSE. Even when the request is brief, you must infer plausible, scene-consistent details for every field. Do not leave fields empty merely because the request did not mention them - the purpose of this task is to upsample a sparse request into a rich, complete annotation. Be creative but stay grounded: your additions must be physically plausible and internally consistent with the request.

Requirements:
- Extract any output parameters that the request states literally (resolution, aspect_ratio) and place them in the corresponding JSON fields exactly as stated.
- For every other field, write rich, specific content inferred from the request's scene, subjects, mood, and context.
- Empty values ("", 0, [], {{}}) are permitted ONLY for truly inapplicable fields:
    * Human-only subject fields (clothing, expression, gender, age, skin_tone_and_texture, facial_features, number_of_arms, number_of_legs, number_of_hands, number_of_fingers) when the subject is non-human.
    * text_and_signage_elements = [] when no visible text or signage is present.
    * aesthetics.patterns = "" when there are no notable repeating patterns.
    * subject_details = {{}} when no image-specific structured attributes apply.
- The resolution_ratio_dict appended to the template lists valid (H, W) pairs for each (resolution, aspect_ratio) combination - the "resolution" object in the JSON must match the pair implied by the request.
- Do not add keys beyond the template. Do not omit keys required by the template.

Return only the JSON object wrapped in a ```json code fence.

{nl_description}

Lists (subjects, text_and_signage_elements) may contain zero or more items of the shape shown. All top-level keys must always be present in the output; fill unused fields with "", 0, {{}}, or [] as appropriate.

{T2I_JSON_TEMPLATE}

resolution_ratio_dict = {_resolution_ratio_dict_text()}"""


def build_t2v_prompt_text(
    prompt: str,
    *,
    resolution: str,
    aspect_ratio: str,
    duration: str,
    fps: int,
    image_conditioned: bool = False,
) -> str:
    nl_description = build_nl_description(
        prompt,
        resolution=resolution,
        aspect_ratio=aspect_ratio,
        duration=duration,
        fps=fps,
    )
    intro = "Given the user's natural-language request below"
    image_note = ""
    if image_conditioned:
        intro = "Given the attached starting frame image and the user's natural-language request below"
        image_note = "\nIMPORTANT - IMAGE INPUT: The attached image is the first frame of the video. Use it as visual ground truth for subject appearance, setting, lighting, and colors. The natural-language request primarily describes temporal/action intent. Your JSON must be consistent with what is visible in the image.\n"
    return f"""{intro}, generate a dense structured JSON that fully describes the video to be produced. The JSON must strictly follow the template provided after the request, including every top-level key and every nested sub-field.

The output is always DENSE. Even when the request is brief, you must infer plausible, scene-consistent details for every field. Do not leave fields empty merely because the request did not mention them - the purpose of this task is to upsample a sparse request into a rich, complete annotation. Be creative but stay grounded: your additions must be physically plausible and internally consistent with the request.

Requirements:
- Extract any output parameters that the request states literally (resolution, aspect_ratio, duration, fps) and place them in the corresponding JSON fields exactly as stated.
- For every other field, write rich, specific content inferred from the request's scene, subjects, mood, and context.
- Empty values ("", 0, [], {{}}) are permitted ONLY for truly inapplicable fields:
    * Human-only subject fields (clothing, expression, gender, age, skin_tone_and_texture, facial_features, number_of_arms, number_of_legs) when the subject is non-human.
    * transitions = [] when the video is a single continuous shot.
    * text_and_signage_elements = [] when no visible text or signage is present.
    * aesthetics.patterns = "" when there are no notable repeating patterns.
- The resolution_ratio_dict appended to the template lists valid (H, W) pairs for each (resolution, aspect_ratio) combination - the "resolution" object in the JSON must match the pair implied by the request.
- Do not add keys beyond the template. Do not omit keys required by the template.

Return only the JSON object wrapped in a ```json code fence.
{image_note}
{nl_description}

Lists (subjects, actions, segments, text_and_signage_elements, transitions) may contain zero or more items of the shape shown. All top-level keys must always be present in the output; fill unused fields with "", 0, {{}}, or [] as appropriate.

{T2V_JSON_TEMPLATE}

resolution_ratio_dict = {_resolution_ratio_dict_text()}"""


def build_t2i_messages(
    prompt: str,
    *,
    resolution: str,
    aspect_ratio: str,
    user_prompt: str | None = None,
) -> list[dict[str, Any]]:
    message_text = user_prompt or build_t2i_prompt_text(prompt, resolution=resolution, aspect_ratio=aspect_ratio)
    return [
        SYSTEM_MESSAGE,
        {
            "role": "user",
            "content": [{"type": "text", "text": message_text}],
        },
    ]


def build_t2v_messages(
    prompt: str,
    *,
    resolution: str,
    aspect_ratio: str,
    duration: str,
    fps: int,
    user_prompt: str | None = None,
) -> list[dict[str, Any]]:
    message_text = user_prompt or build_t2v_prompt_text(
        prompt,
        resolution=resolution,
        aspect_ratio=aspect_ratio,
        duration=duration,
        fps=fps,
    )
    return [
        SYSTEM_MESSAGE,
        {
            "role": "user",
            "content": [{"type": "text", "text": message_text}],
        },
    ]


def build_i2v_messages(
    prompt: str,
    *,
    image_url: str,
    resolution: str,
    aspect_ratio: str,
    duration: str,
    fps: int,
    user_prompt: str | None = None,
) -> list[dict[str, Any]]:
    message_text = user_prompt or build_t2v_prompt_text(
        prompt,
        resolution=resolution,
        aspect_ratio=aspect_ratio,
        duration=duration,
        fps=fps,
        image_conditioned=True,
    )
    return [
        SYSTEM_MESSAGE,
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": message_text},
            ],
        },
    ]


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract a JSON object from a raw model response."""
    cleaned = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("Upsampler response JSON must be an object.")
    return parsed


def extract_json_object_text(text: str) -> str:
    """Extract and normalize a JSON object from a raw model response."""
    parsed = extract_json_object(text)
    return json.dumps(parsed, ensure_ascii=JSON_ENSURE_ASCII)


def image_path_to_data_url(path: str | Path) -> str:
    """Encode a local image path as a data URL for OpenAI-compatible VLM requests."""
    image_path = Path(path)
    mime_type = mimetypes.guess_type(str(image_path))[0] or "image/png"
    encoded = b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _compact_json_object(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=JSON_ENSURE_ASCII)


def _apply_t2i_output_parameters(data: dict[str, Any], *, resolution: str, aspect_ratio: str) -> dict[str, Any]:
    if resolution not in RESOLUTION_RATIO_DICT:
        raise ValueError(f"Unsupported upsampler resolution {resolution!r}.")
    if aspect_ratio not in RESOLUTION_RATIO_DICT[resolution]:
        raise ValueError(f"Unsupported upsampler aspect_ratio {aspect_ratio!r} for resolution {resolution!r}.")
    resolution_pair = RESOLUTION_RATIO_DICT[resolution][aspect_ratio]
    data["resolution"] = {"H": resolution_pair["H"], "W": resolution_pair["W"]}
    data["aspect_ratio"] = aspect_ratio
    return data


def _apply_t2v_output_parameters(
    data: dict[str, Any],
    *,
    resolution: str,
    aspect_ratio: str,
    duration: str,
    fps: int,
) -> dict[str, Any]:
    data = _apply_t2i_output_parameters(data, resolution=resolution, aspect_ratio=aspect_ratio)
    data["duration"] = duration
    data["fps"] = fps
    return data


@dataclass(slots=True)
class PromptUpsamplerConfig:
    endpoint_url: str
    model: str | None = None
    api_token: str | None = None
    timeout_s: float = 300.0
    max_tokens: int = 8192
    max_retries: int = 5
    retry_base_delay_s: float = 1.0
    temperature: float | None = 0.7
    top_p: float | None = 0.8
    top_k: int | None = 20
    min_p: float | None = 0.0
    connection_max_retries: int = 2
    connection_pool_size: int = 4


class PromptUpsamplerClient:
    """Small OpenAI-compatible chat-completions client with explicit retries."""

    def __init__(
        self,
        config: PromptUpsamplerConfig,
        *,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self._base_url = _normalize_openai_base_url(config.endpoint_url)
        self._session = _make_session(config) if session is None else session
        self._sleep = sleep

    def list_models(self) -> list[str]:
        payload = self._with_retries("list models", lambda: self._request_json("GET", f"{self._base_url}/models"))
        data = payload.get("data")
        if not isinstance(data, list):
            raise ValueError("Model list response missing 'data' list.")
        models: list[str] = []
        for item in data:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                models.append(item["id"])
        if not models:
            raise ValueError("Model list response did not include any model ids.")
        return models

    def upsample_t2i(
        self,
        prompt: str,
        *,
        resolution: str,
        aspect_ratio: str,
        user_prompt: str | None = None,
    ) -> str:
        messages = build_t2i_messages(
            prompt,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            user_prompt=user_prompt,
        )
        return self._upsample_messages_with_parameters(
            messages,
            lambda data: _apply_t2i_output_parameters(data, resolution=resolution, aspect_ratio=aspect_ratio),
        )

    def upsample_t2v(
        self,
        prompt: str,
        *,
        resolution: str,
        aspect_ratio: str,
        duration: str,
        fps: int,
        user_prompt: str | None = None,
    ) -> str:
        messages = build_t2v_messages(
            prompt,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            duration=duration,
            fps=fps,
            user_prompt=user_prompt,
        )
        return self._upsample_messages_with_parameters(
            messages,
            lambda data: _apply_t2v_output_parameters(
                data,
                resolution=resolution,
                aspect_ratio=aspect_ratio,
                duration=duration,
                fps=fps,
            ),
        )

    def upsample_i2v(
        self,
        prompt: str,
        *,
        image_url: str,
        resolution: str,
        aspect_ratio: str,
        duration: str,
        fps: int,
        user_prompt: str | None = None,
    ) -> str:
        messages = build_i2v_messages(
            prompt,
            image_url=image_url,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            duration=duration,
            fps=fps,
            user_prompt=user_prompt,
        )
        return self._upsample_messages_with_parameters(
            messages,
            lambda data: _apply_t2v_output_parameters(
                data,
                resolution=resolution,
                aspect_ratio=aspect_ratio,
                duration=duration,
                fps=fps,
            ),
        )

    def upsample_messages(self, messages: list[dict[str, Any]]) -> str:
        def _call() -> str:
            content = self._chat_completion(messages)
            return extract_json_object_text(content)

        return self._with_retries("upsample prompt", _call)

    def _upsample_messages_with_parameters(
        self,
        messages: list[dict[str, Any]],
        apply_parameters: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> str:
        def _call() -> str:
            content = self._chat_completion(messages)
            data = apply_parameters(extract_json_object(content))
            return _compact_json_object(data)

        return self._with_retries("upsample prompt", _call)

    def _get_model(self) -> str:
        if self.config.model:
            return self.config.model
        env_model = os.environ.get("PROMPT_UPSAMPLER_MODEL")
        if env_model:
            self.config.model = env_model
            return env_model
        self.config.model = self.list_models()[0]
        return self.config.model

    def _is_anthropic(self) -> bool:
        """Detect if the endpoint is Anthropic's native API."""
        return "api.anthropic.com" in self._base_url

    def _chat_completion(self, messages: list[dict[str, Any]]) -> str:
        model = self._get_model()
        log.debug(
            f"[prompt-upsampling] _chat_completion: model={model}, base_url={self._base_url}, anthropic={self._is_anthropic()}"
        )

        if self._is_anthropic():
            return self._anthropic_completion(model, messages)

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        if self.config.top_p is not None:
            payload["top_p"] = self.config.top_p
        if self.config.top_k is not None:
            payload["top_k"] = self.config.top_k
        if self.config.min_p is not None:
            payload["min_p"] = self.config.min_p
        response = self._request_json("POST", f"{self._base_url}/chat/completions", payload=payload)
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("Chat completion response missing choices.")
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise ValueError("Chat completion choice must be an object.")
        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise ValueError("Chat completion choice missing message.")
        return _message_content_to_text(message.get("content"))

    def _anthropic_completion(self, model: str, messages: list[dict[str, Any]]) -> str:
        """Call Anthropic's native /v1/messages API."""
        # Extract system message and convert remaining messages
        system_text = ""
        user_messages: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            # Flatten content if it's a list of {"type": "text", "text": ...}
            if isinstance(content, list):
                text_parts = [item["text"] for item in content if isinstance(item, dict) and item.get("type") == "text"]
                content = "\n".join(text_parts)
            if role == "system":
                system_text = content
            else:
                user_messages.append({"role": role, "content": content})

        payload: dict[str, Any] = {
            "model": model,
            "messages": user_messages,
            "max_tokens": self.config.max_tokens,
        }
        if system_text:
            payload["system"] = system_text

        response = self._request_json("POST", f"{self._base_url}/v1/messages", payload=payload)
        content = response.get("content")
        if not isinstance(content, list) or not content:
            raise ValueError(f"Anthropic response missing content: {response}")
        # Extract text from content blocks
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        result = "\n".join(text_parts).strip()
        if not result:
            raise ValueError(f"Anthropic response had no text content: {response}")
        return result

    def _request_json(self, method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        headers = {"Accept": "application/json", "User-Agent": DEFAULT_USER_AGENT}
        if payload is not None:
            headers["Content-Type"] = "application/json"

        token = self.config.api_token

        if self._is_anthropic():
            # Anthropic uses x-api-key header and requires anthropic-version
            if token:
                headers["x-api-key"] = token
            headers["anthropic-version"] = "2023-06-01"
        else:
            if token:
                headers["Authorization"] = f"Bearer {token}"

        log.debug(
            f"[prompt-upsampling] _request_json: {method} {url} token={'***' + token[-4:] if token else '(none)'}"
        )
        try:
            response = self._session.request(
                method,
                url,
                json=payload,
                headers=headers,
                timeout=self.config.timeout_s,
            )
        except requests.RequestException as exc:
            log.debug(f"[prompt-upsampling] _request_json FAILED: {exc}")
            raise RuntimeError(f"Failed to reach {url}: {exc}") from exc

        log.debug(f"[prompt-upsampling] _request_json response: status={response.status_code}")
        if not response.ok:
            raise RuntimeError(f"HTTP {response.status_code} from {url}: {response.text[:1000]}")

        try:
            parsed = response.json()
        except ValueError as exc:
            raise RuntimeError(f"Response from {url} was not valid JSON: {response.text[:1000]}") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError(f"Response from {url} must be a JSON object.")
        return parsed

    def _with_retries(self, operation: str, fn: Callable[[], Any]) -> Any:
        if self.config.max_retries < 1:
            raise ValueError("max_retries must be >= 1.")
        last_exc: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                return fn()
            except Exception as exc:
                last_exc = exc
                if attempt == self.config.max_retries - 1:
                    break
                self._sleep(self.config.retry_base_delay_s * (2**attempt))
        raise RuntimeError(
            f"Prompt upsampler failed to {operation} after {self.config.max_retries} attempts: {last_exc}"
        ) from last_exc


def _normalize_openai_base_url(url: str) -> str:
    normalized = url.strip().rstrip("/")
    if not normalized:
        raise ValueError("endpoint_url cannot be empty.")
    if not normalized.startswith(("http://", "https://")):
        normalized = f"https://{normalized}"
    if normalized.endswith("/chat/completions"):
        normalized = normalized[: -len("/chat/completions")]
    if normalized.endswith("/v1/messages"):
        normalized = normalized[: -len("/v1/messages")]
    # Don't append /v1 for Anthropic — the path is handled per-method.
    if "api.anthropic.com" in normalized:
        return normalized.rstrip("/")
    if not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"
    return normalized


def _make_session(config: PromptUpsamplerConfig) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=config.connection_max_retries,
        connect=config.connection_max_retries,
        read=0,
        status=0,
        backoff_factor=0.25,
        allowed_methods=None,
    )
    adapter = HTTPAdapter(
        pool_connections=config.connection_pool_size,
        pool_maxsize=config.connection_pool_size,
        max_retries=retry,
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
        text = "".join(parts).strip()
        if text:
            return text
    raise ValueError("Chat completion message content is empty or unsupported.")
