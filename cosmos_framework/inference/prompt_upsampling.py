# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Prompt upsampling client for Cosmos3 generation/evaluation scripts.

This module has two layers:

1. Prompt builders that turn a terse user prompt into a model-specific chat
   request for text-to-image, text-to-video, or image-to-video upsampling.
2. A small OpenAI-compatible HTTP client and CLI that send those requests to an
   external upsampler endpoint and write normalized JSON prompts to disk.

The code is intentionally endpoint-light: all prompt shape decisions live here,
while authentication, model choice, and endpoint URL are supplied by callers or
environment variables.
"""

from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import os
import re
import sys
import time
from base64 import b64encode
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from string import Template
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
PROMPT_UPSAMPLER_MODES = ("text2image", "text2video", "image2video", "posttrain_image2video")

# Prompt templates are stored outside this Python file so teams can iterate on
# prompt wording/schema contracts without editing the request/CLI plumbing.
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

PROMPTING_TEMPLATES_DIR = Path(__file__).with_name("prompting_templates") / "external_api"


@lru_cache(maxsize=None)
def _load_prompting_template(filename: str) -> str:
    """Load a built-in template once and strip the trailing newline.

    ``Template.substitute`` is used later, so template files should use
    ``$json_template``, ``$nl_description``, and ``$resolution_ratio_dict``
    placeholders rather than Python ``str.format`` braces.
    """
    return (PROMPTING_TEMPLATES_DIR / filename).read_text(encoding="utf-8").rstrip("\n")


T2I_JSON_TEMPLATE = _load_prompting_template("t2i_json_schema.json")
T2V_JSON_TEMPLATE = _load_prompting_template("t2v_i2v_video_json_schema.json")
T2I_PROMPT_TEMPLATE = Template(_load_prompting_template("t2i_prompt.txt"))
T2V_PROMPT_TEMPLATE = Template(_load_prompting_template("t2v_i2v_video_prompt.txt"))
POSTTRAIN_I2V_JSON_TEMPLATE = _load_prompting_template("posttrained_i2v_json_schema.json")
POSTTRAIN_I2V_PROMPT_TEMPLATE = Template(_load_prompting_template("posttrained_i2v_prompt.txt"))


def _resolution_ratio_dict_text() -> str:
    """Return the valid output resolution table rendered for prompt injection."""
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
    """Append literal output parameters to the user prompt.

    The external upsampler receives one text field, so we make generation
    metadata explicit in prose. The templates instruct the model to copy these
    values back into the JSON output.
    """
    params = [f"resolution {resolution}", f"aspect_ratio {aspect_ratio}"]
    if duration is not None:
        params.append(f"duration {duration}")
    if fps is not None:
        params.append(f"fps {fps}")
    return f"{prompt.strip()}\n\nOutput parameters: {', '.join(params)}."


def derive_duration_label(num_frames: int, fps: int) -> str:
    """Convert frame count and FPS to the compact duration label used by inference."""
    if fps <= 0:
        raise ValueError("fps must be positive.")
    seconds = int(num_frames / fps)
    return f"{seconds}s"


def build_t2i_prompt_text(prompt: str, *, resolution: str, aspect_ratio: str) -> str:
    """Render the complete text-to-image upsampler prompt."""
    nl_description = build_nl_description(prompt, resolution=resolution, aspect_ratio=aspect_ratio)
    return T2I_PROMPT_TEMPLATE.substitute(
        json_template=T2I_JSON_TEMPLATE,
        nl_description=nl_description,
        resolution_ratio_dict=_resolution_ratio_dict_text(),
    )


def build_t2v_prompt_text(
    prompt: str,
    *,
    resolution: str,
    aspect_ratio: str,
    duration: str,
    fps: int,
    image_conditioned: bool = False,
) -> str:
    """Render the complete video upsampler prompt.

    ``image_conditioned`` keeps I2V on the same JSON schema as T2V while adding
    instructions that the attached image is visual ground truth for frame 0.
    """
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
    return T2V_PROMPT_TEMPLATE.substitute(
        image_note=image_note,
        intro=intro,
        json_template=T2V_JSON_TEMPLATE,
        nl_description=nl_description,
        resolution_ratio_dict=_resolution_ratio_dict_text(),
    )


def build_posttrain_i2v_prompt_text(prompt: str) -> str:
    """Render the posttrained image-to-video upsampler prompt.

    Unlike the external-API video template, the posttrained template fills the
    output-parameter fields (resolution/aspect_ratio/duration/fps) post-hoc, so
    the raw user instruction is passed through verbatim without an appended
    ``Output parameters`` summary.
    """
    return POSTTRAIN_I2V_PROMPT_TEMPLATE.substitute(
        json_template=POSTTRAIN_I2V_JSON_TEMPLATE,
        nl_description=prompt.strip(),
    )


def build_t2i_messages(
    prompt: str,
    *,
    resolution: str,
    aspect_ratio: str,
    user_prompt: str | None = None,
) -> list[dict[str, Any]]:
    """Build OpenAI-compatible chat messages for text-to-image upsampling."""
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
    """Build OpenAI-compatible chat messages for text-to-video upsampling."""
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
    """Build OpenAI-compatible chat messages for image-to-video upsampling.

    ``image_url`` may be a remote URL, a data URL, or a gateway-specific raw
    base64 string. The caller owns converting local files to a supported form.
    """
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


def build_posttrain_i2v_messages(
    prompt: str,
    *,
    image_url: str,
    user_prompt: str | None = None,
) -> list[dict[str, Any]]:
    """Build OpenAI-compatible chat messages for posttrained image-to-video upsampling.

    ``image_url`` may be a remote URL, a data URL, or a gateway-specific raw
    base64 string. The caller owns converting local files to a supported form.
    """
    message_text = user_prompt or build_posttrain_i2v_prompt_text(prompt)
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": message_text},
            ],
        },
    ]


def _extract_xml_tag(text: str, tag: str) -> str | None:
    """Return the stripped inner text of the first ``<tag>...</tag>`` block, if present."""
    match = re.search(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", text, flags=re.DOTALL)
    if match is None:
        return None
    inner = match.group(1).strip()
    return inner or None


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract a JSON object from a raw model response.

    Upsamplers may return either a bare JSON object or a fenced `````json```
    block. This helper normalizes both forms and rejects non-object payloads.
    """
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


def image_path_to_base64(path: str | Path) -> str:
    """Encode a local image path as raw base64 for gateways that do not accept data URLs."""
    return b64encode(Path(path).read_bytes()).decode("ascii")


def _compact_json_object(data: dict[str, Any]) -> str:
    """Serialize a JSON object in the compact format expected by inference."""
    return json.dumps(data, ensure_ascii=JSON_ENSURE_ASCII)


def _apply_t2i_output_parameters(data: dict[str, Any], *, resolution: str, aspect_ratio: str) -> dict[str, Any]:
    """Force canonical image metadata into an upsampled T2I JSON object."""
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
    """Force canonical video metadata into an upsampled T2V/I2V JSON object."""
    data = _apply_t2i_output_parameters(data, resolution=resolution, aspect_ratio=aspect_ratio)
    data["duration"] = duration
    data["fps"] = fps
    return data


@dataclass(slots=True)
class PromptUpsamplerConfig:
    """Connection and sampling settings for ``PromptUpsamplerClient``.

    ``endpoint_url`` may be a bare host, a ``/v1`` base URL, or a full
    ``/chat/completions`` URL; it is normalized by the client. Sampling
    parameters set to ``None`` are omitted from the request payload so the same
    client can work across gateways with different accepted knobs.
    """

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
    min_p: float | None = None
    connection_max_retries: int = 2
    connection_pool_size: int = 4


class PromptUpsamplerClient:
    """Small OpenAI-compatible chat-completions client with explicit retries.

    The client deliberately avoids depending on the OpenAI SDK so it can be
    used in lightweight eval scripts.
    """

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
        """Fetch model ids from an OpenAI-compatible ``/models`` endpoint."""
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
        """Upsample a text-to-image prompt and return a compact JSON string."""
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
        """Upsample a text-to-video prompt and return a compact JSON string."""
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
        """Upsample an image-to-video prompt and return a compact JSON string."""
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

    def upsample_posttrain_i2v(
        self,
        prompt: str,
        *,
        image_url: str,
        resolution: str,
        aspect_ratio: str,
        duration: str,
        fps: int,
        user_prompt: str | None = None,
    ) -> dict[str, Any]:
        """Upsample a posttrained image-to-video prompt into a positive/negative record.

        The posttrained Cosmos3-Super-Image2Video contract emits a
        ``<final_prompt>`` JSON block plus a per-sample ``<negative_prompt>``
        block. The positive prompt is returned as a compact JSON string under
        ``prompt`` (with output parameters pinned post-hoc), and the negative
        prompt, when present, is returned as a sibling string.
        """
        messages = build_posttrain_i2v_messages(prompt, image_url=image_url, user_prompt=user_prompt)

        def _call() -> dict[str, Any]:
            content = self._chat_completion(messages)
            # The JSON description must live inside <final_prompt> tags; a
            # response without them violates the posttrained contract and is
            # retried rather than silently parsed from arbitrary text.
            final_prompt = _extract_xml_tag(content, "final_prompt")
            if final_prompt is None:
                raise ValueError("Posttrained upsampler response missing <final_prompt> block.")
            data = _apply_t2v_output_parameters(
                extract_json_object(final_prompt),
                resolution=resolution,
                aspect_ratio=aspect_ratio,
                duration=duration,
                fps=fps,
            )
            record: dict[str, Any] = {"prompt": _compact_json_object(data)}
            negative = _extract_xml_tag(content, "negative_prompt")
            if negative is not None:
                record["negative_prompt"] = negative
            return record

        return self._with_retries("upsample prompt", _call)

    def upsample_messages(self, messages: list[dict[str, Any]]) -> str:
        """Upsample a pre-built chat message list without injecting metadata."""

        def _call() -> str:
            content = self._chat_completion(messages)
            return extract_json_object_text(content)

        return self._with_retries("upsample prompt", _call)

    def _upsample_messages_with_parameters(
        self,
        messages: list[dict[str, Any]],
        apply_parameters: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> str:
        """Call the model, parse the JSON object, and pin output parameters.

        Pinning metadata after the model response makes the output robust to
        small formatting/copy mistakes by the LLM and keeps inference metadata
        exactly aligned with CLI args.
        """

        def _call() -> str:
            content = self._chat_completion(messages)
            data = apply_parameters(extract_json_object(content))
            return _compact_json_object(data)

        return self._with_retries("upsample prompt", _call)

    def _get_model(self) -> str:
        """Resolve the model name from config, env, or endpoint discovery."""
        if self.config.model:
            return self.config.model
        env_model = os.environ.get("PROMPT_UPSAMPLER_MODEL")
        if env_model:
            self.config.model = env_model
            return env_model
        self.config.model = self.list_models()[0]
        return self.config.model

    def _chat_completion(self, messages: list[dict[str, Any]]) -> str:
        """Send messages to the configured endpoint and return assistant text."""
        model = self._get_model()
        log.debug(f"[prompt-upsampling] _chat_completion: model={model}, base_url={self._base_url}")

        # Keep provider-specific sampling params optional. Several gateways
        # reject unsupported keys instead of ignoring them.
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

    def _request_json(self, method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Issue one HTTP request and parse a JSON object response."""
        headers = {"Accept": "application/json", "User-Agent": DEFAULT_USER_AGENT}
        if payload is not None:
            headers["Content-Type"] = "application/json"

        token = self.config.api_token
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
        """Retry transient endpoint/model failures with exponential backoff."""
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
    """Normalize user-provided endpoint strings to a request base URL."""
    normalized = url.strip().rstrip("/")
    if not normalized:
        raise ValueError("endpoint_url cannot be empty.")
    if not normalized.startswith(("http://", "https://")):
        normalized = f"https://{normalized}"
    if normalized.endswith("/chat/completions"):
        normalized = normalized[: -len("/chat/completions")]
    if not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"
    return normalized


def _make_session(config: PromptUpsamplerConfig) -> requests.Session:
    """Create a requests session with connection-level retry behavior."""
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
    """Convert OpenAI message content variants into plain assistant text."""
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


def _normalize_prompt_upsampler_mode(mode: str) -> str:
    """Validate and normalize the CLI mode name."""
    normalized = mode.strip().lower().replace("-", "_")
    if normalized not in PROMPT_UPSAMPLER_MODES:
        valid_modes = ", ".join(PROMPT_UPSAMPLER_MODES)
        raise ValueError(f"Unsupported prompt upsampling mode {mode!r}. Valid modes: {valid_modes}.")
    return normalized


def _read_prompt_lines(path: str | Path) -> list[str]:
    """Read one prompt per non-empty line from a text file."""
    prompts = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()]
    return [prompt for prompt in prompts if prompt]


def _read_optional_image_lines(path: str | Path | None) -> list[str] | None:
    """Read optional one-image-per-prompt entries for I2V mode."""
    if path is None:
        return None
    image_lines = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()]
    return [image_line for image_line in image_lines if image_line]


def _resolve_image_url(image: str) -> str:
    """Resolve an image-list entry into the form sent to the endpoint.

    Remote URLs and explicit data URLs pass through unchanged. Existing local
    files are encoded as data URLs to match the Streamlit/server I2V path. If a
    path-like string cannot be inspected, it is passed through so callers can
    provide gateway-specific references.
    """
    if image.startswith(("http://", "https://", "data:")):
        return image
    image_path = Path(image)
    try:
        if image_path.exists():
            return image_path_to_data_url(image_path)
    except OSError:
        return image
    return image


def _optional_float(value: str) -> float | None:
    """Parse a float CLI argument that can also be disabled with ``none``."""
    if value.strip().lower() in {"none", "null"}:
        return None
    return float(value)


def configure_prompting_templates(
    *,
    mode: str,
    prompt_template_path: str | Path | None = None,
    json_template_path: str | Path | None = None,
) -> None:
    """Override prompt upsampler templates for the selected mode.

    The default templates target the external API prompt format, but eval jobs
    sometimes need reasoner-style or experiment-specific prompts. This function
    swaps the in-memory templates before any prompt text is built. It is global
    by design because the module-level builders use cached ``Template``
    instances.
    """
    if prompt_template_path is None and json_template_path is None:
        return

    global T2I_JSON_TEMPLATE, T2I_PROMPT_TEMPLATE, T2V_JSON_TEMPLATE, T2V_PROMPT_TEMPLATE
    global POSTTRAIN_I2V_JSON_TEMPLATE, POSTTRAIN_I2V_PROMPT_TEMPLATE

    normalized_mode = _normalize_prompt_upsampler_mode(mode)
    if normalized_mode == "posttrain_image2video":
        # Posttrained I2V keeps its own template slots, fully separate from the
        # external-API T2V/I2V template so the two contracts never collide.
        if json_template_path is not None:
            POSTTRAIN_I2V_JSON_TEMPLATE = Path(json_template_path).read_text(encoding="utf-8").rstrip("\n")
        if prompt_template_path is not None:
            prompt_template = Path(prompt_template_path).read_text(encoding="utf-8").rstrip("\n")
            POSTTRAIN_I2V_PROMPT_TEMPLATE = Template(prompt_template)
        return

    if normalized_mode == "text2image":
        # T2I has its own schema and prompt contract.
        if json_template_path is not None:
            T2I_JSON_TEMPLATE = Path(json_template_path).read_text(encoding="utf-8").rstrip("\n")
        if prompt_template_path is not None:
            prompt_template = Path(prompt_template_path).read_text(encoding="utf-8").rstrip("\n")
            T2I_PROMPT_TEMPLATE = Template(prompt_template)
        return

    if normalized_mode in {"text2video", "image2video"}:
        # T2V and I2V both use the video template slots in this module. A
        # caller can still pass I2V-specific files through --prompt-template
        # and --json-template before any prompt text is rendered.
        if json_template_path is not None:
            T2V_JSON_TEMPLATE = Path(json_template_path).read_text(encoding="utf-8").rstrip("\n")
        if prompt_template_path is not None:
            prompt_template = Path(prompt_template_path).read_text(encoding="utf-8").rstrip("\n")
            T2V_PROMPT_TEMPLATE = Template(prompt_template)
        return

    raise AssertionError(f"Unhandled prompt upsampling mode: {normalized_mode}")


def _prompt_record(upsampled_json: str) -> dict[str, Any]:
    """Wrap an upsampled JSON string into the unified record.

    The string is re-parsed through ``extract_json_object`` (validating it is a
    JSON object) and re-serialized to the compact form, so every mode emits a
    consistently-normalized ``prompt`` string.
    """
    return {"prompt": _compact_json_object(extract_json_object(upsampled_json))}


def _upsample_prompt_for_mode(
    client: PromptUpsamplerClient,
    prompt: str,
    *,
    mode: str,
    resolution: str,
    aspect_ratio: str,
    duration: str,
    fps: int,
    image_url: str | None = None,
) -> dict[str, Any]:
    """Dispatch one prompt to the modality-specific client method.

    Returns a unified JSON record for this prompt: every mode produces a compact
    JSON ``prompt`` string, and ``posttrain_image2video`` additionally includes
    a ``negative_prompt`` string when the model returns one.
    """
    normalized_mode = _normalize_prompt_upsampler_mode(mode)
    if normalized_mode == "text2image":
        return _prompt_record(client.upsample_t2i(prompt, resolution=resolution, aspect_ratio=aspect_ratio))
    if normalized_mode == "text2video":
        return _prompt_record(
            client.upsample_t2v(
                prompt,
                resolution=resolution,
                aspect_ratio=aspect_ratio,
                duration=duration,
                fps=fps,
            )
        )
    if normalized_mode == "image2video":
        if image_url is None:
            raise ValueError("image2video mode requires --image-url or --image-list.")
        return _prompt_record(
            client.upsample_i2v(
                prompt,
                image_url=_resolve_image_url(image_url),
                resolution=resolution,
                aspect_ratio=aspect_ratio,
                duration=duration,
                fps=fps,
            )
        )
    if normalized_mode == "posttrain_image2video":
        if image_url is None:
            raise ValueError("posttrain_image2video mode requires --image-url or --image-list.")
        return client.upsample_posttrain_i2v(
            prompt,
            image_url=_resolve_image_url(image_url),
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            duration=duration,
            fps=fps,
        )
    raise AssertionError(f"Unhandled prompt upsampling mode: {normalized_mode}")


def upsample_prompt_file(
    client: PromptUpsamplerClient,
    *,
    input_path: str | Path,
    output_path: str | Path,
    mode: str,
    resolution: str,
    aspect_ratio: str,
    duration: str,
    fps: int,
    image_url: str | None = None,
    image_list_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Upsample one prompt per input line and write JSON records to disk.

    Input is intentionally simple: one non-empty line per prompt. For I2V, the
    optional image list must have the same number of non-empty lines and is
    matched by index. Output is one ``prompt_<index>.json`` file per prompt in
    the requested output directory.

    Every record has the same shape: a compact JSON ``prompt`` string, plus a
    ``negative_prompt`` string for ``posttrain_image2video`` when the model
    returns one.
    """
    prompts = _read_prompt_lines(input_path)
    if not prompts:
        raise ValueError(f"No prompts found in {input_path}.")

    image_urls = _read_optional_image_lines(image_list_path)
    if image_urls is not None and len(image_urls) != len(prompts):
        raise ValueError(f"Expected {len(prompts)} image entries in {image_list_path}, found {len(image_urls)}.")

    results: list[dict[str, Any]] = []
    for index, prompt in enumerate(prompts):
        # ``current_image_url`` remains None for text-only modes and is
        # validated only if the selected mode actually needs it.
        current_image_url = image_urls[index] if image_urls is not None else image_url
        record = _upsample_prompt_for_mode(
            client,
            prompt,
            mode=mode,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            duration=duration,
            fps=fps,
            image_url=current_image_url,
        )
        results.append(record)

    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    for index, result in enumerate(results):
        prompt_output = output / f"prompt_{index}.json"
        prompt_output.write_text(json.dumps(result, ensure_ascii=JSON_ENSURE_ASCII, indent=2) + "\n", encoding="utf-8")
    return results


def build_cli_parser() -> argparse.ArgumentParser:
    """Build the standalone CLI parser for batch prompt upsampling."""
    parser = argparse.ArgumentParser(description="Upsample a text file of prompts into Cosmos3 JSON prompts.")
    parser.add_argument("--input", required=True, help="Text file with one prompt per non-empty line.")
    parser.add_argument("--output", required=True, help="Output directory for per-prompt JSON files.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=PROMPT_UPSAMPLER_MODES,
        help="Prompt upsampling mode, e.g. text2image, text2video, or image2video.",
    )
    parser.add_argument(
        "--endpoint-url",
        default=os.environ.get("PROMPT_UPSAMPLER_ENDPOINT_URL"),
        help="OpenAI-compatible endpoint URL. Defaults to PROMPT_UPSAMPLER_ENDPOINT_URL.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("PROMPT_UPSAMPLER_MODEL"),
        help="Model name. Defaults to PROMPT_UPSAMPLER_MODEL.",
    )
    parser.add_argument(
        "--api-token",
        default=os.environ.get("PROMPT_UPSAMPLER_API_TOKEN"),
        help="API token. Defaults to PROMPT_UPSAMPLER_API_TOKEN.",
    )
    parser.add_argument(
        "--prompt-template",
        default=os.environ.get("PROMPT_UPSAMPLER_PROMPT_TEMPLATE"),
        help=(
            "Prompt template path for the selected mode. Defaults to PROMPT_UPSAMPLER_PROMPT_TEMPLATE, "
            "then the built-in template."
        ),
    )
    parser.add_argument(
        "--json-template",
        default=os.environ.get("PROMPT_UPSAMPLER_JSON_TEMPLATE"),
        help=(
            "JSON schema template path for the selected mode. Defaults to PROMPT_UPSAMPLER_JSON_TEMPLATE, "
            "then the built-in template."
        ),
    )
    parser.add_argument("--resolution", default="480", choices=sorted(RESOLUTION_RATIO_DICT), help="Resolution tier.")
    parser.add_argument("--aspect-ratio", default="16,9", help="Aspect ratio key, e.g. 16,9, 9,16, or 1,1.")
    parser.add_argument("--duration", default="5s", help="Video duration metadata for text2video/image2video.")
    parser.add_argument("--fps", type=int, default=24, help="Video FPS metadata for text2video/image2video.")
    parser.add_argument(
        "--image-url", default=None, help="Shared image URL or local image path for all image2video prompts."
    )
    parser.add_argument(
        "--image-list",
        default=None,
        help="Text file with one image URL or local image path per prompt for image2video.",
    )
    parser.add_argument("--timeout-s", type=float, default=300.0, help="Request timeout in seconds.")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Maximum response tokens.")
    parser.add_argument("--max-retries", type=int, default=5, help="Prompt upsampling retries per prompt.")
    parser.add_argument("--retry-base-delay-s", type=float, default=1.0, help="Base exponential retry delay.")
    parser.add_argument(
        "--temperature", type=_optional_float, default=None, help="Optional sampling temperature, or 'none'."
    )
    parser.add_argument("--top-p", type=_optional_float, default=None, help="Optional sampling top-p, or 'none'.")
    parser.add_argument("--top-k", type=int, default=20, help="Sampling top-k.")
    parser.add_argument("--min-p", type=float, default=None, help="Optional sampling min-p.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser


def cli_main(argv: list[str] | None = None) -> int:
    """CLI entry point used by ``python -m cosmos_framework.inference.prompt_upsampling``."""
    parser = build_cli_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    if args.endpoint_url is None:
        parser.error("--endpoint-url is required unless PROMPT_UPSAMPLER_ENDPOINT_URL is set.")
    if args.image_url is not None and args.image_list is not None:
        parser.error("Pass only one of --image-url or --image-list.")

    configure_prompting_templates(
        mode=args.mode,
        prompt_template_path=args.prompt_template,
        json_template_path=args.json_template,
    )

    # Build the client after CLI validation so endpoint/model/token mistakes
    # fail before any input file is processed.
    config = PromptUpsamplerConfig(
        endpoint_url=args.endpoint_url,
        model=args.model,
        api_token=args.api_token,
        timeout_s=args.timeout_s,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        retry_base_delay_s=args.retry_base_delay_s,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
    )
    client = PromptUpsamplerClient(config)
    results = upsample_prompt_file(
        client,
        input_path=args.input,
        output_path=args.output,
        mode=args.mode,
        resolution=args.resolution,
        aspect_ratio=args.aspect_ratio,
        duration=args.duration,
        fps=args.fps,
        image_url=args.image_url,
        image_list_path=args.image_list,
    )
    log.info("Wrote %d upsampled prompts to %s", len(results), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(cli_main())
