# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import json
import typing
from pathlib import Path
from typing import Any

import pytest
import requests
import typing_extensions

import cosmos_framework.inference.prompt_upsampling as prompt_upsampling
from cosmos_framework.inference.prompt_upsampling import (
    PromptUpsamplerClient,
    PromptUpsamplerConfig,
    build_i2v_messages,
    build_t2i_messages,
    build_t2v_messages,
    derive_duration_label,
    extract_json_object_text,
    image_path_to_data_url,
)

if not hasattr(typing, "Self"):
    typing.Self = typing_extensions.Self  # type: ignore[attr-defined]
if not hasattr(typing, "override"):
    typing.override = typing_extensions.override  # type: ignore[attr-defined]

from cosmos_framework.inference.args import OmniSampleOverrides


class _FakeTokenizerConfig:
    temporal_compression_factor: int = 4


class _FakeVlmConfig:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name


class _FakeModelConfig:
    def __init__(self, model_name: str) -> None:
        self.resolution = "480"
        self.tokenizer = _FakeTokenizerConfig()
        self.vlm_config = _FakeVlmConfig(model_name)


class _FakePromptUpsamplerClient:
    configs: list[Any] = []
    calls: list[dict[str, Any]] = []

    def __init__(self, config: Any) -> None:
        self.config = config
        self.configs.append(config)

    def upsample_t2i(
        self,
        prompt: str,
        *,
        resolution: str,
        aspect_ratio: str,
        user_prompt: str | None = None,
    ) -> str:
        self.calls.append({"prompt": prompt, "resolution": resolution, "aspect_ratio": aspect_ratio})
        return '{"subjects": [], "resolution": {"H": 640, "W": 640}, "aspect_ratio": "1,1"}'

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
        self.calls.append(
            {
                "prompt": prompt,
                "resolution": resolution,
                "aspect_ratio": aspect_ratio,
                "duration": duration,
                "fps": fps,
            }
        )
        return (
            '{"subjects": [], "resolution": {"H": 480, "W": 832}, "aspect_ratio": "16,9", "duration": "8s", "fps": 24}'
        )


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.ok = True
        self.status_code = 200
        self.text = json.dumps(payload)

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeSession:
    def __init__(self, responses: list[dict[str, Any]] | None = None, exc: Exception | None = None) -> None:
        self.responses = responses or []
        self.exc = exc
        self.requests: list[dict[str, Any]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> _FakeResponse:
        self.requests.append({"method": method, "url": url, **kwargs})
        if self.exc is not None:
            raise self.exc
        return _FakeResponse(self.responses.pop(0))


def test_build_templates_include_output_parameters() -> None:
    t2i_messages = build_t2i_messages("a steel robot arm", resolution="480", aspect_ratio="1,1")
    t2v_messages = build_t2v_messages(
        "a steel robot arm assembling a gear",
        resolution="720",
        aspect_ratio="16,9",
        duration="8s",
        fps=24,
    )
    i2v_messages = build_i2v_messages(
        "the object starts rotating",
        image_url="data:image/png;base64,abc",
        resolution="480",
        aspect_ratio="16,9",
        duration="5s",
        fps=20,
    )

    t2i_text = t2i_messages[1]["content"][0]["text"]
    t2v_text = t2v_messages[1]["content"][0]["text"]
    i2v_content = i2v_messages[1]["content"]

    assert "resolution 480" in t2i_text
    assert "aspect_ratio 1,1" in t2i_text
    assert '"1,1": {\n      "H": 640,\n      "W": 640' in t2i_text
    assert "comprehensive_t2i_caption" in t2i_text
    assert "eye shape/color" in t2i_text
    assert "hair color/style" in t2i_text
    assert "lip shape" in t2i_text
    assert "wrinkles" in t2i_text
    assert "moles" in t2i_text
    assert "duration 8s" in t2v_text
    assert "fps 24" in t2v_text
    assert "temporal_caption" in t2v_text
    assert i2v_content[0]["type"] == "image_url"
    assert "first frame of the video" in i2v_content[1]["text"]


def test_build_templates_use_custom_user_prompt() -> None:
    t2i_messages = build_t2i_messages(
        "a steel robot arm",
        resolution="480",
        aspect_ratio="1,1",
        user_prompt="custom t2i prompt",
    )
    t2v_messages = build_t2v_messages(
        "a steel robot arm assembling a gear",
        resolution="720",
        aspect_ratio="16,9",
        duration="8s",
        fps=24,
        user_prompt="custom t2v prompt",
    )
    i2v_messages = build_i2v_messages(
        "the object starts rotating",
        image_url="data:image/png;base64,abc",
        resolution="480",
        aspect_ratio="16,9",
        duration="5s",
        fps=20,
        user_prompt="custom i2v prompt",
    )

    assert t2i_messages[1]["content"] == [{"type": "text", "text": "custom t2i prompt"}]
    assert t2v_messages[1]["content"] == [{"type": "text", "text": "custom t2v prompt"}]
    assert i2v_messages[1]["content"] == [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        {"type": "text", "text": "custom i2v prompt"},
    ]


def test_configure_prompting_templates_uses_custom_t2i_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(prompt_upsampling, "T2I_JSON_TEMPLATE", prompt_upsampling.T2I_JSON_TEMPLATE)
    monkeypatch.setattr(prompt_upsampling, "T2I_PROMPT_TEMPLATE", prompt_upsampling.T2I_PROMPT_TEMPLATE)
    prompt_template_path = tmp_path / "custom_t2i_prompt.txt"
    json_template_path = tmp_path / "custom_t2i_schema.json"
    prompt_template_path.write_text(
        "custom t2i template: $json_template | $nl_description | $resolution_ratio_dict",
        encoding="utf-8",
    )
    json_template_path.write_text('{"custom_t2i_schema": true}', encoding="utf-8")

    prompt_upsampling.configure_prompting_templates(
        mode="text2image",
        prompt_template_path=prompt_template_path,
        json_template_path=json_template_path,
    )
    prompt_text = prompt_upsampling.build_t2i_prompt_text("a steel robot arm", resolution="480", aspect_ratio="1,1")

    assert "custom t2i template" in prompt_text
    assert '{"custom_t2i_schema": true}' in prompt_text
    assert "resolution 480" in prompt_text
    assert '"1,1": {\n      "H": 640,\n      "W": 640' in prompt_text


def test_configure_prompting_templates_uses_custom_t2v_i2v_video_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(prompt_upsampling, "T2V_JSON_TEMPLATE", prompt_upsampling.T2V_JSON_TEMPLATE)
    monkeypatch.setattr(prompt_upsampling, "T2V_PROMPT_TEMPLATE", prompt_upsampling.T2V_PROMPT_TEMPLATE)
    prompt_template_path = tmp_path / "custom_t2v_i2v_video_prompt.txt"
    json_template_path = tmp_path / "custom_t2v_i2v_video_schema.json"
    prompt_template_path.write_text(
        "custom t2v/i2v video template: $image_note | $intro | $json_template | $nl_description",
        encoding="utf-8",
    )
    json_template_path.write_text('{"custom_t2v_i2v_video_schema": true}', encoding="utf-8")

    prompt_upsampling.configure_prompting_templates(
        mode="image2video",
        prompt_template_path=prompt_template_path,
        json_template_path=json_template_path,
    )
    prompt_text = prompt_upsampling.build_t2v_prompt_text(
        "a steel robot arm assembling a gear",
        resolution="720",
        aspect_ratio="16,9",
        duration="8s",
        fps=24,
        image_conditioned=True,
    )

    assert "custom t2v/i2v video template" in prompt_text
    assert "Given the attached starting frame image" in prompt_text
    assert '{"custom_t2v_i2v_video_schema": true}' in prompt_text
    assert "duration 8s" in prompt_text
    assert "fps 24" in prompt_text


def test_derive_duration_label_uses_floor_without_clamping() -> None:
    assert derive_duration_label(189, 24) == "7s"
    assert derive_duration_label(1, 24) == "0s"
    assert derive_duration_label(1000, 24) == "41s"
    with pytest.raises(ValueError, match="fps must be positive"):
        derive_duration_label(100, 0)


def test_extract_json_object_text_accepts_fenced_json() -> None:
    assert extract_json_object_text('```json\n{"subjects":[],"aspect_ratio":"16,9"}\n```') == (
        '{"subjects": [], "aspect_ratio": "16,9"}'
    )


def test_extract_json_object_text_rejects_non_object_json() -> None:
    with pytest.raises(ValueError, match="must be an object"):
        extract_json_object_text("[]")


def test_image_path_to_data_url(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    image_path.write_bytes(b"fake-png")

    assert image_path_to_data_url(image_path) == "data:image/png;base64,ZmFrZS1wbmc="


def test_client_omits_auth_header_when_api_token_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LEPTON_API_TOKEN", "lepton-token")
    session = _FakeSession([{"data": [{"id": "cosmos3-reasoner"}]}])

    client = PromptUpsamplerClient(
        PromptUpsamplerConfig(endpoint_url="https://example.test", max_retries=1),
        session=session,  # type: ignore[arg-type]
    )

    assert client.list_models() == ["cosmos3-reasoner"]
    assert len(session.requests) == 1
    headers = session.requests[0]["headers"]
    assert "Authorization" not in headers
    assert "User-Agent" in headers


def test_client_adds_auth_header_when_api_token_set() -> None:
    session = _FakeSession([{"data": [{"id": "cosmos3-reasoner"}]}])

    client = PromptUpsamplerClient(
        PromptUpsamplerConfig(endpoint_url="https://example.test", api_token="secret-token", max_retries=1),
        session=session,  # type: ignore[arg-type]
    )

    client.list_models()

    assert session.requests[0]["headers"]["Authorization"] == "Bearer secret-token"


def test_client_retries_invalid_json_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LEPTON_API_TOKEN", raising=False)
    responses = [
        {"choices": [{"message": {"content": "not json"}}]},
        {"choices": [{"message": {"content": '```json\n{"subjects":[]}\n```'}}]},
    ]
    sleeps: list[float] = []
    session = _FakeSession(responses)

    client = PromptUpsamplerClient(
        PromptUpsamplerConfig(
            endpoint_url="https://example.test/v1",
            model="cosmos3-reasoner",
            max_retries=2,
            retry_base_delay_s=0.25,
        ),
        session=session,  # type: ignore[arg-type]
        sleep=sleeps.append,
    )

    upsampled = client.upsample_t2i("a red cube", resolution="480", aspect_ratio="1,1")

    assert json.loads(upsampled) == {
        "subjects": [],
        "resolution": {"H": 640, "W": 640},
        "aspect_ratio": "1,1",
    }
    assert sleeps == [0.25]


def test_client_uses_custom_user_prompt_and_enforces_output_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LEPTON_API_TOKEN", raising=False)
    session = _FakeSession([{"choices": [{"message": {"content": '```json\n{"subjects":[]}\n```'}}]}])
    client = PromptUpsamplerClient(
        PromptUpsamplerConfig(endpoint_url="https://example.test/v1", model="cosmos3-reasoner", max_retries=1),
        session=session,  # type: ignore[arg-type]
    )

    upsampled = client.upsample_t2v(
        "a red cube spins",
        resolution="720",
        aspect_ratio="16,9",
        duration="7s",
        fps=24,
        user_prompt="custom t2v user prompt",
    )

    assert json.loads(upsampled) == {
        "subjects": [],
        "resolution": {"H": 720, "W": 1280},
        "aspect_ratio": "16,9",
        "duration": "7s",
        "fps": 24,
    }
    payload = session.requests[0]["json"]
    assert payload["messages"][1]["content"] == [{"type": "text", "text": "custom t2v user prompt"}]


def test_client_uses_custom_i2v_user_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LEPTON_API_TOKEN", raising=False)
    session = _FakeSession([{"choices": [{"message": {"content": '```json\n{"subjects":[]}\n```'}}]}])
    client = PromptUpsamplerClient(
        PromptUpsamplerConfig(endpoint_url="https://example.test/v1", model="cosmos3-reasoner", max_retries=1),
        session=session,  # type: ignore[arg-type]
    )

    upsampled = client.upsample_i2v(
        "a red cube spins",
        image_url="data:image/png;base64,abc",
        resolution="480",
        aspect_ratio="16,9",
        duration="7s",
        fps=24,
        user_prompt="custom i2v user prompt",
    )

    assert json.loads(upsampled)["duration"] == "7s"
    payload = session.requests[0]["json"]
    assert payload["messages"][1]["content"] == [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        {"type": "text", "text": "custom i2v user prompt"},
    ]


def test_client_keeps_sampling_params_for_openai_compatible_claude(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LEPTON_API_TOKEN", raising=False)
    session = _FakeSession([{"choices": [{"message": {"content": '```json\n{"subjects":[]}\n```'}}]}])
    client = PromptUpsamplerClient(
        PromptUpsamplerConfig(
            endpoint_url="https://inference-api.nvidia.com/v1",
            model="claude-opus-4-7",
            max_retries=1,
        ),
        session=session,  # type: ignore[arg-type]
    )

    upsampled = client.upsample_t2i("a red cube", resolution="480", aspect_ratio="1,1")

    assert json.loads(upsampled) == {
        "subjects": [],
        "resolution": {"H": 640, "W": 640},
        "aspect_ratio": "1,1",
    }
    payload = session.requests[0]["json"]
    assert payload["model"] == "claude-opus-4-7"
    assert payload["temperature"] == 0.7
    assert payload["top_p"] == 0.8
    assert payload["top_k"] == 20
    assert "min_p" not in payload


def test_client_raises_after_max_retries() -> None:
    client = PromptUpsamplerClient(
        PromptUpsamplerConfig(endpoint_url="https://example.test/v1", model="cosmos3-reasoner", max_retries=2),
        session=_FakeSession(exc=requests.RequestException("temporary unavailable")),  # type: ignore[arg-type]
        sleep=lambda _: None,
    )

    with pytest.raises(RuntimeError, match="failed to upsample prompt after 2 attempts"):
        client.upsample_t2i("a red cube", resolution="480", aspect_ratio="1,1")


def test_omni_prompt_upsampling_without_endpoint_uses_native_upsampler(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "cosmos_framework.inference.prompt_upsampling.PromptUpsamplerClient",
        _FakePromptUpsamplerClient,
    )
    _FakePromptUpsamplerClient.configs = []
    _FakePromptUpsamplerClient.calls = []

    overrides = OmniSampleOverrides(
        name="sample",
        output_dir=tmp_path / "sample",
        prompt="a red cube",
        prompt_upsampling=True,
        num_frames=1,
        fps=24,
        resolution="480",
        aspect_ratio="1,1",
    )
    overrides._build_prompt_upsampling(model_config=_FakeModelConfig("Qwen/Qwen3-VL-8B-Instruct"))

    assert _FakePromptUpsamplerClient.configs == []
    assert _FakePromptUpsamplerClient.calls == []
    assert overrides.prompt == "a red cube"
    assert overrides.prompt_upsampling_applied is False


def test_omni_prompt_upsampling_rejects_mismatched_variant_for_8b_32b(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "cosmos_framework.inference.prompt_upsampling.PromptUpsamplerClient",
        _FakePromptUpsamplerClient,
    )
    _FakePromptUpsamplerClient.configs = []
    _FakePromptUpsamplerClient.calls = []

    overrides = OmniSampleOverrides(
        name="sample",
        output_dir=tmp_path / "sample",
        prompt="a red cube",
        prompt_upsampling=True,
        prompt_upsampler_endpoint_url="https://custom.example.test",
        prompt_upsampler_variant="32B",
        num_frames=1,
        fps=24,
        resolution="480",
        aspect_ratio="1,1",
    )

    with pytest.raises(ValueError, match="does not match VFM model size"):
        overrides._build_prompt_upsampling(model_config=_FakeModelConfig("Qwen/Qwen3-VL-8B-Instruct"))

    assert _FakePromptUpsamplerClient.calls == []


def test_omni_prompt_upsampling_is_idempotent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "cosmos_framework.inference.prompt_upsampling.PromptUpsamplerClient",
        _FakePromptUpsamplerClient,
    )
    _FakePromptUpsamplerClient.configs = []
    _FakePromptUpsamplerClient.calls = []

    overrides = OmniSampleOverrides(
        name="sample",
        output_dir=tmp_path / "sample",
        prompt="a red cube",
        prompt_upsampling=True,
        prompt_upsampler_endpoint_url="https://custom.example.test",
        num_frames=1,
        fps=24,
        resolution="480",
        aspect_ratio="1,1",
    )

    overrides._build_prompt_upsampling(model_config=_FakeModelConfig("Qwen/Qwen3-VL-8B-Instruct"))
    first_prompt = overrides.prompt
    overrides._build_prompt_upsampling(model_config=_FakeModelConfig("Qwen/Qwen3-VL-8B-Instruct"))

    assert len(_FakePromptUpsamplerClient.calls) == 1
    assert _FakePromptUpsamplerClient.configs[0].endpoint_url == "https://custom.example.test"
    assert overrides.prompt_upsampling_applied is True
    assert overrides.prompt == first_prompt


def test_omni_prompt_upsampling_uses_explicit_endpoint_for_other_vfm_sizes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "cosmos_framework.inference.prompt_upsampling.PromptUpsamplerClient",
        _FakePromptUpsamplerClient,
    )
    _FakePromptUpsamplerClient.configs = []
    _FakePromptUpsamplerClient.calls = []

    overrides = OmniSampleOverrides(
        name="sample",
        output_dir=tmp_path / "sample",
        prompt="a red cube",
        prompt_upsampling=True,
        prompt_upsampler_endpoint_url="https://custom.example.test",
        num_frames=1,
        fps=24,
        resolution="480",
        aspect_ratio="1,1",
    )
    overrides._build_prompt_upsampling(model_config=_FakeModelConfig("Qwen/Qwen3-VL-2B-Instruct"))

    assert overrides.prompt == '{"subjects": [], "resolution": {"H": 640, "W": 640}, "aspect_ratio": "1,1"}'
    assert _FakePromptUpsamplerClient.configs[0].endpoint_url == "https://custom.example.test"
