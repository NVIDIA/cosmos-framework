# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json

import pytest
from PIL import Image

from cosmos_framework.data.generator.augmentors.image_editing_transform import ExtractImageEditingConversation

_STRUCTURED_KEY = "edit_schema_all_inputs_qwen3-vl-235b-a22b-instruct"


def _conversation(instruction: str = "Make the cup red") -> list[list[dict]]:
    return [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "image_0"},
                    {"type": "text", "text": instruction},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "image", "image": "image_1"},
                ],
            },
        ]
    ]


def _media_data() -> tuple[Image.Image, Image.Image, dict[str, Image.Image]]:
    source_image = Image.new("RGB", (16, 16), color="blue")
    target_image = Image.new("RGB", (16, 16), color="red")
    media_list = {
        "image_0": source_image,
        "image_1": target_image,
    }
    return source_image, target_image, media_list


def _base_data_dict() -> tuple[dict, Image.Image, Image.Image]:
    source_image, target_image, media_list = _media_data()
    data_dict = {
        "__key__": "sample_000001",
        "mllm_media_list": media_list,
        "diffusion_media_list": media_list,
    }
    return data_dict, source_image, target_image


def _structured_payload() -> dict:
    return {
        "rewrite_error": None,
        "gemini_rewrite": {
            "edit_type": "adjust",
            "structured_instruction": {
                "target_object": "cup",
                "attribute_type": "color",
                "desired_value": "red",
            },
        },
        "text_json": {
            "content": _conversation("Original dense instruction"),
        },
        "original_instruction": "Original dense instruction",
    }


@pytest.mark.L0
@pytest.mark.CPU
def test_extract_image_editing_conversation_keeps_texts_behavior() -> None:
    data_dict, source_image, target_image = _base_data_dict()
    data_dict["texts"] = {"content": _conversation("Make the cup red")}

    result = ExtractImageEditingConversation()(data_dict)

    assert result is not None
    assert result["source_image"] is source_image
    assert result["target_image"] is target_image
    assert result["editing_instruction"] == "Make the cup red"


@pytest.mark.L0
@pytest.mark.CPU
def test_extract_structured_dict_payload_uses_gemini_rewrite() -> None:
    data_dict, source_image, target_image = _base_data_dict()
    payload = _structured_payload()
    data_dict[_STRUCTURED_KEY] = payload

    result = ExtractImageEditingConversation(
        instruction_key=_STRUCTURED_KEY,
        conversation_key="text_json",
        structured_instruction_field="gemini_rewrite",
    )(data_dict)

    expected_instruction = json.dumps(
        {
            "edit_type": payload["gemini_rewrite"]["edit_type"],
            "structured_instruction": payload["gemini_rewrite"]["structured_instruction"],
        },
        ensure_ascii=False,
    )
    assert result is not None
    assert result["source_image"] is source_image
    assert result["target_image"] is target_image
    assert result["editing_instruction"] == expected_instruction


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("encode_as_bytes", [False, True])
def test_extract_structured_json_payload_uses_gemini_rewrite(encode_as_bytes: bool) -> None:
    data_dict, _, _ = _base_data_dict()
    payload = _structured_payload()
    payload_json = json.dumps(payload, ensure_ascii=False)
    data_dict[_STRUCTURED_KEY] = payload_json.encode("utf-8") if encode_as_bytes else payload_json

    result = ExtractImageEditingConversation(
        instruction_key=_STRUCTURED_KEY,
        conversation_key="text_json",
        structured_instruction_field="gemini_rewrite",
    )(data_dict)

    expected_instruction = json.dumps(
        {
            "edit_type": payload["gemini_rewrite"]["edit_type"],
            "structured_instruction": payload["gemini_rewrite"]["structured_instruction"],
        },
        ensure_ascii=False,
    )
    assert result is not None
    assert result["editing_instruction"] == expected_instruction


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "payload_update",
    [
        {"gemini_rewrite": None},
        {"rewrite_error": "failed to rewrite"},
    ],
)
def test_extract_structured_invalid_payload_returns_none(payload_update: dict) -> None:
    data_dict, _, _ = _base_data_dict()
    payload = _structured_payload()
    payload.update(payload_update)
    data_dict[_STRUCTURED_KEY] = payload

    result = ExtractImageEditingConversation(
        instruction_key=_STRUCTURED_KEY,
        conversation_key="text_json",
        structured_instruction_field="gemini_rewrite",
    )(data_dict)

    assert result is None
