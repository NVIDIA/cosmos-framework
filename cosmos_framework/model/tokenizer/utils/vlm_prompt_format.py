# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Prompt-format helpers shared by tokenizer VLM training and generation."""

DENSEVL_ADD_VISION_ID_PREFIX_BY_MEDIA_TYPE: dict[str, str] = {
    "image": "Picture",
    "video": "Video",
}


def densevl_add_vision_id_text(media_type: str, one_based_index: int) -> str:
    """Render the text DenseVL/Qwen emits for ``add_vision_id=True``."""
    normalized_media_type = str(media_type).strip().lower()
    prefix = DENSEVL_ADD_VISION_ID_PREFIX_BY_MEDIA_TYPE.get(normalized_media_type)
    if prefix is None:
        raise ValueError(
            f"Unsupported DenseVL add_vision_id media_type={media_type!r}; "
            f"expected one of {sorted(DENSEVL_ADD_VISION_ID_PREFIX_BY_MEDIA_TYPE)}."
        )
    if one_based_index < 1:
        raise ValueError(f"DenseVL add_vision_id index must be one-based and positive, got {one_based_index}.")
    return f"{prefix} {one_based_index}: "
