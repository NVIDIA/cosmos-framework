# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Keep the opening temporal paragraph for a short clip at the caption chunk's start."""

from typing import Any

from cosmos_framework.data.imaginaire.webdataset.augmentors.augmentor import Augmentor
from cosmos_framework.utils import log
from cosmos_framework.data.generator.multiview.caption_format import first_caption_paragraph


class FirstCaptionParagraph(Augmentor):
    """Retain the first timestamped paragraph of a plain-text Tier-1 caption.

    Timestamp boundaries also handle consecutive paragraphs separated by only
    one newline. Untimestamped captions keep their first blank-line paragraph.
    Run before duration/resolution metadata is appended and before CFG dropout.
    """

    def __init__(self) -> None:
        super().__init__(input_keys=["ai_caption"], output_keys=["ai_caption"])

    def __call__(self, data: dict[str, Any]) -> dict[str, Any] | None:
        caption = data["ai_caption"]
        if not isinstance(caption, str):
            log.warning(
                f"FirstCaptionParagraph: dropping non-text caption ({type(caption).__name__}); "
                f"sample_key={data.get('__key__')}, url={data.get('__url__')}",
                rank0_only=False,
            )
            return None
        data["ai_caption"] = first_caption_paragraph(caption)
        return data
