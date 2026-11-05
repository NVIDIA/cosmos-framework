# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Five multimodal pool rows with conversation-list prompts, for the data-role tests.

The rows are the shapes a multimodal RL pool carries as JSONL (``{"messages": [...], "reward_spec":
{...}}``): the prompt is a conversation list with the image or video INSIDE the content parts, and
the reference is the grader's spec. Media paths are placeholders and two long texts are shortened;
everything else is as written in the pool. The five shapes: an instruction-following row (image +
text), two exact-match rows (text + image, image + text), a video prompt-following row (text + text +
a 16-frame video block with its sampling metadata), and a text-only instruction-following row whose
turn ``content`` is a plain STRING, not a list of parts -- 3,282 of the pool's 18,270 rows carry this
shape, and it is the one Hugging Face ``datasets`` cannot read beside the list-content rows (Arrow
refuses the file: ``Column(/messages/[]/content) changed from array to string``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CONVERSATION_ROWS: list[dict[str, Any]] = [
    {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "<image-0.png>"},
                    {
                        "type": "text",
                        "text": (
                            "Describe this image in detail. Your entire response should be in English, and in "
                            "all lowercase letters. No capital letters are allowed. The first word of your "
                            "response should be the word broadly."
                        ),
                    },
                ],
            }
        ],
        "reward_spec": {
            "source": "mmif_synth_v1",
            "source_id": "mmif_synth_v1:train:0018917:7daf26067fc9066a",
            "original_source_id": "array/SAT:train:9e89a3b8|tpl:change_case:english_lowercase+first_word:first_word_answer",
            "verifier": "open_instruct_if",
            "payload": {
                "ground_truth": (
                    '[{"instruction_id": ["change_case:english_lowercase", "first_word:first_word_answer"], '
                    '"kwargs": [{}, {"first_word": "broadly"}]}]'
                ),
                "constraint": (
                    "Your entire response should be in English, and in all lowercase letters. No capital "
                    "letters are allowed.\tThe first word of your response should be the word broadly."
                ),
                "constraint_type": "multi",
            },
        },
    },
    {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "As shown in the diagram, in the equilateral $\\triangle ABC$, $D$ is the midpoint of "
                            "side $BC$. An arc is drawn with $A$ as the center and $AD$ as the radius, intersecting "
                            "side $AC$ at point $E$. What is the degree measure of $\\angle ADE$?"
                        ),
                    },
                    {"type": "image", "image": "<image-1.png>"},
                ],
            }
        ],
        "reward_spec": {
            "source": "virl39k",
            "source_id": "virl39k:MMMath-75",
            "original_source_id": "MMMath-75",
            "verifier": "mm_exact_match",
            "payload": {"ground_truth": "75", "answer_type": "numeric"},
        },
    },
    {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "<image-2.png>"},
                    {
                        "type": "text",
                        "text": (
                            "Xiao Ming filled out the following table based on a linear function, but one cell "
                            "was accidentally covered by ink. Think about it, what number was originally in the "
                            "blank cell?"
                        ),
                    },
                ],
            }
        ],
        "reward_spec": {
            "source": "innovator_vl",
            "source_id": "innovator_vl:virl39k_00013_1153",
            "original_source_id": "virl39k_00013_1153",
            "verifier": "mm_exact_match",
            "payload": {"ground_truth": "2", "answer_type": "numeric"},
        },
    },
    {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are an autonomous-driving scene-recognition annotation expert.\n\nPlease analyze "
                            "the input front-view video clip, and output structured annotation results according "
                            "to the following dimensions. [shortened]"
                        ),
                    },
                    {"type": "text", "text": "Analyze this video and return only the JSON result."},
                    {
                        "type": "video",
                        "video": [f"<frame-{index:02d}.jpg>" for index in range(16)],
                        "fps": 0.3972194637537239,
                        "total_num_frames": 16,
                        "frames_indices": list(range(16)),
                    },
                ],
            }
        ],
        "reward_spec": {
            "source": "avif_video_v6",
            "source_id": "avif_video_v6:0000060:002d290d-89d1aea8:en:scene_v0",
            "original_source_id": "bdd100k_video:train:002d290d-89d1aea8|tpl:en:scene_v0",
            "verifier": "av_prompt_following",
            "payload": {"prompt_id": "scene_en_v0", "language": "en", "pair_id": "scene_v0"},
        },
    },
    {
        "messages": [
            {
                "content": (
                    'Q: Translate "And the Pharisees and some of the scribes, coming from Jerusalem, are gathered '
                    'together to him," to Russian?\nYes: [shortened]\n\nAt the end of your response, please '
                    "explicitly add a postscript starting with Addendum:"
                ),
                "role": "user",
            }
        ],
        "reward_spec": {
            "original_source_id": "row:0000009081",
            "payload": {
                "constraint": (
                    "At the end of your response, please explicitly add a postscript starting with {postscript marker}"
                ),
                "constraint_type": "Postscript",
                "ground_truth": (
                    '{"func_name": "verify_postscript", "N": null, "quantifier": null, "end_phrase": null, '
                    '"keyword_list": null, "word": null, "forbidden_words": null, "letter": null, "i": null, '
                    '"first_word": null, "postscript_marker": "Addendum:", "options": null, '
                    '"section_splitter": null, "original_prompt": null}'
                ),
            },
            "source": "allenai/RLVR-IFeval",
            "source_id": "allenai_rlvr_ifeval_9f281316:0000009081:d3248bbeee6addcf",
            "verifier": "open_instruct_if",
        },
        "mix_source": "textif",
    },
]


def write_conversation_jsonl(directory: Path, rows: list[dict[str, Any]], name: str = "pool.jsonl") -> str:
    """Write ``rows`` as one JSON object per line and return the path as a string."""
    path = Path(directory) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return str(path)
