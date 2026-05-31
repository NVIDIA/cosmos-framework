#!/usr/bin/env -S uv run --script
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1


# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "accelerate==1.13.0",
#   "torch==2.11.0",
#   "torchvision",
#   "transformers==5.8.0",
#   "transformers-cosmos3",
# ]
# [tool.uv.sources]
# transformers-cosmos3 = { path = "../", editable = true }
# ///

"""Minimal example of inference with Cosmos-Reason2."""

# Source: https://github.com/QwenLM/Qwen3-VL?tab=readme-ov-file#new-qwen-vl-utils-usage

import warnings

warnings.filterwarnings("ignore")


import torch
import transformers
from transformers import AutoProcessor
from transformers_cosmos3 import Cosmos3ForConditionalGeneration

SEPARATOR = "-" * 20


def main():
    # Ensure reproducibility
    transformers.set_seed(0)

    # Load model
    model_name = "nvidia/Cosmos3-Nano"
    model = Cosmos3ForConditionalGeneration.from_pretrained(
        model_name, trust_remote_code=True, dtype=torch.float16, device_map="auto", attn_implementation="sdpa"
    )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    # Create inputs
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/2b17a2413bd86b2cf9b03823637108851e4ddf2d/inputs/vision/robot_153.jpg",
                },
                {"type": "text", "text": "Caption the image in detail."},
            ],
        },
    ]

    # Process inputs
    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"fps": 4},
    )
    inputs = inputs.to(model.device)

    # Run inference
    generated_ids = model.generate(**inputs, max_new_tokens=4096)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(SEPARATOR)
    print(output_text[0])
    print(SEPARATOR)


if __name__ == "__main__":
    main()
