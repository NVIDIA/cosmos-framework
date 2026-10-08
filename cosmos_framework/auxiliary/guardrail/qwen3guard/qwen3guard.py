# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import argparse
import re
from dataclasses import dataclass
from typing import Any

import torch

from cosmos_framework.auxiliary.guardrail.common.core import ContentSafetyGuardrail, GuardrailRunner
from cosmos_framework.auxiliary.guardrail.qwen3guard.categories import UNSAFE_CATEGORIES
from cosmos_framework.utils import log, misc

SAFE = misc.Color.green("SAFE")
UNSAFE = misc.Color.red("UNSAFE")


@dataclass(frozen=True)
class Qwen3GuardClassification:
    """Structured Qwen3Guard generation result."""

    label: str
    categories: tuple[str, ...]
    raw_output: str


class Qwen3Guard(ContentSafetyGuardrail):
    offload_model: bool
    dtype: torch.dtype
    model_id: str
    model: Any
    tokenizer: Any

    def __init__(
        self,
        offload_model_to_cpu: bool = True,
    ) -> None:
        """Load Qwen3Guard for text filtering safety checks.

        Args:
            offload_model_to_cpu (bool, optional): Whether to offload the model to CPU. Defaults to True.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.offload_model = offload_model_to_cpu
        self.dtype = torch.bfloat16

        self.model_id = "Qwen/Qwen3Guard-Gen-0.6B"

        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        # Move model to GPU unless offload_model_to_cpu is True
        if not offload_model_to_cpu:
            self.model = self.model.to("cuda", dtype=self.dtype).eval()
            log.debug("Moved Qwen3Guard model to GPU")
        else:
            self.model = self.model.to("cpu", dtype=self.dtype).eval()
            log.debug("Moved Qwen3Guard model to CPU")

    @staticmethod
    def parse_output(content: str) -> Qwen3GuardClassification:
        """Parse Qwen3Guard's generated safety label without collapsing its middle class."""
        safe_pattern = r"Safety: (Safe|Unsafe|Controversial)"
        category_pattern = r"(" + "|".join(re.escape(value) for value in UNSAFE_CATEGORIES.values()) + ")"
        safe_label_match = re.search(safe_pattern, content)
        if safe_label_match is None:
            raise ValueError(f"Qwen3Guard output did not contain a safety label: {content!r}")
        label = safe_label_match.group(1)
        categories = tuple(dict.fromkeys(re.findall(category_pattern, content)))
        return Qwen3GuardClassification(label=label, categories=categories, raw_output=content)

    @torch.inference_mode()
    def classify(self, prompt: str) -> Qwen3GuardClassification:
        """Generate and parse one native Safe/Controversial/Unsafe decision."""
        messages = [{"role": "user", "content": prompt}]

        text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)  # [1, input_tokens]
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=128)  # [1, input_tokens + output_tokens]
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()  # [output_tokens]
        content = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return self.parse_output(content)

    def extract_label_and_categories(self, prompt: str) -> tuple[bool, str]:
        """Return the legacy binary result while retaining native classification through classify()."""
        result = self.classify(prompt)
        if result.label.lower() == "unsafe":
            return False, f"Prompt blocked by Qwen3Guard. Safety: {result.label}, Categories: {result.categories}"
        return True, ""

    def is_safe(self, prompt: str) -> tuple[bool, str]:
        """Check if the input prompt is safe according to the Qwen3Guard model."""
        try:
            return self.extract_label_and_categories(prompt)
        except Exception as e:
            log.error(f"Unexpected error occurred when running Qwen3Guard guardrail: {e}")
            return True, "Unexpected error occurred when running Qwen3Guard guardrail."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    qwen3guard = Qwen3Guard()
    runner = GuardrailRunner(safety_models=[qwen3guard])
    with misc.timer("Qwen3Guard safety check"):
        safety, message = runner.run_safety_check(args.prompt)
    log.info(f"Input is: {'SAFE' if safety else 'UNSAFE'}")
    log.info(f"Message: {message}") if not safety else None


if __name__ == "__main__":
    args = parse_args()
    main(args)
