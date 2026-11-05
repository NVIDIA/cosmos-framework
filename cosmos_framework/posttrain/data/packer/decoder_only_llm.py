# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Text decoder-only DataPacker (GRPO now; SFT triad future).

Ported from cosmos-rl `DecoderOnlyLLMDataPacker` (non-multi-turn RL path). `get_rollout_input`
returns a raw `obs.representations["prompt"]` string verbatim, while a conversation (`list[dict]`)
is rendered via the tokenizer's chat template. CPU-side only — `policy_collate_fn` returns CPU
tensors (the trainer moves them to device); cosmos-rl's `torch.cuda.current_device()` is
deliberately not copied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from cosmos_framework.posttrain.data.packer.base import DataPacker
from cosmos_framework.posttrain.data.packer.forward_routing import add_forward_routing, attach_forward_routing
from cosmos_framework.posttrain.schema import ObsBundle, Transition


@dataclass
class RLPolicyInput:
    """One prompt+completion shaped for the policy: concatenated ids + a per-position
    mask (1 where the position's *next-token* prediction is a completion token).

    `mm_kwargs` carries a VLM's extra forward inputs (`pixel_values`, `image_grid_thw`, ...). None
    on the text path, so the LLM behaviour is byte-identical. `forward_routing` is the rollout's
    recorded expert routing for this sequence (`data/packer/forward_routing.py`), None when the
    rollout recorded none.
    """

    input_ids: list[int]
    logprob_masks: list[int]
    mm_kwargs: dict | None = None
    forward_routing: torch.Tensor | None = None  # [T-1,L,K] integer expert ids | None


def shape_policy_input(
    transition: Transition,
    packer: DataPacker | None,
    n_ignore_prefix_tokens: int = 0,
) -> Any:
    """Shape one rollout transition for causal-LM policy scoring.

    Stored ``prompt_ids`` take precedence because they are the exact prefix used for generation,
    but an available packer is still called so a VLM can recover media tensors from the observation.
    When ids are absent, the packer re-tokenizes. The returned mask is next-token aligned:
    ``logprob_masks[:-1]`` selects completion logprobs. Recorded expert routing
    (``policy_info["rollout_routed_experts"]``) rides on the result once it is checked against the
    shaped sequence; this is the only reader of that key.
    """
    if n_ignore_prefix_tokens < 0:
        raise ValueError(f"n_ignore_prefix_tokens must be >= 0, got {n_ignore_prefix_tokens}")

    action = transition.action
    completion = action if isinstance(action, str) else (action.tolist() if hasattr(action, "tolist") else list(action))
    stored_prompt = transition.policy_info.get("prompt_ids") if transition.policy_info else None
    prompt_ids = None
    if stored_prompt is not None:
        prompt_ids = stored_prompt.tolist() if hasattr(stored_prompt, "tolist") else list(stored_prompt)
        prompt_ids = [int(token_id) for token_id in prompt_ids]
        if not prompt_ids:
            raise ValueError("policy_info['prompt_ids'] is empty; a completion requires its generation prefix")

    # Always call the packer when one is available. A VLM needs the original
    # observation even when exact prompt ids were stored, because media tensors
    # cannot be reconstructed from ids alone.
    if packer is not None:
        shaped = packer.get_policy_input(
            transition.obs,
            completion,
            n_ignore_prefix_tokens=n_ignore_prefix_tokens,
            prompt_ids=prompt_ids,
        )
        return attach_forward_routing(shaped, transition.policy_info)
    if prompt_ids is None:
        raise ValueError(
            "policy objective cannot shape this trajectory: it carries no policy_info['prompt_ids'] "
            "and no packer was provided to re-tokenize its observation"
        )
    if isinstance(completion, str):
        raise ValueError("a decoded string completion with stored prompt_ids requires a tokenizer-backed packer")
    else:
        completion_ids = [int(token) for token in completion]
    if n_ignore_prefix_tokens > len(completion_ids):
        raise ValueError(
            f"n_ignore_prefix_tokens={n_ignore_prefix_tokens} exceeds completion length {len(completion_ids)}"
        )
    shaped = RLPolicyInput(
        input_ids=list(prompt_ids) + completion_ids,
        logprob_masks=[0] * (len(prompt_ids) - 1 + n_ignore_prefix_tokens)
        + [1] * (len(completion_ids) - n_ignore_prefix_tokens)
        + [0],
    )
    return attach_forward_routing(shaped, transition.policy_info)


class DecoderOnlyLLMDataPacker(DataPacker):
    """Data-shaping for decoder-only text LLMs (GRPO RL triad; SFT is future work)."""

    def get_rollout_input(self, item: ObsBundle) -> str:
        """Return the generation prompt for `item` (cosmos-rl conditional shape).

        Args:
            item: the prompt observation. `representations["prompt"]` is either a raw prompt
                string (returned verbatim) or a conversation (`list[dict]` of role/content,
                rendered via the tokenizer's chat template).

        Returns:
            The prompt string handed to the rollout engine.

        Raises:
            ValueError: `representations["prompt"]` is neither a str nor a list.
        """
        text = item.representations.get("prompt")
        if isinstance(text, str):
            return text
        if isinstance(text, list):
            return self.tokenizer.apply_chat_template(text, tokenize=False, add_generation_prompt=True)
        raise ValueError(
            f"get_rollout_input expects representations['prompt'] str or conversation list, got {type(text)!r}"
        )

    def get_policy_input(
        self,
        sample: ObsBundle,
        rollout_output: Any,
        n_ignore_prefix_tokens: int = 0,
        *,
        prompt_ids: list[int] | None = None,
    ) -> RLPolicyInput:
        """Build `input_ids` (prompt+completion) and a shifted completion `logprob_masks`.

        Args:
            sample: the prompt observation.
            rollout_output: the completion as a decoded str or a list of token ids.
            n_ignore_prefix_tokens: completion-prefix tokens to exclude from the mask; must be
                <= the completion length (else the mask would desync from `input_ids`).
            prompt_ids: exact rollout prefix. When supplied, it replaces prompt
                re-tokenization while the packer still owns completion shaping.

        Returns:
            An `RLPolicyInput`. `logprob_masks` is `[0]*(len(prompt)-1) + [1]*len(completion) + [0]`
            (length == len(input_ids)), aligned so `mask[:-1]` selects the completion
            positions in the length-(L-1) next-token logprob vector.
        """
        if n_ignore_prefix_tokens < 0:
            raise ValueError(f"n_ignore_prefix_tokens must be >= 0, got {n_ignore_prefix_tokens}")
        if prompt_ids is None:
            # Re-tokenization is only the fallback for producers without stored ids.
            prompt = self.get_rollout_input(sample)
            prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        else:
            prompt_ids = [int(token_id) for token_id in prompt_ids]
        if isinstance(rollout_output, str):
            completion_ids = self.tokenizer(rollout_output, add_special_tokens=False).input_ids
        else:
            completion_ids = [int(t) for t in rollout_output]
        if not prompt_ids:
            raise ValueError("get_policy_input: empty prompt (representations['prompt'] tokenized to no tokens)")
        if n_ignore_prefix_tokens > len(completion_ids):
            raise ValueError(
                f"n_ignore_prefix_tokens={n_ignore_prefix_tokens} exceeds completion length {len(completion_ids)}"
            )
        return RLPolicyInput(
            input_ids=prompt_ids + completion_ids,
            logprob_masks=[0] * (len(prompt_ids) - 1 + n_ignore_prefix_tokens)
            + [1] * (len(completion_ids) - n_ignore_prefix_tokens)
            + [0],
        )

    def policy_compute_max_len(self, processed_samples: list[RLPolicyInput]) -> int:
        return max(len(x.input_ids) for x in processed_samples)

    def policy_collate_fn(self, processed_samples: list[RLPolicyInput], computed_max_len: int) -> dict[str, Any]:
        """Pad/stack processed samples into a CPU tensor dict (`input_ids`, `logprob_masks`).

        Args:
            processed_samples: the per-prompt `RLPolicyInput`s to collate.
            computed_max_len: the padding length (from `policy_compute_max_len`).

        Returns:
            A dict with `"input_ids"` (LongTensor `[B, max_len]`, pad = `pad_token_id`) and
            `"logprob_masks"` (BoolTensor `[B, max_len]`, False in the pad region), both on CPU, plus
            the recorded forward routing when the samples carry it (`add_forward_routing`).

        Raises:
            ValueError: `processed_samples` is empty.
        """
        if not processed_samples:
            raise ValueError("policy_collate_fn got 0 samples (empty batch or zero batch size)")
        pad = self.tokenizer.pad_token_id
        input_ids = torch.tensor(  # [B,L]
            [
                x.input_ids[:computed_max_len] + [pad] * max(0, computed_max_len - len(x.input_ids))
                for x in processed_samples
            ],
            dtype=torch.long,
        )
        logprob_masks = torch.tensor(  # [B,L]
            [
                x.logprob_masks[:computed_max_len] + [0] * max(0, computed_max_len - len(x.logprob_masks))
                for x in processed_samples
            ],
            dtype=torch.bool,
        )
        batch: dict[str, Any] = {"input_ids": input_ids, "logprob_masks": logprob_masks}
        add_forward_routing(batch, processed_samples, computed_max_len)
        return batch
