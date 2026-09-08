# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import numpy as np
import torch

from cosmos_framework.data.generator.processors.qwen3vl_processor import Qwen3VLProcessor


class Qwen3VLNemoChatProcessor(Qwen3VLProcessor):
    """Qwen3-VL processor with the Nemo Chat assistant loss mask."""

    THINK_END_TOKEN: str = "</think>"
    think_end_id: int

    def __init__(
        self,
        name: str = "Qwen/Qwen3-VL-2B-Init",
        credentials: str = "./credentials/s3_training.secret",
        bucket: str = "bucket4",
        cache_dir: str | None = None,
    ) -> None:
        super().__init__(name=name, credentials=credentials, bucket=bucket, cache_dir=cache_dir)
        self.think_end_id = self.processor.tokenizer.convert_tokens_to_ids(self.THINK_END_TOKEN)

    def add_assistant_tokens_mask(
        self,
        tokens: list[int] | torch.Tensor,
    ) -> list[bool] | torch.Tensor:  # tokens: [N_token] or [B,N_token], returns the same shape
        """Mask assistant tokens while excluding the reasoning prefix supplied by the chat template."""
        if isinstance(tokens, torch.Tensor) and tokens.ndim == 2:
            mask = torch.stack(
                [self.add_assistant_tokens_mask(tokens[i]) for i in range(tokens.shape[0])]
            )  # [B,N_token]
            assert mask.shape == tokens.shape
            return mask

        np_tokens = tokens.cpu().numpy() if isinstance(tokens, torch.Tensor) else np.array(tokens)  # [N_token]
        assert np_tokens.ndim == 1

        bos_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        eos_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        role_id = self.processor.tokenizer.convert_tokens_to_ids("assistant")
        role_ids = self.processor.tokenizer.encode("assistant", add_special_tokens=False)
        think_start_id = self.processor.tokenizer.convert_tokens_to_ids("<think>")
        newline_ids = self.processor.tokenizer.encode("\n", add_special_tokens=False)

        start_indices = np.where(np_tokens == bos_token_id)[0]
        end_indices = np.where(np_tokens == eos_token_id)[0]
        masks = np.zeros_like(np_tokens, dtype=bool)
        assert len(start_indices) == len(end_indices)

        # The generation prompt supplies the opening <think> token. Empty thinking blocks may place
        # a tokenizer-specific newline before </think>, so the complete prefix should receive no loss.
        for start, end in zip(start_indices, end_indices):
            content_start = None
            if np_tokens[start + 1] == role_id:
                content_start = start + 3
            elif all(np_tokens[start + 1 : start + 1 + len(role_ids)] == role_ids):
                content_start = start + 2 + len(role_ids)

            if content_start is None:
                continue

            masks[content_start : end + 1] = True
            if np_tokens[content_start] == think_start_id:
                masks[content_start] = False
                if np_tokens[content_start + 1] == self.think_end_id:
                    masks[content_start + 1] = False
                else:
                    think_end_position = content_start + 1 + len(newline_ids)
                    has_empty_think_block = (
                        bool(newline_ids)
                        and think_end_position < end
                        and np.array_equal(
                            np_tokens[content_start + 1 : think_end_position],
                            newline_ids,
                        )
                        and np_tokens[think_end_position] == self.think_end_id
                    )
                    if has_empty_think_block:
                        masks[content_start + 1 : think_end_position + 1] = False

        assert masks.shape == np_tokens.shape
        if isinstance(tokens, torch.Tensor):
            mask = torch.from_numpy(masks)  # [N_token]
            return mask
        return masks.tolist()
