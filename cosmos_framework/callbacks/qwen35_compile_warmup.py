# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Warm Qwen3.5 sequence-shaped kernels before real training or resumed steps."""

import gc
import time

import torch

from cosmos_framework.utils import log
from cosmos_framework.utils.callback import Callback
from cosmos_framework.model.generator.hf_model import HFModel
from cosmos_framework.model.generator.reasoner.qwen35_caption import Qwen35CaptionLoss
from cosmos_framework.model.generator.vlm_model import VLMModel


def warmup_caption_kernels(model: HFModel, max_length: int, bucket_multiple: int) -> None:
    """Run one text-only forward/backward per padded length without optimizer updates.

    This warms FLA/Triton even when torch.compile is disabled. It does not read a
    dataloader or touch VLMModel's accumulation statistics. All FSDP ranks must
    participate, including when restoring a checkpoint into a fresh process.
    """
    if bucket_multiple < 2 or max_length < bucket_multiple or max_length % bucket_multiple:
        raise ValueError("Warmup max_length must be a positive multiple of bucket_multiple >= 2")
    if model.hf_config.model_type != "qwen3_5":
        raise ValueError("Caption kernel warmup requires dense Qwen3.5")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("Caption warmup must run before accumulating training gradients")
    device = next(model.parameters()).device
    cuda_devices = [device.index] if device.type == "cuda" else []
    pad_id = model.hf_config.text_config.pad_token_id or 0
    was_training = model.training
    started = time.perf_counter()
    model.train()
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            # Largest first: reuse zeroed sharded gradients across smaller buckets.
            for length in range(max_length, 0, -bucket_multiple):
                ids = torch.full((1, length), pad_id, dtype=torch.long, device=device)
                labels = torch.full_like(ids, -100)
                labels[:, -1] = pad_id
                result = model(
                    input_ids=ids,
                    labels=labels,
                    attention_mask=torch.ones_like(ids),
                    mm_token_type_ids=torch.zeros_like(ids),
                )
                if not isinstance(result, Qwen35CaptionLoss):
                    raise TypeError("Caption warmup requires enable_fused_weighted_ce=True")
                if not torch.isfinite(result.loss).item():
                    raise RuntimeError(f"Non-finite caption warmup loss at sequence length {length}")
                result.loss.backward()
                model.zero_grad(set_to_none=False)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                log.info(f"Caption compile warmup: sequence length {length} complete")
                del result, labels, ids
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    log.info(
        f"Compiled {max_length // bucket_multiple} sequence buckets in "
        f"{time.perf_counter() - started:.2f}s; no optimizer steps or training samples consumed"
    )


class Qwen35CompileWarmup(Callback):
    """Standard unified callback; kernel warmup is independent of torch.compile."""

    enabled: bool
    max_length: int
    bucket_multiple: int

    def __init__(self, enabled: bool = True, max_length: int = 36864, bucket_multiple: int = 2048) -> None:
        super().__init__()
        self.enabled = enabled
        self.max_length = max_length
        self.bucket_multiple = bucket_multiple

    def on_train_start(self, model: VLMModel, iteration: int = 0) -> None:
        del iteration
        if not self.enabled:
            return
        if not model.config.policy.enable_fused_weighted_ce:
            log.info("Skipping caption kernel warmup because enable_fused_weighted_ce=False")
            return
        warmup_caption_kernels(model.model, self.max_length, self.bucket_multiple)
