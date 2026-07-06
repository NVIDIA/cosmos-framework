# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Built-in SampleBatcher implementations."""

from __future__ import annotations

from collections import deque
from enum import Enum
from typing import Callable, Iterator, Optional

from cosmos_framework.data.generator.dataflow.base import SampleBatcher


class SimpleBatcher(SampleBatcher):
    """Fixed-size batching — stock DataLoader behavior. Never needs sample_size."""

    def __init__(self, batch_size: int, drop_last: bool = False):
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.batch_size = batch_size
        self.drop_last = drop_last

    def batches(self, samples: Iterator[dict]) -> Iterator[list[dict]]:
        buf: list[dict] = []
        for s in samples:
            buf.append(s)
            if len(buf) == self.batch_size:
                yield buf
                buf = []
        if buf and not self.drop_last:
            yield buf


class _Modality(Enum):
    IMAGE = "image"
    VIDEO = "video"
    TEXT = "text"


class PoolPackingBatcher(SampleBatcher):
    """Pool-based greedy bin-packing batcher (re-homed from PackingIterableDataset).

    Buffers ``pool_size`` samples and assembles each batch by greedily selecting
    candidates that fit within the padded token budget, never mixing modalities
    within a batch. ``sample_size`` defaults to ``len(sample["input_ids"])``;
    pass ``size_fn`` to override, or subclass and override the method.
    """

    def __init__(
        self,
        max_tokens: int,
        pool_size: int = 16,
        max_batch_size: int = 1,
        long_threshold: int = 6400,
        batching_strategy: str = "prefer_closest",
        apply_long_sample_halving: bool = True,
        size_fn: Optional[Callable[[dict], int]] = None,
    ):
        assert batching_strategy in ("prefer_first", "prefer_closest"), (
            f"batching_strategy must be 'prefer_first' or 'prefer_closest', got {batching_strategy!r}"
        )
        self.max_tokens = max_tokens
        self.pool_size = pool_size
        self.max_batch_size = max_batch_size
        self.long_threshold = long_threshold
        self.batching_strategy = batching_strategy
        self.apply_long_sample_halving = apply_long_sample_halving
        self._size_fn = size_fn

    def sample_size(self, sample: dict) -> int:
        if self._size_fn is not None:
            return self._size_fn(sample)
        # len() == shape[0] for a 1-D tensor and also works for list input_ids.
        return len(sample["input_ids"])

    def batches(self, samples: Iterator[dict]) -> Iterator[list[dict]]:
        pool: deque[dict] = deque()
        src = iter(samples)
        exhausted = False
        while True:
            while not exhausted and len(pool) < self.pool_size:
                try:
                    pool.append(next(src))
                except StopIteration:
                    exhausted = True
            if not pool:
                return
            yield self._best_fit_batch(pool)

    def _max_tokens(self, cur_max: int) -> int:
        if not self.apply_long_sample_halving:
            return self.max_tokens
        if cur_max < 1000:
            return self.max_tokens
        return self.max_tokens // 2

    @staticmethod
    def _get_modality(sample: dict) -> "_Modality":
        if "pixel_values" in sample:
            return _Modality.IMAGE
        elif "pixel_values_videos" in sample:
            return _Modality.VIDEO
        return _Modality.TEXT

    @staticmethod
    def _padded_cost(cur_max: int, k: int) -> int:
        return cur_max * k

    def _best_fit_batch(self, pool: deque) -> list[dict]:
        seed = pool.popleft()
        seed_modality = self._get_modality(seed)
        L0 = self.sample_size(seed)
        if L0 >= self.long_threshold or L0 >= self._max_tokens(L0):
            return [seed]
        chosen = [seed]
        cur_max = L0
        while pool:
            if self.max_batch_size and len(chosen) >= self.max_batch_size:
                break
            best_idx = self._find_best_candidate(pool, cur_max, len(chosen), seed_modality)
            if best_idx is None:
                break
            cand = self._remove_from_pool(pool, best_idx)
            chosen.append(cand)
            cur_max = max(cur_max, self.sample_size(cand))
        return chosen

    def _find_best_candidate(self, pool, cur_max, num_chosen, seed_modality):
        if self.batching_strategy == "prefer_first":
            return self._find_best_candidate_prefer_first(pool, cur_max, num_chosen, seed_modality)
        return self._find_best_candidate_prefer_closest(pool, cur_max, num_chosen, seed_modality)

    def _find_best_candidate_prefer_first(self, pool, cur_max, num_chosen, seed_modality):
        best_idx = None
        best_new_tokens = None
        for idx, cand in enumerate(pool):
            if self._get_modality(cand) != seed_modality:
                continue
            L = self.sample_size(cand)
            new_max = max(cur_max, L)
            new_tokens = self._padded_cost(new_max, num_chosen + 1)
            if new_tokens <= self._max_tokens(cur_max):
                if best_new_tokens is None or new_tokens < best_new_tokens:
                    best_new_tokens = new_tokens
                    best_idx = idx
        return best_idx

    def _find_best_candidate_prefer_closest(self, pool, cur_max, num_chosen, seed_modality):
        best_idx = None
        best_new_tokens = None
        smallest_length_diff = None
        for idx, cand in enumerate(pool):
            if self._get_modality(cand) != seed_modality:
                continue
            L = self.sample_size(cand)
            new_max = max(cur_max, L)
            new_tokens = self._padded_cost(new_max, num_chosen + 1)
            if new_tokens <= self._max_tokens(cur_max):
                length_diff = abs(L - cur_max)
                if (
                    best_new_tokens is None
                    or new_tokens < best_new_tokens
                    or (new_tokens == best_new_tokens and length_diff < smallest_length_diff)
                ):
                    best_new_tokens = new_tokens
                    best_idx = idx
                    smallest_length_diff = length_diff
        return best_idx

    @staticmethod
    def _remove_from_pool(pool: deque, idx: int) -> dict:
        if idx == 0:
            return pool.popleft()
        elif idx == len(pool) - 1:
            return pool.pop()
        else:
            pool.rotate(-idx)
            item = pool.popleft()
            pool.rotate(idx)
            return item


from collections import deque as _deque

from cosmos_framework.utils import log


class SequentialPackingBatcher(SampleBatcher):
    """Order-preserving pull-until-budget packing (port of PackingDataLoader.__iter__).

    Accumulates samples in stream order until `max_sequence_length` (or
    `max_samples_per_batch`); a sample that would overflow a non-empty batch is
    carried to the next batch (bounded by `lookahead_limit`); a sample that alone
    exceeds the budget is discarded with a log. `sample_size` ports the VFM VAE
    token formula (needs the tokenizer compression factors + patch size + optional
    sound params).
    """

    def __init__(
        self,
        max_sequence_length: Optional[int] = None,
        tokenizer_spatial_compression_factor: int = 16,
        tokenizer_temporal_compression_factor: int = 4,
        patch_spatial: int = 2,
        max_samples_per_batch=None,
        lookahead_limit: int = 10,
        sound_latent_fps: float = 0,
        audio_sample_rate: int = 48000,
    ):
        self.max_sequence_length = max_sequence_length
        self.tokenizer_spatial_compression_factor = tokenizer_spatial_compression_factor
        self.tokenizer_temporal_compression_factor = tokenizer_temporal_compression_factor
        self.patch_spatial = patch_spatial
        self.max_samples_per_batch = max_samples_per_batch
        self.lookahead_limit = lookahead_limit
        self.sound_latent_fps = sound_latent_fps
        self.audio_sample_rate = audio_sample_rate
        assert (self.max_sequence_length is None) != (self.max_samples_per_batch is None), (
            "Exactly one of max_sequence_length or max_samples_per_batch must be set "
            "(token-budget mode vs count-only mode), matching legacy PackingDataLoader."
        )

    def sample_size(self, sample: dict) -> int:
        # PORT of _compute_num_tokens_per_sample (joint_dataloader.py:325-400),
        # operating on a SINGLE sample.
        #
        # In the original batched method:
        #   - text_token_ids is a list of tensors  → num_text_tokens = text_token_ids[0].shape[0]
        #   - text_token_ids is a 2-D tensor [B,S] → num_text_tokens = text_token_ids.shape[1]
        # For a single sample:
        #   - 1-D tensor [S]      → shape[0]  (torch.arange(N) case from tests)
        #   - list of tensors     → text_token_ids[0].shape[0]
        #   - list of ints        → len(text_token_ids)
        #   - 2-D tensor [1,S]    → shape[1]  (mirrors original .shape[1] branch)
        import torch as _torch
        text_token_ids = sample["text_token_ids"]
        if isinstance(text_token_ids, list):
            if len(text_token_ids) > 0 and isinstance(text_token_ids[0], _torch.Tensor):
                num_text_tokens = text_token_ids[0].shape[0]
            else:
                num_text_tokens = len(text_token_ids)
        else:
            # tensor: 1-D → shape[0], 2-D → shape[1]
            if text_token_ids.ndim == 1:
                num_text_tokens = text_token_ids.shape[0]
            else:
                num_text_tokens = text_token_ids.shape[1]

        num_tokens = num_text_tokens + 1

        # Vision part — single sample has "images" or "video" as a tensor,
        # not a list. Wrap in [media] to mirror the original's iteration loop.
        is_image_batch = "images" in sample
        input_images_or_videos = sample["images" if is_image_batch else "video"]

        for media in input_images_or_videos if isinstance(input_images_or_videos, list) else [input_images_or_videos]:
            if is_image_batch:
                _, H, W = media.shape
                T = 1
            else:
                _, T, H, W = media.shape

            vae_spatial_downsample = self.tokenizer_spatial_compression_factor * self.patch_spatial
            vae_temporal_downsample = self.tokenizer_temporal_compression_factor

            latent_h_shape = H // vae_spatial_downsample
            latent_w_shape = W // vae_spatial_downsample
            latent_t_shape = 1 + (T - 1) // vae_temporal_downsample

            num_vision_tokens = latent_h_shape * latent_w_shape * latent_t_shape + 2
            num_tokens += num_vision_tokens

        # Action part — single sample: action is a tensor [T_action, D] or None,
        # not wrapped in a list. Mirror the original: iterate as list for uniform handling.
        if "action" in sample:
            action = sample["action"]
            action_list = action if isinstance(action, list) else [action]
            for act in action_list:
                if act is None:
                    continue
                num_tokens += act.shape[0]

        # Sound part — estimate sound tokens from audio waveform length
        if self.sound_latent_fps > 0 and "sound" in sample:
            sound_data = sample["sound"]
            if isinstance(sound_data, list) and len(sound_data) > 0:
                first_sound = sound_data[0]
                if isinstance(first_sound, list):
                    first_sound = first_sound[0]
                if first_sound is not None and isinstance(first_sound, _torch.Tensor):
                    num_audio_samples = first_sound.shape[-1]
                    audio_duration = num_audio_samples / self.audio_sample_rate
                    num_sound_tokens = int(audio_duration * self.sound_latent_fps)
                    num_tokens += num_sound_tokens
            elif isinstance(sound_data, _torch.Tensor):
                num_audio_samples = sound_data.shape[-1]
                audio_duration = num_audio_samples / self.audio_sample_rate
                num_sound_tokens = int(audio_duration * self.sound_latent_fps)
                num_tokens += num_sound_tokens

        return num_tokens

    def batches(self, samples):
        src = iter(samples)
        carry = _deque()
        exhausted = False
        while True:
            current_len = 0
            num_samples = 0
            group = []
            skipped = _deque()
            lookahead = 0

            def _next():
                if carry:
                    return carry.popleft()
                return next(src)

            while True:
                if self.max_samples_per_batch is not None and num_samples >= self.max_samples_per_batch:
                    break
                if group and lookahead >= self.lookahead_limit:
                    break
                try:
                    s = _next()
                except StopIteration:
                    exhausted = True
                    break
                n = self.sample_size(s)
                if self.max_sequence_length is not None and current_len + n >= self.max_sequence_length:
                    if not group:
                        log.error(
                            f"SequentialPackingBatcher: discarding oversized sample with {n} "
                            f"tokens (max_sequence_length={self.max_sequence_length})",
                            rank0_only=False,
                        )
                        continue
                    skipped.append(s)
                    lookahead += 1
                    continue
                current_len += n
                num_samples += 1
                group.append(s)
            for s in reversed(skipped):
                carry.appendleft(s)
            if group:
                yield group
            if exhausted and not carry:
                return
