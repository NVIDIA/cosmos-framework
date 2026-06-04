# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Golden-batch equality: legacy PackingDataLoader+RankPartitionedDataLoader vs the
new four-role VFM dataflow stack on the same fixed, deterministic source.

This test proves that the new CosmosDataLoader(RankPartitionedDistributor,
IdentityProcessor, SequentialPackingBatcher, VFMListCollator) yields the SAME
packed batches as PackingDataLoader(RankPartitionedDataLoader(...)) given an
identical input stream.

Architecture differences that are EXPECTED and explicitly excluded:
  - Keys starting with '_' (internal bookkeeping: _num_tokens, _num_samples, etc.)
    are emitted by the legacy PackingDataLoader but not by the new stack.
  - 'dataset_name': set by the legacy packing loop but not by SequentialPackingBatcher.
  - Nesting depth for multi-item keys:
    * Legacy: video / text_token_ids stored as list[list[tensor]] (inner list wraps
      each sample via v[i:i+1] in _get_next_sample).
    * New:    video / text_token_ids stored as list[tensor] (VFMListCollator puts
              each sample's tensor directly in the list, no inner wrapping).
    The per-sample tensor CONTENTS are identical; we compare them elementwise.
  - 'image_size': both paths flatten to a plain list of tensors (same shape).
"""

from __future__ import annotations

import os
import torch
import torch.distributed as dist
import torch.utils.data


# ──────────────────────────────────────────────────────────────────────────────
# Deterministic stub dataset
# ──────────────────────────────────────────────────────────────────────────────

# Fixed sample specs: (text_len, T, H, W).  Varied so token counts differ and
# multiple samples actually pack per batch.
_SAMPLE_SPECS = [
    (10, 1, 64, 64),
    (5,  1, 32, 32),
    (8,  2, 64, 64),
    (3,  1, 32, 64),
    (12, 1, 64, 64),
    (6,  2, 32, 32),
    (4,  1, 64, 32),
    (9,  1, 32, 32),
    (7,  2, 64, 32),
    (11, 1, 32, 64),
    (2,  1, 32, 32),
    (15, 1, 64, 64),
    (5,  2, 32, 64),
    (8,  1, 32, 32),
    (6,  1, 64, 64),
]

def _make_fixed_samples():
    """Return a deterministic list of SFT-shaped sample dicts."""
    samples = []
    for idx, (tlen, T, H, W) in enumerate(_SAMPLE_SPECS):
        # Use constant tensors so equality checks are trivially deterministic.
        video = torch.full((3, T, H, W), float(idx), dtype=torch.float32)
        text_token_ids = torch.arange(tlen, dtype=torch.long)
        # image_size: a small tensor exercising the collator's list path.
        image_size = torch.tensor([H, W], dtype=torch.long)
        samples.append({
            "video": video,
            "text_token_ids": text_token_ids,
            "image_size": image_size,
        })
    return samples


class _FixedSFTDataset(torch.utils.data.IterableDataset):
    """Yields the fixed sample list, cycling indefinitely.

    Exposes shard_world_size / shard_rank / shard_id attributes so
    RankPartitionedDataLoader and RankPartitionedDistributor can set them.
    For world_size=1 (single-process) we simply ignore them and yield all.
    """

    def __init__(self):
        super().__init__()
        self._samples = _make_fixed_samples()
        self.shard_world_size = 1
        self.shard_rank = 0
        self.shard_id = 0

    def __len__(self):
        # Twice the fixed list so the packer can fill N=5 batches comfortably.
        return len(self._samples) * 2

    def __iter__(self):
        # Yield ALL samples (world_size=1 case; repeating twice so the packer
        # can fill N=5 batches without exhausting the stream).
        yield from self._samples
        yield from self._samples


# ──────────────────────────────────────────────────────────────────────────────
# Token-budget: sized to guarantee multi-sample packing
# ──────────────────────────────────────────────────────────────────────────────
# With (spatial_factor=16, patch_spatial=2, temporal_factor=4):
#   32x32 video, T=1 → latent 1x1x1 + 2 = 3 vision tokens
#   64x64 video, T=1 → latent 2x2x1 + 2 = 6 vision tokens
# Smallest sample: text_len=2 → 2+1+3=6 tokens.
# Budget of 80 → many samples pack per batch.
_BUDGET = 80

_PACKER_KWARGS = dict(
    tokenizer_spatial_compression_factor=16,
    tokenizer_temporal_compression_factor=4,
    patch_spatial=2,
    sound_latent_fps=0,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _setup_dist():
    """Init a single-process gloo group; return True if we used gloo, False for monkeypatch."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29557")
    if not dist.is_initialized():
        try:
            dist.init_process_group(backend="gloo", rank=0, world_size=1)
            return True
        except Exception:
            pass
    # Fallback: monkeypatch so RankPartitionedDataLoader.__init__ succeeds.
    return False


def _monkeypatch_dist(monkeypatch):
    """Patch the three distributed calls used by RankPartitionedDataLoader."""
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)


def _drain(loader, n: int) -> list[dict]:
    it = iter(loader)
    return [next(it) for _ in range(n)]


# ──────────────────────────────────────────────────────────────────────────────
# Payload-key extraction helpers
# ──────────────────────────────────────────────────────────────────────────────

_SKIP_KEYS = {"dataset_name"}  # bookkeeping only, not in new stack


def _payload_keys(batch: dict) -> set[str]:
    """Return non-bookkeeping keys for comparison."""
    return {k for k in batch if not k.startswith("_") and k not in _SKIP_KEYS}


def _unwrap_video(val):
    """Normalise legacy list[list[tensor]] and new list[tensor] to list[tensor]."""
    out = []
    for item in val:
        if isinstance(item, list):
            # legacy wraps each sample as a single-element list
            out.extend(item)
        else:
            out.append(item)
    return out


def _unwrap_text(val):
    """Same unwrapping for text_token_ids."""
    return _unwrap_video(val)


def _compare_batches(legacy: dict, new: dict) -> None:
    """Assert that the meaningful payload keys match between the two batches."""
    lk = _payload_keys(legacy)
    nk = _payload_keys(new)
    assert lk == nk, f"Key mismatch: legacy={sorted(lk)}, new={sorted(nk)}"

    for key in sorted(lk):
        lv = legacy[key]
        nv = new[key]

        if key in ("video", "text_token_ids"):
            # Both are lists; legacy may wrap each element in an inner list.
            lu = _unwrap_video(lv)
            nu = _unwrap_text(nv)
            assert len(lu) == len(nu), (
                f"key={key}: per-sample count mismatch: "
                f"legacy_unwrapped={len(lu)}, new_unwrapped={len(nu)}"
            )
            for i, (a, b) in enumerate(zip(lu, nu)):
                assert torch.equal(a, b), (
                    f"key={key} sample[{i}] mismatch: legacy={a}, new={b}"
                )

        elif key == "image_size":
            # Both should be flat lists of equal-shaped tensors.
            assert len(lv) == len(nv), (
                f"key={key}: list length mismatch: legacy={len(lv)}, new={len(nv)}"
            )
            for i, (a, b) in enumerate(zip(lv, nv)):
                assert torch.equal(a, b), (
                    f"key={key}[{i}] mismatch: legacy={a}, new={b}"
                )

        elif isinstance(lv, torch.Tensor) and isinstance(nv, torch.Tensor):
            assert torch.equal(lv, nv), f"key={key} tensor mismatch"

        elif isinstance(lv, list) and isinstance(nv, list):
            assert len(lv) == len(nv), f"key={key} list length mismatch"
            for i, (a, b) in enumerate(zip(lv, nv)):
                if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                    assert torch.equal(a, b), f"key={key}[{i}] tensor mismatch"
                else:
                    assert a == b, f"key={key}[{i}] value mismatch"
        else:
            assert lv == nv, f"key={key} value mismatch: {lv!r} vs {nv!r}"


# ──────────────────────────────────────────────────────────────────────────────
# The golden-batch equality test
# ──────────────────────────────────────────────────────────────────────────────

N_BATCHES = 5


def test_vfm_golden_batches_match(monkeypatch):
    """New four-role stack yields same packed batches as legacy PackingDataLoader."""
    from cosmos_framework.data.vfm.joint_dataloader import (
        PackingDataLoader,
        RankPartitionedDataLoader,
    )
    from cosmos_framework.data.vfm.dataflow import (
        CosmosDataLoader,
        RankPartitionedDistributor,
        SequentialPackingBatcher,
        VFMListCollator,
        IdentityProcessor,
    )

    # ── distributed bootstrap ──────────────────────────────────────────────
    used_gloo = _setup_dist()
    if not used_gloo:
        _monkeypatch_dist(monkeypatch)

    try:
        # ── legacy stack ──────────────────────────────────────────────────
        stub_legacy = _FixedSFTDataset()
        legacy = PackingDataLoader(
            dataloader=RankPartitionedDataLoader(
                datasets={"video": {"dataset": stub_legacy, "ratio": 1}},
                batch_size=1,
            ),
            max_sequence_length=_BUDGET,
            max_samples_per_batch=None,
            **_PACKER_KWARGS,
        )

        # ── new stack ─────────────────────────────────────────────────────
        stub_new = _FixedSFTDataset()
        new = CosmosDataLoader(
            distributor=RankPartitionedDistributor(
                {"video": {"dataset": stub_new, "ratio": 1}}
            ),
            processor=IdentityProcessor(),
            batcher=SequentialPackingBatcher(
                max_sequence_length=_BUDGET,
                max_samples_per_batch=None,
                audio_sample_rate=48000,
                **_PACKER_KWARGS,
            ),
            collator=VFMListCollator(),
            num_workers=0,
        )

        # ── drain N batches and compare ───────────────────────────────────
        legacy_batches = _drain(legacy, N_BATCHES)
        new_batches = _drain(new, N_BATCHES)

        assert len(legacy_batches) == N_BATCHES, f"Expected {N_BATCHES} legacy batches"
        assert len(new_batches) == N_BATCHES, f"Expected {N_BATCHES} new batches"

        for i, (lb, nb) in enumerate(zip(legacy_batches, new_batches)):
            # Number of packed samples (inferred from video list length after unwrap)
            n_legacy = len(_unwrap_video(lb["video"]))
            n_new = len(_unwrap_text(nb["video"]))
            assert n_legacy == n_new, (
                f"Batch {i}: sample count mismatch: legacy={n_legacy}, new={n_new}"
            )
            _compare_batches(lb, nb)

    finally:
        if used_gloo and dist.is_initialized():
            dist.destroy_process_group()
