# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Golden-batch EXACT equality: legacy PackingDataLoader+RankPartitionedDataLoader
vs the new four-role VFM dataflow stack on the same fixed, deterministic source.

This test proves that the new CosmosDataLoader(RankPartitionedDistributor,
IdentityProcessor, SequentialPackingBatcher, VFMListCollator) yields STRUCTURALLY
IDENTICAL packed batches to PackingDataLoader(RankPartitionedDataLoader(...)) given
an identical input stream — INCLUDING the list[list[Tensor]] nesting for
_MULTI_ITEM_KEYS.

Comparison covers all payload keys (those not starting with ``_`` and not
``dataset_name``).  No nesting normalization is applied.

  - ``video``, ``text_token_ids``: both stacks produce ``list[list[Tensor]]``
    (each inner list is a single-element list wrapping the sample tensor).
  - ``image_size``: both stacks produce a flat ``list[Tensor]``
    (``_FLATTEN_LIST_KEYS`` path → extend rather than append).

Bookkeeping keys excluded from comparison:
  - Keys starting with ``_`` (e.g. ``_num_tokens``): legacy internal metadata.
  - ``dataset_name``: set by the legacy packing loop; not emitted by
    SequentialPackingBatcher.
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
        # image_size: a small tensor exercising the _FLATTEN_LIST_KEYS path.
        image_size = torch.tensor([H, W], dtype=torch.long)
        samples.append({
            "video": video,
            "text_token_ids": text_token_ids,
            "image_size": image_size,
        })
    return samples


class _FixedSFTDataset(torch.utils.data.IterableDataset):
    """Yields the fixed sample list, cycling twice.

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

def _setup_dist(monkeypatch):
    """Init a single-process gloo group; return True if we used gloo, False for monkeypatch.

    Uses monkeypatch.setenv (auto-restored at teardown) so the test does not leave
    MASTER_ADDR/MASTER_PORT dirtied in os.environ — the repo conftest enforces this.
    """
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29557")
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
# Exact structural comparison helpers
# ──────────────────────────────────────────────────────────────────────────────

_SKIP_KEYS = {"dataset_name"}  # bookkeeping only, not in new stack


def _payload_keys(batch: dict) -> set[str]:
    """Return non-bookkeeping keys for comparison."""
    return {k for k in batch if not k.startswith("_") and k not in _SKIP_KEYS}


def _assert_tensors_equal(a, b, label: str) -> None:
    assert isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor), (
        f"{label}: expected two Tensors, got {type(a)} and {type(b)}"
    )
    assert torch.equal(a, b), f"{label}: tensor mismatch:\n  legacy={a}\n  new={b}"


def _assert_exact(legacy_val, new_val, key: str) -> None:
    """Recursively assert exact structural and value equality."""
    if isinstance(legacy_val, list) and isinstance(new_val, list):
        assert len(legacy_val) == len(new_val), (
            f"key={key}: list length mismatch: legacy={len(legacy_val)}, new={len(new_val)}"
        )
        for i, (a, b) in enumerate(zip(legacy_val, new_val)):
            _assert_exact(a, b, f"{key}[{i}]")
    elif isinstance(legacy_val, torch.Tensor) and isinstance(new_val, torch.Tensor):
        _assert_tensors_equal(legacy_val, new_val, f"key={key}")
    else:
        assert type(legacy_val) == type(new_val), (
            f"key={key}: type mismatch: legacy={type(legacy_val)}, new={type(new_val)}"
        )
        assert legacy_val == new_val, (
            f"key={key}: value mismatch: legacy={legacy_val!r}, new={new_val!r}"
        )


def _compare_batches_exact(legacy: dict, new: dict, batch_idx: int) -> None:
    """Assert EXACT structural identity (including list[list[Tensor]] nesting)."""
    lk = _payload_keys(legacy)
    nk = _payload_keys(new)
    assert lk == nk, (
        f"Batch {batch_idx}: key mismatch: legacy={sorted(lk)}, new={sorted(nk)}"
    )
    for key in sorted(lk):
        _assert_exact(legacy[key], new[key], f"batch[{batch_idx}][{key}]")


# ──────────────────────────────────────────────────────────────────────────────
# The golden-batch EXACT equality test
# ──────────────────────────────────────────────────────────────────────────────

N_BATCHES = 5


def test_vfm_golden_batches_match(monkeypatch):
    """New four-role stack yields STRUCTURALLY IDENTICAL packed batches as legacy PackingDataLoader.

    Asserts exact list[list[Tensor]] nesting for _MULTI_ITEM_KEYS (video, text_token_ids)
    and flat list[Tensor] for image_size — no nesting normalization.
    """
    from cosmos_framework.data.generator.joint_dataloader import (
        PackingDataLoader,
        RankPartitionedDataLoader,
    )
    from cosmos_framework.data.generator.dataflow import (
        CosmosDataLoader,
        RankPartitionedDistributor,
        SequentialPackingBatcher,
        VFMListCollator,
        IdentityProcessor,
    )

    # ── distributed bootstrap ──────────────────────────────────────────────
    used_gloo = _setup_dist(monkeypatch)
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
            _compare_batches_exact(lb, nb, batch_idx=i)

    finally:
        if used_gloo and dist.is_initialized():
            dist.destroy_process_group()
