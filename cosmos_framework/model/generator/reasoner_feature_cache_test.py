# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from cosmos_framework.data.generator.sequence_packing import PackedSequence
from cosmos_framework.model.generator.reasoner_feature_cache import (
    REASONER_FEATURE_CACHE_MANIFEST,
    IncrementalReasonerFeatureCacheWriter,
    OfflineReasonerFeatureProvider,
    ReasonerFeatureCacheEntry,
    ReasonerFeatureCacheIdentity,
    build_reasoner_feature_request_from_text_tokens,
    build_reasoner_feature_requests,
    compute_reasoner_feature_fingerprint,
    finalize_incremental_reasoner_feature_cache,
    write_reasoner_feature_cache,
)
from cosmos_framework.model.generator.reasoner_features import ReasonerFeatureBatch, ReasonerFeatureRequest


def _identity(**overrides: str) -> ReasonerFeatureCacheIdentity:
    fields = {
        "reasoner": "reasoner-checkpoint-sha256",
        "tokenizer": "tokenizer-sha256",
        "framing": "framing-v1-sha256",
    }
    fields.update(overrides)
    return ReasonerFeatureCacheIdentity(**fields)


def _features(
    sample_key: str,
    length: int,
    *,
    value_offset: int = 0,
    fingerprint: str | None = None,
) -> ReasonerFeatureBatch:
    layers_k: list[torch.Tensor] = []
    layers_v: list[torch.Tensor] = []
    for layer_idx in range(2):
        values = torch.arange(length * 2 * 4, dtype=torch.float32).reshape(length, 2, 4)
        values = (values + value_offset + 100 * layer_idx).to(torch.bfloat16)
        layers_k.append(values)
        layers_v.append(values + 0.5)
    return ReasonerFeatureBatch(
        cross_k=tuple(layers_k),
        cross_v=tuple(layers_v),
        causal_offsets=torch.tensor([0, length], dtype=torch.int64),
        fingerprints=(fingerprint or f"fingerprint-{sample_key}",),
    )


def _request(sample_key: str, length: int, *, fingerprint: str | None = None) -> ReasonerFeatureRequest:
    return ReasonerFeatureRequest(
        sample_key=sample_key,
        token_ids=torch.arange(length, dtype=torch.int64),
        position_ids=torch.arange(length, dtype=torch.int64),
        causal_offsets=torch.tensor([0, length], dtype=torch.int64),
        fingerprint=fingerprint or f"fingerprint-{sample_key}",
    )


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_content_fingerprint_covers_positions_offsets_and_identity() -> None:
    token_ids = torch.tensor([10, 20], dtype=torch.int64)
    position_ids = torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], dtype=torch.float32)
    offsets = torch.tensor([0, 2], dtype=torch.int64)
    baseline = compute_reasoner_feature_fingerprint(
        token_ids,
        position_ids,
        offsets,
        identity=_identity(),
    )
    assert baseline == compute_reasoner_feature_fingerprint(token_ids, position_ids, offsets, identity=_identity())
    assert baseline != compute_reasoner_feature_fingerprint(
        token_ids,
        position_ids + 1,
        offsets,
        identity=_identity(),
    )
    assert baseline != compute_reasoner_feature_fingerprint(
        token_ids,
        position_ids,
        offsets,
        identity=_identity(reasoner="other"),
    )


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_text_only_producer_framing_matches_finalized_training_pack() -> None:
    direct = build_reasoner_feature_request_from_text_tokens(
        sample_key="sample",
        text_ids=[10, 20],
        special_tokens={"eos_token_id": 30, "start_of_generation": 40},
        use_float_positions=True,
        identity=_identity(),
    )
    assert direct.token_ids.tolist() == [10, 20, 30, 40]
    assert direct.position_ids.dtype == torch.float32

    packed = PackedSequence(
        sample_lens=[5],
        split_lens=[4, 1],
        attn_modes=["causal", "full"],
        sequence_length=5,
        text_ids=direct.token_ids,
        text_indexes=torch.arange(4),
        position_ids=torch.cat((direct.position_ids, torch.zeros(3, 1)), dim=1),
    )
    from_training = build_reasoner_feature_requests(packed, ("sample",), _identity())[0]

    torch.testing.assert_close(from_training.token_ids, direct.token_ids)
    torch.testing.assert_close(from_training.position_ids, direct.position_ids)
    torch.testing.assert_close(from_training.causal_offsets, direct.causal_offsets)
    assert from_training.fingerprint == direct.fingerprint


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_sharded_cache_round_trip_preserves_request_order(tmp_path: Path) -> None:
    entries = [
        ReasonerFeatureCacheEntry("first", _features("first", 3, value_offset=0)),
        ReasonerFeatureCacheEntry("second", _features("second", 2, value_offset=1000)),
    ]
    cache_root = tmp_path / "cache"
    manifest_path = write_reasoner_feature_cache(
        cache_root,
        entries,
        identity=_identity(),
        # The first entry is 192 bytes and the second is 128 bytes, forcing two shards.
        max_shard_bytes=200,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["shards"]) == 2
    assert all(len(shard["records"]) == 1 for shard in manifest["shards"])

    provider = OfflineReasonerFeatureProvider(cache_root, expected_identity=_identity())
    future = provider.submit([_request("second", 2), _request("first", 3)])
    assert future.done()
    loaded = future.result()

    assert loaded.causal_offsets.tolist() == [0, 2, 5]
    assert loaded.fingerprints == ("fingerprint-second", "fingerprint-first")
    for layer_idx in range(2):
        torch.testing.assert_close(
            loaded.cross_k[layer_idx],
            torch.cat((entries[1].features.cross_k[layer_idx], entries[0].features.cross_k[layer_idx])),
        )
        torch.testing.assert_close(
            loaded.cross_v[layer_idx],
            torch.cat((entries[1].features.cross_v[layer_idx], entries[0].features.cross_v[layer_idx])),
        )


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_writer_rejects_duplicate_keys_and_incompatible_shapes(tmp_path: Path) -> None:
    duplicate_entries = [
        ReasonerFeatureCacheEntry("same", _features("same", 2)),
        ReasonerFeatureCacheEntry("same", _features("same", 3)),
    ]
    with pytest.raises(ValueError, match="Duplicate"):
        write_reasoner_feature_cache(tmp_path / "duplicate", duplicate_entries, identity=_identity())

    incompatible = _features("bad", 2)
    incompatible = ReasonerFeatureBatch(
        cross_k=(torch.zeros(2, 1, 4), torch.zeros(2, 1, 4)),
        cross_v=(torch.zeros(2, 1, 4), torch.zeros(2, 1, 4)),
        causal_offsets=incompatible.causal_offsets,
        fingerprints=incompatible.fingerprints,
    )
    with pytest.raises(ValueError, match="signature"):
        write_reasoner_feature_cache(
            tmp_path / "shape",
            [
                ReasonerFeatureCacheEntry("good", _features("good", 2)),
                ReasonerFeatureCacheEntry("bad", incompatible),
            ],
            identity=_identity(),
        )


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_writer_is_immutable_and_requires_single_sample_fingerprint(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    write_reasoner_feature_cache(
        cache_root,
        [ReasonerFeatureCacheEntry("one", _features("one", 2))],
        identity=_identity(),
    )
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_reasoner_feature_cache(
            cache_root,
            [ReasonerFeatureCacheEntry("two", _features("two", 2))],
            identity=_identity(),
        )

    no_fingerprint = ReasonerFeatureBatch(
        cross_k=(torch.zeros(2, 1, 4),),
        cross_v=(torch.zeros(2, 1, 4),),
        causal_offsets=torch.tensor([0, 2]),
    )
    with pytest.raises(ValueError, match="fingerprint"):
        ReasonerFeatureCacheEntry("missing", no_fingerprint)


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_provider_fails_closed_on_identity_and_record_fingerprint(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    write_reasoner_feature_cache(
        cache_root,
        [ReasonerFeatureCacheEntry("one", _features("one", 2))],
        identity=_identity(),
    )

    with pytest.raises(ValueError, match="requires an expected cache identity"):
        OfflineReasonerFeatureProvider(cache_root, expected_identity=None)
    with pytest.raises(ValueError, match="identity mismatch"):
        OfflineReasonerFeatureProvider(cache_root, expected_identity=_identity(reasoner="wrong"))
    with pytest.raises(ValueError, match="dtype mismatch"):
        OfflineReasonerFeatureProvider(
            cache_root,
            expected_identity=_identity(),
            expected_dtype=torch.float16,
        )

    provider = OfflineReasonerFeatureProvider(
        cache_root,
        expected_identity=_identity(),
        expected_dtype=torch.bfloat16,
    )
    future = provider.submit([_request("one", 2, fingerprint="stale")])
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        future.result()


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_provider_can_reuse_content_addressed_null_caption(tmp_path: Path) -> None:
    shared = _features("null", 2)
    cache_root = tmp_path / "cache"
    write_reasoner_feature_cache(
        cache_root,
        [ReasonerFeatureCacheEntry("__null__", shared)],
        identity=_identity(),
    )
    provider = OfflineReasonerFeatureProvider(cache_root, expected_identity=_identity())

    loaded = provider.submit([_request("unrelated-video", 2, fingerprint=shared.fingerprints[0])]).result()
    torch.testing.assert_close(loaded.cross_k[0], shared.cross_k[0])


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_provider_supports_repeated_sample_keys_with_distinct_prompt_variants(tmp_path: Path) -> None:
    first = _features("same", 2, value_offset=0, fingerprint="prompt-a")
    second = _features("same", 3, value_offset=1000, fingerprint="prompt-b")
    cache_root = tmp_path / "cache"
    write_reasoner_feature_cache(
        cache_root,
        [
            ReasonerFeatureCacheEntry("same", first),
            ReasonerFeatureCacheEntry("same", second),
        ],
        identity=_identity(),
    )
    provider = OfflineReasonerFeatureProvider(cache_root, expected_identity=_identity())

    loaded = provider.submit(
        [
            _request("same", 3, fingerprint="prompt-b"),
            _request("same", 2, fingerprint="prompt-a"),
            _request("same", 2, fingerprint="prompt-a"),
        ]
    ).result()

    assert loaded.causal_offsets.tolist() == [0, 3, 5, 7]
    torch.testing.assert_close(
        loaded.cross_k[0],
        torch.cat((second.cross_k[0], first.cross_k[0], first.cross_k[0])),
    )


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_provider_rejects_cache_miss_token_count_and_multi_document_request(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    write_reasoner_feature_cache(
        cache_root,
        [ReasonerFeatureCacheEntry("one", _features("one", 2))],
        identity=_identity(),
    )
    provider = OfflineReasonerFeatureProvider(cache_root, expected_identity=_identity())

    with pytest.raises(KeyError, match="cache miss"):
        provider.submit([_request("missing", 2)]).result()
    with pytest.raises(ValueError, match="token count mismatch"):
        provider.submit([_request("one", 3)]).result()
    multi_document = ReasonerFeatureRequest(
        sample_key="one",
        token_ids=torch.arange(2),
        position_ids=torch.arange(2),
        causal_offsets=torch.tensor([0, 1, 2]),
        fingerprint="fingerprint-one",
    )
    with pytest.raises(ValueError, match="one causal document"):
        provider.submit([multi_document]).result()


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_provider_rejects_stale_manifest_and_corrupt_shard(tmp_path: Path) -> None:
    stale_root = tmp_path / "stale"
    write_reasoner_feature_cache(
        stale_root,
        [ReasonerFeatureCacheEntry("one", _features("one", 2))],
        identity=_identity(),
    )
    manifest_path = stale_root / REASONER_FEATURE_CACHE_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"]["reasoner"] = "silently-changed"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest fingerprint"):
        OfflineReasonerFeatureProvider(stale_root, expected_identity=_identity())

    corrupt_root = tmp_path / "corrupt"
    write_reasoner_feature_cache(
        corrupt_root,
        [ReasonerFeatureCacheEntry("one", _features("one", 2))],
        identity=_identity(),
    )
    corrupt_manifest = json.loads((corrupt_root / REASONER_FEATURE_CACHE_MANIFEST).read_text(encoding="utf-8"))
    shard_path = corrupt_root / corrupt_manifest["shards"][0]["path"]
    with shard_path.open("r+b") as handle:
        handle.seek(-1, 2)
        original = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([original[0] ^ 0xFF]))
    corrupt_provider = OfflineReasonerFeatureProvider(corrupt_root, expected_identity=_identity())
    with pytest.raises(ValueError, match="checksum mismatch"):
        corrupt_provider.verify_all_shards()
    with pytest.raises(ValueError, match="checksum mismatch"):
        corrupt_provider.submit([_request("one", 2)]).result()


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_incremental_writer_resumes_skips_and_publishes_provider_compatible_cache(tmp_path: Path) -> None:
    cache_root = tmp_path / "incremental"
    first = ReasonerFeatureCacheEntry("first", _features("first", 3, value_offset=0))
    second = ReasonerFeatureCacheEntry("second", _features("second", 2, value_offset=1000))

    rank0 = IncrementalReasonerFeatureCacheWriter(
        cache_root,
        identity=_identity(),
        rank=0,
        world_size=2,
        max_shard_bytes=1024,
    )
    assert rank0.append(first)
    assert rank0.buffered_bytes > 0
    committed_path = rank0.flush()
    assert committed_path is not None and committed_path.is_file()
    assert rank0.buffered_bytes == 0

    resumed = IncrementalReasonerFeatureCacheWriter.resume(
        cache_root,
        identity=_identity(),
        rank=0,
        world_size=2,
        max_shard_bytes=1024,
    )
    assert resumed.committed_records == 1
    assert resumed.contains("first", first.features.fingerprints[0])
    assert not resumed.contains("missing", "missing-fingerprint")
    assert not resumed.append(first)
    completion = resumed.finalize()
    assert completion.is_file()
    assert resumed.finalize() == completion

    rank1 = IncrementalReasonerFeatureCacheWriter(
        cache_root,
        identity=_identity(),
        rank=1,
        world_size=2,
        max_shard_bytes=1024,
    )
    assert rank1.append(second)
    rank1.finalize()

    manifest_path = finalize_incremental_reasoner_feature_cache(
        cache_root,
        identity=_identity(),
        world_size=2,
    )
    assert manifest_path.is_file()
    assert (
        finalize_incremental_reasoner_feature_cache(
            cache_root,
            identity=_identity(),
            world_size=2,
        )
        == manifest_path
    )

    staging_root = cache_root.parent / f".{cache_root.name}.reasoner-kv-staging"
    staged_shard = next(staging_root.glob("rank-00000/*.safetensors"))
    assert not staged_shard.samefile(cache_root / staged_shard.name)

    provider = OfflineReasonerFeatureProvider(
        cache_root,
        expected_identity=_identity(),
        expected_dtype=torch.bfloat16,
    )
    loaded = provider.submit([_request("second", 2), _request("first", 3)]).result()
    assert loaded.causal_offsets.tolist() == [0, 2, 5]
    torch.testing.assert_close(
        loaded.cross_k[0],
        torch.cat((second.features.cross_k[0], first.features.cross_k[0])),
    )


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_incremental_writer_bounds_payload_and_commits_oversized_records(tmp_path: Path) -> None:
    cache_root = tmp_path / "bounded"
    writer = IncrementalReasonerFeatureCacheWriter(
        cache_root,
        identity=_identity(),
        rank=0,
        world_size=1,
        # A length-1 entry is 64 bytes; two entries must become separate shards.
        max_shard_bytes=100,
    )
    assert writer.append(ReasonerFeatureCacheEntry("one", _features("one", 1)))
    assert writer.buffered_bytes == 64
    assert writer.append(ReasonerFeatureCacheEntry("two", _features("two", 1)))
    assert writer.buffered_bytes == 64
    writer.finalize()

    staging_root = cache_root.parent / f".{cache_root.name}.reasoner-kv-staging"
    assert len(list(staging_root.glob("rank-00000/*.safetensors"))) == 2

    oversized_root = tmp_path / "oversized"
    oversized = IncrementalReasonerFeatureCacheWriter(
        oversized_root,
        identity=_identity(),
        rank=0,
        world_size=1,
        max_shard_bytes=32,
    )
    assert oversized.append(ReasonerFeatureCacheEntry("large", _features("large", 2)))
    assert oversized.buffered_bytes == 0
    assert oversized.committed_records == 1


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_incremental_finalizer_rejects_missing_completion_duplicates_and_signature_mismatch(tmp_path: Path) -> None:
    incomplete_root = tmp_path / "incomplete"
    incomplete = IncrementalReasonerFeatureCacheWriter(
        incomplete_root,
        identity=_identity(),
        rank=0,
        world_size=1,
    )
    incomplete.append(ReasonerFeatureCacheEntry("one", _features("one", 2)))
    incomplete.flush()
    with pytest.raises(FileNotFoundError, match="completion marker"):
        finalize_incremental_reasoner_feature_cache(
            incomplete_root,
            identity=_identity(),
            world_size=1,
        )

    duplicate_root = tmp_path / "duplicate-ranks"
    duplicate_entry = ReasonerFeatureCacheEntry("same", _features("same", 2))
    for rank in range(2):
        writer = IncrementalReasonerFeatureCacheWriter(
            duplicate_root,
            identity=_identity(),
            rank=rank,
            world_size=2,
        )
        writer.append(duplicate_entry)
        writer.finalize()
    with pytest.raises(ValueError, match="Duplicate Reasoner feature record across ranks"):
        finalize_incremental_reasoner_feature_cache(
            duplicate_root,
            identity=_identity(),
            world_size=2,
        )

    signature_root = tmp_path / "signature-ranks"
    rank0 = IncrementalReasonerFeatureCacheWriter(
        signature_root,
        identity=_identity(),
        rank=0,
        world_size=2,
    )
    rank0.append(ReasonerFeatureCacheEntry("normal", _features("normal", 2)))
    rank0.finalize()
    incompatible = ReasonerFeatureBatch(
        cross_k=(torch.zeros(2, 1, 4, dtype=torch.bfloat16),) * 2,
        cross_v=(torch.zeros(2, 1, 4, dtype=torch.bfloat16),) * 2,
        causal_offsets=torch.tensor([0, 2]),
        fingerprints=("different-signature",),
    )
    rank1 = IncrementalReasonerFeatureCacheWriter(
        signature_root,
        identity=_identity(),
        rank=1,
        world_size=2,
    )
    rank1.append(ReasonerFeatureCacheEntry("incompatible", incompatible))
    rank1.finalize()
    with pytest.raises(ValueError, match="inconsistent signatures"):
        finalize_incremental_reasoner_feature_cache(
            signature_root,
            identity=_identity(),
            world_size=2,
        )


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_incremental_resume_and_finalizer_reject_corrupt_committed_shard(tmp_path: Path) -> None:
    cache_root = tmp_path / "corrupt-incremental"
    writer = IncrementalReasonerFeatureCacheWriter(
        cache_root,
        identity=_identity(),
        rank=0,
        world_size=1,
    )
    writer.append(ReasonerFeatureCacheEntry("one", _features("one", 2)))
    writer.finalize()

    staging_root = cache_root.parent / f".{cache_root.name}.reasoner-kv-staging"
    shard_path = next(staging_root.glob("rank-00000/*.safetensors"))
    with shard_path.open("r+b") as handle:
        handle.seek(-1, 2)
        original = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([original[0] ^ 0xFF]))

    with pytest.raises(ValueError, match="checksum mismatch"):
        IncrementalReasonerFeatureCacheWriter.resume(
            cache_root,
            identity=_identity(),
            rank=0,
            world_size=1,
        )
    with pytest.raises(ValueError, match="checksum mismatch"):
        finalize_incremental_reasoner_feature_cache(
            cache_root,
            identity=_identity(),
            world_size=1,
        )


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_incremental_finalizer_allows_an_empty_completed_rank(tmp_path: Path) -> None:
    cache_root = tmp_path / "empty-rank"
    rank0 = IncrementalReasonerFeatureCacheWriter(
        cache_root,
        identity=_identity(),
        rank=0,
        world_size=2,
    )
    rank0.append(ReasonerFeatureCacheEntry("one", _features("one", 2)))
    rank0.finalize()
    IncrementalReasonerFeatureCacheWriter(
        cache_root,
        identity=_identity(),
        rank=1,
        world_size=2,
    ).finalize()

    manifest_path = finalize_incremental_reasoner_feature_cache(
        cache_root,
        identity=_identity(),
        world_size=2,
    )
    assert manifest_path.is_file()
