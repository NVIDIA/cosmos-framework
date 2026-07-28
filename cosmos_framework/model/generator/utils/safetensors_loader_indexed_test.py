# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Tests for the indexed-snapshot loading branch of load_vlm_model
(nvidia/Cosmos3-Edge layout: no top-level shards; a root
model.safetensors.index.json maps reasoner keys into subdir shard files).

Tests needing the real HF snapshot or the canonical converter output skip
themselves when those files are absent, so the suite is green on CI runners
without the Edge checkpoint cached.
"""

import json
import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

import cosmos_framework.model.generator.utils.safetensors_loader as safetensors_loader
from cosmos_framework.model.generator.utils.safetensors_loader import (
    _EDGE_INDEX_GEN_ONLY_KEY_RE,
    _detect_indexed_snapshot,
    _verify_edge_vision_shard_hash,
    convert_key_from_cosmos3_edge_index,
    load_vlm_model,
)
from cosmos_framework.model.generator.utils.safetensors_loader_test import _StubConfig, _StubModel

_REPO_ROOT = Path(__file__).resolve().parents[4]

# Canonical converter output: its key list is the authoritative target set
# (identical to Cosmos3EdgeForConditionalGeneration.state_dict().keys()).
_CANONICAL_VLM_FILE = _REPO_ROOT / "examples" / "checkpoints" / "Cosmos3-Edge-Reasoner-VLM" / "model.safetensors"


def _find_local_edge_snapshot() -> str | None:
    """Locate a locally cached nvidia/Cosmos3-Edge indexed snapshot, if any."""
    roots: list[str] = []
    if os.environ.get("HF_HOME"):
        roots.append(os.path.join(os.environ["HF_HOME"], "hub"))
    if os.environ.get("HF_HUB_CACHE"):
        roots.append(os.environ["HF_HUB_CACHE"])
    roots.append(os.path.expanduser("~/.cache/huggingface/hub"))
    for root in roots:
        snapshots_dir = os.path.join(root, "models--nvidia--Cosmos3-Edge", "snapshots")
        if not os.path.isdir(snapshots_dir):
            continue
        for name in sorted(os.listdir(snapshots_dir)):
            snapshot = os.path.join(snapshots_dir, name)
            try:
                if _detect_indexed_snapshot(snapshot) is not None:
                    return snapshot
            except (ValueError, FileNotFoundError):
                continue  # partially downloaded / foreign snapshot — keep looking
    return None


def _write_indexed_snapshot(root: Path, shards: dict[str, dict[str, torch.Tensor]]) -> Path:
    """Write a fake indexed snapshot: subdir shard files + root weight-map index."""
    root.mkdir(parents=True, exist_ok=True)
    weight_map: dict[str, str] = {}
    for rel_path, tensors in shards.items():
        shard_path = root / rel_path
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(shard_path))
        for key in tensors:
            weight_map[key] = rel_path
    with open(root / "model.safetensors.index.json", "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f)
    return root


# --- convert_key_from_cosmos3_edge_index (pure remap) ------------------------


@pytest.mark.L0
@pytest.mark.CPU
def test_edge_index_remap_spot_pairs():
    """Representative index → model pairs from the mapping table."""
    pairs = {
        "layers.0.self_attn.to_q.weight": "model.language_model.layers.0.mixer.q_proj.weight",
        "layers.3.self_attn.to_k.weight": "model.language_model.layers.6.mixer.k_proj.weight",
        "layers.5.self_attn.to_v.weight": "model.language_model.layers.10.mixer.v_proj.weight",
        "layers.27.self_attn.to_out.weight": "model.language_model.layers.54.mixer.o_proj.weight",
        "layers.0.input_layernorm.weight": "model.language_model.layers.0.norm.weight",
        "layers.4.mlp.up_proj.weight": "model.language_model.layers.9.mixer.up_proj.weight",
        "layers.4.mlp.down_proj.weight": "model.language_model.layers.9.mixer.down_proj.weight",
        "layers.27.post_attention_layernorm.weight": "model.language_model.layers.55.norm.weight",
        "embed_tokens.weight": "model.language_model.embeddings.weight",
        "norm.weight": "model.language_model.norm_f.weight",
        "lm_head.weight": "lm_head.weight",
        # visual + projector pass through unchanged
        "model.visual.encoder.layers.0.self_attn.q_proj.weight": (
            "model.visual.encoder.layers.0.self_attn.q_proj.weight"
        ),
        "model.projector.linear_fc1.bias": "model.projector.linear_fc1.bias",
    }
    for raw, expected in pairs.items():
        assert convert_key_from_cosmos3_edge_index(raw) == expected

    # Keys outside the reasoner manifest must return None (fail loudly upstream).
    for bad in (
        "layers.0.self_attn.q_norm.weight",
        "layers.0.mlp.gate_proj.weight",
        "action_proj_in.fc.weight",
        "blocks.0.attn.to_q_moe_gen.weight",
        "model.language_model.layers.0.mixer.q_proj.weight",  # already-canonical keys are not index keys
    ):
        assert convert_key_from_cosmos3_edge_index(bad) is None


@pytest.mark.L0
@pytest.mark.CPU
def test_edge_index_remap_miniature_manifest_bijection():
    """Full per-layer pattern for a miniature 2-layer manifest: the remap must
    be a bijection onto the expected canonical key set."""
    raw_keys = ["embed_tokens.weight", "norm.weight", "lm_head.weight"]
    expected = {
        "model.language_model.embeddings.weight",
        "model.language_model.norm_f.weight",
        "lm_head.weight",
    }
    for n in range(2):
        raw_keys += [
            f"layers.{n}.self_attn.to_q.weight",
            f"layers.{n}.self_attn.to_k.weight",
            f"layers.{n}.self_attn.to_v.weight",
            f"layers.{n}.self_attn.to_out.weight",
            f"layers.{n}.input_layernorm.weight",
            f"layers.{n}.mlp.up_proj.weight",
            f"layers.{n}.mlp.down_proj.weight",
            f"layers.{n}.post_attention_layernorm.weight",
        ]
        expected |= {
            f"model.language_model.layers.{2 * n}.mixer.q_proj.weight",
            f"model.language_model.layers.{2 * n}.mixer.k_proj.weight",
            f"model.language_model.layers.{2 * n}.mixer.v_proj.weight",
            f"model.language_model.layers.{2 * n}.mixer.o_proj.weight",
            f"model.language_model.layers.{2 * n}.norm.weight",
            f"model.language_model.layers.{2 * n + 1}.mixer.up_proj.weight",
            f"model.language_model.layers.{2 * n + 1}.mixer.down_proj.weight",
            f"model.language_model.layers.{2 * n + 1}.norm.weight",
        }
    raw_keys += ["model.visual.post_layernorm.weight", "model.projector.norm.weight"]
    expected |= {"model.visual.post_layernorm.weight", "model.projector.norm.weight"}

    mapped = [convert_key_from_cosmos3_edge_index(k) for k in raw_keys]
    assert None not in mapped
    assert len(set(mapped)) == len(mapped)  # injective
    assert set(mapped) == expected  # onto


@pytest.mark.L0
@pytest.mark.CPU
def test_edge_index_remap_full_root_index_bijection_onto_canonical():
    """Map ALL reasoner root-index keys of the real snapshot and assert an exact
    bijection onto the canonical Cosmos3-Edge-Reasoner-VLM key list (the 28
    gen-only k_norm_und_for_gen keys are skipped by detection)."""
    snapshot = _find_local_edge_snapshot()
    if snapshot is None:
        pytest.skip("no local nvidia/Cosmos3-Edge indexed snapshot in the HF cache")
    if not os.path.exists(_CANONICAL_VLM_FILE):
        pytest.skip(f"canonical key list not available: {_CANONICAL_VLM_FILE}")

    with open(os.path.join(snapshot, "model.safetensors.index.json")) as f:
        raw_keys = sorted(json.load(f)["weight_map"])
    with safe_open(str(_CANONICAL_VLM_FILE), framework="pt") as f:
        canonical_keys = set(f.keys())

    gen_only = [k for k in raw_keys if _EDGE_INDEX_GEN_ONLY_KEY_RE.match(k)]
    assert len(gen_only) in (0, 28), gen_only
    reasoner_keys = [k for k in raw_keys if not _EDGE_INDEX_GEN_ONLY_KEY_RE.match(k)]

    mapped = {k: convert_key_from_cosmos3_edge_index(k) for k in reasoner_keys}
    assert None not in mapped.values(), sorted(k for k, v in mapped.items() if v is None)
    assert len(set(mapped.values())) == len(mapped)  # injective
    assert set(mapped.values()) == canonical_keys  # onto the authoritative target set
    assert len(mapped) == 670


# --- _detect_indexed_snapshot layout gating ----------------------------------


@pytest.mark.L0
@pytest.mark.CPU
def test_detect_returns_none_for_flat_layout(tmp_path):
    """Top-level *.safetensors present → existing flat path, even when an
    index file also exists (regression guard for nano/super/llava snapshots)."""
    save_file({"model.embed_tokens.weight": torch.zeros(2, 2)}, str(tmp_path / "model.safetensors"))
    with open(tmp_path / "model.safetensors.index.json", "w") as f:
        json.dump({"weight_map": {"model.embed_tokens.weight": "model.safetensors"}}, f)
    assert _detect_indexed_snapshot(str(tmp_path)) is None


@pytest.mark.L0
@pytest.mark.CPU
def test_detect_returns_none_without_subdir_references(tmp_path):
    """No index at all, and an index whose weight_map has no subdir references,
    both fall through to the existing path."""
    assert _detect_indexed_snapshot(str(tmp_path)) is None  # empty dir, no index

    with open(tmp_path / "model.safetensors.index.json", "w") as f:
        json.dump({"weight_map": {"embed_tokens.weight": "model-00001-of-00002.safetensors"}}, f)
    assert _detect_indexed_snapshot(str(tmp_path)) is None

    # Non-directory inputs (S3 URI / bare repo id) are never indexed snapshots.
    assert _detect_indexed_snapshot("s3://bucket/model") is None
    assert _detect_indexed_snapshot("nvidia/Cosmos3-Edge") is None


@pytest.mark.L0
@pytest.mark.CPU
def test_detect_raises_on_unmappable_index_key(tmp_path):
    _write_indexed_snapshot(
        tmp_path,
        {"transformer/part-00001.safetensors": {"layers.0.mlp.gate_proj.weight": torch.zeros(2, 2)}},
    )
    with pytest.raises(ValueError, match="no canonical model-key mapping"):
        _detect_indexed_snapshot(str(tmp_path))


@pytest.mark.L0
@pytest.mark.CPU
def test_detect_skips_gen_only_k_norm_keys(tmp_path):
    """Gen-only k_norm_und_for_gen index keys are recognized and skipped;
    genuinely unknown keys keep failing loudly."""
    _write_indexed_snapshot(
        tmp_path,
        {
            "transformer/part-00001.safetensors": {
                "layers.0.self_attn.to_q.weight": torch.zeros(2, 2),
                "layers.0.self_attn.k_norm_und_for_gen.weight": torch.zeros(2),
            }
        },
    )
    snapshot = _detect_indexed_snapshot(str(tmp_path))
    assert snapshot is not None
    assert "layers.0.self_attn.k_norm_und_for_gen.weight" not in snapshot.key_map
    assert snapshot.key_map == {"layers.0.self_attn.to_q.weight": "model.language_model.layers.0.mixer.q_proj.weight"}


@pytest.mark.L0
@pytest.mark.CPU
def test_detect_raises_on_missing_shard_file(tmp_path):
    _write_indexed_snapshot(
        tmp_path,
        {"transformer/part-00001.safetensors": {"embed_tokens.weight": torch.zeros(2, 2)}},
    )
    with open(tmp_path / "model.safetensors.index.json", "w") as f:
        json.dump({"weight_map": {"embed_tokens.weight": "transformer/does-not-exist.safetensors"}}, f)
    with pytest.raises(FileNotFoundError, match="shard file"):
        _detect_indexed_snapshot(str(tmp_path))


# --- G2 vision-shard hash guard ----------------------------------------------


@pytest.mark.L0
@pytest.mark.CPU
def test_g2_hash_mismatch_raises(tmp_path):
    """A vision_encoder/model.safetensors with any other content must be refused."""
    snapshot = _write_indexed_snapshot(
        tmp_path,
        {"vision_encoder/model.safetensors": {"model.visual.post_layernorm.weight": torch.zeros(4)}},
    )
    with pytest.raises(ValueError, match="upstream vision weights changed"):
        _verify_edge_vision_shard_hash(str(snapshot), ["vision_encoder/model.safetensors"])


@pytest.mark.L0
@pytest.mark.CPU
def test_g2_hash_guard_skipped_without_vision_shard(tmp_path):
    """Snapshots whose index does not reference the vision shard skip the guard."""
    _verify_edge_vision_shard_hash(str(tmp_path), ["transformer/part-00001.safetensors"])  # no raise


@pytest.mark.L0
@pytest.mark.CPU
def test_g2_hash_pass_with_matching_digest(tmp_path, monkeypatch):
    """Positive path: with the constant patched to the actual digest, no raise."""
    import hashlib

    snapshot = _write_indexed_snapshot(
        tmp_path,
        {"vision_encoder/model.safetensors": {"model.visual.post_layernorm.weight": torch.zeros(4)}},
    )
    with open(snapshot / "vision_encoder" / "model.safetensors", "rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    monkeypatch.setattr(safetensors_loader, "_EDGE_VISION_SHARD_SHA256", digest)
    _verify_edge_vision_shard_hash(str(snapshot), ["vision_encoder/model.safetensors"])  # no raise


@pytest.mark.L0
@pytest.mark.CPU
def test_load_vlm_model_runs_g2_guard_on_indexed_snapshot(tmp_path):
    """End-to-end: the indexed branch of load_vlm_model hits the G2 guard
    BEFORE copying any tensor."""
    snapshot = _write_indexed_snapshot(
        tmp_path,
        {"vision_encoder/model.safetensors": {"model.visual.post_layernorm.weight": torch.ones(4)}},
    )
    config = _StubConfig(
        text_config=_StubConfig(num_experts=None, num_local_experts=None),
        tie_word_embeddings=False,
    )
    model = _StubModel({"model.visual.post_layernorm.weight": (4,)}, config)
    with pytest.raises(ValueError, match="upstream vision weights changed"):
        load_vlm_model(model=model, checkpoint_path=str(snapshot), credential_path=None, parallel_dims=None)
    # Guard fired before any tensor copy: the model param is untouched.
    assert torch.equal(model._params["model.visual.post_layernorm.weight"].data, torch.zeros(4))


# --- load_vlm_model indexed branch (fake snapshot, single-rank CPU) -----------


def _edge_stub(param_shapes: dict[str, tuple[int, ...]]) -> _StubModel:
    config = _StubConfig(
        text_config=_StubConfig(num_experts=None, num_local_experts=None),
        tie_word_embeddings=False,
    )
    return _StubModel(param_shapes, config)


@pytest.mark.L0
@pytest.mark.CPU
def test_load_vlm_model_indexed_end_to_end(tmp_path):
    """Fake indexed snapshot (no vision shard → G2 skipped): every root-index
    key lands on its remapped canonical key; DiT-only tensors in the shared
    shard files are never loaded."""
    fills = {
        "layers.0.self_attn.to_q.weight": 1.0,
        "layers.0.self_attn.to_k.weight": 2.0,
        "layers.0.self_attn.to_v.weight": 3.0,
        "layers.0.self_attn.to_out.weight": 4.0,
        "layers.0.input_layernorm.weight": 5.0,
        "layers.0.mlp.up_proj.weight": 6.0,
        "layers.0.mlp.down_proj.weight": 7.0,
        "layers.0.post_attention_layernorm.weight": 8.0,
        "embed_tokens.weight": 9.0,
        "norm.weight": 10.0,
        "lm_head.weight": 11.0,
        "model.projector.norm.weight": 12.0,
    }

    def t(raw_key: str) -> torch.Tensor:
        shape = (4,) if "norm" in raw_key else (4, 4)
        return torch.full(shape, fills[raw_key])

    shard1 = {k: t(k) for k in list(fills)[:5]}
    # DiT-generation tensor sharing the shard file but absent from the index.
    shard1["blocks.0.attn.to_q_moe_gen.weight"] = torch.full((4, 4), -1.0)
    shard2 = {k: t(k) for k in list(fills)[5:]}
    snapshot = _write_indexed_snapshot(
        tmp_path,
        {"transformer/part-00001.safetensors": shard1, "transformer/part-00002.safetensors": shard2},
    )
    # The DiT extra must not appear in the index: rewrite weight_map without it.
    with open(snapshot / "model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]
    del weight_map["blocks.0.attn.to_q_moe_gen.weight"]
    with open(snapshot / "model.safetensors.index.json", "w") as f:
        json.dump({"weight_map": weight_map}, f)

    expected = {convert_key_from_cosmos3_edge_index(k): t(k) for k in fills}
    model = _edge_stub({k: tuple(v.shape) for k, v in expected.items()})

    keys_loaded = load_vlm_model(
        model=model,
        checkpoint_path=str(snapshot),
        credential_path=None,
        parallel_dims=None,
    )

    assert keys_loaded == set(expected)
    for canonical_key, tensor in expected.items():
        assert torch.equal(model._params[canonical_key].data, tensor), canonical_key
    # The DiT tensor never landed anywhere.
    assert not any(torch.equal(p.data, torch.full((4, 4), -1.0)) for p in model._params.values())


@pytest.mark.L0
@pytest.mark.CPU
def test_load_vlm_model_indexed_unexpected_key_raises(tmp_path):
    """A root-index key that remaps to a key absent from the model must fail
    loudly (no silent extra-key tolerance on the indexed branch)."""
    snapshot = _write_indexed_snapshot(
        tmp_path,
        {
            "transformer/part-00001.safetensors": {
                "embed_tokens.weight": torch.ones(4, 4),
                # remaps to model.language_model.layers.3.mixer.up_proj.weight — not in the model
                "layers.1.mlp.up_proj.weight": torch.ones(4, 4),
            }
        },
    )
    model = _edge_stub({"model.language_model.embeddings.weight": (4, 4)})
    with pytest.raises(ValueError, match="do not exist in the model state dict"):
        load_vlm_model(model=model, checkpoint_path=str(snapshot), credential_path=None, parallel_dims=None)


@pytest.mark.L0
@pytest.mark.CPU
def test_load_vlm_model_indexed_missing_model_key_raises(tmp_path):
    """A model key not covered by the root index trips the Phase-6 completeness
    check (nothing about the indexed branch weakens it)."""
    snapshot = _write_indexed_snapshot(
        tmp_path,
        {"transformer/part-00001.safetensors": {"embed_tokens.weight": torch.ones(4, 4)}},
    )
    model = _edge_stub(
        {
            "model.language_model.embeddings.weight": (4, 4),
            "model.language_model.norm_f.weight": (4,),  # not in the index
        }
    )
    with pytest.raises(ValueError, match="required model parameter"):
        load_vlm_model(model=model, checkpoint_path=str(snapshot), credential_path=None, parallel_dims=None)


# --- integration: real snapshot → fake canonical-shaped model -----------------


@pytest.mark.L0
@pytest.mark.CPU
def test_load_real_edge_snapshot_into_canonical_state_dict():
    """Load the real snapshot through the indexed branch into a dict-backed
    model whose keys/shapes mirror the canonical converter output; all 670
    tensors must land, three of them verified bitwise.

    Non-persistent buffers (e.g. rotary inv_freq) never appear in a state
    dict, so the canonical key list already excludes them by construction.
    """
    snapshot = _find_local_edge_snapshot()
    if snapshot is None:
        pytest.skip("no local nvidia/Cosmos3-Edge indexed snapshot in the HF cache")
    if not os.path.exists(_CANONICAL_VLM_FILE):
        pytest.skip(f"canonical reference not available: {_CANONICAL_VLM_FILE}")

    with safe_open(str(_CANONICAL_VLM_FILE), framework="pt") as f:
        canonical_keys = sorted(f.keys())
        shapes = {k: tuple(f.get_slice(k).get_shape()) for k in canonical_keys}

    config = _StubConfig(
        text_config=_StubConfig(num_experts=None, num_local_experts=None),
        tie_word_embeddings=False,
    )
    # bfloat16 params: halves memory AND makes the bitwise spot checks exact
    # (checkpoint tensors are bf16; copy_ into bf16 preserves bits).
    model = _StubModel(param_shapes={}, config=config)
    model._params = {
        k: torch.nn.Parameter(torch.zeros(shapes[k], dtype=torch.bfloat16), requires_grad=False) for k in canonical_keys
    }

    keys_loaded = load_vlm_model(
        model=model,
        checkpoint_path=snapshot,
        credential_path=None,
        parallel_dims=None,
    )

    assert keys_loaded == set(canonical_keys)
    assert len(keys_loaded) == 670

    # Bitwise spot checks against the canonical converter output: one remapped
    # attention key, one flat key, one pass-through visual key.
    spot_keys = [
        "model.language_model.layers.0.mixer.q_proj.weight",
        "model.language_model.embeddings.weight",
        "model.visual.embeddings.patch_embedding.weight",
    ]
    with safe_open(str(_CANONICAL_VLM_FILE), framework="pt") as f:
        for key in spot_keys:
            reference = f.get_tensor(key)
            loaded = model._params[key].data
            assert loaded.dtype == reference.dtype, key
            assert torch.equal(loaded, reference), key
