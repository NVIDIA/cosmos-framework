# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Tests for the export_model pure helpers (Edge detection, reasoner-vision capability,
include_visual writing, vision_encoder bundle config merging, processor bundling and its
failure cleanup, stale-artifact cleanup, manifest building/sanitization, the --verify
constant-image heuristic, ViT prefix selection, parallelism reset, completion-marker removal,
LiDAR VAE bundling, per-view caption layout of the Auto recipes) and the Edge checkpoint
registry entry. GPU/download paths are exercised by the e2e validation, not here."""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cosmos_framework.inference.common.checkpoints import register_checkpoints
from cosmos_framework.inference.common.public_model_config import (
    build_public_model_config,
    restore_model_config_from_public_model_config,
)
from cosmos_framework.scripts import _export_model_helpers as helpers
from cosmos_framework.utils.checkpoint_db import CheckpointConfig, sanitize_uri

_EDGE_BACKBONE_URI = "s3://bucket0/cosmos3/pretrained/huggingface/nvidia/Cosmos3-Edge-Reasoner-590c1c0/"

_VISION_CONFIG = {"hidden_size": 1152, "num_hidden_layers": 27, "patch_size": 16}
_PROJECTOR_CONFIG_TOP = {
    "input_hidden_size": 1152,
    "merger_intermediate_size": 11520,
    "out_hidden_size": 2048,
    "spatial_merge_size": 2,
}
_PROJECTOR_CONFIG_STANDALONE = {
    "input_hidden_size": 1152,
    "merger_intermedia": 11520,
    "out_hidden_size": 2048,
    "spatial_merge_size": 2,
}
_TOKEN_IDS = {"image_token_id": 19, "video_token_id": 18, "vision_start_token_id": 20}


def test_local_training_tokenizer_export_round_trip(tmp_path: Path) -> None:
    from transformers import Qwen2Tokenizer

    from cosmos_framework.inference.inference import _bundled_processor_serves_node, _point_tokenizer_node_at_dir
    from cosmos_framework.utils.lazy_config import instantiate

    source = tmp_path / "training" / "resources" / "tokenizer"
    source.mkdir(parents=True)
    # Real tokenizer files, including an added token whose ID must survive.
    (source / "vocab.json").write_text(json.dumps({"h": 0, "i": 1, "<|endoftext|>": 2}))
    (source / "merges.txt").write_text("#version: 0.2\n")
    tokenizer = Qwen2Tokenizer(str(source / "vocab.json"), str(source / "merges.txt"))
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|vision_start|>"]})
    tokenizer.save_pretrained(source)
    expected = tokenizer.encode("hi<|vision_start|>")
    model = {
        "config": {
            "vlm_config": {
                "tokenizer": {
                    "_target_": "cosmos_framework.data.generator.local_simulation.dataset.load_bundle_tokenizer",
                    "directory": str(source),
                }
            }
        }
    }
    exported = tmp_path / "exported"
    exported.mkdir()
    assert helpers.prepare_local_bundle_tokenizer(model, exported, config_only=False)
    public = build_public_model_config(model)
    public["config"]["vlm_config"]["tokenizer"]["pretrained_model_name"] = "."
    assert str(source) not in json.dumps(public)
    assert "local_simulation" not in json.dumps(public)
    # Remove access to the original location and move the exported artifact.
    source.rename(source.with_name("unavailable"))
    relocated = tmp_path / "relocated"
    exported.rename(relocated)
    restored = restore_model_config_from_public_model_config(public)
    node = restored["config"]["vlm_config"]["tokenizer"]
    assert _bundled_processor_serves_node(node, relocated)
    assert _point_tokenizer_node_at_dir(node, relocated)
    reloaded = instantiate(node)
    assert reloaded.tokenizer.encode("hi<|vision_start|>") == expected


def test_local_training_tokenizer_export_requires_files(tmp_path: Path) -> None:
    model = {
        "config": {
            "vlm_config": {
                "tokenizer": {
                    "_target_": "cosmos_framework.data.generator.local_simulation.dataset.load_bundle_tokenizer",
                    "directory": str(tmp_path / "missing"),
                }
            }
        }
    }
    with pytest.raises(FileNotFoundError, match="Local training tokenizer is missing"):
        helpers.prepare_local_bundle_tokenizer(model, tmp_path, config_only=False)
    assert helpers.prepare_local_bundle_tokenizer(model, tmp_path, config_only=True)
    assert not helpers.prepare_local_bundle_tokenizer(model, tmp_path, config_only=False)


def _model_dict(*, model_name="nvidia/Cosmos3-Edge-Reasoner", target="X.Nemotron3DenseVLTextForCausalLM"):
    return {
        "_target_": "X.OmniMoTModel",
        "config": {
            "vlm_config": {
                "model_name": model_name,
                "model_instance": {"_target_": target, "config": {"_target_": "X.create_vlm_config"}},
                "pretrained_weights": {"backbone_path": _EDGE_BACKBONE_URI},
            },
        },
    }


def _make_snapshot(tmp_path, *, standalone_config=None, top_config=None):
    """Build a fake nvidia/Cosmos3-Edge snapshot under an HF-cache-style path."""
    snapshot = tmp_path / "snapshots" / ("0" * 40)
    vision_dir = snapshot / "vision_encoder"
    vision_dir.mkdir(parents=True)
    (vision_dir / "model.safetensors").write_bytes(b"tower-weights")
    if standalone_config is not None:
        (vision_dir / "config.json").write_text(json.dumps(standalone_config))
    if top_config is not None:
        (snapshot / "config.json").write_text(json.dumps(top_config))
    return snapshot


def test_edge_reasoner_uri_resolves_in_registry():
    register_checkpoints()
    checkpoint = CheckpointConfig.maybe_from_uri(sanitize_uri(_EDGE_BACKBONE_URI))
    assert checkpoint is not None
    assert checkpoint.hf.repository == "nvidia/Cosmos3-Edge"
    # Deliberately unfiltered: training's backbone seeding ('load_language_model'
    # -> 'download_checkpoint') downloads the full repo through this entry (the
    # root weight index points into transformer/*.safetensors). Export narrows
    # its ViT-bundle download at the call site instead.
    assert checkpoint.hf.include == ()
    assert "vision_encoder/*" in helpers.EDGE_VIT_BUNDLE_HF_INCLUDE
    assert set(helpers.PROCESSOR_HF_INCLUDE) <= set(helpers.EDGE_VIT_BUNDLE_HF_INCLUDE)


class TestIsEdgeModel:
    def test_by_model_name(self):
        assert helpers.is_edge_model(_model_dict())

    def test_by_target_fallback(self):
        model_dict = _model_dict(model_name="something-else")
        model_dict["config"]["vlm_config"]["pretrained_weights"]["backbone_path"] = ""
        assert helpers.is_edge_model(model_dict)

    def test_by_backbone_uri_fallback(self):
        model_dict = _model_dict(model_name="something-else", target="X.Qwen3VLTextForCausalLM")
        assert helpers.is_edge_model(model_dict)

    def test_qwen_is_not_edge(self):
        model_dict = _model_dict(model_name="nvidia/Cosmos3-Nano-Reasoner", target="X.Qwen3VLTextForCausalLM")
        model_dict["config"]["vlm_config"]["pretrained_weights"]["backbone_path"] = (
            "s3://bucket0/cosmos3/pretrained/huggingface/Cosmos-Reason/Cosmos3-Nano-Reasoner-bb9c6f5/"
        )
        assert not helpers.is_edge_model(model_dict)


class TestReasonerVisionCapable:
    """Gates the --verify reasoner-image check; mirrors 'inference.args._reasoner_vision_capable'."""

    @staticmethod
    def _qwen_dict(include_visual=None):
        model_dict = _model_dict(model_name="nvidia/Cosmos3-Nano-Reasoner", target="X.Qwen3VLTextForCausalLM")
        model_dict["config"]["vlm_config"]["pretrained_weights"]["backbone_path"] = ""
        if include_visual is not None:
            model_dict["config"]["vlm_config"]["model_instance"]["config"]["include_visual"] = include_visual
        return model_dict

    def test_edge_is_always_capable(self):
        # Lazy SigLIP2 tower: capable even without in-checkpoint vision weights.
        assert helpers.reasoner_vision_capable(_model_dict())

    def test_edge_capable_even_with_include_visual_false(self):
        model_dict = _model_dict()
        assert helpers.set_include_visual(model_dict, False)
        assert helpers.reasoner_vision_capable(model_dict)

    def test_qwen_include_visual_absent_is_not_capable(self):
        # The shipped Nano/Super SFT configs leave include_visual unset.
        assert not helpers.reasoner_vision_capable(self._qwen_dict())

    def test_qwen_include_visual_false_is_not_capable(self):
        assert not helpers.reasoner_vision_capable(self._qwen_dict(include_visual=False))

    def test_qwen_include_visual_true_is_capable(self):
        assert helpers.reasoner_vision_capable(self._qwen_dict(include_visual=True))

    def test_unknown_family_is_capable(self):
        # Never block a family this heuristic doesn't know.
        model_dict = _model_dict(model_name="acme/some-model", target="X.SomeOtherForCausalLM")
        model_dict["config"]["vlm_config"]["pretrained_weights"]["backbone_path"] = ""
        assert helpers.reasoner_vision_capable(model_dict)


class TestSetIncludeVisual:
    def test_writes_into_create_vlm_config_node(self):
        model_dict = _model_dict()
        assert helpers.set_include_visual(model_dict, False)
        assert model_dict["config"]["vlm_config"]["model_instance"]["config"]["include_visual"] is False

    def test_no_model_instance(self):
        model_dict = _model_dict()
        model_dict["config"]["vlm_config"]["model_instance"] = None
        assert not helpers.set_include_visual(model_dict, False)


class TestVisionEncoderBundleConfig:
    def test_merged_top_level_only(self, tmp_path):
        # Like nvidia/Cosmos3-Edge@5bb63b97: spec folded into the top-level config.
        snapshot = _make_snapshot(
            tmp_path,
            top_config={"vision_config": _VISION_CONFIG, "projector_config": _PROJECTOR_CONFIG_TOP, **_TOKEN_IDS},
        )
        vision_dir, root = helpers.resolve_vision_bundle_dirs(snapshot)
        assert root == snapshot
        merged = helpers.build_vision_encoder_bundle_config(vision_dir, root)
        assert merged == {
            "vision_config": _VISION_CONFIG,
            "projector_config": _PROJECTOR_CONFIG_TOP,
            **_TOKEN_IDS,
        }

    def test_standalone_first_with_token_ids_from_top(self, tmp_path):
        # Like nvidia/Cosmos3-Edge@28a0b8e: standalone vision_encoder/config.json
        # (no token ids) wins for the specs; ids are folded in from the top level.
        snapshot = _make_snapshot(
            tmp_path,
            standalone_config={"vision_config": _VISION_CONFIG, "projector_config": _PROJECTOR_CONFIG_STANDALONE},
            top_config={"vision_config": {"other": 1}, "projector_config": _PROJECTOR_CONFIG_TOP, **_TOKEN_IDS},
        )
        vision_dir, root = helpers.resolve_vision_bundle_dirs(snapshot)
        merged = helpers.build_vision_encoder_bundle_config(vision_dir, root)
        assert merged["projector_config"] == _PROJECTOR_CONFIG_STANDALONE
        assert merged["vision_config"] == _VISION_CONFIG
        assert {k: merged[k] for k in _TOKEN_IDS} == _TOKEN_IDS

    def test_direct_vision_encoder_dir_uses_parent_root(self, tmp_path):
        # --vit-checkpoint-path pointed straight at <snapshot>/vision_encoder.
        snapshot = _make_snapshot(
            tmp_path,
            top_config={"vision_config": _VISION_CONFIG, "projector_config": _PROJECTOR_CONFIG_TOP, **_TOKEN_IDS},
        )
        vision_dir, root = helpers.resolve_vision_bundle_dirs(snapshot / "vision_encoder")
        assert vision_dir == snapshot / "vision_encoder"
        assert root == snapshot
        merged = helpers.build_vision_encoder_bundle_config(vision_dir, root)
        assert {k: merged[k] for k in _TOKEN_IDS} == _TOKEN_IDS

    def test_missing_token_ids_is_actionable(self, tmp_path):
        snapshot = _make_snapshot(
            tmp_path,
            standalone_config={"vision_config": _VISION_CONFIG, "projector_config": _PROJECTOR_CONFIG_STANDALONE},
        )
        vision_dir, root = helpers.resolve_vision_bundle_dirs(snapshot)
        with pytest.raises(ValueError, match="image_token_id.*--vit-checkpoint-path"):
            helpers.build_vision_encoder_bundle_config(vision_dir, root)

    def test_no_weights_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="vision_encoder/model.safetensors"):
            helpers.resolve_vision_bundle_dirs(tmp_path)


def test_bundle_vision_encoder_writes_verbatim_weights_and_merged_config(tmp_path):
    snapshot = _make_snapshot(
        tmp_path / "src",
        top_config={"vision_config": _VISION_CONFIG, "projector_config": _PROJECTOR_CONFIG_TOP, **_TOKEN_IDS},
    )
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    vision_dir, root = helpers.resolve_vision_bundle_dirs(snapshot)
    helpers.bundle_vision_encoder(vision_dir, root, output_dir)
    assert (output_dir / "vision_encoder" / "model.safetensors").read_bytes() == b"tower-weights"
    written = json.loads((output_dir / "vision_encoder" / "config.json").read_text())
    assert set(written) == {
        "vision_config",
        "projector_config",
        "image_token_id",
        "video_token_id",
        "vision_start_token_id",
    }


def test_bundle_processor_files_skips_config_json(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    for name in [
        "config.json",  # must never be copied over the exported config.json
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "video_preprocessor_config.json",
        "model.safetensors.index.json",  # weights index: not a processor file
    ]:
        (snapshot / name).write_text("{}")
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    copied = helpers.bundle_processor_files(snapshot, output_dir)
    assert sorted(copied) == [
        "chat_template.jinja",
        "preprocessor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "video_preprocessor_config.json",
    ]
    assert not (output_dir / "config.json").exists()
    assert not (output_dir / "model.safetensors.index.json").exists()


def _flaky_copyfile(monkeypatch, *, fail_on_call, exception):
    """Patch shutil.copyfile to raise on the Nth call (quota/IO mid-copy failure)."""
    real_copyfile = shutil.copyfile
    calls = {"count": 0}

    def flaky(src, dst, **kwargs):
        calls["count"] += 1
        if calls["count"] == fail_on_call:
            raise exception
        return real_copyfile(src, dst, **kwargs)

    monkeypatch.setattr(shutil, "copyfile", flaky)


def test_bundle_processor_files_removes_partial_set_on_midcopy_failure(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    for name in ["preprocessor_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        (snapshot / name).write_text("{}")
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    _flaky_copyfile(monkeypatch, fail_on_call=3, exception=OSError("disk quota exceeded"))
    with pytest.raises(OSError, match="disk quota exceeded"):
        helpers.bundle_processor_files(snapshot, output_dir)
    # A partial set would flip inference's local-first preference onto a broken
    # processor: the export root must be left with zero processor files.
    assert list(output_dir.iterdir()) == []


class TestProcessorSourceFromTokenizerNode:
    def test_repository_mode(self):
        # Edge-style local-artifact node (edge_model_config.py).
        node = {"_target_": "X.build_processor_lazy", "repository": "nvidia/Cosmos3-Edge", "revision": "main"}
        assert helpers.processor_source_from_tokenizer_node(node) == {
            "repository": "nvidia/Cosmos3-Edge",
            "revision": "main",
            "subdir": "",
        }

    def test_repository_mode_respects_subdir(self):
        node = {"repository": "nvidia/Cosmos3-Edge", "revision": "abc", "subdir": "processor"}
        source = helpers.processor_source_from_tokenizer_node(node)
        assert source == {"repository": "nvidia/Cosmos3-Edge", "revision": "abc", "subdir": "processor"}

    def test_tokenizer_type_mode(self):
        # Qwen-family build_processor_lazy node (configs.base.defaults.reasoner).
        node = {
            "_target_": "X.build_processor_lazy",
            "tokenizer_type": "Qwen/Qwen3-VL-8B-Instruct",
            "config_variant": "gcp",
        }
        assert helpers.processor_source_from_tokenizer_node(node) == {
            "repository": "Qwen/Qwen3-VL-8B-Instruct",
            "revision": None,
            "subdir": "",
        }

    def test_pretrained_model_name_mode(self):
        # Legacy create_qwen2_tokenizer_with_download node.
        node = {"pretrained_model_name": "Qwen/Qwen3-VL-8B-Instruct", "config_variant": "gcp"}
        source = helpers.processor_source_from_tokenizer_node(node)
        assert source is not None
        assert source["repository"] == "Qwen/Qwen3-VL-8B-Instruct"
        assert source["revision"] is None

    @pytest.mark.parametrize(
        "node",
        [
            None,
            "not-a-dict",
            {},
            {"tokenizer_type": "/local/snapshot/dir"},  # local dir, not a repo id
            {"tokenizer_type": "s3://bucket/some/path"},  # URI, not a repo id
            {"tokenizer_type": "single-token"},  # no org/name shape
        ],
    )
    def test_unusable_nodes_return_none(self, node):
        assert helpers.processor_source_from_tokenizer_node(node) is None


class TestResolveProcessorDownload:
    def test_plain_repo_id_pinned_via_registry(self):
        register_checkpoints()
        spec = helpers.resolve_processor_download(
            {"repository": "Qwen/Qwen3-VL-8B-Instruct", "revision": None, "subdir": ""}
        )
        assert spec.repository == "Qwen/Qwen3-VL-8B-Instruct"
        assert re.fullmatch(r"[0-9a-f]{40}", spec.revision)  # registry pin, not 'main'
        assert spec.include == helpers.PROCESSOR_HF_INCLUDE

    def test_unregistered_repo_falls_back_to_main(self):
        register_checkpoints()
        spec = helpers.resolve_processor_download({"repository": "some-org/some-repo", "revision": None, "subdir": ""})
        assert spec.repository == "some-org/some-repo"
        assert spec.revision == "main"

    def test_explicit_revision_used_as_is(self):
        spec = helpers.resolve_processor_download(
            {"repository": "nvidia/Cosmos3-Edge", "revision": "deadbeef", "subdir": "processor"}
        )
        assert spec.repository == "nvidia/Cosmos3-Edge"
        assert spec.revision == "deadbeef"
        assert spec.subdirectory == "processor"


class TestBundleProcessorFromTokenizerNode:
    _PROCESSOR_FILES = [
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "merges.txt",
        "vocab.json",
    ]

    def _make_processor_snapshot(self, tmp_path):
        snapshot = tmp_path / "snapshots" / ("b" * 40)
        snapshot.mkdir(parents=True)
        for name in [*self._PROCESSOR_FILES, "config.json"]:
            (snapshot / name).write_text("{}")
        return snapshot

    def test_bundles_from_mocked_snapshot(self, tmp_path):
        register_checkpoints()
        snapshot = self._make_processor_snapshot(tmp_path)
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        (output_dir / "config.json").write_text('{"model_type": "cosmos3_omni"}')  # the exported config
        seen = {}

        def fake_download(spec):
            seen["spec"] = spec
            return str(snapshot)

        node = {"tokenizer_type": "Qwen/Qwen3-VL-8B-Instruct", "config_variant": "gcp"}
        source = helpers.bundle_processor_from_tokenizer_node(node, output_dir, download=fake_download)
        assert source == {"repo": "Qwen/Qwen3-VL-8B-Instruct", "revision": "b" * 40, "bundled": True}
        assert seen["spec"].repository == "Qwen/Qwen3-VL-8B-Instruct"
        for name in self._PROCESSOR_FILES:
            assert (output_dir / name).is_file()
        # The exported config.json is never overwritten by bundling.
        assert json.loads((output_dir / "config.json").read_text()) == {"model_type": "cosmos3_omni"}

    def test_bundles_edge_node_without_vit_snapshot(self, tmp_path):
        # The --no-vit path: no tower snapshot was resolved, so the processor
        # comes from the tokenizer node's explicit repository/revision.
        snapshot = self._make_processor_snapshot(tmp_path)
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        seen = {}

        def fake_download(spec):
            seen["spec"] = spec
            return str(snapshot)

        node = {"repository": "nvidia/Cosmos3-Edge", "revision": "main"}
        source = helpers.bundle_processor_from_tokenizer_node(node, output_dir, download=fake_download)
        assert seen["spec"].repository == "nvidia/Cosmos3-Edge"
        assert seen["spec"].revision == "main"  # explicit node revision, no registry pin
        assert source == {"repo": "nvidia/Cosmos3-Edge", "revision": "b" * 40, "bundled": True}
        for name in self._PROCESSOR_FILES:
            assert (output_dir / name).is_file()

    def test_download_failure_is_best_effort(self, tmp_path):
        register_checkpoints()
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        def failing_download(spec):
            raise RuntimeError("offline")

        node = {"tokenizer_type": "Qwen/Qwen3-VL-8B-Instruct", "config_variant": "gcp"}
        source = helpers.bundle_processor_from_tokenizer_node(node, output_dir, download=failing_download)
        assert source is None  # warning + continue; never an exception
        assert list(output_dir.iterdir()) == []

    def test_unusable_node_is_best_effort(self, tmp_path):
        assert helpers.bundle_processor_from_tokenizer_node(None, tmp_path) is None
        assert helpers.bundle_processor_from_tokenizer_node({"tokenizer_type": "/local/dir"}, tmp_path) is None

    def test_snapshot_without_processor_files_returns_none(self, tmp_path):
        snapshot = tmp_path / "snapshots" / ("c" * 40)
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}")  # weights-repo config only
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        node = {"repository": "some-org/some-repo", "revision": "main"}
        source = helpers.bundle_processor_from_tokenizer_node(node, output_dir, download=lambda spec: str(snapshot))
        assert source is None
        assert list(output_dir.iterdir()) == []

    def test_midcopy_failure_is_best_effort_and_leaves_no_partial_set(self, tmp_path, monkeypatch):
        snapshot = self._make_processor_snapshot(tmp_path)
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        _flaky_copyfile(monkeypatch, fail_on_call=3, exception=OSError("disk quota exceeded"))
        node = {"repository": "some-org/some-repo", "revision": "main"}
        source = helpers.bundle_processor_from_tokenizer_node(node, output_dir, download=lambda spec: str(snapshot))
        assert source is None  # warning + continue; never an exception
        assert list(output_dir.iterdir()) == []

    def test_keyboard_interrupt_propagates(self, tmp_path, monkeypatch):
        # Only 'Exception' is caught (best-effort); BaseExceptions must escape.
        snapshot = self._make_processor_snapshot(tmp_path)
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        _flaky_copyfile(monkeypatch, fail_on_call=1, exception=KeyboardInterrupt())
        node = {"repository": "some-org/some-repo", "revision": "main"}
        with pytest.raises(KeyboardInterrupt):
            helpers.bundle_processor_from_tokenizer_node(node, output_dir, download=lambda spec: str(snapshot))


class TestCleanStaleExportArtifacts:
    def test_removes_only_managed_paths(self, tmp_path):
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        # Managed leftovers of a prior export.
        vision_dir = output_dir / "vision_encoder"
        vision_dir.mkdir()
        (vision_dir / "model.safetensors").write_bytes(b"stale-tower")
        (vision_dir / "config.json").write_text("{}")
        stale = ["preprocessor_config.json", "tokenizer.json", "chat_template.jinja", "export_manifest.json"]
        for name in stale:
            (output_dir / name).write_text("{}")
        # Never touched: the exported config/metadata, model shards
        # (save_pretrained manages those), and user files.
        kept = ["config.json", "checkpoint.json", "model.safetensors", "model.safetensors.index.json", "notes.txt"]
        for name in kept:
            (output_dir / name).write_text("keep")

        removed = helpers.clean_stale_export_artifacts(output_dir)

        assert sorted(removed) == sorted(["vision_encoder/", *stale])
        assert not vision_dir.exists()
        for name in stale:
            assert not (output_dir / name).exists()
        for name in kept:
            assert (output_dir / name).read_text() == "keep"

    def test_fresh_dir_is_a_noop(self, tmp_path):
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        assert helpers.clean_stale_export_artifacts(output_dir) == []
        assert list(output_dir.iterdir()) == []


class TestSanitizeExportArgs:
    def test_redacts_local_paths_and_keeps_flags(self):
        export_args = {
            "config_only": False,
            "student_only_checkpoint_metadata": True,
            "verify": True,
            "vit": True,
            "vit_checkpoint_path": "/data/users/someone/hf_snapshot",
        }
        sanitized = helpers.sanitize_export_args(export_args)
        assert sanitized == {
            "config_only": False,
            "student_only_checkpoint_metadata": True,
            "verify": True,
            "vit": True,
            "vit_checkpoint_path": None,  # redacted, key kept (stable schema)
        }
        assert json.dumps(sanitized)  # JSON-serializable

    def test_none_path_and_enum_strings_pass_through(self):
        assert helpers.sanitize_export_args({"vit_checkpoint_path": None}) == {"vit_checkpoint_path": None}
        assert helpers.sanitize_export_args({"mode": "latency"}) == {"mode": "latency"}
        # Any string carrying a path separator is treated as a local path.
        assert helpers.sanitize_export_args({"note": "sub/dir"}) == {"note": None}
        # '_path'/'_dir'-keyed fields are redacted even when relative.
        assert helpers.sanitize_export_args({"cache_dir": "relative"}) == {"cache_dir": None}


class TestConstantImageFailure:
    """Heuristic --verify check for the NaN-latent decode class (constant frames)."""

    def test_all_black_fails(self):
        failure = helpers.constant_image_failure(np.zeros((64, 64, 3), dtype=np.uint8))
        assert failure is not None and "constant" in failure

    def test_uniform_gray_fails(self):
        assert helpers.constant_image_failure(np.full((64, 64, 3), 90, dtype=np.uint8)) is not None

    def test_nearly_constant_fails(self):
        # Sub-threshold ripple, e.g. JPEG rounding on a flat NaN-decode frame.
        rng = np.random.default_rng(0)
        pixels = np.full((64, 64, 3), 12.0) + rng.normal(0.0, 0.1, size=(64, 64, 3))
        assert helpers.constant_image_failure(pixels) is not None

    def test_empty_fails(self):
        assert helpers.constant_image_failure(np.zeros((0, 0, 3), dtype=np.uint8)) is not None

    def test_noisy_image_passes(self):
        rng = np.random.default_rng(0)
        pixels = rng.integers(0, 256, size=(64, 64, 3)).astype(np.uint8)
        assert helpers.constant_image_failure(pixels) is None


def test_hf_revision_from_snapshot_path():
    commit = "5bb63b97f461726dc3480a1ad872e275cd479c25"
    path = f"/cache/hub/models--nvidia--Cosmos3-Edge/snapshots/{commit}"
    assert helpers.hf_revision_from_snapshot_path(path) == commit
    assert helpers.hf_revision_from_snapshot_path(f"{path}/vision_encoder") == commit
    assert helpers.hf_revision_from_snapshot_path("/some/local/snapshot") is None


def test_build_export_manifest_schema():
    manifest = helpers.build_export_manifest(
        vision_tower_source=helpers.build_artifact_source(
            repo="nvidia/Cosmos3-Edge",
            resolved_path="/cache/hub/models--nvidia--Cosmos3-Edge/snapshots/" + "a" * 40,
            bundled=True,
        ),
        processor_source=None,
        export_args={"vit": True},
        framework_commit="deadbeef",
    )
    assert set(manifest) == {"vision_tower_source", "processor_source", "export_args", "framework_commit"}
    assert manifest["vision_tower_source"] == {"repo": "nvidia/Cosmos3-Edge", "revision": "a" * 40, "bundled": True}
    assert manifest["processor_source"] is None
    assert json.dumps(manifest)  # JSON-serializable


def test_read_framework_commit_reads_git_head():
    commit = helpers.read_framework_commit()
    assert commit is not None  # this test runs from a git worktree
    assert re.fullmatch(r"[0-9a-f]{40}", commit)


def test_read_framework_commit_supports_linked_worktree(tmp_path):
    common_git_dir = tmp_path / "main" / ".git"
    linked_git_dir = common_git_dir / "worktrees" / "linked"
    worktree = tmp_path / "linked"
    linked_git_dir.mkdir(parents=True)
    worktree.mkdir()
    (worktree / ".git").write_text(f"gitdir: {linked_git_dir}\n")
    (linked_git_dir / "HEAD").write_text("ref: refs/heads/test\n")
    (linked_git_dir / "commondir").write_text("../..\n")
    ref = common_git_dir / "refs" / "heads" / "test"
    ref.parent.mkdir(parents=True)
    ref.write_text("a" * 40 + "\n")

    assert helpers.read_framework_commit(worktree) == "a" * 40


# ---------------------------------------------------------------------------
# ViT checkpoint prefixes ('model.visual.' or 'visual.', never both)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prefix", ["model.visual.", "visual."])
def test_visual_checkpoint_selector_accepts_supported_prefixes(prefix):
    names = [f"{prefix}patch_embed.weight", f"{prefix}blocks.0.attn.qkv.weight", "lm_head.weight"]
    selected_prefix, selected_names = helpers.select_visual_checkpoint_tensors(names)
    assert selected_prefix == prefix
    assert selected_names == names[:2]


def test_visual_checkpoint_selector_rejects_mixed_prefixes():
    with pytest.raises(ValueError, match="mixes both supported key prefixes"):
        helpers.select_visual_checkpoint_tensors(["visual.patch_embed.weight", "model.visual.patch_embed.weight"])


def test_visual_checkpoint_selector_without_visual_tensors():
    assert helpers.select_visual_checkpoint_tensors(["lm_head.weight"]) == ("", [])


@pytest.mark.parametrize("source_prefix", ["model.visual.", "visual."])
def test_visual_checkpoint_rewrite_normalizes_supported_prefixes(source_prefix):
    state = {f"{source_prefix}patch_embed.weight": object()}
    assert list(helpers.rewrite_visual_fqns_for_vfm(state)) == ["model.net.language_model.visual.patch_embed.weight"]


def test_visual_checkpoint_rewrite_rejects_mixed_prefixes():
    with pytest.raises(ValueError, match="mixes both"):
        helpers.rewrite_visual_fqns_for_vfm({"visual.a": 0, "model.visual.a": 1})


# ---------------------------------------------------------------------------
# Exported-config fixes
# ---------------------------------------------------------------------------


def test_reset_export_parallelism_sets_degrees_to_one():
    model_dict = {
        "config": {
            "parallelism": {
                "data_parallel_shard_degree": 32,
                "data_parallel_replicate_degree": -1,
                "context_parallel_shard_degree": 32,
                "cfg_parallel_shard_degree": 2,
                "fsdp_cpu_offload": True,
            }
        }
    }
    assert helpers.reset_export_parallelism(model_dict) is True
    assert model_dict["config"]["parallelism"] == {
        "data_parallel_shard_degree": 1,
        "data_parallel_replicate_degree": 1,
        "context_parallel_shard_degree": 1,
        "cfg_parallel_shard_degree": 1,
        "fsdp_cpu_offload": True,  # non-degree settings are untouched
    }


def test_reset_export_parallelism_without_a_parallelism_section():
    model_dict = {"config": {}}
    assert helpers.reset_export_parallelism(model_dict) is False
    assert model_dict == {"config": {}}


def test_remove_export_completion_marker(tmp_path):
    (tmp_path / "checkpoint.json").write_text("{}")
    (tmp_path / "config.json").write_text("{}")
    assert helpers.remove_export_completion_marker(tmp_path) is True
    assert not (tmp_path / "checkpoint.json").exists()
    assert (tmp_path / "config.json").exists()
    assert helpers.remove_export_completion_marker(tmp_path) is False


def test_clean_stale_export_artifacts_removes_a_stale_lidar_vae(tmp_path):
    lidar_dir = tmp_path / "lidar_vae"
    lidar_dir.mkdir()
    (lidar_dir / "diffusion_pytorch_model.safetensors").write_bytes(b"stale-vae")
    # A bundle is only managed (and so only removed) when bundling is requested.
    assert helpers.clean_stale_export_artifacts(tmp_path) == []
    assert lidar_dir.exists()
    assert helpers.clean_stale_export_artifacts(tmp_path, include_lidar_vae=True) == ["lidar_vae/"]
    assert not lidar_dir.exists()


# ---------------------------------------------------------------------------
# LiDAR VAE bundling
# ---------------------------------------------------------------------------


def _lidar_tokenizer_node(vae_path):
    return {
        "_target_": "cosmos_framework.model.generator.tokenizers.lidar.lidar_tokenizer_v1.LidarTokenizerV1Interface",
        "vae_path": vae_path,
        "bucket_name": "",
        "object_store_credential_path_pretrained": None,
        "network_config": {"resolution": [128, 1808], "patch_size": [2, 2]},
        "spatial_compression_factor": 16,
    }


def _fake_lidar_vae_state():
    import torch

    return {
        "encoder.proj.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "latent_mean": torch.zeros(4),
        "latent_std": torch.ones(4),
    }


def test_bundled_lidar_vae_layout_matches_the_loader():
    # 'Cosmos3OmniModel.from_pretrained_dcp' binds exactly this file when present.
    assert helpers.LIDAR_VAE_BUNDLE_RELATIVE_PATH == "lidar_vae/diffusion_pytorch_model.safetensors"


def test_bundle_lidar_vae_copies_a_local_safetensors_verbatim(tmp_path):
    import safetensors.torch

    source = tmp_path / "src" / "lidar_vae_weights.safetensors"
    source.parent.mkdir()
    safetensors.torch.save_file(_fake_lidar_vae_state(), str(source))
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    node = _lidar_tokenizer_node(str(source))

    def must_not_load(path):
        raise AssertionError("a local safetensors VAE is copied, not re-serialized")

    assert helpers.bundle_lidar_vae(node, output_dir, load_payload=must_not_load) == str(source)

    bundled = output_dir / helpers.LIDAR_VAE_BUNDLE_RELATIVE_PATH
    assert bundled.read_bytes() == source.read_bytes()
    bundle_config = json.loads((output_dir / "lidar_vae" / "config.json").read_text())
    # The architecture travels; the training-time location does not.
    assert bundle_config["network_config"] == node["network_config"]
    assert bundle_config["spatial_compression_factor"] == 16
    assert not {"vae_path", "bucket_name", "object_store_credential_path_pretrained"} & set(bundle_config)


def test_bundle_lidar_vae_reserializes_a_torch_checkpoint_loadably(tmp_path):
    import torch

    from cosmos_framework.model.generator.tokenizers.lidar.checkpoint import (
        load_artifact,
        parse_lidar_checkpoint_stats,
        remap_checkpoint_state,
    )

    state = {
        f"network.{key}" if key.startswith("encoder") else key: value for key, value in _fake_lidar_vae_state().items()
    }
    payload = {"model": {**state, "min_range": 0.5, "max_range": 120.0, "note": "dropped"}, "iteration": 7}
    source = tmp_path / "lidar_vae.pt"
    torch.save(payload, source)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    helpers.bundle_lidar_vae(_lidar_tokenizer_node(str(source)), output_dir)

    loaded = load_artifact(str(output_dir / helpers.LIDAR_VAE_BUNDLE_RELATIVE_PATH))
    assert set(loaded) == {"network.encoder.proj.weight", "latent_mean", "latent_std", "min_range", "max_range"}
    torch.testing.assert_close(loaded["network.encoder.proj.weight"], state["network.encoder.proj.weight"])
    mean, std, min_range, max_range = parse_lidar_checkpoint_stats(loaded)
    torch.testing.assert_close(mean, torch.zeros(4))
    assert (min_range, max_range) == (0.5, 120.0)
    remapped, _ = remap_checkpoint_state(loaded, {"encoder.proj.weight"})
    assert set(remapped) == {"encoder.proj.weight"}


def test_bundle_lidar_vae_rejects_a_payload_without_latent_stats_and_cleans_up(tmp_path):
    import torch

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    node = _lidar_tokenizer_node("s3://example-bucket/lidar_vae.pt")
    with pytest.raises(ValueError, match="latent_mean"):
        helpers.bundle_lidar_vae(node, output_dir, load_payload=lambda path: {"weight": torch.zeros(1)})
    assert not (output_dir / "lidar_vae").exists()


def test_bundle_lidar_vae_resolves_an_env_interpolation(tmp_path, monkeypatch):
    import safetensors.torch

    source = tmp_path / "vae.safetensors"
    safetensors.torch.save_file(_fake_lidar_vae_state(), str(source))
    monkeypatch.setenv("EXPORT_TEST_LIDAR_VAE", str(source))
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    node = _lidar_tokenizer_node("${oc.env:EXPORT_TEST_LIDAR_VAE}")
    assert helpers.bundle_lidar_vae(node, output_dir) == str(source)
    assert (output_dir / helpers.LIDAR_VAE_BUNDLE_RELATIVE_PATH).read_bytes() == source.read_bytes()


def test_bundle_lidar_vae_requires_a_vae_path(tmp_path):
    with pytest.raises(ValueError, match="Omit --bundle-lidar-vae"):
        helpers.bundle_lidar_vae(_lidar_tokenizer_node(None), tmp_path)
    assert not (tmp_path / "lidar_vae").exists()


def test_bundle_lidar_vae_passes_the_resolved_remote_uri_to_the_loader(tmp_path):
    seen = []
    node = {**_lidar_tokenizer_node("pretrained/lidar_vae.pt"), "bucket_name": "example-bucket"}

    def load(path):
        seen.append(path)
        return _fake_lidar_vae_state()

    helpers.bundle_lidar_vae(node, tmp_path, load_payload=load)
    assert seen == ["s3://example-bucket/pretrained/lidar_vae.pt"]
    assert (tmp_path / helpers.LIDAR_VAE_BUNDLE_RELATIVE_PATH).is_file()


def test_bind_bundled_lidar_vae_rewrites_the_exported_config(tmp_path):
    hf_config_json = {
        "model": {
            "config": {
                "lidar_tokenizer": {
                    **_lidar_tokenizer_node("pretrained/tokenizers/lidar/diffusion_pytorch_model.safetensors"),
                    "bucket_name": "bucket",
                    "object_store_credential_path_pretrained": "credentials/s3.secret",
                }
            }
        }
    }
    # No bundle on disk: the config is left alone.
    assert helpers.bind_bundled_lidar_vae(hf_config_json, tmp_path) is False
    assert hf_config_json["model"]["config"]["lidar_tokenizer"]["bucket_name"] == "bucket"

    bundled = tmp_path / helpers.LIDAR_VAE_BUNDLE_RELATIVE_PATH
    bundled.parent.mkdir()
    bundled.write_bytes(b"vae")
    assert helpers.bind_bundled_lidar_vae(hf_config_json, tmp_path) is True
    node = hf_config_json["model"]["config"]["lidar_tokenizer"]
    assert node["vae_path"] == "lidar_vae/diffusion_pytorch_model.safetensors"
    assert node["bucket_name"] == ""
    assert node["object_store_credential_path_pretrained"] is None
    assert node["network_config"]  # the architecture is untouched


def test_bind_bundled_lidar_vae_ignores_models_without_lidar(tmp_path):
    bundled = tmp_path / helpers.LIDAR_VAE_BUNDLE_RELATIVE_PATH
    bundled.parent.mkdir()
    bundled.write_bytes(b"vae")
    hf_config_json = {"model": {"config": {"tokenizer": {"vae_path": "x"}}}}
    assert helpers.bind_bundled_lidar_vae(hf_config_json, tmp_path) is False
    assert hf_config_json == {"model": {"config": {"tokenizer": {"vae_path": "x"}}}}


# ---------------------------------------------------------------------------
# Per-view caption layout recorded for the Auto recipes
# ---------------------------------------------------------------------------


@pytest.fixture
def requires_torchcodec() -> None:
    """Skip when torchcodec cannot load; Auto recipes and the exporter import video augmentors."""
    try:
        import torchcodec.decoders  # noqa: F401
    except Exception as error:  # ImportError, or RuntimeError when FFmpeg's libraries do not load
        pytest.skip(f"torchcodec does not load here ({type(error).__name__})")


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.usefixtures("requires_torchcodec")
@pytest.mark.parametrize("device", [None, "cpu", "cuda"])
def test_export_initialization_preserves_device_selection(device: str | None) -> None:
    env = os.environ.copy()
    for key in ("COSMOS_DEVICE", "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE"):
        env.pop(key, None)
    if device is not None:
        env["COSMOS_DEVICE"] = device
    env["COSMOS_TRAINING"] = "0"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, os\n"
            "import cosmos_framework.scripts.export_model\n"
            "print(json.dumps({key: os.environ[key] for key in ('COSMOS_DEVICE', 'COSMOS_TRAINING')}))",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == {
        "COSMOS_DEVICE": device or "cpu",
        "COSMOS_TRAINING": "1",
    }


@pytest.mark.parametrize("phase, per_view", [(1, False), (2, True), (3, True), (4, True)])
def test_transfer_auto_recipe_caption_layout(phase: int, per_view: bool, requires_torchcodec: None) -> None:
    from cosmos_framework.configs.simulation.common.transfer_auto_recipes import make_transfer_auto_experiment
    from cosmos_framework.inference.common.multiview_captions import (
        JOINT_CAM_LIDAR_TASK,
        PER_VIEW_CAPTION_TASK_KEY,
        PER_VIEW_CAPTIONS_KEY,
        per_view_caption_export_fields,
    )

    fields, streams, _ = per_view_caption_export_fields(make_transfer_auto_experiment(phase))
    if per_view:
        assert fields == {PER_VIEW_CAPTIONS_KEY: True, PER_VIEW_CAPTION_TASK_KEY: JOINT_CAM_LIDAR_TASK}
        assert {path.split(".")[1] for path in streams} == {"joint_480", "joint_720"}
    else:
        assert fields == {PER_VIEW_CAPTIONS_KEY: False}


@pytest.mark.parametrize("stage", [1, 2, 3])
def test_sim_auto_recipe_caption_layout(stage):
    from cosmos_framework.configs.simulation.common.auto_profiles import make_auto_experiment
    from cosmos_framework.inference.common.multiview_captions import (
        JOINT_CAM_LIDAR_TASK,
        PER_VIEW_CAPTION_TASK_KEY,
        PER_VIEW_CAPTIONS_KEY,
        per_view_caption_export_fields,
    )

    fields, _, _ = per_view_caption_export_fields(make_auto_experiment(stage))
    assert fields == {PER_VIEW_CAPTIONS_KEY: True, PER_VIEW_CAPTION_TASK_KEY: JOINT_CAM_LIDAR_TASK}


def _per_view_training_config(*, per_view=True, joint=True):
    from types import SimpleNamespace

    dataset = {"separate_view_text_tokenization": per_view, "joint_cam_lidar_transfer": joint}
    return SimpleNamespace(dataloader_train={"dataloaders": {"mv": {"dataloader": {"dataset": dataset}}}})


class TestResolvePerViewCaptionFields:
    _JOINT = "av_joint_camera_lidar_transfer"

    def test_auto_derives_from_the_dataloaders(self):
        fields = helpers.resolve_per_view_caption_fields("auto", None, _per_view_training_config)
        assert fields == {"per_view_captions": True, "per_view_caption_task": self._JOINT}

    def test_auto_records_merged_captions_explicitly(self):
        fields = helpers.resolve_per_view_caption_fields(
            "auto", None, lambda: _per_view_training_config(per_view=False)
        )
        assert fields == {"per_view_captions": False}

    def test_auto_records_merged_without_dataloaders(self):
        # As i4: no derivable caption streams records the merged layout.
        from types import SimpleNamespace

        assert helpers.resolve_per_view_caption_fields("auto", None, lambda: SimpleNamespace(model={})) == {
            "per_view_captions": False
        }

    def test_auto_records_nothing_when_the_config_does_not_load(self):
        def fail():
            raise ValueError("not a training config")

        assert helpers.resolve_per_view_caption_fields("auto", None, fail) == {}

    def test_auto_with_an_explicit_task(self):
        fields = helpers.resolve_per_view_caption_fields(
            "auto", "av_multiview_transfer", lambda: _per_view_training_config(joint=False)
        )
        assert fields == {"per_view_captions": True, "per_view_caption_task": "av_multiview_transfer"}

    def test_auto_rejects_a_task_for_merged_captions(self):
        with pytest.raises(ValueError, match="do not tokenize per-view captions"):
            helpers.resolve_per_view_caption_fields(
                "auto", self._JOINT, lambda: _per_view_training_config(per_view=False)
            )

    def test_on_without_dataloaders(self):
        from types import SimpleNamespace

        assert helpers.resolve_per_view_caption_fields("on", None, lambda: SimpleNamespace(model={})) == {
            "per_view_captions": True
        }
        assert helpers.resolve_per_view_caption_fields("on", self._JOINT, lambda: SimpleNamespace(model={})) == {
            "per_view_captions": True,
            "per_view_caption_task": self._JOINT,
        }

    def test_on_fills_a_derivable_task(self):
        fields = helpers.resolve_per_view_caption_fields("on", None, _per_view_training_config)
        assert fields == {"per_view_captions": True, "per_view_caption_task": self._JOINT}

    def test_off_never_loads_the_config(self):
        def must_not_load():
            raise AssertionError("off does not need the training config")

        assert helpers.resolve_per_view_caption_fields("off", None, must_not_load) == {"per_view_captions": False}
        with pytest.raises(ValueError, match="cannot be combined"):
            helpers.resolve_per_view_caption_fields("off", self._JOINT, must_not_load)
