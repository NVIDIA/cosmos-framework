# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Lossless tensor streaming, stable sources and the real export transaction."""

import json
import os
from contextlib import ExitStack

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForImageTextToText

from cosmos_framework.posttrain.checkpoint.adapters import (
    prepare_merged_state,
    read_adapter_snapshot,
    save_adapter_artifact,
)
from cosmos_framework.posttrain.checkpoint.conversion import adapter_stream, hf
from cosmos_framework.posttrain.checkpoint.conversion.adapter_stream import AdapterTensorSource
from cosmos_framework.posttrain.checkpoint.tools import materialize_hf as mat
from cosmos_framework.posttrain.checkpoint.tools.adapters import (
    export_merged_adapter,
    merged_export_metadata,
    validate_export_targets,
)
from cosmos_framework.posttrain.config.adapters import AdapterExtraModuleConfig
from cosmos_framework.posttrain.config.model import Qwen3VLModelConfig
from cosmos_framework.posttrain.model.qwen3_vl import build_qwen3_vl_model
from cosmos_framework.posttrain.tests.test_adapters_native import adapter_config, tiny_snapshot


def case(tmp_path, method="mixed", *, tied=True, output_dtype="float32"):
    torch.manual_seed(41)
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    tiny_snapshot(base, tied=tied)
    config = adapter_config("lora" if method == "mixed" else method)
    if method == "mixed":
        config.components[0].use_rslora = True
        config.components[1].method = "dora"
        config.components[2].method = "pissa"
        config.components[2].execution = "pissa_dense_v1"
    config.extra_modules = [AdapterExtraModuleConfig(targets=["model.language_model.norm"])]
    model = build_qwen3_vl_model(Qwen3VLModelConfig(name_or_path=str(base), dtype="bfloat16"), adapters=config)
    with torch.no_grad():
        for p in model.net.parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p) * 0.01)
    save_adapter_artifact(model.net, str(adapter))
    record, state = read_adapter_snapshot(str(adapter))
    identity = record["identity"]
    expected = {n: t.bfloat16().float() for n, t in load_file(str(base / "model-00001-of-00001.safetensors")).items()}
    expected.update(state)
    expected = prepare_merged_state(expected, {**identity, "base_storage_dtype": "float32"})
    if tied:
        expected.pop("lm_head.weight", None)
    if output_dtype == "bfloat16":
        expected = {n: t.bfloat16() for n, t in expected.items()}
    source = AdapterTensorSource(
        str(base),
        str(adapter),
        record,
        state,
        validate_export_targets(str(base), identity),
        output_dtype,
        shard_bytes=16000,
    )
    return base, adapter, source, expected


def run(source, out):
    metadata = merged_export_metadata(source.adapter, source.identity, output_dtype=source.output_dtype)
    return mat.run_request(
        base=source.base,
        out=str(out),
        tensor_source=source,
        output_dtype=source.output_dtype,
        min_tensors=1,
        expect_tensors=len(source),
        export_metadata=metadata,
    )


def test_source_metadata_cannot_label_one_adapters_tensors_as_another(tmp_path):
    base, _, source, _ = case(tmp_path)
    metadata = merged_export_metadata(source.adapter, source.identity)
    metadata["adapter_weights_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="captured adapter tensor source identity"):
        mat.run_request(
            base=str(base),
            out=str(tmp_path / "bad"),
            tensor_source=source,
            output_dtype="float32",
            export_metadata=metadata,
            min_tensors=1,
        )
    assert not (tmp_path / "bad.lock").exists()
    assert not (tmp_path / "bad.tmp").exists()


@pytest.mark.parametrize("destination", ["base", "base/nested", "adapter", "."])
def test_output_cannot_replace_or_modify_its_sources(tmp_path, destination):
    _, _, source, _ = case(tmp_path)
    with pytest.raises(ValueError, match="must not overlap"):
        run(source, tmp_path / destination)


@pytest.mark.parametrize("method", ["mixed", "lora", "dora", "pissa"])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_stream_exactly_matches_full_state_math_and_shard_bound(tmp_path, method, dtype):
    base, adapter, source, expected = case(tmp_path, method, output_dtype=dtype)
    with ExitStack() as held:
        source.open(held)
        assert set(source) == set(expected)
        for name in source:
            assert torch.equal(source[name], expected[name]), name
        target = tmp_path / "shards"
        source.build(str(target))
    index = json.loads((target / hf.INDEX_FILE).read_text())["weight_map"]
    for shard in set(index.values()):
        state = load_file(str(target / shard))
        assert len(state) == 1 or sum(t.numel() * t.element_size() for t in state.values()) <= source.shard_bytes
        for name, tensor in state.items():
            assert torch.equal(tensor, expected[name]), name
    assert not list(tmp_path.rglob("model.pt"))


def test_public_export_reload_tying_and_exact_completed_reuse(tmp_path):
    base, adapter, source, expected = case(tmp_path)
    out = tmp_path / "merged"
    assert export_merged_adapter(str(adapter), str(base), str(out)) == "built"
    marker = (out / mat.MATERIALIZED_MARKER).read_bytes()
    assert export_merged_adapter(str(adapter), str(base), str(out)) == "skipped"
    assert (out / mat.MATERIALIZED_MARKER).read_bytes() == marker
    model = AutoModelForImageTextToText.from_pretrained(out, dtype="auto")
    assert model.lm_head.weight is model.model.language_model.embed_tokens.weight
    for name, tensor in expected.items():
        assert torch.equal(model.state_dict()[name], tensor), name
    index = json.loads((out / hf.INDEX_FILE).read_text())["weight_map"]
    name = "model.language_model.layers.0.mlp.up_proj.weight"
    file = out / index[name]
    values = load_file(str(file))
    values[name].flatten()[0] += 1
    save_file(values, str(file), metadata={"format": "pt"})
    with pytest.raises(hf.ConversionError, match="differ bitwise"):
        export_merged_adapter(str(adapter), str(base), str(out))
    assert (out / mat.MATERIALIZED_MARKER).read_bytes() == marker


def test_legacy_completed_export_reuse_keeps_identity_and_rejects_legacy_staging(tmp_path):
    base, adapter, source, expected = case(tmp_path)
    checkpoint, out = tmp_path / "checkpoint", tmp_path / "legacy"
    checkpoint.mkdir()
    torch.save({"net": expected}, checkpoint / "model.pt")
    assert (
        mat.run_request(
            base=str(base),
            out=str(out),
            ckpt_dir=str(checkpoint),
            min_tensors=1,
            output_dtype="float32",
            export_metadata=merged_export_metadata(str(adapter), source.identity),
        )
        == "built"
    )
    before = (out / mat.MATERIALIZED_MARKER).read_bytes()
    assert run(source, out) == "skipped"
    assert (out / mat.MATERIALIZED_MARKER).read_bytes() == before
    os.rename(out, str(out) + mat.STAGING_SUFFIX)
    with pytest.raises(mat.MaterializeRefusedError, match="only complete marked export"):
        run(source, out)
    assert (tmp_path / "legacy.tmp" / mat.MATERIALIZED_MARKER).read_bytes() == before


@pytest.mark.parametrize("stage", ["first_shard", "serialized", "marked"])
def test_interruption_recovery_does_not_repeat_complete_serialization(tmp_path, monkeypatch, stage):
    _, _, source, _ = case(tmp_path)
    out = tmp_path / "merged"
    with monkeypatch.context() as patch:
        if stage == "first_shard":
            original = adapter_stream.save_file

            def kill(*args, **kwargs):
                original(*args, **kwargs)
                raise KeyboardInterrupt("simulated process death after one shard")

            patch.setattr(adapter_stream, "save_file", kill)
        elif stage == "serialized":
            patch.setattr(mat, "sync_sidecars", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
        else:
            patch.setattr(mat, "publish_staging", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
        with pytest.raises(KeyboardInterrupt):
            run(source, out)
    assert not out.exists()
    assert (tmp_path / "merged.tmp").exists()
    if stage != "first_shard":
        monkeypatch.setattr(source, "build", lambda *a: pytest.fail("complete staged tensors must be reused"))
    assert run(source, out) == "built"
    assert run(source, out) == "skipped"


@pytest.mark.parametrize("damage", ["missing", "wrong_shape", "wrong_bytes", "tied_head"])
def test_incomplete_or_wrong_base_refused_before_publication(tmp_path, damage):
    base, adapter, _, _ = case(tmp_path)
    file = base / "model-00001-of-00001.safetensors"
    state = load_file(str(file))
    name = "model.language_model.layers.0.mlp.up_proj.weight"
    if damage == "missing":
        del state[name]
    elif damage == "wrong_shape":
        state[name] = state[name][:-1].clone()
    elif damage == "wrong_bytes":
        state[name].flatten()[0] += 1
    else:
        state["lm_head.weight"] = state["model.language_model.embed_tokens.weight"].clone() + 1
    save_file(state, str(file))
    index = json.loads((base / hf.INDEX_FILE).read_text())
    index["weight_map"] = {name: file.name for name in state}
    (base / hf.INDEX_FILE).write_text(json.dumps(index))
    with pytest.raises(ValueError, match="missing architecture|shape|wrong base identity"):
        export_merged_adapter(str(adapter), str(base), str(tmp_path / "merged"))
    assert not (tmp_path / "merged").exists()


@pytest.mark.parametrize("change", ["shard_write", "processor"])
def test_changed_source_during_export_cannot_publish(tmp_path, monkeypatch, change):
    base, _, source, _ = case(tmp_path)
    original = source.build

    def change_after_build(out):
        original(out)
        if change == "shard_write":
            file = base / "model-00001-of-00001.safetensors"
            with file.open("r+b") as handle:
                handle.seek(-4, os.SEEK_END)
                handle.write(b"xxxx")
        else:
            file = base / "tokenizer_config.json"
            file.write_text(file.read_text() + "\n")

    monkeypatch.setattr(source, "build", change_after_build)
    with pytest.raises(hf.ConversionError, match="changed during adapter export"):
        run(source, tmp_path / "merged")
    assert not (tmp_path / "merged").exists()
    assert not (tmp_path / "merged.tmp").exists()
