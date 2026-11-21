# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Small real tensor/portable/DCP/HF contracts; not accelerator qualification."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from cosmos_framework.posttrain.checkpoint import dense_adapter as dense
from cosmos_framework.posttrain.checkpoint.adapter import save_adapter_artifact
from cosmos_framework.posttrain.checkpoint.manager import CheckpointManager
from cosmos_framework.posttrain.model.checkpoint import build_adapter_checkpoint_codec
from cosmos_framework.posttrain.utils import adapter_source
from cosmos_framework.utils.generator.lora import LoraInjectedLinear

pytestmark = [pytest.mark.L0, pytest.mark.CPU]
BUILD = {"code_revision": "a" * 40, "wheel_sha256": "b" * 64}


@pytest.mark.parametrize("failure", [None, "serialization", "failure_diagnostic"])
def test_cli_records_export_interval_and_preserves_primary_failure(tmp_path, monkeypatch, failure):
    from safetensors import torch as safetensors_torch

    from cosmos_framework.posttrain.checkpoint.tools import export_adapter_dense as cli
    from cosmos_framework.posttrain.config import build as model_build
    from cosmos_framework.posttrain.data.packer import sft as packer_build
    from cosmos_framework.posttrain.entrypoints import sft
    from cosmos_framework.posttrain.utils import adapter_identity, determinism, dist, qlora_metrics
    from cosmos_framework.utils import config as config_utils

    net, source, checkpoint = fixture(tmp_path)
    output, journal = tmp_path / "dense", tmp_path / "memory.jsonl"
    rl = SimpleNamespace(seed=19, max_length=32, processor_kwargs={}, mask_think_opener=False)
    monkeypatch.setattr(config_utils, "load_config", lambda *_: object())
    monkeypatch.setattr(sft, "validate_config", lambda _: rl)
    monkeypatch.setattr(sft, "sft_policy_source", lambda _: (str(source), False))
    monkeypatch.setattr(packer_build, "build_sft_packer", lambda *_, **__: object())
    monkeypatch.setattr(model_build, "build_backbone", lambda _: SimpleNamespace(net=net))
    monkeypatch.setattr(
        adapter_identity,
        "adapter_processor_identity",
        lambda *_, **__: net._cosmos_adapter_checkpoint_codec.identity["processor_contract"],
    )
    monkeypatch.setattr(determinism, "enable_determinism", lambda **_: None)
    monkeypatch.setattr(dist, "init_local_process_group", lambda: None)
    events = []
    monkeypatch.setattr(dist, "destroy_process_group_if_initialized", lambda: events.append("cleanup"))
    for name, value in {
        "WORLD_SIZE": "1",
        "RANK": "0",
        "COSMOS_TRAINING_CODE_REVISION": BUILD["code_revision"],
        "COSMOS_TRAINING_WHEEL_SHA256": BUILD["wheel_sha256"],
        "COSMOS_QLORA_METRICS_FILE": str(journal),
    }.items():
        monkeypatch.setenv(name, value)
    original_record = qlora_metrics.record_qlora_phase

    def record(phase, *, reset_peak=False):
        events.append(phase)
        committed = (output / dense.DENSE_EXPORT_MANIFEST).exists()
        assert committed == (phase == "dense_export_complete")
        if phase == "dense_export_failed" and failure == "failure_diagnostic":
            raise OSError("diagnostic destination unavailable")
        return original_record(phase, reset_peak=reset_peak)

    monkeypatch.setattr(qlora_metrics, "record_qlora_phase", record)
    primary = ValueError("injected shard serialization failure")

    def failed_save(*_, **__):
        raise primary

    if failure:
        monkeypatch.setattr(safetensors_torch, "save_file", failed_save)
    argv = [
        "--config",
        "unused",
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--output-dtype",
        "bfloat16",
    ]
    if failure:
        with pytest.raises(ValueError) as error:
            cli.main(argv)
        assert error.value is primary
        assert not (output / dense.DENSE_EXPORT_MANIFEST).exists()
    else:
        cli.main(argv)
        assert dense.read_dense_export(output)["complete"] is True
    end = "dense_export_failed" if failure else "dense_export_complete"
    assert events == ["dense_export_start", end, "cleanup"]
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    assert [row["phase"] for row in rows] == (
        ["dense_export_start"] if failure == "failure_diagnostic" else ["dense_export_start", end]
    )
    assert [row["reset_peak_after_record"] for row in rows] == [True] + [False] * (len(rows) - 1)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = LoraInjectedLinear(
            nn.Linear(4, 3, dtype=torch.bfloat16),
            rank=2,
            alpha=4,
            adapter_dtype=torch.float32,
        )
        self.embedding = nn.Embedding(5, 4, dtype=torch.bfloat16)
        for name, parameter in self.named_parameters():
            parameter.requires_grad_("lora_" in name)
        self._cosmos_adapter_parameters = {
            f"linear.{name}": parameter for name, parameter in self.linear.adapter_parameters().items()
        }
        frozen = {name: p.detach().clone() for name, p in self.named_parameters() if not p.requires_grad}

        def validate():
            for name, parameter in self.named_parameters():
                if name in frozen:
                    assert torch.equal(parameter, frozen[name])

        self._cosmos_validate_frozen_base = validate

    def state_dict(self, *args, **kwargs):
        raise AssertionError("full-model state extraction is forbidden")


def fixture(tmp_path, *, dcp=False):
    torch.manual_seed(19)
    net = Tiny()
    source = tmp_path / "base"
    source.mkdir()
    config = {
        "model_type": "qwen3_vl",
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "text_config": {"num_hidden_layers": 36, "hidden_size": 4096, "intermediate_size": 12288},
    }
    for name, value in {
        "config.json": config,
        "tokenizer_config.json": {"chat_template": "exact template"},
        "preprocessor_config.json": {},
        "video_preprocessor_config.json": {},
        "tokenizer.json": {"test": "processor bytes"},
    }.items():
        (source / name).write_text(json.dumps(value))
    save_file(
        {name: p.detach() for name, p in net.named_parameters() if "lora_" not in name}, source / "model.safetensors"
    )
    identity = adapter_source._identity(source)
    (source / adapter_source.SOURCE_MANIFEST).write_text(json.dumps(identity))
    model_identity = {
        "format": "cosmos.adapter-model.v1",
        "model_format": "cosmos_native_qwen3_vl",
        "base": identity,
        "targets": {"linear": [3, 4]},
        "quantization": None,
    }
    codec = build_adapter_checkpoint_codec(net, identity=model_identity)
    codec.bind_processor_identity({"template": "exact template", "max_length": 32})
    net._cosmos_adapter_checkpoint_codec = codec
    if dcp:
        import torch.distributed.checkpoint as dcp_io
        from torch.distributed.checkpoint.state_dict import get_model_state_dict

        checkpoint = tmp_path / "checkpoint" / "iter_000002"
        checkpoint.mkdir(parents=True)
        dcp_io.save(get_model_state_dict(codec.view), checkpoint_id=checkpoint / "model")
        torch.save(codec.metadata(), checkpoint / "model_metadata.pt")
        torch.save({"optimizer_step": 2}, checkpoint / "trainer.pt")
    else:
        checkpoint = Path(
            save_adapter_artifact(
                str(tmp_path / "checkpoints" / "iter_000002"),
                codec.parameters,
                codec.metadata(),
                manager=CheckpointManager(checkpoint_format="consolidated"),
                publish=False,
            )
        )
    return net, source, checkpoint


def export(net, source, checkpoint, output, **kw):
    return dense.export_dense_adapter(
        net,
        source=str(source),
        checkpoint=str(checkpoint),
        output=str(output),
        output_dtype=kw.pop("output_dtype", "bfloat16"),
        exporter_build=kw.pop("exporter_build", BUILD),
        **kw,
    )


def test_output_cannot_enter_source_through_a_symlink_alias(tmp_path):
    net, source, checkpoint = fixture(tmp_path)
    alias = tmp_path / "alias"
    alias.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="separate from source"):
        export(net, source, checkpoint, alias / "dense")
    assert not (source / "dense").exists()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_merge_is_fp32_sum_then_output_cast_and_does_not_mutate(dtype):
    torch.manual_seed(7)
    base = torch.randn(513, 4).bfloat16()
    a = torch.randn(2, 4)
    b = torch.randn(513, 2)
    originals = [x.clone() for x in (base, a, b)]
    expected = (base.float() + 2.5 * (b @ a)).to(dtype)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = dense.merge_effective_weight(base, a, b, scale=2.5, dtype=dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for old, new in zip(originals, (base, a, b), strict=True):
        assert torch.equal(old, new)


@pytest.mark.parametrize("dcp", [False, True])
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_streaming_export_real_checkpoint_weights_sidecars_and_no_model_mutation(tmp_path, dcp, dtype):
    net, source, checkpoint = fixture(tmp_path, dcp=dcp)
    before = {n: p.clone() for n, p in net.named_parameters()}
    rng = torch.get_rng_state().clone()
    flags = [m.training for m in net.modules()]
    report = export(net, source, checkpoint, tmp_path / "dense", output_dtype=dtype)
    assert torch.equal(rng, torch.get_rng_state()) and flags == [m.training for m in net.modules()]
    assert all(torch.equal(before[n], p) for n, p in net.named_parameters())
    assert report["complete"] and not report["resumable_training_checkpoint"]
    assert report["source_metadata"] == net._cosmos_adapter_checkpoint_codec.metadata()
    assert set(report["tensors"]) == {"linear.weight", "linear.bias", "embedding.weight"}
    expected = (
        before["linear.weight"].float() + 2 * (before["linear.lora_B.weight"] @ before["linear.lora_A.weight"])
    ).to(getattr(torch, dtype))
    shard = tmp_path / "dense" / report["tensors"]["linear.weight"]["file"]
    torch.testing.assert_close(load_file(shard)["linear.weight"], expected, rtol=0, atol=0)
    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
    ]:
        assert (tmp_path / "dense" / name).read_bytes() == (source / name).read_bytes()
    assert json.loads((tmp_path / "dense/config.json").read_text())["torch_dtype"] == dtype
    assert dense.read_dense_export(tmp_path / "dense") == report
    with pytest.raises(FileExistsError):
        export(net, source, checkpoint, tmp_path / "dense", output_dtype=dtype)


@pytest.mark.parametrize(
    "limit,groups",
    [
        (70, [["embedding.weight", "linear.bias", "linear.weight"]]),
        (44, [["embedding.weight"], ["linear.bias", "linear.weight"]]),
        (24, [["embedding.weight"], ["linear.bias"], ["linear.weight"]]),
    ],
)
def test_size_bounded_shards_preserve_weights_and_use_standard_index(tmp_path, limit, groups):
    net, source, checkpoint = fixture(tmp_path)
    root = tmp_path / "dense"
    report = export(net, source, checkpoint, root, max_shard_size_bytes=limit)
    assert report["protocol"] == "cosmos.native-adapter-dense-export.v2"
    assert report["sharding"]["max_shard_size_bytes"] == limit
    index = json.loads((root / "model.safetensors.index.json").read_text())
    assert index["metadata"]["total_size"] == 70
    for i, names in enumerate(groups, start=1):
        filename = f"model-{i:05d}-of-{len(groups):05d}.safetensors"
        loaded = load_file(root / filename)
        assert set(loaded) == set(names)
        assert all(index["weight_map"][name] == filename for name in names)
        size = sum(t.numel() * t.element_size() for t in loaded.values())
        assert size <= limit or len(loaded) == 1
        for name, value in loaded.items():
            assert dense.tensor_digest(value) == {k: v for k, v in report["tensors"][name].items() if k != "file"}
    assert len(list(root.glob("*.safetensors"))) == len(groups)
    # Packaging changes file boundaries, never merge arithmetic or tensor bytes.
    reference = export(net, source, checkpoint, tmp_path / "reference", max_shard_size_bytes=1)
    assert {name: {k: v for k, v in record.items() if k != "file"} for name, record in report["tensors"].items()} == {
        name: {k: v for k, v in record.items() if k != "file"} for name, record in reference["tensors"].items()
    }
    assert dense.read_dense_export(root) == report


def test_shard_plan_is_order_independent_and_accounts_for_output_dtype():
    shapes = {"c": (2,), "a": (3,), "b": (2,)}
    expected = {
        "a": "model-00001-of-00002.safetensors",
        "b": "model-00001-of-00002.safetensors",
        "c": "model-00002-of-00002.safetensors",
    }
    assert dense._shard_plan(shapes, dtype="bfloat16", max_bytes=10) == expected
    assert dense._shard_plan(dict(reversed(list(shapes.items()))), dtype="bfloat16", max_bytes=10) == expected
    assert len(set(dense._shard_plan(shapes, dtype="float32", max_bytes=10).values())) == 3


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "512MiB", None])
def test_invalid_shard_size_refuses_before_output(tmp_path, limit):
    net, source, checkpoint = fixture(tmp_path)
    with pytest.raises(ValueError, match="positive integer"):
        export(net, source, checkpoint, tmp_path / "dense", max_shard_size_bytes=limit)
    assert not (tmp_path / "dense").exists()


def test_interrupted_packed_export_never_publishes_commit(tmp_path, monkeypatch):
    from safetensors import torch as safetensors_torch

    net, source, checkpoint = fixture(tmp_path)
    root = tmp_path / "dense"
    original = safetensors_torch.save_file
    calls = []

    def interrupted(tensors, filename, **kwargs):
        calls.append(set(tensors))
        if len(calls) == 2:
            raise OSError("interrupted second packed shard")
        original(tensors, filename, **kwargs)

    monkeypatch.setattr(safetensors_torch, "save_file", interrupted)
    with pytest.raises(OSError, match="second packed shard"):
        export(net, source, checkpoint, root, max_shard_size_bytes=44)
    assert calls == [{"embedding.weight"}, {"linear.bias", "linear.weight"}]
    assert (root / "model-00001-of-00002.safetensors").is_file()
    assert not (root / dense.DENSE_EXPORT_MANIFEST).exists()
    with pytest.raises(ValueError, match="incomplete"):
        dense.read_dense_export(root)
    with pytest.raises(FileExistsError):
        export(net, source, checkpoint, root, max_shard_size_bytes=44)


@pytest.mark.parametrize("fault", ["shard_policy", "shard_size", "total_size", "extra_tensor"])
def test_reader_checks_packing_and_membership_even_with_updated_file_hashes(tmp_path, fault):
    net, source, checkpoint = fixture(tmp_path)
    root = tmp_path / "dense"
    report = export(net, source, checkpoint, root, max_shard_size_bytes=44)
    if fault == "shard_policy":
        del report["sharding"]
    elif fault == "shard_size":
        report["sharding"]["max_shard_size_bytes"] = 70
    elif fault == "total_size":
        index_path = root / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        index["metadata"]["total_size"] += 2
        index_path.write_text(json.dumps(index))
        report["files"][index_path.name] = dense._file_record(index_path)
    else:
        shard_path = root / report["tensors"]["embedding.weight"]["file"]
        values = load_file(shard_path)
        values["unindexed"] = torch.ones(1, dtype=torch.bfloat16)
        save_file(values, shard_path)
        report["files"][shard_path.name] = dense._file_record(shard_path)
    (root / dense.DENSE_EXPORT_MANIFEST).write_text(json.dumps(report))
    with pytest.raises(ValueError, match="shard policy|shard plan|total_size|membership"):
        dense.read_dense_export(root)


def test_v1_one_tensor_artifacts_remain_readable(tmp_path):
    net, source, checkpoint = fixture(tmp_path)
    root = tmp_path / "dense"
    report = export(net, source, checkpoint, root, max_shard_size_bytes=1)
    report["protocol"] = "cosmos.native-adapter-dense-export.v1"
    del report["sharding"]
    for number, record in enumerate(report["tensors"].values(), start=1):
        old = record["file"]
        new = f"model-{number:05d}.safetensors"
        (root / old).rename(root / new)
        record["file"] = new
        report["files"][new] = report["files"].pop(old)
    index_path = root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"] = {name: record["file"] for name, record in report["tensors"].items()}
    index_path.write_text(json.dumps(index))
    report["files"][index_path.name] = dense._file_record(index_path)
    (root / dense.DENSE_EXPORT_MANIFEST).write_text(json.dumps(report))
    assert dense.read_dense_export(root) == report


@pytest.mark.parametrize("mutation", ["file", "extra", "missing_commit", "tensor_digest", "index"])
def test_dense_reader_refuses_tamper_and_incomplete_artifact(tmp_path, mutation):
    net, source, checkpoint = fixture(tmp_path)
    target = tmp_path / "dense"
    report = export(net, source, checkpoint, target)
    if mutation == "file":
        with (target / "tokenizer.json").open("ab") as stream:
            stream.write(b" ")
    elif mutation == "extra":
        (target / "unexpected.txt").write_text("extra")
    elif mutation == "missing_commit":
        (target / dense.DENSE_EXPORT_MANIFEST).unlink()
    elif mutation == "index":
        (target / "model.safetensors.index.json").write_text("{}")
    else:
        report["tensors"]["linear.weight"]["sha256"] = "0" * 64
        (target / dense.DENSE_EXPORT_MANIFEST).write_text(json.dumps(report))
    with pytest.raises((ValueError, KeyError)):
        dense.read_dense_export(target)


@pytest.mark.parametrize("fault", ["source", "checkpoint", "adapter", "write"])
def test_changed_inputs_or_io_failure_never_publish_completion(tmp_path, monkeypatch, fault):
    net, source, checkpoint = fixture(tmp_path)
    original = dense._dense_tensors

    def changed(*args, **kwargs):
        yield from original(*args, **kwargs)
        if fault == "source":
            (source / "tokenizer.json").write_text("changed")
        elif fault == "checkpoint":
            (checkpoint / "foreign").write_text("changed")
        elif fault == "adapter":
            net.linear.lora_A.weight.add_(1)
        else:
            raise OSError("injected write failure")

    monkeypatch.setattr(dense, "_dense_tensors", changed)
    with pytest.raises((RuntimeError, ValueError, OSError)):
        export(net, source, checkpoint, tmp_path / "dense")
    assert not (tmp_path / "dense" / dense.DENSE_EXPORT_MANIFEST).exists()


@pytest.mark.parametrize("field,value", [("output_dtype", "float16"), ("exporter_build", {})])
def test_invalid_contract_refuses_before_output(tmp_path, field, value):
    net, source, checkpoint = fixture(tmp_path)
    with pytest.raises(ValueError):
        export(net, source, checkpoint, tmp_path / "dense", **{field: value})
    assert not (tmp_path / "dense").exists()


@pytest.mark.parametrize("fault", ["shape", "dtype", "nan", "scale"])
def test_merge_refuses_invalid_inputs(fault):
    base = torch.ones(3, 4, dtype=torch.bfloat16)
    a = torch.ones(2, 4)
    b = torch.ones(3, 2)
    scale = 2.0
    if fault == "shape":
        a = a[:, :3]
    elif fault == "dtype":
        a = a.bfloat16()
    elif fault == "nan":
        a[0, 0] = float("nan")
    else:
        scale = float("inf")
    with pytest.raises(ValueError):
        dense.merge_effective_weight(base, a, b, scale=scale, dtype=torch.bfloat16)


def test_cli_parser_requires_explicit_output_dtype_and_retains_original_overrides():
    from cosmos_framework.posttrain.checkpoint.tools.export_adapter_dense import parser

    args = parser().parse_args(
        [
            "--config",
            "recipe.py",
            "--checkpoint",
            "source",
            "--output",
            "dense",
            "--output-dtype",
            "bfloat16",
            "--override",
            "model.adapter.rank=32",
        ]
    )
    assert args.override == ["model.adapter.rank=32"]
    with pytest.raises(SystemExit):
        parser().parse_args(["--config", "recipe.py", "--checkpoint", "source", "--output", "dense"])


@pytest.mark.parametrize("field,value", [("tie_word_embeddings", True), ("quantization_config", {"bits": 4})])
def test_incompatible_hf_config_refused_before_output(tmp_path, field, value):
    net, source, checkpoint = fixture(tmp_path)
    config = json.loads((source / "config.json").read_text())
    config[field] = value
    (source / "config.json").write_text(json.dumps(config))
    (source / adapter_source.SOURCE_MANIFEST).write_text(json.dumps(adapter_source._identity(source)))
    with pytest.raises(ValueError, match="tied|quantization"):
        export(net, source, checkpoint, tmp_path / "dense")
    assert not (tmp_path / "dense").exists()


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_fresh_standard_hf_qwen3_vl_reload_has_exact_dense_weights_and_text_logits(tmp_path, monkeypatch, dtype):
    """Use a tiny architecture only for format/numerical testing, not an 8B capability claim."""
    from transformers import AutoModelForImageTextToText, Qwen3VLConfig, Qwen3VLForConditionalGeneration

    config = Qwen3VLConfig(
        text_config={
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "max_position_embeddings": 64,
            "tie_word_embeddings": False,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 16,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": [0],
        },
        image_token_id=29,
        video_token_id=30,
        vision_start_token_id=27,
        vision_end_token_id=28,
        tie_word_embeddings=False,
    )
    torch.manual_seed(29)
    net = Qwen3VLForConditionalGeneration(config).to(dtype=torch.bfloat16)
    source = tmp_path / "base"
    net.save_pretrained(source)
    expected = {name: p.detach().clone().to(getattr(torch, dtype)) for name, p in net.named_parameters()}
    # Only source-size certification is replaced: this CPU fixture cannot be the
    # immutable 8B profile. The saved HF files/headers, codec, merge, publication
    # and fresh standard HF loader are real and source changes still fail.
    source_identity = {
        "repository": adapter_source.SOURCE_REPOSITORY,
        "revision": adapter_source.SOURCE_REVISION,
        "files": {p.name: dense._file_record(p) for p in source.iterdir()},
    }

    def tiny_verified(path):
        assert Path(path) == source
        assert {p.name: dense._file_record(p) for p in source.iterdir()} == source_identity["files"]
        return source_identity

    monkeypatch.setattr(dense, "verify_adapter_snapshot", tiny_verified)
    target = "model.language_model.layers.0.self_attn.q_proj"
    parent, _, leaf = target.rpartition(".")
    old = net.get_submodule(target)
    module = LoraInjectedLinear(old, rank=2, alpha=4, adapter_dtype=torch.float32)
    net.get_submodule(parent).add_module(leaf, module)
    with torch.no_grad():
        module.lora_A.weight.fill_(0.03125)
        module.lora_B.weight.fill_(0.0625)
    for name, p in net.named_parameters():
        p.requires_grad_("lora_" in name)
    net._cosmos_adapter_parameters = {f"{target}.{n}": p for n, p in module.adapter_parameters().items()}
    identity = {
        "format": "cosmos.adapter-model.v1",
        "model_format": "cosmos_native_qwen3_vl",
        "base": source_identity,
        "targets": {target: list(old.weight.shape)},
        "quantization": None,
    }
    codec = build_adapter_checkpoint_codec(net, identity=identity)
    codec.bind_processor_identity({"fixture": "tiny format-only processor contract"})
    net._cosmos_adapter_checkpoint_codec = codec
    checkpoint = save_adapter_artifact(
        str(tmp_path / "checkpoints/iter_000002"),
        codec.parameters,
        codec.metadata(),
        manager=CheckpointManager(checkpoint_format="consolidated"),
        publish=False,
    )
    expected[target + ".weight"] = (old.weight.float() + 2 * (module.lora_B.weight @ module.lora_A.weight)).to(
        getattr(torch, dtype)
    )
    report = export(net, source, checkpoint, tmp_path / "dense", output_dtype=dtype)
    assert dense.read_dense_export(tmp_path / "dense") == report
    loaded = AutoModelForImageTextToText.from_pretrained(
        tmp_path / "dense", dtype="auto", local_files_only=True, trust_remote_code=False, attn_implementation="sdpa"
    ).eval()
    reference = Qwen3VLForConditionalGeneration(copy.deepcopy(loaded.config)).to(dtype=getattr(torch, dtype)).eval()
    reference.load_state_dict(expected, strict=True)
    assert set(dict(loaded.named_parameters())) == set(expected)
    for name, p in loaded.named_parameters():
        assert p.dtype == getattr(torch, dtype)
        torch.testing.assert_close(p, expected[name], atol=0, rtol=0)
    text_inputs = {"input_ids": torch.tensor([[1, 2, 3, 4]])}
    image_inputs = {
        "input_ids": torch.tensor([[1, 27, 29, 28, 2]]),
        "mm_token_type_ids": torch.tensor([[0, 0, 1, 0, 0]]),
        "pixel_values": torch.randn(4, 24),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
    }
    video_inputs = {
        "input_ids": torch.tensor([[1, 27, 30, 28, 2]]),
        "mm_token_type_ids": torch.tensor([[0, 0, 2, 0, 0]]),
        "pixel_values_videos": torch.randn(4, 24),
        "video_grid_thw": torch.tensor([[1, 2, 2]]),
    }
    with torch.no_grad():
        # Test the actual standard loader before changing any storage. BF16 is
        # exact here. Raw FP32 multimodal CPU results showed a 4.47e-8 difference
        # with byte-identical weights: mmap versus owning-storage alignment can
        # select a differently rounded BLAS path. One FP32 epsilon is an explicit
        # small-fixture tolerance, not an accelerator or release parity claim.
        for inputs in (text_inputs, image_inputs, video_inputs):
            expected_logits = reference(**inputs).logits
            assert expected_logits.abs().max() < 1
            torch.testing.assert_close(
                loaded(**inputs).logits,
                expected_logits,
                atol=0 if dtype == "bfloat16" else torch.finfo(torch.float32).eps,
                rtol=0,
            )
        # Separate layout-attribution check: owning copies preserve every weight
        # value and produce exact logits. This does not replace the raw HF check.
        for parameter in loaded.parameters():
            parameter.data = parameter.detach().clone()
        for inputs in (text_inputs, image_inputs, video_inputs):
            torch.testing.assert_close(loaded(**inputs).logits, reference(**inputs).logits, atol=0, rtol=0)


@pytest.mark.parametrize("wrong", ["manifest", "base", "processor", "adapter"])
def test_external_dense_identity_refuses_wrong_model_even_if_artifact_is_self_consistent(tmp_path, wrong):
    net, source, checkpoint = fixture(tmp_path)
    target = tmp_path / "dense"
    report = export(net, source, checkpoint, target)
    if wrong == "manifest":
        kwargs = {"expected_manifest_sha256": "0" * 64}
    elif wrong == "base":
        kwargs = {"expected_identity": {"base_identity": {"revision": "0" * 40}}}
    elif wrong == "processor":
        kwargs = {"expected_identity": {"processor_identity": {"template": "wrong"}}}
    else:
        name = next(iter(report["adapter_tensors"]))
        kwargs = {"expected_identity": {"adapter_tensors": {name: {"sha256": "0" * 64}}}}
    with pytest.raises(ValueError, match="differs"):
        dense.read_dense_export(target, **kwargs)
    assert (
        dense.read_dense_export(
            target,
            expected_manifest_sha256=dense._file_record(target / dense.DENSE_EXPORT_MANIFEST)["sha256"],
            expected_identity={
                "base_identity": report["base_identity"],
                "processor_identity": report["processor_identity"],
                "adapter_tensors": report["adapter_tensors"],
            },
        )
        == report
    )
