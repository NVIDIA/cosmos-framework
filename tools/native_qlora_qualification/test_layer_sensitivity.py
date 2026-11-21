# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU contract tests; the separate real-CUDA canary remains mandatory."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from tools.native_qlora_qualification import layer_sensitivity as subject


def native_dense():
    from cosmos_framework.utils.generator.lora import LoraInjectedLinear

    generator = torch.Generator().manual_seed(41)
    base = torch.nn.Linear(8, 6, bias=False, dtype=torch.bfloat16)
    module = LoraInjectedLinear(base, 2, 6, adapter_dtype=torch.float32, adapter_compute_dtype=torch.float32)
    module.weight.requires_grad_(False)
    with torch.no_grad():
        module.lora_A.weight.copy_(torch.randn(2, 8, generator=generator) / 10)
        module.lora_B.weight.copy_(torch.randn(6, 2, generator=generator) / 10)
    return module.eval()


def test_native_dense_forward_preserves_nonzero_adapter_arithmetic_and_rng():
    original = native_dense()
    source = torch.arange(48, dtype=torch.bfloat16).reshape(6, 8) / 100
    rng = torch.get_rng_state().clone()
    replacement = subject.dense_replacement(original, source)
    assert torch.equal(rng, torch.get_rng_state())
    assert replacement.lora_A is original.lora_A and replacement.lora_B is original.lora_B
    assert replacement.lora_dropout is original.lora_dropout
    assert not replacement.weight.requires_grad
    x = torch.arange(16, dtype=torch.bfloat16).reshape(2, 8) / 7
    expected = torch.nn.functional.linear(x, source) + original.adapter_delta(x, output_dtype=torch.bfloat16)
    torch.testing.assert_close(replacement(x), expected, rtol=0, atol=0)
    assert not torch.equal(replacement(x), torch.nn.functional.linear(x, source))


@pytest.mark.parametrize("bad", ["shape", "dtype", "nonfinite", "math", "rslora", "bias"])
def test_dense_restoration_rejects_unsupported_source_or_math(bad):
    original, source = native_dense(), torch.ones(6, 8, dtype=torch.bfloat16)
    if bad == "shape":
        source = source[:, :4]
    if bad == "dtype":
        source = source.float()
    if bad == "nonfinite":
        source[0, 0] = float("nan")
    if bad == "math":
        original._adapter_compute_dtype = torch.bfloat16
    if bad == "rslora":
        original._lora_use_rslora = True
    if bad == "bias":
        original.bias = torch.nn.Parameter(torch.zeros(6), requires_grad=False)
    with pytest.raises(ValueError, match="restoration requires"):
        subject.dense_replacement(original, source)


def test_context_restores_on_partial_source_failure_and_refuses_checkpoint(monkeypatch):
    from cosmos_framework.posttrain.model import checkpoint
    from cosmos_framework.posttrain.model.quantization import loader

    net = torch.nn.Module()
    net.first, net.second = native_dense(), native_dense()
    net._cosmos_adapter_parameters = {
        f"{name}.{key}": p for name in ("first", "second") for key, p in getattr(net, name).adapter_parameters().items()
    }
    codec = checkpoint.AdapterCheckpointCodec(net, identity={"test": "CPU synthetic base"})
    net.eval()
    originals = (net.first, net.second)
    monkeypatch.setattr(
        loader, "quantized_state_record", lambda module: {"cpu_synthetic_weight": loader.tensor_digest(module.weight)}
    )
    targets = {name: (net, name, getattr(net, name)) for name in ("first", "second")}
    with torch.no_grad():
        with subject.restore_block(net, targets, lambda _: torch.ones(6, 8, dtype=torch.bfloat16)):
            assert net.first is not originals[0]
            with pytest.raises(RuntimeError, match="model parameters changed"):
                codec.metadata()
            with pytest.raises(ValueError, match="nested"):
                with subject.restore_block(net, targets, lambda _: None):
                    pass
        codec.assert_live()

        def source(name):
            if name == "second":
                raise OSError("source failed")
            return torch.ones(6, 8, dtype=torch.bfloat16)

        with pytest.raises(OSError, match="source failed"):
            with subject.restore_block(net, targets, source):
                pass
    assert (net.first, net.second) == originals
    assert not net._layer_sensitivity_active
    codec.assert_live()


def test_context_refuses_grad_enabled_or_training_model():
    net = torch.nn.Sequential(native_dense()).eval()
    with pytest.raises(ValueError, match="eval mode"):
        with subject.restore_block(net, {}, lambda _: None):
            pass
    with torch.no_grad():
        net.train()
        with pytest.raises(ValueError, match="eval mode"):
            with subject.restore_block(net, {}, lambda _: None):
                pass


def test_native_cache_reuses_cpu_prepared_samples_without_redecode(tmp_path):
    from cosmos_framework.posttrain.data.packer.sft import SFTPolicyInput
    from cosmos_framework.posttrain.data.packer.sft_preprocess import SFTPreprocessingPipeline, _WorkerSpec

    class Packer:
        max_length, mask_think_opener, processor_kwargs, processor = 100, False, {}, None
        tokenizer = SimpleNamespace(chat_template="test")
        calls = 0

        def validation_semantics_identity(self):
            return {"fixture": "native cache"}

        def sft_process_sample(self, target):
            self.calls += 1
            return SFTPolicyInput([1, 2], [-100, 2], {"pixel_values_videos": torch.ones(2, 8)})

    packer = Packer()
    spec = _WorkerSpec(
        str(tmp_path), str(tmp_path), 100, {}, False, packer.validation_semantics_identity(), 100_000, {}
    )
    pipeline = SFTPreprocessingPipeline(packer, spec, prefetch=False, max_pending=1)
    rows = [{"messages": [{"role": "assistant", "content": str(i)}]} for i in range(3)]
    warmed = subject.warm_cache(pipeline, rows)
    assert packer.calls == 3
    for row in rows:
        sample = pipeline.sft_process_sample(subject.target_for(row))
        sample.mm_kwargs["pixel_values_videos"].zero_()
        assert pipeline.sft_process_sample(subject.target_for(row)).mm_kwargs["pixel_values_videos"].sum() == 16
    subject.assert_cache_hits(pipeline, warmed)
    assert packer.calls == 3
    pipeline.close()
    tiny = SFTPreprocessingPipeline(
        Packer(), _WorkerSpec(str(tmp_path), str(tmp_path), 100, {}, False, {}, 0, {}), prefetch=False, max_pending=1
    )
    with pytest.raises(ValueError, match="do not fit"):
        subject.warm_cache(tiny, rows)
    tiny.close()


def test_pairing_rejects_changed_inputs_and_retains_worsening():
    base = dict(
        id="a",
        input_sha256="x",
        media_sha256="y",
        native_supervised_tokens=2,
        restricted_choice_correct=True,
        answer_nll=1.0,
        restricted_choice_prediction="A",
    )
    changed = dict(base, restricted_choice_correct=False, answer_nll=2.0, restricted_choice_prediction="B")
    assert subject.paired_change([base], [changed]) == {
        "restricted_choice_accuracy_delta": -1.0,
        "answer_nll_delta": 1.0,
        "restricted_choice_disagreements": 1,
    }
    with pytest.raises(ValueError, match="paired native inputs"):
        subject.paired_change([base], [dict(changed, input_sha256="other")])


def test_training_manifest_is_accepted_and_role_or_row_change_rejected(tmp_path):
    values = [
        dict(
            id=str(i),
            messages=[
                dict(
                    role="user",
                    content=[dict(type="video", video="clip.mp4"), dict(type="text", text="Question?\nA: one\nB: two")],
                ),
                dict(role="assistant", content="A"),
            ],
        )
        for i in range(64)
    ]
    data, manifest = tmp_path / "training.jsonl", tmp_path / "manifest.json"
    data.write_text("".join(json.dumps(row) + "\n" for row in values))
    value = {
        "protocol": "cosmos.wts-group-split.v1",
        "role": "training_calibration",
        "release_qualified": False,
        "provenance": {"historical_exposure": "training_exposed", "preexisting_final_data_consumed": False, "rows": 64},
        "data_sha256": subject.score.file_digest(data),
        "rows": [{"id": row["id"], "raw_row_sha256": subject.score.json_digest(row)} for row in values],
    }
    manifest.write_text(json.dumps(value))
    rows, _, _ = subject.training_rows(data, manifest, subject.score.file_digest(manifest))
    assert len(rows) == 64
    for change in ("role", "row"):
        edited = json.loads(json.dumps(value))
        if change == "role":
            edited["role"] = "development"
        else:
            edited["rows"][0]["raw_row_sha256"] = "0" * 64
        path = tmp_path / (change + ".json")
        path.write_text(json.dumps(edited))
        with pytest.raises(ValueError):
            subject.training_rows(data, path, subject.score.file_digest(path))
