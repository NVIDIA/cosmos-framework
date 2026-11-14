# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Real tiny checkpoint/public loader controls, separate from A100 qualification."""

import copy
import json
import os
import subprocess
import sys

import pytest
import torch

from cosmos_framework.inference.reasoner import base_deployment
from cosmos_framework.inference.reasoner.runtime import CosmosFrameworkRuntime
from cosmos_framework.posttrain.tests.test_gemma_fixed_mma import cpu_projection  # noqa: F401
from cosmos_framework.posttrain.tests.test_gemma_sdpa_layout import tiny  # noqa: F401
from cosmos_framework.utils.generator import gemma_sdpa_layout as profile
from cosmos_framework.utils.hf_snapshot import snapshot_identity


@pytest.fixture
def base(tmp_path, tiny, monkeypatch):  # noqa: F811
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    path = tmp_path / "base"
    tiny.to(torch.bfloat16).save_pretrained(path)
    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
    vocab.update({f"word{i}": i for i in range(4, tiny.config.vocab_size)})
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    processor = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, bos_token="<bos>", eos_token="<eos>", pad_token="<pad>", unk_token="<unk>"
    )
    processor.chat_template = "{% for message in messages %}{{ message['content'] }} {% endfor %}"
    processor.save_pretrained(path)
    # Fixture-only admission: production retains exact2B content/architecture/A100.
    monkeypatch.setattr(profile, "BASE_SHA256", snapshot_identity(path, weights=True)["sha256"])
    monkeypatch.setattr(profile, "PROCESSOR_SHA256", snapshot_identity(path, weights=False)["sha256"])
    monkeypatch.setattr(base_deployment, "_validate_device", lambda device: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return path


def load(base, **kwargs):
    return CosmosFrameworkRuntime(
        str(base), dtype="float32", attn_implementation="sdpa", base_execution_profile=profile.SDPA64_PROFILE, **kwargs
    )


def test_public_unadapted_identity_tensors_aliases_cache_and_restore(base, cpu_projection):  # noqa: F811
    from safetensors.torch import load_file

    runtime = load(base)
    model = runtime.model
    assert runtime.adapter_deployment is runtime.portable_adapter is None
    assert runtime.base_deployment["kind"] == "unadapted_model_deployment"
    identity = runtime.base_deployment["identity"]
    assert identity["format"] == profile.BASE_FORMAT and "resolved" not in identity
    assert all(not p.requires_grad and not p.is_meta and p.dtype == torch.float32 for p in model.parameters())
    assert model.model.embed_tokens.weight is model.lm_head.weight
    assert (
        model.model.embed_tokens.embed_scale.item()
        == profile.BUFFER_CONTRACT["buffers"]["model.embed_tokens.embed_scale"]["value"]
    )
    saved = load_file(base / "model.safetensors")
    for name, tensor in saved.items():
        assert torch.equal(tensor.float(), model.state_dict()[name]), name
    assert not any(hasattr(m, "lora_A") for m in model.modules())
    ownership = profile._ownership(model)
    ids = torch.arange(39)[None].remainder(model.config.vocab_size)
    with torch.no_grad():
        full = model(ids, use_cache=False).logits
        prefix = model(ids[:, :32], use_cache=True).past_key_values
        snapshots = [(layer.keys.clone(), layer.values.clone()) for layer in prefix.layers]
        split = model(ids[:, 32:], past_key_values=copy.deepcopy(prefix), use_cache=True).logits
        again = model(ids[:, 32:], past_key_values=copy.deepcopy(prefix), use_cache=True).logits
        cache = copy.deepcopy(prefix)
        incremental = []
        for position in range(32, ids.shape[1]):
            result = model(ids[:, position : position + 1], past_key_values=cache, use_cache=True)
            incremental.append(result.logits)
            cache = result.past_key_values
        generated = model.generate(ids[:, :32], max_new_tokens=4, do_sample=False)
    torch.testing.assert_close(full[:, 32:], split, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(torch.cat(incremental, dim=1), split, atol=2e-6, rtol=2e-5)
    assert torch.equal(split, again) and generated.shape[1] > 32
    assert all(
        torch.equal(layer.keys, k) and torch.equal(layer.values, v) for layer, (k, v) in zip(prefix.layers, snapshots)
    )
    assert profile._ownership(model) == ownership
    with pytest.raises(RuntimeError, match="no_grad"):
        model(ids)
    with torch.no_grad(), torch.autocast("cpu"), pytest.raises(RuntimeError, match="autocast"):
        model(ids)
    runtime.restore_inference_execution()
    assert runtime.base_deployment["execution_profile"] == "stock_fp32"
    assert profile._ownership(model) == ownership
    profile.install(model, identity=identity, profile=profile.SDPA64_PROFILE)
    with torch.no_grad(), pytest.raises(ValueError, match="prefix from this installation"):
        model(ids[:, 32:], past_key_values=prefix)


@pytest.mark.parametrize("name", ["model.safetensors", "tokenizer_config.json"])
def test_changed_snapshot_rejected_before_model_construction(base, monkeypatch, name):
    from transformers import AutoModelForCausalLM

    data = (base / name).read_bytes()
    (base / name).write_bytes(data + b" ")

    def forbidden(*args, **kwargs):
        pytest.fail("wrong identity reached model allocation")

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", forbidden)
    with pytest.raises(ValueError, match="exact unadapted"):
        load(base)


@pytest.mark.parametrize("field", ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"])
def test_loading_diagnostics_never_silently_initialize(base, monkeypatch, field):
    from transformers import AutoModelForCausalLM

    native = AutoModelForCausalLM.from_pretrained

    def incomplete(*args, **kwargs):
        net, info = native(*args, **kwargs)
        info[field] = ["synthetic-missing-tensor"]
        return net, info

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", incomplete)
    with pytest.raises(ValueError, match="no weights may be initialized"):
        load(base)


@pytest.mark.parametrize("conflict", ["portable", "config", "export", "vit", "lora", "base", "map"])
def test_public_base_conflicts_fail_before_loading(monkeypatch, conflict):
    kwargs = {
        "portable": {"portable_adapter": {}},
        "config": {"config_file": "x"},
        "export": {"export_dir": "x"},
        "vit": {"vit_checkpoint_path": "x"},
        "lora": {"enable_lora": True},
        "base": {"base_model_path": "x"},
        "map": {"device_map": "balanced"},
    }[conflict]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="base_execution_profile"):
        load("nonexistent-before-allocation", **kwargs)


@pytest.mark.parametrize("marker", ["adapter_manifest.json", ".materialized"])
def test_export_markers_cannot_be_reinterpreted_as_base(base, marker):
    (base / marker).write_text("{}")
    with pytest.raises(ValueError, match="unadapted snapshot"):
        load(base)


def test_identity_and_profile_changes_are_rejected_before_mutation(base, cpu_projection):  # noqa: F811
    runtime = load(base)
    model = runtime.model
    identity = runtime.base_deployment["identity"]
    before = profile._ownership(model)
    for key, value in (("dtype", "bfloat16"), ("execution_profile", profile.PROFILE), ("resolved", {"targets": []})):
        wrong = copy.deepcopy(identity)
        wrong[key] = value
        with pytest.raises(ValueError, match="exact unadapted"):
            profile.install(model, identity=wrong, profile=profile.SDPA64_PROFILE)
    assert profile._ownership(model) == before
    with torch.inference_mode(), pytest.raises(ValueError, match="outside inference_mode"):
        load(base)
    with pytest.raises(ValueError, match="dtype='float32'"):
        base_deployment.load_base_model(base, profile=profile.SDPA64_PROFILE, dtype="bfloat16", device="cpu")
    with pytest.raises(ValueError, match="unsupported base_execution_profile"):
        CosmosFrameworkRuntime(str(base), base_execution_profile=profile.PROFILE)


def test_shared_identity_retains_old_public_import_and_buffer_contract(base):
    from cosmos_framework.posttrain.checkpoint.adapters import snapshot_identity as old_identity
    from cosmos_framework.posttrain.checkpoint.conversion.hf import SIDECAR_FILES
    from cosmos_framework.posttrain.model.hf_adapter_buffers import prepare_adapter_buffers as old_buffers
    from cosmos_framework.utils import hf_buffers, hf_snapshot

    assert old_identity is hf_snapshot.snapshot_identity
    assert old_buffers is hf_buffers.prepare_adapter_buffers
    assert SIDECAR_FILES is hf_snapshot.SIDECAR_FILES
    assert old_identity(base, weights=True) == snapshot_identity(base, weights=True)


def test_fresh_process_public_generation_has_base_provenance_without_training_imports(base, tmp_path):
    script = tmp_path / "fresh.py"
    script.write_text("""import json,sys,torch
from transformers import AutoConfig
from cosmos_framework.inference.reasoner import base_deployment,generate
from cosmos_framework.utils.generator import gemma_sdpa_layout as profile,gemma_fixed_mma
from cosmos_framework.utils.hf_snapshot import snapshot_identity
from cosmos_framework.utils.hf_buffers import buffer_contract
config=AutoConfig.from_pretrained(sys.argv[1],local_files_only=True)
profile.ARCHITECTURE={k:getattr(config,k) for k in profile.ARCHITECTURE}
profile.BUFFER_CONTRACT=buffer_contract(config)
profile.BASE_SHA256=snapshot_identity(sys.argv[1],weights=True)['sha256']
profile.PROCESSOR_SHA256=snapshot_identity(sys.argv[1],weights=False)['sha256']
profile._validate_device=lambda model:None
base_deployment._validate_device=lambda device:None
torch.cuda.is_available=lambda:False
torch.cuda.is_current_stream_capturing=lambda:False
gemma_fixed_mma._project=lambda x,w,b=None:torch.nn.functional.linear(x.double(),w.double(),None if b is None else b.double())
configuration,output=sys.argv[2:]
sys.argv=['generate','--config',configuration,'--output',output]
generate.main()
assert not any(n.startswith('cosmos_framework.posttrain') for n in sys.modules)
record=json.load(open(output))
assert record['deployment']['kind']=='unadapted_model_deployment'
assert 'resolved' not in record['deployment']['identity']
assert len(record['answers'])==1 and record['deployment']['effective_generation']['max_tokens']==4
""")
    config = tmp_path / "generation.toml"
    config.write_text(
        "[model]\nmodel_name="
        + json.dumps(str(base))
        + '\ndtype="float32"\nattn_implementation="sdpa"\nbase_execution_profile='
        + json.dumps(profile.SDPA64_PROFILE)
        + '\n[generation]\nmax_tokens=4\ntemperature=0.0\nseed=42\n[[tasks]]\nquestion="word4 word5 word6"\n'
    )
    result = subprocess.run(
        [sys.executable, "-B", str(script), str(base), str(config), str(tmp_path / "answer.json")],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_evaluation_passes_explicit_base_profile(monkeypatch):
    from types import SimpleNamespace

    from cosmos_framework.evaluation.reasoner import base as evaluation
    from cosmos_framework.inference.reasoner import framework_torchcodec_video as video

    class Evaluator(evaluation.BaseEvaluator):
        def make_tasks(self, *args):
            pass

        def save(self, *args):
            pass

        def compute_metrics(self, *args):
            pass

        def aggregate_metrics_if_all_shards_present(self, *args):
            pass

    calls = []
    runtime = SimpleNamespace(processor=object())

    def create(path, **kwargs):
        calls.append((path, kwargs))
        return runtime

    monkeypatch.setattr(evaluation, "CosmosFrameworkRuntime", create)
    monkeypatch.setattr(video, "FrameworkTorchCodecVideoPreprocessor", lambda **kwargs: None)
    evaluator = Evaluator(
        {
            "model": {"model_name": "immutable-base", "base_execution_profile": profile.SDPA64_PROFILE},
            "vision": {"video_decoder": "torchcodec-cuda-on-demand", "num_frames": 4},
        }
    )
    evaluator._send_status_callback = lambda *args: None
    assert evaluator.load_model() == (runtime, runtime.processor)
    assert calls[0][1]["base_execution_profile"] == profile.SDPA64_PROFILE


def test_live_base_parameter_change_invalidates_cached_profile(base, cpu_projection):  # noqa: F811
    runtime = load(base)
    with torch.no_grad():
        prefix = runtime.model(torch.tensor([[1, 4, 5]]), use_cache=True).past_key_values
        runtime.model.model.layers[0].self_attn.q_proj.weight.add_(1e-6)
        with pytest.raises(ValueError, match="changed"):
            runtime.model(torch.tensor([[6]]), past_key_values=prefix)


def test_explicit_untracked_write_boundary_invalidates_base_and_external_prefix(base, cpu_projection):  # noqa: F811
    runtime = load(base)
    identity = runtime.base_deployment["identity"]
    with torch.no_grad():
        prefix = runtime.model(torch.tensor([[1, 4, 5]]), use_cache=True).past_key_values
        runtime.model.model.layers[0].self_attn.q_proj.weight.data.add_(0.25)
    runtime.clear_adapter_caches()
    assert runtime.base_deployment["state_modified_since_load"] is True
    with torch.no_grad(), pytest.raises(ValueError, match="untracked writes"):
        runtime.model(torch.tensor([[6]]), past_key_values=prefix)
    with torch.no_grad(), pytest.raises(ValueError, match="untracked writes"):
        runtime.model(torch.tensor([[1, 4, 5, 6]]), use_cache=False)
    runtime.restore_inference_execution()
    with pytest.raises(ValueError, match="reload"):
        profile.install(runtime.model, identity=identity, profile=profile.SDPA64_PROFILE)
    fresh = load(base)
    assert "state_modified_since_load" not in fresh.base_deployment
    with torch.no_grad():
        assert torch.isfinite(fresh.model(torch.tensor([[1, 4, 5, 6]])).logits).all()
