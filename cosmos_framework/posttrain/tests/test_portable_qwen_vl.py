# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Public ordered HF VLM assembly; CPU fixtures confer no real-checkpoint credit."""

import hashlib
import json
import os
import subprocess
import sys

import pytest
import torch
import transformers
from safetensors.torch import load_file, save_file

from cosmos_framework.posttrain.checkpoint.adapters import save_adapter_artifact
from cosmos_framework.posttrain.config.adapters import AdapterComponentConfig as Component
from cosmos_framework.posttrain.config.adapters import AdapterConfig, AdapterExtraModuleConfig
from cosmos_framework.posttrain.config.model import HFModelConfig
from cosmos_framework.posttrain.model.hf import build_hf_model
from cosmos_framework.posttrain.tests.helpers.sft_packer import CHAT_TEMPLATE, local_processor, local_tokenizer
from cosmos_framework.posttrain.tests.test_adapter_qwen2_vl_families import small_config
from cosmos_framework.utils.generator import portable_adapters, portable_qwen_vl

pytestmark = pytest.mark.skipif(
    transformers.__version__ != "5.8.1", reason="ordered public VLM inference pins Transformers 5.8.1"
)


def snapshot(path, family="qwen2_vl", *, tied=False):
    torch.manual_seed(61)
    tokenizer = local_tokenizer()
    if family == "qwen3_vl":
        from cosmos_framework.posttrain.tests.test_model_qwen3_vl_gpu import _dense_config

        config = _dense_config()
    else:
        config = small_config(family)
    config.text_config.vocab_size = len(tokenizer)
    config.tie_word_embeddings = config.text_config.tie_word_embeddings = tied
    for name, token in [
        ("image_token_id", "<|image_pad|>"),
        ("video_token_id", "<|video_pad|>"),
        ("vision_start_token_id", "<|vision_start|>"),
        ("vision_end_token_id", "<|vision_end|>"),
    ]:
        setattr(config, name, tokenizer.convert_tokens_to_ids(token))
    net = transformers.AutoModelForImageTextToText.from_config(config, attn_implementation="sdpa", dtype=torch.bfloat16)
    net.save_pretrained(path)
    media = local_processor(tokenizer)
    cls = {
        "qwen2_vl": transformers.Qwen2VLProcessor,
        "qwen2_5_vl": transformers.Qwen2_5_VLProcessor,
        "qwen3_vl": transformers.Qwen3VLProcessor,
    }[family]
    processor = cls(
        image_processor=media.image_processor,
        tokenizer=tokenizer,
        chat_template=CHAT_TEMPLATE,
        video_processor=(
            transformers.Qwen3VLVideoProcessor if family == "qwen3_vl" else transformers.Qwen2VLVideoProcessor
        )(
            patch_size=2,
            temporal_patch_size=2,
            merge_size=2,
        ),
    )
    processor.save_pretrained(path)
    tensors = load_file(path / "model.safetensors")
    if family in ("qwen2_vl", "qwen2_5_vl"):
        tensors = {
            n.replace("model.language_model.", "model.").replace("model.visual.", "visual."): t
            for n, t in tensors.items()
        }
    save_file(tensors, path / "model.safetensors")
    return config


def artifact(
    tmp_path,
    monkeypatch,
    family="qwen2_vl",
    method="mixed",
    rslora=False,
    *,
    tied=False,
    dense=False,
    model_execution=portable_qwen_vl.PROFILE,
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    base, portable = tmp_path / "base", tmp_path / "portable"
    config = snapshot(base, family, tied=tied)
    projector = "model.visual.merger.linear_fc1" if family == "qwen3_vl" else "model.visual.merger.mlp.2"
    cfg = AdapterConfig(
        model_execution=model_execution,
        components=[
            Component(
                component=component,
                targets=[target],
                method=kind,
                rank=rank,
                alpha=5,
                use_rslora=rslora,
                dropout=0 if kind == "pissa" else 0.1,
                execution="pissa_dense_v1" if dense and kind == "pissa" else "factorized_v1",
            )
            for component, target, kind, rank in [
                (
                    "language",
                    "model.language_model.layers.0.self_attn.q_proj",
                    "lora" if method == "mixed" else method,
                    2,
                ),
                ("projector", projector, "dora" if method == "mixed" else method, 3),
                ("vision", "model.visual.blocks.0.attn.qkv", "pissa" if method == "mixed" else method, 2),
            ]
        ],
        extra_modules=[AdapterExtraModuleConfig(component="language", targets=["model.language_model.norm"])],
    )
    reference = build_hf_model(
        HFModelConfig(name_or_path=str(base), dtype="bfloat16", attn_impl="sdpa", grad_ckpt=False), adapters=cfg
    ).net
    with torch.no_grad():
        for parameter in reference.parameters():
            if parameter.requires_grad:
                parameter.add_(0.003)
    reference.eval()
    save_adapter_artifact(reference, str(portable))
    reference.float()
    return base, portable, reference, config


def load(base, portable, monkeypatch):
    # Only hardware admission is substituted for structural CPU fixtures.
    monkeypatch.setattr(portable_adapters, "_validate_device", lambda device: None)
    return portable_adapters.load_portable_model(
        str(base),
        {"directory": str(portable), "execution_profile": portable_qwen_vl.PROFILE},
        dtype="float32",
        device="cpu",
        attn_implementation="sdpa",
    )


def data(config):
    text = {"input_ids": torch.tensor([[30, 31, 32, 33]])}
    result = [text]
    for video in (False, True):
        ids = torch.tensor(
            [
                [
                    30,
                    config.vision_start_token_id,
                    config.video_token_id if video else config.image_token_id,
                    config.vision_end_token_id,
                    31,
                ]
            ]
        )
        kwargs = {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "mm_token_type_ids": (ids == config.image_token_id).long() + 2 * (ids == config.video_token_id).long(),
            "pixel_values_videos" if video else "pixel_values": torch.arange(96).reshape(4, 24).float() / 96,
            "video_grid_thw" if video else "image_grid_thw": torch.tensor([[1, 2, 2]]),
        }
        if video and config.model_type == "qwen2_5_vl":
            kwargs["second_per_grid_ts"] = torch.tensor([0.5])
        result.append(kwargs)
    return result


@pytest.mark.parametrize("method", ["lora", "dora", "pissa"])
@pytest.mark.parametrize("rslora", [False, True])
def test_six_settings_strict_public_load_no_reinitialization(tmp_path, monkeypatch, method, rslora):
    base, portable, reference, config = artifact(tmp_path, monkeypatch, method=method, rslora=rslora)
    monkeypatch.setattr(torch.linalg, "svd", lambda *a, **k: pytest.fail("portable loading must not decompose PiSSA"))
    monkeypatch.setattr(torch, "svd_lowrank", lambda *a, **k: pytest.fail("portable loading must not decompose PiSSA"))
    loaded, processor, provenance = load(base, portable, monkeypatch)
    assert type(processor) is transformers.Qwen2VLProcessor
    assert provenance["precision_contract"]["artifact_base_storage"] == "bfloat16"
    assert provenance["identity"] == reference._posttrain_adapter_identity
    assert {n: p.requires_grad for n, p in loaded.named_parameters()} == {
        n: p.requires_grad for n, p in reference.named_parameters()
    }
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value, rtol=0, atol=0)
    owners = {n: id(p) for n, p in loaded.named_parameters(remove_duplicate=False)}
    with torch.no_grad():
        for batch in data(config):
            torch.testing.assert_close(loaded(**batch).logits, reference(**batch).logits, rtol=0, atol=0)
    assert owners == {n: id(p) for n, p in loaded.named_parameters(remove_duplicate=False)}


@pytest.mark.parametrize("family", ["qwen2_vl", "qwen2_5_vl", "qwen3_vl"])
def test_mixed_dense_ties_cached_generation_and_stock_merge(tmp_path, monkeypatch, family):
    from cosmos_framework.inference.reasoner.runtime import CosmosFrameworkRuntime
    from cosmos_framework.posttrain.checkpoint.tools.adapters import export_merged_adapter

    base, portable, reference, config = artifact(tmp_path, monkeypatch, family=family, tied=True, dense=True)
    loaded, processor, _ = load(base, portable, monkeypatch)
    assert loaded.get_input_embeddings().weight is loaded.get_output_embeddings().weight
    merged = tmp_path / "merged"
    assert export_merged_adapter(str(portable), str(base), str(merged)) == "built"
    runtime = CosmosFrameworkRuntime(str(merged), dtype="float32", attn_implementation="sdpa")
    with torch.no_grad():
        for batch in data(config):
            expected = reference(**batch).logits
            torch.testing.assert_close(loaded(**batch).logits, expected, rtol=0, atol=0)
            torch.testing.assert_close(runtime.model(**batch).logits, expected, rtol=5e-5, atol=5e-6)
            common = loaded.generate(**batch, max_new_tokens=3, do_sample=False)
            assert torch.equal(common, reference.generate(**batch, max_new_tokens=3, do_sample=False))
    assert processor.tokenizer.padding_side == "left"


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda i: i.update(base_storage_dtype="float32"), "BF16-base"),
        (lambda i: i["resolved"].update(model_execution="qwen3_vl_hf_stock_v1"), "stock HF|ordered"),
        (lambda i: i["dependencies"].update(transformers="4.57.1"), "pinned runtime"),
        (lambda i: i["resolved"]["targets"][0]["settings"].update(rank=1), "invalid adapter tensor"),
        (lambda i: i["resolved"]["targets"][0].update(component="vision"), "selects outside"),
        (lambda i: i["resolved"]["extra_parameters"][0].update(component="vision"), "selects outside"),
    ],
)
def test_identity_mismatch_rejected_before_injection(tmp_path, monkeypatch, mutation, match):
    base, portable, _, _ = artifact(tmp_path, monkeypatch)
    path = portable / "adapter_manifest.json"
    record = json.loads(path.read_text())
    mutation(record["identity"])
    path.write_text(json.dumps(record))
    from cosmos_framework.posttrain.model import adapters

    monkeypatch.setattr(adapters, "install_adapters", lambda *a, **k: pytest.fail("must reject before injection"))
    with pytest.raises(ValueError, match=match):
        load(base, portable, monkeypatch)


@pytest.mark.parametrize("filename", ["config.json", "tokenizer_config.json", "model.safetensors"])
def test_snapshot_edit_rejected_before_construction(tmp_path, monkeypatch, filename):
    base, portable, _, _ = artifact(tmp_path, monkeypatch)
    with (base / filename).open("ab") as stream:
        stream.write(b" ")
    monkeypatch.setattr(portable_qwen_vl, "load_base", lambda *a: pytest.fail("invalid source reached construction"))
    with pytest.raises(ValueError, match="identity mismatch"):
        load(base, portable, monkeypatch)


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda m: m.bfloat16(), "ownership"),
        (lambda m: setattr(m, "_cosmos_deployment_profile", "foreign"), "ownership"),
        (lambda m: setattr(m, "_posttrain_model_execution", "default_v1"), "ownership"),
        (lambda m: m.train(), "eval/no_grad"),
        (lambda m: setattr(m.model.language_model.layers[0].self_attn.q_proj, "_lora_alpha", 999), "ownership"),
    ],
)
def test_profile_and_ownership_mutations_refused(tmp_path, monkeypatch, change, match):
    base, portable, _, config = artifact(tmp_path, monkeypatch)
    model, _, _ = load(base, portable, monkeypatch)
    change(model)
    with torch.no_grad(), pytest.raises(ValueError, match=match):
        model(**data(config)[0])


def test_no_autograd_autocast_or_profile_reinterpretation(tmp_path, monkeypatch):
    base, portable, _, config = artifact(tmp_path, monkeypatch)
    model, _, _ = load(base, portable, monkeypatch)
    with pytest.raises(ValueError, match="eval/no_grad"):
        model(**data(config)[0])
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16), pytest.raises(ValueError, match="autocast"):
        model(**data(config)[0])
    with pytest.raises(ValueError, match="already installed"):
        portable_qwen_vl.install_inference_guard(model, model._posttrain_adapter_identity)
    for dtype in ("bfloat16", "float16"):
        with pytest.raises(ValueError, match="float32"):
            portable_adapters.load_portable_model(
                str(base),
                {"directory": str(portable), "execution_profile": portable_qwen_vl.PROFILE},
                dtype=dtype,
                device="cpu",
            )
    with pytest.raises(ValueError, match="cache"):
        portable_adapters.PortableAdapterConfig(str(portable), portable_qwen_vl.PROFILE, cache_effective_weights=True)


@pytest.mark.parametrize("stock_factorized", [False, True])
def test_fresh_public_runtime_without_training_builder_or_optional_dependencies(
    tmp_path, monkeypatch, stock_factorized
):
    profile = portable_qwen_vl.STOCK_FACTORIZED_PROFILE if stock_factorized else portable_qwen_vl.PROFILE
    base, portable, reference, config = artifact(
        tmp_path,
        monkeypatch,
        family="qwen3_vl" if stock_factorized else "qwen2_vl",
        model_execution="qwen3_vl_hf_stock_v1" if stock_factorized else portable_qwen_vl.PROFILE,
    )
    output = tmp_path / "fresh.pt"
    script = r"""
import sys
from importlib.abc import MetaPathFinder
class NoTraining(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in ("cosmos_framework.posttrain.model.hf", "cosmos_framework.posttrain.entrypoints", "cosmos_framework.posttrain.trainer")):
            raise RuntimeError("inference imported training-only implementation: " + fullname)
sys.meta_path.insert(0, NoTraining())
for name in ("megatron", "apex", "deepspeed", "ray"):
    sys.modules[name] = None  # Normal unavailable-optional-package semantics, including find_spec.
import torch
torch.set_float32_matmul_precision("highest")
torch.cuda.is_current_stream_capturing = lambda: False  # No CUDA graphs in the CPU structural fixture.
from cosmos_framework.utils.generator import portable_adapters
portable_adapters._validate_device = lambda device: None  # CPU fixture admission only.
from cosmos_framework.inference.reasoner.runtime import CosmosFrameworkRuntime
runtime = CosmosFrameworkRuntime(sys.argv[1], dtype="float32", attn_implementation="sdpa", portable_adapter={"directory":sys.argv[2],"execution_profile":sys.argv[4]})
with torch.no_grad():
    logits = runtime.model(input_ids=torch.tensor([[30,31,32,33]])).logits
    output = runtime.model.generate(input_ids=torch.tensor([[30,31,32,33]]), max_new_tokens=3, do_sample=False)
torch.save({"logits":logits,"output":output,"deployment":runtime.adapter_deployment},sys.argv[3])
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(base), str(portable), str(output), profile],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    actual = torch.load(output, weights_only=False)
    with torch.no_grad():
        torch.testing.assert_close(actual["logits"], reference(**data(config)[0]).logits, rtol=0, atol=0)
        assert torch.equal(actual["output"], reference.generate(**data(config)[0], max_new_tokens=3, do_sample=False))
    assert actual["deployment"]["execution_profile"] == profile


@pytest.mark.parametrize(
    "family,stock_factorized", [("qwen2_vl", False), ("qwen2_5_vl", False), ("qwen3_vl", False), ("qwen3_vl", True)]
)
def test_public_task_generation_and_repeated_video_prefixes(tmp_path, monkeypatch, family, stock_factorized):
    import numpy as np
    from PIL import Image

    from cosmos_framework.inference.reasoner.runtime import CosmosFrameworkRuntime

    profile = portable_qwen_vl.STOCK_FACTORIZED_PROFILE if stock_factorized else portable_qwen_vl.PROFILE
    base, portable, _, _ = artifact(
        tmp_path,
        monkeypatch,
        family=family,
        dense=not stock_factorized,
        model_execution="qwen3_vl_hf_stock_v1" if stock_factorized else portable_qwen_vl.PROFILE,
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(portable_adapters, "_validate_device", lambda device: None)
    runtime = CosmosFrameworkRuntime(
        str(base),
        dtype="float32",
        attn_implementation="sdpa",
        portable_adapter={
            "directory": str(portable),
            "execution_profile": profile,
        },
    )
    generation = {"max_tokens": 3, "temperature": 0, "seed": 42}
    vision = {"num_frames": 4, "min_pixels": 16, "max_pixels": 64}
    frames = [Image.fromarray(np.full((8, 8, 3), n * 40, dtype=np.uint8)) for n in range(4)]
    video = {
        "question": "What happens?",
        "media_mode": "video",
        "media_paths": ["fixed-video"],
        "_framework_decoded_media": [
            {
                "frames": frames,
                "fps": 4 / 31 * 30,
                "video_metadata": {"fps": 30.0, "total_num_frames": 31, "frames_indices": [0, 10, 20, 30]},
            }
        ],
    }
    image = {
        "prompt": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": Image.new("RGB", (32, 32)) if family == "qwen3_vl" else frames[0],
                        "min_pixels": 1024 if family == "qwen3_vl" else 16,
                        "max_pixels": 1024 if family == "qwen3_vl" else 64,
                    },
                    {"type": "text", "text": "What happens?"},
                ],
            }
        ]
    }
    for task in ({"question": "What happens?"}, image):
        answers, losses = runtime.generate_tasks([task], generation_config=generation, vision_config=vision)
        assert len(answers) == len(losses) == 1 and torch.isfinite(torch.tensor(losses)).all()
    expected = runtime.generate_tasks([video], generation_config=generation, vision_config=vision)
    cached_vision = {
        **vision,
        "video_feature_cache_size": 4,
        "video_processor_cache_size": 1,
        "video_prefix_cache_size": 1,
    }
    first = runtime.generate_tasks([video], generation_config=generation, vision_config=cached_vision)
    entry = next(iter(runtime._video_prefix_cache.values()))["past_key_values"]
    before = [(layer.keys.clone(), layer.values.clone()) for layer in entry.layers]
    second = runtime.generate_tasks([video], generation_config=generation, vision_config=cached_vision)
    assert first[0] == second[0] == expected[0]
    torch.testing.assert_close(torch.tensor(first[1]), torch.tensor(expected[1]), rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(torch.tensor(second[1]), torch.tensor(first[1]), rtol=0, atol=0)
    assert runtime._video_prefix_cache_hits == 1
    for layer, (keys, values) in zip(entry.layers, before, strict=True):
        assert torch.equal(layer.keys, keys) and torch.equal(layer.values, values)
    with torch.no_grad():
        runtime.model.model.language_model.layers[0].self_attn.q_proj.lora_B.weight.add_(0.001)
    runtime._refresh_adapter_caches()
    assert not runtime._video_prefix_cache and not runtime._video_feature_cache
    assert runtime.adapter_deployment["state_modified_since_load"]


@pytest.mark.parametrize("kind", ["missing", "unexpected", "canonical_collision", "tied_disagreement"])
def test_source_tensor_inventory_fails_before_weight_copy(tmp_path, monkeypatch, kind):
    from cosmos_framework.posttrain.checkpoint.adapters import snapshot_identity
    from cosmos_framework.posttrain.model import weight_loader

    base, portable, _, _ = artifact(tmp_path, monkeypatch, tied=True)
    identity = json.loads((portable / "adapter_manifest.json").read_text())["identity"]
    path = base / "model.safetensors"
    state = load_file(path)
    if kind == "missing":
        del state["model.layers.0.self_attn.q_proj.weight"]
    elif kind == "unexpected":
        state["unrelated.weight"] = torch.ones(1)
    elif kind == "canonical_collision":
        state["model.language_model.layers.0.self_attn.q_proj.weight"] = state[
            "model.layers.0.self_attn.q_proj.weight"
        ].clone()
    else:
        state["lm_head.weight"] = state["model.embed_tokens.weight"].clone() + 1
    save_file(state, path)
    identity["base"] = snapshot_identity(
        str(base), weights=True
    )  # Deliberately malformed NEW test fixture, not historical evidence.
    monkeypatch.setattr(
        weight_loader, "load_weights", lambda *a, **k: pytest.fail("invalid inventory reached weight copy")
    )
    with pytest.raises(ValueError, match="missing portable|unexpected or conflicting|conflicting tied"):
        portable_qwen_vl.load_base(
            str(base), transformers.AutoConfig.from_pretrained(base), torch.device("cpu"), identity
        )


def test_foreign_processor_rejected_before_base_allocation(tmp_path, monkeypatch):
    base, portable, _, _ = artifact(tmp_path, monkeypatch)
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(
        portable_qwen_vl, "load_base", lambda *a: pytest.fail("foreign processor reached base allocation")
    )
    with pytest.raises(ValueError, match="processor must match"):
        load(base, portable, monkeypatch)


def test_target_count_mismatch_rejected_on_meta_before_base_copy(tmp_path, monkeypatch):
    from cosmos_framework.posttrain.model import weight_loader

    base, portable, _, _ = artifact(tmp_path, monkeypatch)
    path = portable / "adapter_manifest.json"
    record = json.loads(path.read_text())
    record["identity"]["resolved"]["targets"][0]["trainable_count"] += 1
    path.write_text(json.dumps(record))
    monkeypatch.setattr(weight_loader, "load_weights", lambda *a, **k: pytest.fail("invalid target reached base copy"))
    with pytest.raises(ValueError, match="target/extra identity mismatch before base materialization"):
        load(base, portable, monkeypatch)


@pytest.mark.parametrize("suffix", ["magnitude", "initial_A", "lora_B.weight"])
def test_incomplete_method_state_is_never_reinitialized(tmp_path, monkeypatch, suffix):
    base, portable, _, _ = artifact(tmp_path, monkeypatch)
    path = portable / "adapter.safetensors"
    state = load_file(path)
    del state[next(name for name in state if name.endswith(suffix))]
    save_file(state, path)
    manifest = portable / "adapter_manifest.json"
    record = json.loads(manifest.read_text())
    record["weights_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(record))
    monkeypatch.setattr(portable_qwen_vl, "load_base", lambda *a: pytest.fail("partial state reached construction"))
    with pytest.raises(ValueError, match="incomplete/extra adapter state"):
        load(base, portable, monkeypatch)


def test_live_config_and_batch_mutations_fail_before_forward(tmp_path, monkeypatch):
    base, portable, _, config = artifact(tmp_path, monkeypatch)
    model, _, _ = load(base, portable, monkeypatch)
    with torch.no_grad(), pytest.raises(ValueError, match="batch-one"):
        model(input_ids=data(config)[0]["input_ids"].repeat(2, 1))
    model.config.text_config.max_position_embeddings += 1
    with torch.no_grad(), pytest.raises(ValueError, match="ownership"):
        model(**data(config)[0])
