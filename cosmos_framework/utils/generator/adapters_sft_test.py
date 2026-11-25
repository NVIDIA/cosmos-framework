# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Adapter SFT integration: recipes and preflight, HF and native builders, checkpoints and exports.

These tests drive the posttrain SFT hooks. The L0 tests run with Transformers 4 (the repository environment,
selected by the GPU CI unit-test job) and in the posttrain SFT environment (Transformers >= 5.8.1,
``docker/posttrain/environment`` with the ``sft`` group).
``test_hf_qwen3_vl_adapters_match_stock_and_merged_forward`` needs Transformers 4 (the HF Qwen3-VL forward patch).
The L1 end-to-end recipe runs need Transformers >= 5.8.1 (SFT requires it) and skip elsewhere::

    uv run pytest --num-gpus=0 -m "L0 or L1" cosmos_framework/utils/generator/adapters_sft_test.py
    PYTHONPATH=. "$SFT_PYTHON" -m pytest --num-gpus=0 -m "L0 or L1" cosmos_framework/utils/generator/adapters_sft_test.py

(the SFT environment also needs ``pytest-env`` and ``pytest-custom-exit-code`` for the root ``conftest.py``).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from cosmos_framework.utils.generator.adapter_artifacts import read_adapter_artifact, write_adapter_artifact
from cosmos_framework.utils.generator.adapter_config import AdapterComponentConfig, AdapterConfig, AdapterSetup
from cosmos_framework.utils.generator.adapters import AdapterLinear, adapter_state_spec

pytestmark = [pytest.mark.CPU]

RECIPES = (
    "cosmos_framework/posttrain/configs/sft_qwen3_vl_8b_robovqa_adapters.py",
    "cosmos_framework/posttrain/configs/sft_qwen3p8_27b_adapters.py",
)
METHODS = ("lora", "rslora", "dora", "rsdora", "pissa", "rspissa")


# Posttrain is imported inside the tests, so an import failure there fails these tests, not the collection.
def export_adapters(argv: list[str]) -> None:
    from cosmos_framework.posttrain.checkpoint.tools.export_adapters import main

    main(argv)


def checkpoint_adapter_state(checkpoint_dir: str) -> Any:
    from cosmos_framework.posttrain.checkpoint.adapters import checkpoint_adapter_state as read

    return read(checkpoint_dir)


def require_resumable_adapters(checkpoint_dir: str, identity: Any, training: Any = None) -> None:
    from cosmos_framework.posttrain.checkpoint.adapters import require_resumable_adapters as check

    check(checkpoint_dir, identity, training)


def builder_with_keyword_arguments(model_id: str, cfg: Any, pdims: Any = None, **kwargs: Any) -> Any:
    """A native builder that forwards ``**kwargs`` (accepted by the adapter preflight)."""
    raise NotImplementedError


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """These are CPU tests: keep the builders from materializing on a GPU when the host has one."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


# Tiny models and builders, shared with ``adapters_sft_distributed_test.py``.


def tiny_text(path: Path, model_type: str = "qwen3", *, seed: int = 0, dtype: torch.dtype = torch.float32) -> Path:
    """Save a tiny Hugging Face Qwen2/Qwen3 causal LM with tied embeddings."""
    from transformers import AutoModelForCausalLM, Qwen2Config, Qwen3Config

    common = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        tie_word_embeddings=True,
    )
    config = Qwen3Config(head_dim=16, **common) if model_type == "qwen3" else Qwen2Config(**common)
    torch.manual_seed(seed)
    AutoModelForCausalLM.from_config(config).to(dtype).save_pretrained(path)
    return path


def text_adapters(**pissa_overrides: Any) -> AdapterConfig:
    """DoRA (rank-stabilized) on q_proj, PiSSA on down_proj and LoRA with dropout on one v_proj."""
    pissa = dict(targets=["model.layers.*.mlp.down_proj"], method="pissa", rank=2, alpha=4.0, lr_multiplier=0.5)
    pissa.update(weight_decay=0.1, **pissa_overrides)
    return AdapterConfig(
        components=[
            AdapterComponentConfig(
                targets=["model.layers.*.self_attn.q_proj"], method="dora", rank=4, alpha=8.0, use_rslora=True
            ),
            AdapterComponentConfig(**pissa),
            AdapterComponentConfig(targets=["model.layers.1.self_attn.v_proj"], rank=4, alpha=4.0, dropout=0.1),
        ]
    )


def build_hf(
    path: Path,
    adapters: AdapterConfig | AdapterSetup | None,
    *,
    pdims: Any = None,
    dtype: str = "float32",
    reduce: str = "float32",
    grad_ckpt: bool = False,
) -> Any:
    """``build_hf_model`` with storage == compute ``dtype`` and an ``AdapterSetup``."""
    from cosmos_framework.posttrain.config.build import Precision
    from cosmos_framework.posttrain.config.model import HFModelConfig
    from cosmos_framework.posttrain.model.hf import build_hf_model

    if isinstance(adapters, AdapterConfig):
        adapters = AdapterSetup(adapters)
    precision = Precision(master=dtype, param=dtype, reduce=reduce, transfer=dtype)
    return build_hf_model(
        HFModelConfig(name_or_path=str(path), dtype=dtype, attn_impl="sdpa", grad_ckpt=grad_ckpt),
        pdims,
        precision,
        adapters=adapters,
    )


def tiny_qwen3_vl(path: Path, tokenizer: Any = None, *, tie_word_embeddings: bool = False) -> Path:
    """Save a tiny Qwen3-VL (bfloat16 weights) for the native builder."""
    from transformers import AutoModelForImageTextToText

    config = tiny_vlm_config(tokenizer, "qwen3_vl")
    config.tie_word_embeddings = tie_word_embeddings
    torch.manual_seed(0)
    AutoModelForImageTextToText.from_config(config).to(torch.bfloat16).save_pretrained(path)
    return path


def tiny_vlm_config(tokenizer: Any, model_type: str) -> Any:
    """A tiny Qwen3-VL (native seat) or Qwen3.5 (HF seat) with the local test tokenizer's media tokens."""
    text = dict(
        vocab_size=len(tokenizer) if tokenizer is not None else 64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
    )
    vision = dict(
        depth=2,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=32,
        num_position_embeddings=16,
    )
    names = ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id")
    pads = ("<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>")
    if tokenizer is None:
        tokens = dict(zip(names, (61, 62, 60, 63)))
    else:
        tokens = {name: tokenizer.convert_tokens_to_ids(pad) for name, pad in zip(names, pads)}
    if model_type == "qwen3_vl":
        text["rope_scaling"] = {"rope_type": "default", "mrope_section": [2, 3, 3], "mrope_interleaved": True}
        vision["deepstack_visual_indexes"] = [0]
        from transformers import Qwen3VLConfig

        return Qwen3VLConfig(text_config=text, vision_config=vision, tie_word_embeddings=False, **tokens)
    text.update(
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        layer_types=["linear_attention", "full_attention"],
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0,
            "mrope_section": [2, 3, 3],
        },
    )
    from transformers import Qwen3_5Config

    return Qwen3_5Config(text_config=text, vision_config=vision, **tokens)


@pytest.fixture
def restore_ray_client_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """The local SFT runtime starts Ray, whose ``ray._private.client_mode_hook`` sets ``RAY_CLIENT_MODE``.

    Record the variable's absence so it is removed again after the test (the root conftest refuses tests that
    change the environment).
    """
    monkeypatch.setenv("RAY_CLIENT_MODE", "0")
    monkeypatch.delenv("RAY_CLIENT_MODE")


def _recipe_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str, model_type: str = "qwen3_vl") -> Path:
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps({"model_type": model_type}))
    for split in ("train", "validation"):
        (tmp_path / f"{split}.jsonl").write_text("{}\n")
    monkeypatch.setenv("SFT_MODEL_PATH", str(model_dir))
    monkeypatch.setenv("SFT_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("SFT_ADAPTER_METHOD", method)
    return model_dir


@pytest.mark.L0
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("recipe", RECIPES)
def test_adapter_recipes_pass_root_and_sft_validation(
    recipe: str, method: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cosmos_framework.posttrain.config.config import PosttrainConfig
    from cosmos_framework.posttrain.entrypoints.sft import validate_config
    from cosmos_framework.utils.config import load_config

    _recipe_inputs(tmp_path, monkeypatch, method, "qwen3_vl" if recipe == RECIPES[0] else "qwen3_5")
    config = load_config(recipe, ["--", "job.name=adapter_check"])
    assert isinstance(config, PosttrainConfig)
    config.__attrs_post_init__()  # the root config's backend validation
    assert config.model_backend in ("native", "hf")
    rl = validate_config(config)
    assert rl.adapters is not None and rl.optimizer.name == "adamw"
    for component in rl.adapters.components:
        assert component.method == method.removeprefix("rs")
        assert component.use_rslora == method.startswith("rs")


@pytest.mark.L0
@pytest.mark.parametrize("model_type", ["qwen2", "qwen3"])
def test_text_architectures_pass_sft_validation(
    model_type: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cosmos_framework.posttrain.entrypoints.sft import validate_config
    from cosmos_framework.utils.config import load_config

    _recipe_inputs(tmp_path, monkeypatch, "lora", model_type)
    config = load_config(RECIPES[1], ["--", "rl.processor_kwargs={}"])
    config.rl.adapters = text_adapters()
    rl = validate_config(config)
    assert [c.targets for c in rl.adapters.components][0] == ["model.layers.*.self_attn.q_proj"]


@pytest.mark.L0
def test_adapter_preflight_refusals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from cosmos_framework.posttrain.entrypoints.sft import validate_config
    from cosmos_framework.utils.config import load_config

    _recipe_inputs(tmp_path, monkeypatch, "lora", "qwen3_5")
    hf = RECIPES[1]
    for override, message in (
        ("rl.optimizer.name=fusedadam", "optimizer.name='adamw'"),
        ("rl.optimizer.decay_groups=hf", "decay_groups"),
        ("rl.optimizer.train_keys=[lora_]", "train_keys"),
        ("rl.reduce_dtype=bfloat16", "set the reduce dtype to float32"),
        ("rl.master_dtype=float32", r"storage \(float32\) must equal compute \(bfloat16\)"),
    ):
        with pytest.raises(ValueError, match=message):
            validate_config(load_config(hf, ["--", override]))
    config = load_config(hf, [])
    config.rl.optimizer.lr_multipliers = {"model.visual": 0.1}
    with pytest.raises(ValueError, match="lr_multipliers"):
        validate_config(config)
    (tmp_path / "model" / "config.json").write_text(json.dumps({"model_type": "llama"}))
    with pytest.raises(ValueError, match="no adapter binding for architecture 'llama'"):
        validate_config(load_config(hf, []))

    _recipe_inputs(tmp_path, monkeypatch, "lora", "qwen3_vl")
    from cosmos_framework.posttrain.model.adapters import validate_sft_adapters

    native = load_config(RECIPES[0], [])
    native.model.builder = "cosmos_framework.posttrain.model.qwen3_vl.build_qwen3_vl_native_value_model"
    source = (str(tmp_path / "model"), False)  # the test builder below has no registered SFT source
    with pytest.raises(ValueError, match="build_qwen3_vl_native_value_model does not accept adapters"):
        validate_sft_adapters(native, native.rl, source)
    native.model.builder = f"{__name__}.builder_with_keyword_arguments"
    validate_sft_adapters(native, native.rl, source)  # a builder taking **kwargs is accepted
    unadapted = load_config(RECIPES[0], ["--", f"rl.adapter_warm_start={tmp_path}"])
    unadapted.rl.adapters = None
    with pytest.raises(ValueError, match="requires adapters"):
        validate_config(unadapted)
    with pytest.raises(ValueError, match="is not a portable adapter"):  # checked before any rank loads the base
        validate_config(load_config(RECIPES[0], ["--", f"rl.adapter_warm_start={tmp_path}"]))
    for architecture in ("qwen3_vl", "qwen3"):
        adapter = tmp_path / f"adapter_{architecture}"
        _portable_adapter(adapter, architecture)
        opts = ["--", f"rl.adapter_warm_start={adapter}", "rl.resume=auto"]
        if architecture == "qwen3_vl":
            # warm start with any rl.resume is valid: a run that finds its own checkpoint resumes from it instead.
            validate_config(load_config(RECIPES[0], opts))
        else:
            with pytest.raises(ValueError, match="was trained on 'qwen3', not 'qwen3_vl'"):
                validate_config(load_config(RECIPES[0], opts))


def _portable_adapter(path: Path, architecture: str) -> None:
    from cosmos_framework.utils.generator.adapter_artifacts import IDENTITY_FORMAT

    row = {
        "name": "model.layers.0.self_attn.q_proj",
        "component": "language",
        "shape": [4, 4],
        "method": "lora",
        "rank": 2,
        "alpha": 4.0,
        "use_rslora": False,
    }
    identity = {
        "format": IDENTITY_FORMAT,
        "base_dtype": "bfloat16",
        "base_fingerprint": "0" * 64,
        "architecture": architecture,
        "targets": [row],
    }
    write_adapter_artifact(str(path), identity, {n: torch.zeros(s) for n, s in adapter_state_spec(identity).items()})


@pytest.mark.L0
def test_legacy_native_tree_refuses_adapters() -> None:
    from types import SimpleNamespace

    from cosmos_framework.posttrain.config.build import build_backbone

    config = SimpleNamespace(model_backend="native", model=SimpleNamespace(), model_hf=None)
    with pytest.raises(ValueError, match="NativePolicyBuildSpec"):
        build_backbone(config, adapters=AdapterSetup(text_adapters()))


@pytest.mark.L0
@pytest.mark.parametrize("model_type", ["qwen2", "qwen3"])
def test_hf_adapters_train_export_and_warm_start(model_type: str, tmp_path: Path) -> None:
    from transformers import AutoModelForCausalLM

    from cosmos_framework.utils.generator.adapter_training import adapter_optimizer_groups

    base = tiny_text(tmp_path / "base", model_type)
    ids = torch.randint(0, 64, (2, 9), generator=torch.Generator().manual_seed(0))
    reference = AutoModelForCausalLM.from_pretrained(base)(input_ids=ids).logits
    net = build_hf(base, text_adapters()).net.eval()
    assert isinstance(net.model.layers[0].self_attn.q_proj, AdapterLinear)
    torch.testing.assert_close(net(input_ids=ids).logits, reference, atol=1e-5, rtol=1e-5)

    groups = adapter_optimizer_groups(net, lr=1e-3, weight_decay=0.01)
    assert groups[0].lr == 1e-3  # the first group trains at the base rate
    assert sorted((g.lr, g.weight_decay, len(g.parameter_names)) for g in groups) == [
        (5e-4, 0.1, 4),  # PiSSA: lora_A/B of two down_proj targets, explicit decay
        (1e-3, 0.0, 2),  # DoRA magnitudes are not decayed
        (1e-3, 0.01, 6),  # DoRA and LoRA factors inherit the optimizer's decay
    ]

    _perturb(net)
    adapted = net(input_ids=ids).logits
    identity = net.adapter_identity
    assert identity["architecture"] == model_type and identity["base_dtype"] == "float32"
    state = net.state_dict()
    write_adapter_artifact(str(tmp_path / "adapter"), identity, {n: state[n] for n in adapter_state_spec(identity)})

    export_adapters(
        ["--adapter", str(tmp_path / "adapter"), "--merged", "--base", str(base), "--out", str(tmp_path / "merged")]
    )
    merged = AutoModelForCausalLM.from_pretrained(tmp_path / "merged")
    assert not any(isinstance(m, AdapterLinear) for m in merged.modules())
    torch.testing.assert_close(merged(input_ids=ids).logits, adapted, atol=1e-4, rtol=1e-4)

    # Training-only settings may change between a run and its warm start; tensor-defining ones may not.
    retrained = text_adapters(initialization="svd_lowrank")
    retrained.components[0].lr_multiplier, retrained.components[2].dropout = 2.0, 0.0
    warm = build_hf(base, AdapterSetup(retrained, warm_start=str(tmp_path / "adapter"))).net.eval()
    torch.testing.assert_close(warm(input_ids=ids).logits, adapted)
    # A different base storage dtype is allowed too: the adapter tensors are float32 either way.
    bf16 = build_hf(base, AdapterSetup(text_adapters(), warm_start=str(tmp_path / "adapter")), dtype="bfloat16").net
    assert bf16.adapter_identity["base_dtype"] == "bfloat16"
    torch.testing.assert_close(
        bf16.state_dict()["model.layers.0.self_attn.q_proj.lora_A.weight"],
        state["model.layers.0.self_attn.q_proj.lora_A.weight"],
    )
    changed = text_adapters()
    changed.components[0].rank = 2
    with pytest.raises(ValueError, match=r"model\.layers\.0\.self_attn\.q_proj\.rank: 4 \(saved\) != 2"):
        build_hf(base, AdapterSetup(changed, warm_start=str(tmp_path / "adapter")))


def _perturb(net: torch.nn.Module) -> None:
    generator = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for parameter in net.parameters():
            if parameter.requires_grad:
                parameter.add_(0.05 * torch.randn(parameter.shape, generator=generator))


@pytest.mark.L0
def test_hf_adapter_refusals_and_checkpoint_restore_placeholders(tmp_path: Path) -> None:
    from cosmos_framework.posttrain.config.build import Precision
    from cosmos_framework.posttrain.config.model import HFModelConfig
    from cosmos_framework.posttrain.model.hf import build_hf_model

    base = tiny_text(tmp_path / "base")
    hf = HFModelConfig(name_or_path=str(base), dtype="float32", attn_impl="sdpa")
    split = Precision(master="float32", param="bfloat16", reduce="float32", transfer="bfloat16")
    with pytest.raises(ValueError, match=r"storage \(float32\) must equal compute \(bfloat16\)"):
        build_hf_model(hf, None, split, adapters=AdapterSetup(text_adapters()))
    with pytest.raises(ValueError, match="reduce dtype to float32"):
        build_hf(base, text_adapters(), reduce="bfloat16")
    with pytest.raises(ValueError, match="not both"):
        AdapterSetup(text_adapters(), from_checkpoint=True, warm_start=str(tmp_path))
    net = build_hf(base, AdapterSetup(text_adapters(), from_checkpoint=True)).net
    assert all(not m.lora_B.weight.any() for m in net.modules() if isinstance(m, AdapterLinear))


@pytest.mark.L0
def test_native_qwen3_vl_refuses_tied_embeddings_and_trains_in_bfloat16(tmp_path: Path) -> None:
    from cosmos_framework.posttrain.config.model import Qwen3VLModelConfig
    from cosmos_framework.posttrain.model.qwen3_vl import build_qwen3_vl_model

    adapters = AdapterSetup(
        AdapterConfig(
            components=[
                AdapterComponentConfig(targets=["model.language_model.layers.*.self_attn.q_proj"], method="dora"),
                AdapterComponentConfig(
                    component="vision", targets=["model.visual.blocks.*.attn.qkv"], layers=[-1], method="pissa", rank=2
                ),
            ]
        )
    )
    tied = tiny_qwen3_vl(tmp_path / "tied", tie_word_embeddings=True)
    with pytest.raises(ValueError, match="tie_word_embeddings=False"):
        build_qwen3_vl_model(
            Qwen3VLModelConfig(name_or_path=str(tied), dtype="bfloat16", attn_impl="sdpa"), adapters=adapters
        )

    base = tiny_qwen3_vl(tmp_path / "untied")
    cfg = Qwen3VLModelConfig(
        name_or_path=str(base), dtype="bfloat16", attn_impl="sdpa", activation_checkpointing="full"
    )
    net = build_qwen3_vl_model(cfg, adapters=adapters).net
    assert net.adapter_identity["base_dtype"] == "bfloat16"
    ids = torch.randint(0, 60, (2, 8), generator=torch.Generator().manual_seed(0))
    trainable = {name: p for name, p in net.named_parameters() if p.requires_grad}
    before = {name: p.detach().clone() for name, p in trainable.items()}
    optimizer = torch.optim.AdamW(trainable.values(), lr=1e-2)
    net(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits.float().square().mean().backward()
    optimizer.step()
    for name, parameter in net.named_parameters():
        if parameter.requires_grad:
            assert parameter.dtype == torch.float32 and parameter.grad is not None, name
        else:
            assert parameter.dtype == torch.bfloat16, name
    assert any(not torch.equal(p, before[name]) for name, p in trainable.items())  # the step moved the adapters


@pytest.mark.L0
@pytest.mark.parametrize("layout", ["consolidated", "dcp"])
def test_checkpoint_extraction_resume_check_and_cli(tmp_path: Path, layout: str) -> None:
    from safetensors.torch import load_file

    from cosmos_framework.utils.generator.adapters import training_identity

    base = tiny_text(tmp_path / "base")  # a float32 snapshot, trained below with bfloat16 storage
    net = build_hf(base, text_adapters(), dtype="bfloat16").net
    _perturb(net)
    identity, training = net.adapter_identity, training_identity(net.adapter_manifest)
    state = {name: tensor.detach().clone() for name, tensor in net.state_dict().items()}
    checkpoint = tmp_path / "iter_000002"
    checkpoint.mkdir()
    torch.save({"objective_identity": {"adapters": identity, "adapter_training": training}}, checkpoint / "trainer.pt")
    if layout == "consolidated":
        torch.save({"net": state}, checkpoint / "model.pt")
    else:
        import torch.distributed.checkpoint as dcp

        dcp.save(state, checkpoint_id=str(checkpoint / "model"), no_dist=True)
    extracted_identity, extracted = checkpoint_adapter_state(str(checkpoint))
    assert extracted_identity == identity
    assert set(extracted) == set(adapter_state_spec(identity))
    for name, tensor in extracted.items():
        torch.testing.assert_close(tensor, state[name], atol=0, rtol=0)

    require_resumable_adapters(str(checkpoint), identity, training)
    other = build_hf(base, text_adapters(alpha=8), dtype="bfloat16").net.adapter_identity
    with pytest.raises(
        ValueError, match=r"different adapters: model\.layers\.0\.mlp\.down_proj\.alpha: 4\.0 \(saved\) != 8\.0"
    ):
        require_resumable_adapters(str(checkpoint), other)
    retrained = {name: {**settings, "lr_multiplier": 2.0} for name, settings in training.items()}
    with pytest.raises(
        ValueError, match=r"different adapter training settings.*q_proj\.lr_multiplier: 1\.0 \(saved\) != 2\.0"
    ):
        require_resumable_adapters(str(checkpoint), identity, retrained)

    export_adapters(["--checkpoint", str(checkpoint), "--out", str(tmp_path / "adapter")])
    assert read_adapter_artifact(str(tmp_path / "adapter"))[0] == identity
    with pytest.raises(SystemExit):
        export_adapters(["--checkpoint", str(checkpoint), "--out", str(tmp_path / "x"), "--dtype", "bfloat16"])
    merged = tmp_path / "merged"
    export_adapters(["--checkpoint", str(checkpoint), "--merged", "--base", str(base), "--out", str(merged)])
    saved, exported = (json.loads((path / "config.json").read_text()) for path in (base, merged))
    dtype_key = "dtype" if "dtype" in saved else "torch_dtype"
    assert saved[dtype_key] == "float32" and exported[dtype_key] == "bfloat16"  # default: the training storage dtype
    assert load_file(str(merged / "model.safetensors"))["model.norm.weight"].dtype == torch.bfloat16


@pytest.mark.L0
def test_checkpoint_without_adapters_is_refused(tmp_path: Path) -> None:
    checkpoint = tmp_path / "iter_000001"
    checkpoint.mkdir()
    torch.save({"objective_identity": {}}, checkpoint / "trainer.pt")
    torch.save({"net": {}}, checkpoint / "model.pt")
    with pytest.raises(ValueError, match="not trained with adapters"):
        checkpoint_adapter_state(str(checkpoint))
    with pytest.raises(ValueError, match="trained without adapters"):
        require_resumable_adapters(str(checkpoint), {"format": "cosmos-adapter-v1"})


@pytest.mark.L1
@pytest.mark.parametrize(
    ("recipe", "model_type", "method", "opts"),
    [
        (
            RECIPES[0],
            "qwen3_vl",
            "rspissa",
            [
                "model.model_config.dtype=float32",
                "rl.adapters.components.2.layers=[1]",
                "rl.processor_kwargs.videos_kwargs.do_sample_frames=True",
            ],
        ),
        (  # The recipe's own bfloat16 base.
            RECIPES[0],
            "qwen3_vl",
            "dora",
            ["rl.adapters.components.2.layers=[1]", "rl.processor_kwargs.videos_kwargs.do_sample_frames=True"],
        ),
        (
            RECIPES[1],
            "qwen3_5",
            "rsdora",
            [
                "rl.master_dtype=float32",
                "rl.param_dtype=float32",
                "rl.processor_kwargs.images_kwargs.size.shortest_edge=16",
                "rl.processor_kwargs.images_kwargs.size.longest_edge=64",
                "rl.processor_kwargs.videos_kwargs.size.shortest_edge=16",
                "rl.processor_kwargs.videos_kwargs.size.longest_edge=64",
                "rl.processor_kwargs.videos_kwargs.num_frames=2",
            ],
        ),
    ],
)
def test_adapter_recipe_trains_resumes_and_exports(
    recipe: str,
    model_type: str,
    method: str,
    opts: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    restore_ray_client_mode: None,
) -> None:
    """A recipe end to end on a tiny model: SFT with saves, exact resume, both exports, then a warm-started run
    that is requeued (``rl.resume=auto``) and resumes from its own checkpoint instead of the warm start."""
    pytest.importorskip("transformers", minversion="5.8.1")
    from safetensors.torch import load_file
    from transformers import AutoModelForImageTextToText

    from cosmos_framework.posttrain.entrypoints.sft import run
    from cosmos_framework.posttrain.scripts.prepare_sft_example import prepare_example
    from cosmos_framework.posttrain.tests.helpers.sft_packer import local_processor, local_tokenizer
    from cosmos_framework.posttrain.utils.entry import out_dir
    from cosmos_framework.utils.config import load_config

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr("cosmos_framework.posttrain.utils.dist.init_local_process_group", lambda: None)
    monkeypatch.setattr("cosmos_framework.posttrain.utils.report.make_reporters", lambda *a, **kw: [])
    monkeypatch.setenv("IMAGINAIRE_OUTPUT_ROOT", str(tmp_path / "outputs"))
    tokenizer = local_tokenizer()
    model_path, data_path = tmp_path / "model", tmp_path / "data"
    AutoModelForImageTextToText.from_config(tiny_vlm_config(tokenizer, model_type)).save_pretrained(model_path)
    local_processor(tokenizer).save_pretrained(model_path)
    prepare_example(data_path, train_samples=4, validation_samples=4)
    monkeypatch.setenv("SFT_MODEL_PATH", str(model_path))
    monkeypatch.setenv("SFT_DATA_ROOT", str(data_path))
    monkeypatch.setenv("SFT_ADAPTER_METHOD", method)
    opts = [
        "rl.max_steps=2",
        "rl.micro_batch_size=2",
        "rl.grad_accum=1",
        "rl.save_every=1",
        "rl.keep_optim_last=0",
        "rl.resume=off",
        *opts,
    ]
    config = load_config(recipe, ["--", *opts])
    summary = run(config, recipe, opts, out_dir(config))
    assert summary["final_step"] == 2
    checkpoint = next((tmp_path / "outputs").rglob("iter_000001"))
    resume_opts = [item for item in opts if not item.startswith("rl.resume=")]
    resume_opts += [f"rl.resume={checkpoint}", "job.name=resumed"]
    resumed_config = load_config(recipe, ["--", *resume_opts])
    resumed = run(resumed_config, recipe, resume_opts, out_dir(resumed_config))
    assert resumed["start"] == 1 and resumed["final_loss"] == pytest.approx(summary["final_loss"], rel=1e-6)
    # Step 2 after the resume used the restored adapters and optimizer moments: identical adapter tensors.
    finals = sorted((tmp_path / "outputs").rglob("iter_000002"), key=lambda path: "resumed" in str(path))
    (_, uninterrupted), (_, continued) = (checkpoint_adapter_state(str(path)) for path in finals)
    for name, tensor in uninterrupted.items():
        torch.testing.assert_close(continued[name], tensor, atol=0, rtol=0)

    export_adapters(["--checkpoint", str(checkpoint), "--out", str(tmp_path / "adapter")])
    identity, _ = read_adapter_artifact(str(tmp_path / "adapter"))
    assert identity["architecture"] == model_type
    export_adapters(
        [
            "--adapter",
            str(tmp_path / "adapter"),
            "--merged",
            "--base",
            str(model_path),
            "--out",
            str(tmp_path / "merged"),
        ]
    )
    merged, base = load_file(tmp_path / "merged" / "model.safetensors"), load_file(model_path / "model.safetensors")
    assert merged.keys() == base.keys()
    stored = getattr(torch, identity["base_dtype"])  # The default export dtype is the training storage dtype.
    changed = {name for name in base if not torch.equal(merged[name].float(), base[name].to(stored).float())}
    assert changed and changed <= {f"{row['name']}.weight" for row in identity["targets"]}
    assert type(AutoModelForImageTextToText.from_pretrained(tmp_path / "merged")).__name__.startswith(
        {"qwen3_vl": "Qwen3VL", "qwen3_5": "Qwen3_5"}[model_type]
    )

    warm_opts = [item for item in opts if not item.startswith("rl.resume=")]
    warm_opts += ["rl.resume=auto", f"rl.adapter_warm_start={tmp_path / 'adapter'}", "job.name=warm"]
    warm_config = load_config(recipe, ["--", *warm_opts])
    warm = run(warm_config, recipe, warm_opts, out_dir(warm_config))
    assert warm["start"] == 0 and warm["final_step"] == 2
    requeued_config = load_config(recipe, ["--", *warm_opts])
    requeued = run(requeued_config, recipe, warm_opts, out_dir(requeued_config))
    assert requeued["start"] == 2


@pytest.mark.L0
def test_hf_qwen3_vl_adapters_match_stock_and_merged_forward(tmp_path: Path) -> None:
    """All three components on a VLM, through the HF body (Qwen3-VL SFT itself uses the native builder)."""
    import transformers
    from packaging.version import Version
    from transformers import AutoModelForImageTextToText, Qwen3VLConfig

    if Version(transformers.__version__) >= Version("5"):
        pytest.skip("the HF Qwen3-VL text-only forward patch supports Transformers 4 only")

    config = Qwen3VLConfig(
        text_config=dict(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            rope_scaling={"rope_type": "default", "mrope_section": [2, 3, 3], "mrope_interleaved": True},
        ),
        vision_config=dict(
            depth=2,
            hidden_size=16,
            intermediate_size=32,
            num_heads=2,
            patch_size=2,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=32,
            num_position_embeddings=16,
            deepstack_visual_indexes=[0],
        ),
        image_token_id=61,
        video_token_id=62,
        vision_start_token_id=60,
        vision_end_token_id=63,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    AutoModelForImageTextToText.from_config(config).save_pretrained(tmp_path / "base")
    adapters = AdapterConfig(
        components=[
            AdapterComponentConfig(
                component="language", targets=["model.language_model.layers.*.self_attn.q_proj"], method="pissa", rank=4
            ),
            AdapterComponentConfig(
                component="projector", targets=["model.visual.merger.linear_fc*"], method="dora", rank=4
            ),
            AdapterComponentConfig(component="vision", targets=["model.visual.blocks.*.attn.qkv"], layers=[-1], rank=2),
        ]
    )
    ids = torch.randint(0, 60, (2, 8), generator=torch.Generator().manual_seed(0))
    stock = AutoModelForImageTextToText.from_pretrained(tmp_path / "base")(input_ids=ids).logits
    net = build_hf(tmp_path / "base", adapters).net.eval()
    assert {row["component"] for row in net.adapter_identity["targets"]} == {"language", "projector", "vision"}
    torch.testing.assert_close(net(input_ids=ids).logits, stock, atol=1e-5, rtol=1e-5)
    _perturb(net)
    adapted = net(input_ids=ids).logits
    state = net.state_dict()
    identity = net.adapter_identity
    write_adapter_artifact(str(tmp_path / "adapter"), identity, {n: state[n] for n in adapter_state_spec(identity)})
    export_adapters(
        [
            "--adapter",
            str(tmp_path / "adapter"),
            "--merged",
            "--base",
            str(tmp_path / "base"),
            "--out",
            str(tmp_path / "merged"),
        ]
    )
    merged = AutoModelForImageTextToText.from_pretrained(tmp_path / "merged")
    torch.testing.assert_close(merged(input_ids=ids).logits, adapted, atol=1e-4, rtol=1e-4)


@pytest.mark.L0
def test_native_adapter_sharding_matches_apply_fsdp_units(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``_shard_with_adapters`` must wrap exactly the units ``parallelize_vlm.apply_fsdp`` wraps for dense Qwen3-VL."""
    import torch.distributed.fsdp as fsdp
    from accelerate import init_on_device
    from transformers import AutoConfig

    from cosmos_framework.configs.base.defaults.activation_checkpointing import ActivationCheckpointingConfig
    from cosmos_framework.configs.base.defaults.parallelism import ParallelismConfig
    from cosmos_framework.model.generator import parallelize_vlm
    from cosmos_framework.posttrain.model import qwen3_vl
    from cosmos_framework.utils.generator import parallelism

    model_config = AutoConfig.from_pretrained(tiny_qwen3_vl(tmp_path / "base"))
    config_cls, model_cls = qwen3_vl._native_qwen3_vl_classes("qwen3_vl")

    def wrapped_units(shard: Any) -> list[tuple[str, Any, Any]]:
        with init_on_device("meta", include_buffers=False):
            net = model_cls(qwen3_vl._coerce_native_config(config_cls, model_config, "sdpa"))
        shim = SimpleNamespace(model=net, sound_und=False, hf_config=model_config)
        slots = parallelize_vlm.apply_ac(shim, ActivationCheckpointingConfig(mode="full"))
        names = {id(module): name for name, module in net.named_modules()}
        calls: list[tuple[str, Any, Any]] = []

        def record(module: Any, **kwargs: Any) -> None:  # unit, mesh, resharding: what must not drift
            calls.append((names[id(module)], kwargs.get("mesh"), kwargs.get("reshard_after_forward", True)))

        monkeypatch.setattr(parallelize_vlm, "fully_shard", record)
        monkeypatch.setattr(fsdp, "fully_shard", record)
        shard(shim, net, slots)
        return calls

    mesh = object()
    monkeypatch.setattr(parallelize_vlm, "fsdp_mesh", lambda dims: mesh)
    monkeypatch.setattr(parallelism, "fsdp_mesh", lambda dims: mesh)
    dims = SimpleNamespace(dp_enabled=True)
    parallelism_config = ParallelismConfig(fsdp_master_dtype="bfloat16", fsdp_reduce_dtype="float32")
    stock = wrapped_units(
        lambda shim, net, slots: parallelize_vlm.apply_fsdp(shim, slots, dims, parallelism_config, "bfloat16")
    )
    adapted = wrapped_units(lambda shim, net, slots: qwen3_vl._shard_with_adapters(net, slots, dims))
    assert adapted == stock and len(stock) > 2 and all(call[1] is mesh for call in stock)


@pytest.mark.L0
def test_full_base_fit_and_missing_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from functools import partial

    from cosmos_framework.posttrain.entrypoints.sft import validate_config
    from cosmos_framework.utils.config import load_config
    from cosmos_framework.utils.generator import adapter_training

    base = tiny_text(tmp_path / "base")
    check = adapter_training.check_full_base_fits
    monkeypatch.setattr(adapter_training, "check_full_base_fits", partial(check, device_memory=50_000))
    with pytest.raises(ValueError, match="does not fit"):  # float32: about 70 KB of tensors
        build_hf(base, text_adapters())
    monkeypatch.setattr(adapter_training, "check_full_base_fits", partial(check, device_memory=2**30))
    assert build_hf(base, text_adapters()).net.adapter_identity["base_dtype"] == "float32"

    _recipe_inputs(tmp_path / "recipe", monkeypatch, "lora", "qwen3_5")
    (tmp_path / "recipe" / "model" / "config.json").unlink()
    with pytest.raises(ValueError, match="has no config.json"):
        validate_config(load_config(RECIPES[1], []))
