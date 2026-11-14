# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Original-factorized public reference; CPU fixtures are not GPU qualification."""

import copy
from inspect import signature
from types import MethodType

import pytest
import torch
import transformers

from cosmos_framework.posttrain.model.adapters import PosttrainAdapterLinear
from cosmos_framework.posttrain.tests.test_portable_qwen_vl import artifact, data
from cosmos_framework.utils.generator import factorized_inference, portable_adapters, portable_qwen_vl

PROFILE = portable_qwen_vl.STOCK_FACTORIZED_PROFILE
pytestmark = pytest.mark.skipif(transformers.__version__ != "5.8.1", reason="public reference pins Transformers5.8.1")


@pytest.fixture(autouse=True)
def matmul_policy():
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


def load(base, portable, monkeypatch):
    # Substitute only physical device admission for small CPU structure tests.
    monkeypatch.setattr(portable_adapters, "_validate_device", lambda device: None)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    return portable_adapters.load_portable_model(
        str(base),
        {"directory": str(portable), "execution_profile": PROFILE},
        dtype="float32",
        device="cpu",
        attn_implementation="sdpa",
    )


@pytest.mark.parametrize("method", ["lora", "dora", "pissa", "mixed"])
@pytest.mark.parametrize("rslora", [False, True])
def test_exact_original_forward_all_modalities_ownership_and_saved_initials(tmp_path, monkeypatch, method, rslora):
    base, portable, reference, config = artifact(
        tmp_path, monkeypatch, "qwen3_vl", method, rslora, tied=True, model_execution="qwen3_vl_hf_stock_v1"
    )
    monkeypatch.setattr(torch.linalg, "svd", lambda *a, **k: pytest.fail("reload must not decompose"))
    monkeypatch.setattr(torch, "svd_lowrank", lambda *a, **k: pytest.fail("reload must not decompose"))
    loaded, processor, provenance = load(base, portable, monkeypatch)
    assert type(processor) is transformers.Qwen3VLProcessor
    assert provenance["execution_profile"] == PROFILE
    assert provenance["identity"] == reference._posttrain_adapter_identity
    assert provenance["precision_contract"]["original_storage_equivalence"].startswith("not claimed")
    assert provenance["precision_contract"]["cache_consistent_numerics"].startswith("not claimed")
    assert loaded.get_input_embeddings().weight is loaded.get_output_embeddings().weight
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value, rtol=0, atol=0)
    owners = {n: (id(p), p.clone(), p.requires_grad) for n, p in loaded.named_parameters(remove_duplicate=False)}
    targets = [module for module in loaded.modules() if isinstance(module, PosttrainAdapterLinear)]
    assert all(module.forward.__func__ is PosttrainAdapterLinear.forward for module in targets)
    if method != "pissa":
        assert any(getattr(module.lora_dropout, "p", 0) > 0 for module in targets)
    rng = torch.get_rng_state().clone()
    with torch.no_grad():
        for batch in data(config):
            torch.testing.assert_close(loaded(**batch).logits, reference(**batch).logits, atol=0, rtol=0)
            assert torch.equal(
                loaded.generate(**batch, max_new_tokens=3, do_sample=False),
                reference.generate(**batch, max_new_tokens=3, do_sample=False),
            )
    assert torch.equal(rng, torch.get_rng_state())
    assert all(
        id(p) == owners[n][0] and torch.equal(p, owners[n][1]) and p.requires_grad == owners[n][2]
        for n, p in loaded.named_parameters(remove_duplicate=False)
    )
    # The separate stable-FP64 admission and its historical zero-dropout limit stay intact.
    if method != "pissa":
        with pytest.raises(ValueError, match="zero dropout"):
            factorized_inference.validate_identity(provenance["identity"])


@pytest.mark.parametrize(
    "mutation",
    ["dropout_train", "dropout_replace", "method", "scale", "precision", "profile", "forward", "child_forward"],
)
def test_changed_execution_rejected_before_cache_mutation(tmp_path, monkeypatch, mutation):
    from transformers.cache_utils import DynamicCache

    base, portable, _, config = artifact(tmp_path, monkeypatch, "qwen3_vl", model_execution="qwen3_vl_hf_stock_v1")
    loaded, _, _ = load(base, portable, monkeypatch)
    target = next(
        module
        for module in loaded.modules()
        if isinstance(module, PosttrainAdapterLinear) and getattr(module.lora_dropout, "p", 0) > 0
    )
    if mutation == "dropout_train":
        target.lora_dropout.train()
    elif mutation == "dropout_replace":
        target.lora_dropout = torch.nn.Dropout(target.lora_dropout.p).eval()
    elif mutation == "method":
        target.method = "lora" if target.method == "dora" else "dora"
    elif mutation == "scale":
        target._lora_alpha += 1
    elif mutation == "precision":
        loaded.bfloat16()
    elif mutation in ("forward", "child_forward"):
        owner = target if mutation == "forward" else target.lora_A
        original = owner.forward
        owner.forward = MethodType(lambda self, value: original(value) + 1, owner)
    else:
        loaded._cosmos_deployment_profile = factorized_inference.PROFILE
    cache = DynamicCache(config=loaded.config)
    with torch.no_grad(), pytest.raises(ValueError, match="eval|ownership"):
        loaded(**data(config)[0], past_key_values=cache)
    assert cache.get_seq_length() == 0


def test_autograd_autocast_static_cache_and_incompatible_artifact_are_rejected(tmp_path, monkeypatch):
    from transformers.cache_utils import StaticCache

    base, portable, _, config = artifact(tmp_path, monkeypatch, "qwen3_vl", model_execution="qwen3_vl_hf_stock_v1")
    loaded, _, provenance = load(base, portable, monkeypatch)
    batch = data(config)[0]
    with pytest.raises(ValueError, match="no_grad"):
        loaded(**batch)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16), pytest.raises(ValueError, match="autocast"):
        loaded(**batch)
    with torch.no_grad(), pytest.raises(ValueError, match="DynamicCache"):
        loaded(**batch, past_key_values=StaticCache(config=loaded.config, max_cache_len=16))
    for change in ["model_execution", "execution"]:
        bad = copy.deepcopy(provenance["identity"])
        if change == "model_execution":
            bad["resolved"][change] = portable_qwen_vl.PROFILE
        else:
            bad["resolved"]["targets"][0]["settings"][change] = "merged_weight_bf16_v1"
        with pytest.raises(ValueError):
            portable_qwen_vl.validate_identity(bad, profile=PROFILE)
    with pytest.raises(ValueError, match="effective-weight cache"):
        portable_adapters.PortableAdapterConfig("adapter", PROFILE, cache_effective_weights=True)


@pytest.mark.parametrize("kind", ["sliding", "lazy_sliding", "missing_layer", "offload"])
def test_foreign_dynamic_cache_rejected_before_mutation(tmp_path, monkeypatch, kind):
    from transformers.cache_utils import DynamicCache, DynamicSlidingWindowLayer

    base, portable, _, config = artifact(tmp_path, monkeypatch, "qwen3_vl", model_execution="qwen3_vl_hf_stock_v1")
    loaded, _, _ = load(base, portable, monkeypatch)
    cache = DynamicCache(config=config)
    if kind == "sliding":
        foreign = copy.deepcopy(config)
        foreign.text_config.sliding_window = 2
        foreign.text_config.layer_types = ["sliding_attention"] * foreign.text_config.num_hidden_layers
        cache = DynamicCache(config=foreign)
    elif kind == "lazy_sliding":
        cache = DynamicCache()
        cache.layer_class_to_replicate = DynamicSlidingWindowLayer
    elif kind == "missing_layer":
        cache.layers.pop()
    else:
        cache.offloading = True
    with torch.no_grad(), pytest.raises(ValueError, match="DynamicCache"):
        loaded(**data(config)[0], past_key_values=cache)
    assert cache.get_seq_length() == 0


def test_ordinary_lazy_cache_preserves_original_outputs(tmp_path, monkeypatch):
    from transformers.cache_utils import DynamicCache, DynamicLayer

    base, portable, reference, config = artifact(
        tmp_path, monkeypatch, "qwen3_vl", model_execution="qwen3_vl_hf_stock_v1"
    )
    loaded, _, _ = load(base, portable, monkeypatch)
    cache = DynamicCache()
    with torch.no_grad():
        expected = reference(**data(config)[0], past_key_values=DynamicCache()).logits
        actual = loaded(**data(config)[0], past_key_values=cache).logits
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert len(cache.layers) == config.text_config.num_hidden_layers
    assert all(type(layer) is DynamicLayer for layer in cache.layers)


@pytest.mark.parametrize("name", ["input_ids", "inputs_embeds"])
@pytest.mark.parametrize("positional", [False, True])
def test_batch_one_contract_applies_to_positional_and_keyword_inputs(tmp_path, monkeypatch, name, positional):
    from transformers.cache_utils import DynamicCache

    base, portable, _, config = artifact(tmp_path, monkeypatch, "qwen3_vl", model_execution="qwen3_vl_hf_stock_v1")
    loaded, _, _ = load(base, portable, monkeypatch)
    value = (
        torch.ones(2, 4, dtype=torch.long) if name == "input_ids" else torch.zeros(2, 4, config.text_config.hidden_size)
    )
    cache = DynamicCache(config=config)
    args, kwargs = [], {"past_key_values": cache}
    if positional:
        names = list(signature(loaded.forward).parameters)
        args = [None] * (names.index(name) + 1)
        args[-1] = value
        if names.index("past_key_values") < len(args):
            args[names.index("past_key_values")] = kwargs.pop("past_key_values")
    else:
        kwargs[name] = value
    with torch.no_grad(), pytest.raises(ValueError, match="batch-one"):
        loaded(*args, **kwargs)
    assert cache.get_seq_length() == 0
