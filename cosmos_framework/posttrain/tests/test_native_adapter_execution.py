# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU execution-policy contracts; these do not qualify CUDA compilation or NF4 kernels."""

from __future__ import annotations

import copy
import re
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.checkpoint import CheckpointPolicy

from cosmos_framework.configs.base.defaults.activation_checkpointing import ActivationCheckpointingConfig
from cosmos_framework.model.generator import parallelize_vlm
from cosmos_framework.posttrain.config.model import (
    NATIVE_ADAPTER_SAC_SAVE_OPS,
    NativeLoRAConfig,
    Qwen3VLModelConfig,
    native_adapter_execution_policy,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.fixture(autouse=True)
def strict_compiler_errors():
    with torch._dynamo.config.patch(suppress_errors=False):
        yield


def config(**overrides):
    return Qwen3VLModelConfig(name_or_path="test/source", attn_impl="sdpa", adapter=NativeLoRAConfig(), **overrides)


def test_default_execution_identity_keeps_legacy_artifact_contract():
    cfg = config()
    assert native_adapter_execution_policy(cfg) is None
    assert cfg.adapter.parameter_dtype == "float32"
    assert cfg.adapter_compute_dtype == "float32"
    assert cfg.sft_logits_mode == "full"
    assert cfg.compile_language_model is False


def test_execution_options_survive_omegaconf_structured_roundtrip():
    cfg = config(
        adapter_compute_dtype="bfloat16",
        sft_logits_mode="supervised",
        activation_checkpointing="selective",
        compile_language_model=True,
    )
    structured = OmegaConf.structured(cfg)
    restored = OmegaConf.to_object(structured)
    assert isinstance(restored, Qwen3VLModelConfig)
    assert native_adapter_execution_policy(restored) == native_adapter_execution_policy(cfg)
    structured.adapter_compute_dtype = "float16"
    with pytest.raises(ValueError, match="adapter_compute_dtype"):
        OmegaConf.to_object(structured)


@pytest.mark.parametrize(
    "overrides",
    [
        {"activation_checkpointing": "none"},
        {"activation_checkpointing": "selective"},
        {"adapter_compute_dtype": "bfloat16"},
        {"sft_logits_mode": "supervised"},
        {"compile_language_model": True},
    ],
)
def test_each_opt_in_has_explicit_execution_identity(overrides):
    cfg = config(**overrides)
    policy = native_adapter_execution_policy(cfg)
    assert policy["protocol"] == "cosmos.native-adapter-execution.v1"
    assert policy["adapter_compute_dtype"] == cfg.adapter_compute_dtype
    assert policy["sft_logits_mode"] == cfg.sft_logits_mode
    assert policy["language_compile"]["enabled"] is cfg.compile_language_model
    ac = policy["activation_checkpointing"]
    assert ac["mode"] == cfg.activation_checkpointing
    assert ac["preserve_rng_state"] is True
    assert ac["save_ops_regex"] == (list(NATIVE_ADAPTER_SAC_SAVE_OPS) if ac["mode"] == "selective" else [])
    assert cfg.adapter.parameter_dtype == "float32"
    assert policy["language_compile"]["fail_on_recompile_limit_hit"] is True


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"adapter_compute_dtype": "float16"}, ValueError),
        ({"sft_logits_mode": "last"}, ValueError),
        ({"compile_language_model": "true"}, TypeError),
        ({"compile_language_model": 1}, TypeError),
    ],
)
def test_execution_options_reject_invalid_types_and_values(overrides, error):
    with pytest.raises(error):
        config(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [{"adapter_compute_dtype": "bfloat16"}, {"sft_logits_mode": "supervised"}, {"compile_language_model": True}],
)
def test_new_execution_options_require_explicit_adapter(overrides):
    with pytest.raises(ValueError, match="require an explicit adapter"):
        Qwen3VLModelConfig(**overrides)


class AdapterTestBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(8, 8, bias=False).requires_grad_(False)
        self.a = nn.Linear(8, 2, bias=False)
        self.b = nn.Linear(2, 8, bias=False)
        nn.init.zeros_(self.b.weight)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        return torch.tanh(self.base(x) + self.b(self.a(self.dropout(x))))


class TinyNative(nn.Module):
    _no_split_modules = ["AdapterTestBlock"]

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList([AdapterTestBlock(), AdapterTestBlock()])
        self.model.visual = nn.Module()
        self.model.visual.blocks = nn.ModuleList([AdapterTestBlock().requires_grad_(False)])

    def forward(self, x):
        for layer in self.model.language_model.layers:
            x = layer(x)
        return x


def ac(mode):
    return ActivationCheckpointingConfig(mode=mode, save_ops_regex=list(NATIVE_ADAPTER_SAC_SAVE_OPS))


def update(model, inputs):
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    torch.manual_seed(19)
    losses = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        output = model(inputs)
        loss = output.square().sum()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
        optimizer.step()
        losses.append(loss.detach())
    return torch.stack(losses), [p.detach().clone() for p in model.parameters()]


@pytest.mark.parametrize("mode", ["none", "full", "selective"])
def test_nonreentrant_ac_updates_adapters_from_frozen_inputs_and_preserves_rng(mode):
    torch.manual_seed(7)
    baseline = TinyNative()
    candidate = copy.deepcopy(baseline)
    initial = [p.detach().clone() for p in candidate.parameters()]
    parallelize_vlm.apply_ac(SimpleNamespace(model=candidate), ac(mode))
    inputs = torch.randn(3, 8)  # frozen base inputs need no requires_grad hook
    expected_loss, expected = update(baseline, inputs)
    actual_loss, actual = update(candidate, inputs)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    for before, after, reference, parameter in zip(initial, actual, expected, candidate.parameters(), strict=True):
        torch.testing.assert_close(after, reference, rtol=0, atol=0)
        assert (not torch.equal(before, after)) if parameter.requires_grad else torch.equal(before, after)
    assert inputs.requires_grad is False


def test_selective_policy_saves_actual_dispatched_sdpa_and_matmul_ops(monkeypatch):
    captured = []
    original = parallelize_vlm.create_selective_checkpoint_contexts

    def capture(policy):
        captured.append(policy)
        return original(policy)

    monkeypatch.setattr(parallelize_vlm, "create_selective_checkpoint_contexts", capture)
    block = parallelize_vlm._apply_selective_ac(AdapterTestBlock(), ac("selective"))
    block(torch.randn(3, 8)).sum().backward()
    policy = captured[0]
    for name in (
        "_scaled_dot_product_flash_attention",
        "_scaled_dot_product_efficient_attention",
        "_scaled_dot_product_cudnn_attention",
        "_scaled_dot_product_attention_math",
        "mm",
        "addmm",
        "bmm",
    ):
        op = getattr(torch.ops.aten, name).default
        assert policy(None, op) == CheckpointPolicy.MUST_SAVE
    assert policy(None, torch.ops.aten.sin.default) == CheckpointPolicy.MUST_RECOMPUTE
    assert not any(re.search(pattern, "some_mm_like_op") for pattern in NATIVE_ADAPTER_SAC_SAVE_OPS)


def prepare_builder(monkeypatch, net):
    from cosmos_framework.posttrain.model import qwen3_vl
    from cosmos_framework.posttrain.model.quantization import loader

    class Lifecycle:
        def __init__(self, *_args):
            pass

        prepare = materialize = load = finalize = lambda self, *_args: None

    def build(**kwargs):
        kwargs["hooks"].parallelize(net, None)
        return net

    monkeypatch.setattr(
        "transformers.AutoConfig.from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(model_type="qwen3_vl", tie_word_embeddings=False),
    )
    monkeypatch.setattr(loader, "NativeAdapterLifecycle", Lifecycle)
    monkeypatch.setattr(qwen3_vl, "build_qwen3_vl_policy", build)
    return qwen3_vl.build_qwen3_vl_model


@pytest.mark.parametrize("mode", ["none", "full", "selective"])
def test_native_builder_compiles_only_language_after_ac_and_keeps_parameter_references(monkeypatch, mode):
    net = TinyNative()
    refs = tuple(net.parameters())
    build = prepare_builder(monkeypatch, net)
    compiled = []

    class Wrapper(nn.Module):
        def __init__(self, module):
            super().__init__()
            self._orig_mod = module

        def forward(self, x):
            return self._orig_mod(x)

    def compile_block(block, **kwargs):
        assert hasattr(block, "_checkpoint_wrapped_module") is (mode != "none")
        assert kwargs["dynamic"] is True and kwargs["fullgraph"] is False
        compiled.append(block)
        return Wrapper(block)

    monkeypatch.setattr(torch, "compile", compile_block)
    assert build(config(activation_checkpointing=mode, compile_language_model=True), None) is net
    assert len(compiled) == net._cosmos_compiled_language_blocks == 2
    assert all(isinstance(layer, Wrapper) for layer in net.model.language_model.layers)
    assert not isinstance(net.model.visual.blocks[0], Wrapper)
    assert tuple(net.parameters()) == refs
    net(torch.randn(3, 8))
    diagnostics = net._cosmos_compile_diagnostics()
    assert diagnostics["guarded_modules"] == diagnostics["guarded_modules_called"] == 2
    assert diagnostics["guarded_module_forward_calls"] == [1, 1]


def test_native_builder_default_never_compiles(monkeypatch):
    net = TinyNative()
    build = prepare_builder(monkeypatch, net)
    monkeypatch.setattr(torch, "compile", lambda *_args, **_kwargs: pytest.fail("compile is opt-in"))
    build(config(), None)
    assert not hasattr(net, "_cosmos_compiled_language_blocks")


def test_native_builder_rejects_compiler_parameter_replacement(monkeypatch):
    net = TinyNative()
    build = prepare_builder(monkeypatch, net)
    monkeypatch.setattr(torch, "compile", lambda *_args, **_kwargs: nn.Linear(8, 8))
    with pytest.raises(RuntimeError, match="changed native model parameter"):
        build(config(compile_language_model=True), None)


def test_native_builder_refuses_suppressed_compiler_errors(monkeypatch):
    net = TinyNative()
    build = prepare_builder(monkeypatch, net)
    with torch._dynamo.config.patch(suppress_errors=True):
        with pytest.raises(ValueError, match="suppress_errors=False"):
            build(config(compile_language_model=True), None)


@pytest.mark.parametrize("mode", ["none", "full", "selective"])
def test_compiled_checkpoint_wrapper_preserves_cpu_training_with_frozen_inputs(monkeypatch, mode):
    """Exercise Dynamo/AOT autograd; the real CUDA Inductor/NF4 row is a separate gate."""
    torch.manual_seed(31)
    baseline = TinyNative()
    candidate = copy.deepcopy(baseline)
    build = prepare_builder(monkeypatch, candidate)
    compile_module = torch.compile

    def compile_aot(module, **kwargs):
        return compile_module(module, backend="aot_eager", **kwargs)

    monkeypatch.setattr(torch, "compile", compile_aot)
    torch._dynamo.reset()
    try:
        build(config(activation_checkpointing=mode, compile_language_model=True), None)
        inputs = torch.randn(3, 8)
        expected_loss, expected = update(baseline, inputs)
        actual_loss, actual = update(candidate, inputs)
        torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
        for observed, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(observed, reference, rtol=0, atol=0)
        assert inputs.requires_grad is False
    finally:
        torch._dynamo.reset()


def test_native_compiler_limit_is_fatal_and_global_policy_is_restored():
    from cosmos_framework.posttrain.utils.native_compile import guard_native_compiled_call, native_compile_diagnostics

    class Specialized(nn.Module):
        def forward(self, x, flag):
            return x.sin() if flag else x.cos()

    torch._dynamo.reset()
    try:
        module = guard_native_compiled_call(torch.compile(Specialized(), backend="eager", dynamic=True))
        assert guard_native_compiled_call(module) is module
        with torch._dynamo.config.patch(
            recompile_limit=1,
            accumulated_recompile_limit=8,
            fail_on_recompile_limit_hit=False,
            suppress_errors=True,
        ):
            value = torch.randn(3)
            torch.testing.assert_close(module(value, True), value.sin())
            assert torch._dynamo.config.fail_on_recompile_limit_hit is False
            assert torch._dynamo.config.suppress_errors is True
            with pytest.raises(torch._dynamo.exc.FailOnRecompileLimitHit):
                module(value, False)
            assert torch._dynamo.config.fail_on_recompile_limit_hit is False
            assert torch._dynamo.config.suppress_errors is True
        diagnostics = native_compile_diagnostics([module])
        assert diagnostics["unique_graphs"] > 0 and diagnostics["captured_calls"] > 0
        assert diagnostics["guard_failures"] > 0
        assert diagnostics["guarded_module_forward_calls"] == [2]
        assert all(
            isinstance(value, int)
            for key, value in diagnostics.items()
            if key not in {"scope", "guarded_module_forward_calls"}
        )
    finally:
        torch._dynamo.reset()
