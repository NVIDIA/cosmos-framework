# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Additional real-CUDA combination gate, separate from the existing 14-case gate.

This direct tiny-model fixture exercises the execution modules, not config
admission, native checkpoint I/O, full-model recovery, or performance.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager

import pytest
import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl, checkpoint_wrapper

from cosmos_framework.posttrain.model.qwen3_vl import (
    Qwen3VLBuildHooks,
    build_qwen3_vl_policy,
    override_qwen3_vl_forward,
)
from cosmos_framework.posttrain.tests.test_model_qwen3_vl_gpu import _batch, _dense_config
from cosmos_framework.posttrain.tests.test_model_qwen3_vl_qlora_gpu import restore_determinism_state  # noqa: F401

pytestmark = [pytest.mark.GPU, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA kernels")]

_TARGETS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


class _BlockContexts:
    """Count actual original/recompute entries without replacing native scopes."""

    def __init__(self, modules):
        self.modules = tuple(modules)
        self.phase = None
        self.entries = {"forward": 0, "recompute": 0}
        self.calls = {phase: [0] * len(modules) for phase in self.entries}
        self.hooks = [module.register_forward_pre_hook(self._hook(index)) for index, module in enumerate(modules)]

    def _hook(self, index):
        def record(_module, _args):
            assert self.phase in self.calls
            self.calls[self.phase][index] += 1

        return record

    @contextmanager
    def _count(self, phase, native_context):
        assert self.phase is None
        with native_context:
            self.phase = phase
            self.entries[phase] += 1
            try:
                yield
            finally:
                self.phase = None

    def contexts(self):
        from cosmos_framework.posttrain.model.quantization.bnb import checkpointed_dequant_contexts

        forward, recompute = checkpointed_dequant_contexts(self.modules)
        return self._count("forward", forward), self._count("recompute", recompute)

    def snapshot(self):
        return copy.deepcopy((self.entries, self.calls))

    def assert_backward(self, before):
        entries, calls = before
        assert self.phase is None
        for phase in self.entries:
            assert self.entries[phase] - entries[phase] == 1
            # Every one of the seven packed matrices participates again during
            # autograd recomputation, including with checkpoint early-stop on.
            assert all(after - prior == 1 for after, prior in zip(self.calls[phase], calls[phase], strict=True))

    def close(self):
        for hook in self.hooks:
            hook.remove()


def _build(execution, device):
    from cosmos_framework.posttrain.config.model import NativeLoRAConfig, QuantizedBaseTrainingConfig
    from cosmos_framework.posttrain.model.quantization.bnb import NativeLoraLinear4bit

    torch.manual_seed(37)
    policy = build_qwen3_vl_policy(
        model_type="qwen3_vl",
        model_config=_dense_config(),
        dtype=torch.bfloat16,
        initialize_on_meta=False,
        attn_implementation="sdpa",
        hooks=Qwen3VLBuildHooks(prepare_model=lambda net: override_qwen3_vl_forward(net.model)),
        sft_logits_mode="supervised",
    )
    policy.net.requires_grad_(False)
    targets, contexts = {}, []
    layers = policy.net.model.language_model.layers
    for layer_index, block in enumerate(layers):
        members = []
        for target_index, name in enumerate(_TARGETS):
            parent_name, attribute = name.rsplit(".", 1)
            parent = block.get_submodule(parent_name)
            source = getattr(parent, attribute)
            assert type(source) is nn.Linear and source.bias is None and source.weight.dtype == torch.bfloat16
            module = NativeLoraLinear4bit(
                nn.Linear(source.in_features, source.out_features, bias=False, device="meta"),
                NativeLoRAConfig(rank=32, alpha=64),
                QuantizedBaseTrainingConfig(),
                quantized_linear_execution=execution,
            )
            module.lora_A.weight = nn.Parameter(torch.empty(32, source.in_features, device=device, dtype=torch.float32))
            module.lora_B.weight = nn.Parameter(
                torch.empty(source.out_features, 32, device=device, dtype=torch.float32)
            )
            module.load_dense_weight_once(source.weight.detach().cpu(), device)
            generator = torch.Generator(device=device).manual_seed(38 + 7 * layer_index + target_index)
            module.reset_adapter_parameters(generator=generator)
            # A nonzero restored B requires both factors to receive gradients.
            with torch.no_grad():
                module.lora_B.weight.normal_(std=0.01, generator=generator)
            setattr(parent, attribute, module)
            targets[f"{layer_index}.{name}"] = module
            members.append(module)
        kwargs = {"preserve_rng_state": True, "determinism_check": "default"}
        if execution == "checkpointed_dequant":
            context = _BlockContexts(members)
            contexts.append(context)
            kwargs["context_fn"] = context.contexts
        layers[layer_index] = checkpoint_wrapper(block, checkpoint_impl=CheckpointImpl.NO_REENTRANT, **kwargs)
    assert len(targets) == 14
    policy.net.train()
    parameters = {name: parameter for name, parameter in policy.net.named_parameters() if parameter.requires_grad}
    expected = [parameter for module in targets.values() for parameter in module.adapter_parameters().values()]
    assert len(parameters) == len(expected) == 28
    assert {id(parameter) for parameter in parameters.values()} == {id(parameter) for parameter in expected}
    assert all(parameter.dtype == torch.float32 for parameter in parameters.values())
    assert not policy.net.lm_head.weight.requires_grad
    assert not policy.net.model.language_model.embed_tokens.weight.requires_grad
    optimizer = torch.optim.AdamW(list(parameters.values()), lr=1e-4, fused=True)
    assert optimizer.param_groups[0]["fused"] is True
    assert [id(parameter) for group in optimizer.param_groups for parameter in group["params"]] == [
        id(parameter) for parameter in parameters.values()
    ]
    return policy, targets, contexts, parameters, optimizer


def _exact(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _exact(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            _exact(left, right)
    else:
        assert actual == expected


def _frozen(policy, targets):
    from cosmos_framework.posttrain.model.quantization.loader import quantized_state_record

    values = {
        "parameters": {
            name: value.detach().cpu().clone()
            for name, value in policy.net.named_parameters()
            if not value.requires_grad
        },
        "buffers": {name: value.detach().cpu().clone() for name, value in policy.net.named_buffers()},
        "packed": {name: quantized_state_record(module) for name, module in targets.items()},
    }
    aliases = {}
    for name, module in targets.items():
        state = module.weight.quant_state
        tensors = [module.weight, state.absmax, state.code, state.offset, state.state2.absmax, state.state2.code]
        aliases[name] = (id(module.weight), id(state), id(state.state2), [(id(t), t.data_ptr()) for t in tensors])
        assert module.quant_state is state and module.weight.grad is None
    assert all(parameter.grad is None for parameter in policy.net.parameters() if not parameter.requires_grad)
    return values, aliases


def _backward(built, batch, valid, labels, *, selected):
    from cosmos_framework.posttrain.model.quantization.bnb import _CHECKPOINTED_DEQUANT_OWNERS

    policy, _targets, contexts, parameters, optimizer = built
    optimizer.zero_grad(set_to_none=True)
    before = [context.snapshot() for context in contexts]
    shapes = []
    hook = policy.net.lm_head.register_forward_pre_hook(lambda _module, args: shapes.append(tuple(args[0].shape)))
    try:
        logits = (
            policy.forward_sft_batch(batch, valid).logits
            if selected
            else policy.forward_policy_batch(batch).logits[:, :-1][valid]
        ).float()
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
    finally:
        hook.remove()
    assert shapes == ([(2, 32)] if selected else [(1, 5, 32)])
    for context, snapshot in zip(contexts, before, strict=True):
        context.assert_backward(snapshot)
    assert _CHECKPOINTED_DEQUANT_OWNERS.get() == frozenset()
    gradients = {}
    for name, parameter in parameters.items():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.count_nonzero() > 0
        gradients[name] = parameter.grad.detach().cpu().clone()
    return {"logits": logits.detach().cpu(), "loss": loss.detach().cpu(), "gradients": gradients}


def _updated(built):
    _policy, _targets, _contexts, parameters, optimizer = built
    previous = [parameter.detach().clone() for parameter in parameters.values()]
    optimizer.step()
    assert any(not torch.equal(parameter, old) for parameter, old in zip(parameters.values(), previous, strict=True))
    assert set(optimizer.state) == set(parameters.values())
    for parameter in parameters.values():
        assert torch.isfinite(parameter).all()
        state = optimizer.state[parameter]
        assert set(state) == {"step", "exp_avg", "exp_avg_sq"}
        assert all(value.dtype == torch.float32 and torch.isfinite(value).all() for value in state.values())
        assert state["exp_avg"].shape == state["exp_avg_sq"].shape == parameter.shape
    return {
        "parameters": {name: parameter.detach().cpu().clone() for name, parameter in parameters.items()},
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "cpu_rng": torch.get_rng_state().clone(),
        "cuda_rng": torch.cuda.get_rng_state(next(iter(parameters.values())).device).clone(),
    }


def test_selected_dequant_native_multistep_and_boundary_replay(record_property):
    from cosmos_framework.posttrain.model.quantization.bnb import _CHECKPOINTED_DEQUANT_OWNERS
    from cosmos_framework.posttrain.utils.determinism import enable_determinism

    enable_determinism(strict=True)
    device = torch.device("cuda", torch.cuda.current_device())
    ordinary = _build("bnb", device)
    candidate = _build("checkpointed_dequant", device)
    built_models = [ordinary, candidate]
    try:
        frozen = [_frozen(built[0], built[1]) for built in built_models]
        _exact(frozen[0][0], frozen[1][0])
        _exact(ordinary[3], candidate[3])
        batch = candidate[0].prepare_policy_batch(_batch())
        valid = torch.tensor([[False, False, True, True]], device=device)
        labels = torch.tensor([3, 4], device=device)
        results, boundary = [], None
        for iteration in range(3):
            expected = _backward(ordinary, batch, valid, labels, selected=True)
            expected_update = _updated(ordinary)
            full = _backward(candidate, batch, valid, labels, selected=False)
            actual = _backward(candidate, batch, valid, labels, selected=True)
            # Keep the existing selected-head tolerances for the changed BF16
            # projection shape. Same-shape execution and replay remain exact.
            torch.testing.assert_close(actual["logits"], full["logits"], rtol=0.02, atol=0.002)
            torch.testing.assert_close(actual["loss"], full["loss"], rtol=0.002, atol=0.002)
            for name in actual["gradients"]:
                torch.testing.assert_close(actual["gradients"][name], full["gradients"][name], rtol=0.03, atol=1e-5)
            _exact(actual, expected)
            updated = _updated(candidate)
            _exact(updated, expected_update)
            results.append((actual, updated))
            for built, snapshot in zip(built_models, frozen, strict=True):
                _exact(_frozen(built[0], built[1]), snapshot)
            if iteration == 0:
                boundary = (
                    copy.deepcopy(updated),
                    torch.get_rng_state().clone(),
                    torch.cuda.get_rng_state(device).clone(),
                )
        # Fresh construction and adapter/optimizer restoration at the first
        # update boundary; this deliberately does not claim native DCP recovery.
        replay = _build("checkpointed_dequant", device)
        built_models.append(replay)
        replay_frozen = _frozen(replay[0], replay[1])
        _exact(replay_frozen[0], frozen[1][0])
        saved, cpu_rng, cuda_rng = boundary
        with torch.no_grad():
            for name, parameter in replay[3].items():
                parameter.copy_(saved["parameters"][name])
        replay[4].load_state_dict(saved["optimizer"])
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng, device)
        for expected, expected_update in results[1:]:
            _exact(_backward(replay, batch, valid, labels, selected=True), expected)
            _exact(_updated(replay), expected_update)
            _exact(_frozen(replay[0], replay[1]), replay_frozen)
        assert _CHECKPOINTED_DEQUANT_OWNERS.get() == frozenset()
        record_property(
            "selected_dequant_scope", "tiny_native_vlm_2_blocks_14_nf4_linears_r32_alpha64_fp32_fused_adamw"
        )
        record_property("exact_selected_updates", 3)
        record_property("fresh_boundary_replayed_updates", 2)
        record_property("full_selected_tolerances", "logits=.02/.002;ce=.002/.002;gradients=.03/1e-5")
        record_property("frozen_parameters_buffers_packed_nested_state_exact", True)
        record_property("optimizer_adapter_only_fp32_state_exact", True)
        record_property("all_14_linears_forward_and_recompute_observed", True)
    finally:
        for built in built_models:
            built[4].zero_grad(set_to_none=True)
            for context in built[2]:
                context.close()
