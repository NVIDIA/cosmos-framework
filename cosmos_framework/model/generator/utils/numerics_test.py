# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU checks for numerical safeguards; GPU quality and performance require separate validation."""

import unittest
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from cosmos_framework.configs.base.defaults.compile import CompileConfig
from cosmos_framework.model.generator.mot.parallelize_unified_mot import apply_compile
from cosmos_framework.model.generator.mot.unified_mot import _apply_request_rms_norm
from cosmos_framework.model.generator.reasoner.qwen3_vl.qwen3_vl import Qwen3VLTextRMSNorm
from cosmos_framework.model.generator.utils.memory import MemoryValue
from cosmos_framework.model.generator.utils.numerics import native_rms_norm


@dataclass
class RequestMemory(MemoryValue):
    preserve_transfer_numerics: bool = False


class RequestNorm(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm: Qwen3VLTextRMSNorm = Qwen3VLTextRMSNorm(width)

    def forward(
        self,
        value: torch.Tensor,
        attention_mask: object = None,
        packed_position_embeddings: object = None,
        natten_metadata: object = None,
        memory_value: MemoryValue | None = None,
    ) -> torch.Tensor:
        return _apply_request_rms_norm(self.norm, value, memory_value)


class NumericsTests(unittest.TestCase):
    def setUp(self) -> None:
        # Each case builds fresh capture backends; isolate their Dynamo cache entries
        # so repeated test runs do not consume the shared forward's recompile budget.
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)

    def test_compiled_inference_preserves_eager_rmsnorm(self) -> None:
        for dtype in (torch.float32, torch.bfloat16):
            for weight_dtype in (torch.float32, torch.bfloat16):
                with self.subTest(dtype=dtype, weight_dtype=weight_dtype):
                    norm = RequestNorm(256).to(weight_dtype).eval()
                    norm.norm.weight.data.copy_(torch.linspace(0.1, 2, 256, dtype=weight_dtype))
                    value = torch.linspace(-3, 5, 768, dtype=dtype).reshape(3, 256)
                    graphs: list[torch.fx.GraphModule] = []

                    def capture(
                        graph: torch.fx.GraphModule, _inputs: list[torch.Tensor]
                    ) -> Callable[..., torch.Tensor]:
                        graphs.append(graph)
                        return graph.forward

                    compiled = torch.compile(norm, backend=capture, fullgraph=True)
                    with torch.no_grad():
                        expected = norm(value)
                        actual = compiled(value, memory_value=RequestMemory(preserve_transfer_numerics=True))
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertEqual(actual.dtype, expected.dtype)
                    self.assertTrue(any("cosmos_numerics.native_rms_norm" in str(g.graph) for g in graphs))

    def test_training_and_eval_autograd_keep_original_gradients(self) -> None:
        for training in (True, False):
            with self.subTest(training=training):
                eager = Qwen3VLTextRMSNorm(16).train(training)
                compiled_model = deepcopy(eager)
                expected_input = torch.linspace(-2, 3, 32).reshape(2, 16).requires_grad_()
                actual_input = expected_input.detach().clone().requires_grad_()
                expected = eager(expected_input)
                expected.square().sum().backward()
                actual = torch.compile(compiled_model, backend="aot_eager", fullgraph=True)(actual_input)
                actual.square().sum().backward()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(actual_input.grad, expected_input.grad, rtol=0, atol=0)
                torch.testing.assert_close(compiled_model.weight.grad, eager.weight.grad, rtol=0, atol=0)

    def test_native_op_fake_metadata_and_fullgraph_capture(self) -> None:
        value = torch.arange(16, dtype=torch.bfloat16).reshape(2, 8)
        weight = torch.ones(8, dtype=torch.float32)
        with FakeTensorMode() as mode:
            fake = native_rms_norm(mode.from_tensor(value), mode.from_tensor(weight), 1e-6)
            self.assertEqual(fake.shape, value.shape)
            self.assertEqual(fake.dtype, torch.float32)
        compiled = torch.compile(native_rms_norm, backend="aot_eager", fullgraph=True)
        torch.testing.assert_close(compiled(value, weight, 1e-6), native_rms_norm(value, weight, 1e-6), rtol=0, atol=0)

    def test_request_routing_preserves_normal_graph_and_state_ownership(self) -> None:
        for use_graphs, scope in ((False, "block"), (True, "block"), (True, "forward")):
            with self.subTest(use_graphs=use_graphs, scope=scope):
                torch._dynamo.reset()
                layer = RequestNorm(16).eval()
                model = torch.nn.Module()
                layers = torch.nn.ModuleList([layer])
                model.add_module("model", torch.nn.ModuleDict({"layers": layers}))
                config = CompileConfig(
                    enabled=True,
                    use_cuda_graphs=use_graphs,
                    cuda_graph_scope=scope,
                    compile_dynamic=True,
                    max_autotune_pointwise=True,
                )
                compile_calls: list[dict[str, Any]] = []
                graphs: list[torch.fx.GraphModule] = []
                real_compile = torch.compile

                def capture(graph: torch.fx.GraphModule, _inputs: list[torch.Tensor]) -> Callable[..., torch.Tensor]:
                    graphs.append(graph)
                    return graph.forward

                def compile_for_cpu(block: torch.nn.Module, **kwargs: Any) -> torch.nn.Module:
                    compile_calls.append(kwargs)
                    compiled = real_compile(block, backend=capture, fullgraph=True)
                    assert isinstance(compiled, torch.nn.Module)
                    return compiled

                with patch("torch.compile", side_effect=compile_for_cpu):
                    apply_compile(model, config)
                    compiled = layers[0]
                    state_keys = tuple(model.state_dict())
                    parameter_ids = tuple(id(p) for p in model.parameters())
                    value = torch.linspace(-2, 3, 32).reshape(2, 16)
                    ordinary = RequestMemory(preserve_transfer_numerics=False)
                    transfer = RequestMemory(preserve_transfer_numerics=True)
                    with torch.no_grad():
                        expected = layer(value)
                        torch.testing.assert_close(compiled(value, memory_value=ordinary), expected, rtol=0, atol=0)
                        self.assertEqual(len(compile_calls), 1)
                        self.assertFalse(any("cosmos_numerics.native_rms_norm" in str(g.graph) for g in graphs))
                        # Positional metadata is part of the decoder's existing API.
                        torch.testing.assert_close(
                            compiled(value, None, None, None, transfer), expected, rtol=0, atol=0
                        )
                        self.assertEqual(len(compile_calls), 2)
                        self.assertTrue(any("cosmos_numerics.native_rms_norm" in str(g.graph) for g in graphs))
                        for metadata in (ordinary, transfer, ordinary):
                            torch.testing.assert_close(compiled(value, memory_value=metadata), expected, rtol=0, atol=0)
                        self.assertEqual(len(compile_calls), 2)
                    self.assertEqual(tuple(model.state_dict()), state_keys)
                    self.assertEqual(tuple(id(p) for p in model.parameters()), parameter_ids)
                    self.assertEqual(len(parameter_ids), 1)
                    self.assertIs(compiled._orig_mod, layer)
                    # Even a marked request keeps training/autograd on the original path.
                    for training in (True, False):
                        compiled.train(training)
                        grad_value = value.detach().clone().requires_grad_(True)
                        compiled(grad_value, memory_value=transfer).sum().backward()
                        self.assertIsNotNone(grad_value.grad)
                        self.assertEqual(len(compile_calls), 2)
                normal_kwargs, transfer_kwargs = compile_calls
                self.assertEqual(normal_kwargs["mode"], "reduce-overhead" if use_graphs and scope == "block" else None)
                self.assertEqual(normal_kwargs["options"], {"max_autotune_pointwise": True})
                self.assertNotIn("mode", transfer_kwargs)
                options = transfer_kwargs["options"]
                self.assertEqual(options.get("emulate_precision_casts", False), use_graphs)
                self.assertEqual(options.get("triton.cudagraphs", False), use_graphs and scope == "block")
                self.assertEqual(transfer_kwargs["dynamic"], not (use_graphs and scope == "forward"))

    def test_shared_qwen_norm_is_unmodified_even_in_compiled_eval(self) -> None:
        norm = Qwen3VLTextRMSNorm(16).eval()
        graphs: list[torch.fx.GraphModule] = []

        def capture(graph: torch.fx.GraphModule, _inputs: list[torch.Tensor]) -> Callable[..., torch.Tensor]:
            graphs.append(graph)
            return graph.forward

        value = torch.linspace(-2, 3, 32).reshape(2, 16)
        with torch.no_grad():
            actual = torch.compile(norm, backend=capture, fullgraph=True)(value)
            torch.testing.assert_close(actual, norm(value), rtol=0, atol=0)
        self.assertFalse(any("cosmos_numerics.native_rms_norm" in str(g.graph) for g in graphs))


if __name__ == "__main__":
    unittest.main()
