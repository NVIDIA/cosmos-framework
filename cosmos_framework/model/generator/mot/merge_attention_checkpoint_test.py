# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Check maskless merge gradients, including non-reentrant checkpoint recomputation."""

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from cosmos_framework.model.generator.mot import multiview_maskless_attention as maskless


def _reference(query, key, value):
    scores = torch.einsum("bqhd,bkhd->bhqk", query, key) * query.shape[-1] ** -0.5
    return torch.einsum("bhqk,bkhd->bqhd", scores.softmax(-1), value).contiguous()


class _FusedAttentionForCPU(torch.autograd.Function):
    """CPU oracle for a fused kernel that saves its output/LSE by data pointer."""

    @staticmethod
    def forward(ctx, query, key, value):
        scores = torch.einsum("bqhd,bkhd->bhqk", query, key) * query.shape[-1] ** -0.5
        lse = scores.logsumexp(-1).transpose(1, 2).contiguous()
        output = torch.einsum("bhqk,bkhd->bqhd", scores.softmax(-1), value).contiguous()
        ctx.save_for_backward(query, key, value, output, lse)
        ctx.mark_non_differentiable(lse)
        return output, lse

    @staticmethod
    def backward(ctx, grad_output, grad_lse):
        query, key, value, output, lse = ctx.saved_tensors
        scale = query.shape[-1] ** -0.5
        scores = torch.einsum("bqhd,bkhd->bhqk", query, key) * scale
        probability = (scores - lse.transpose(1, 2).unsqueeze(-1)).exp()
        grad_probability = torch.einsum("bqhd,bkhd->bhqk", grad_output, value)
        delta = (grad_output * output).sum(-1).transpose(1, 2).unsqueeze(-1)
        grad_scores = probability * (grad_probability - delta)
        grad_query = torch.einsum("bhqk,bkhd->bqhd", grad_scores, key) * scale
        grad_key = torch.einsum("bhqk,bqhd->bkhd", grad_scores, query) * scale
        grad_value = torch.einsum("bhqk,bqhd->bkhd", probability, grad_output)
        return grad_query, grad_key, grad_value


def _check_merge(attention_op, *, device, dtype, splits, checkpointed, atol, rtol):
    torch.manual_seed(3701)
    query_length, heads, dim = (128, 4, 64) if device == "cuda" else (4, 2, 8)
    lengths = [128, 256, 128][:splits] if device == "cuda" else [3, 5, 2][:splits]
    tensors = [torch.randn(1, query_length, heads, dim, device=device, dtype=dtype)]
    tensors += [torch.randn(1, size, heads, dim, device=device, dtype=dtype) for size in lengths]
    tensors += [torch.randn(1, size, heads, dim, device=device, dtype=dtype) for size in lengths]
    inputs = [tensor.detach().clone().requires_grad_() for tensor in tensors]
    reference_dtype = torch.float32 if device == "cuda" else torch.float64
    golden = [tensor.detach().to(reference_dtype).requires_grad_() for tensor in tensors]
    probe = torch.randn(1, query_length, heads, dim, device=device, dtype=reference_dtype)

    def combined(query, *keys_values):
        outputs, lses = [], []
        for key, value in zip(keys_values[:splits], keys_values[splits:], strict=True):
            output, lse = attention_op(query, key, value)
            outputs.append(output)
            lses.append(lse)
        output, _ = maskless.merge_attentions(outputs, lses, torch_compile=False)
        return output

    actual = checkpoint(combined, *inputs, use_reentrant=False) if checkpointed else combined(*inputs)
    expected = _reference(golden[0], torch.cat(golden[1 : splits + 1], 1), torch.cat(golden[splits + 1 :], 1))
    actual_gradients = torch.autograd.grad((actual.to(reference_dtype) * probe).sum(), inputs)
    expected_gradients = torch.autograd.grad((expected * probe).sum(), golden)
    torch.testing.assert_close(actual.to(reference_dtype), expected, atol=atol, rtol=rtol)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
        assert torch.isfinite(actual_gradient).all()
        torch.testing.assert_close(actual_gradient.to(reference_dtype), expected_gradient, atol=atol, rtol=rtol)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("splits", [2, 3])
@pytest.mark.parametrize("checkpointed", [False, True])
def test_maskless_merge_matches_dense_gradients_cpu(splits, checkpointed):
    _check_merge(
        _FusedAttentionForCPU.apply,
        device="cpu",
        dtype=torch.float64,
        splits=splits,
        checkpointed=checkpointed,
        atol=1e-10,
        rtol=1e-9,
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_maskless_inference_keeps_native_merge(monkeypatch):
    calls = []
    output = torch.zeros(1, 2, 1, 8)
    lse = torch.zeros(1, 2, 1)

    def native(**kwargs):
        calls.append(kwargs)
        return output, lse

    monkeypatch.setattr(maskless, "_native_merge_attentions", native)
    with torch.no_grad():
        actual = maskless.merge_attentions([output, output], [lse, lse], torch_compile=True)
    assert actual[0] is output
    assert len(calls) == 1 and calls[0]["torch_compile"] is True


@pytest.mark.L0
@pytest.mark.GPU
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Fused-kernel gradients require CUDA")
@pytest.mark.parametrize("splits", [2, 3])
@pytest.mark.parametrize("checkpointed", [False, True])
def test_maskless_merge_matches_fused_gradients_gpu(splits, checkpointed):
    from cosmos_framework.model.attention import attention

    def fused(query, key, value):
        return attention(query, key, value, backend="natten", return_lse=True)

    _check_merge(
        fused,
        device="cuda",
        dtype=torch.bfloat16,
        splits=splits,
        checkpointed=checkpointed,
        atol=0.05,
        rtol=0.05,
    )
