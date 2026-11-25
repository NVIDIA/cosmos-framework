# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The real posttrain adapter build paths under FSDP2 and HSDP on CPU processes (gloo).

``build_hf_model`` and the native Qwen3-VL builder, each at FSDP 1x2 and HSDP 2x2 with a bfloat16 base: every
rank seeds differently, so equal results prove every rank received the same initial adapters; gradients must
match one process on the whole batch. L1 (spawns up to four processes per case); the core sharding and
initialization mechanics are covered at L0 by ``adapters_distributed_test.py``.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.multiprocessing as mp

from cosmos_framework.utils.generator.adapter_config import AdapterComponentConfig, AdapterConfig, AdapterSetup
from cosmos_framework.utils.generator.adapters_sft_test import build_hf, text_adapters, tiny_qwen3_vl, tiny_text

pytestmark = [pytest.mark.L1, pytest.mark.CPU]

_ROWS = 4


def _native_adapters() -> AdapterConfig:
    return AdapterConfig(
        components=[
            AdapterComponentConfig(
                targets=["model.language_model.layers.*.self_attn.q_proj"], method="dora", rank=4, use_rslora=True
            ),
            AdapterComponentConfig(
                component="vision",
                targets=["model.visual.blocks.*.attn.qkv"],
                layers=[-1],
                method="pissa",
                initialization="svd_lowrank",
                rank=2,
            ),
        ]
    )


def _build(kind: str, base: str, pdims: Any) -> torch.nn.Module:
    if kind == "hf":
        adapters = text_adapters(initialization="svd_lowrank")
        adapters.components[2].dropout = 0.0  # Dropout masks differ between the sharded and the reference batches.
        return build_hf(Path(base), adapters, pdims=pdims, dtype="bfloat16", grad_ckpt=True).net
    from cosmos_framework.posttrain.config.model import Qwen3VLModelConfig
    from cosmos_framework.posttrain.model.qwen3_vl import build_qwen3_vl_model

    cfg = Qwen3VLModelConfig(name_or_path=base, dtype="bfloat16", attn_impl="sdpa", activation_checkpointing="full")
    return build_qwen3_vl_model(cfg, pdims, adapters=AdapterSetup(_native_adapters())).net


def _ids() -> torch.Tensor:
    return torch.randint(0, 60, (_ROWS, 8), generator=torch.Generator().manual_seed(5))


def _loss(net: torch.nn.Module, ids: torch.Tensor) -> torch.Tensor:
    logits = net(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
    return logits.float().square().mean()


def _full(tensor: torch.Tensor) -> torch.Tensor:
    from torch.distributed.tensor import DTensor

    return (tensor.full_tensor() if isinstance(tensor, DTensor) else tensor).detach().float().clone()


def _trainable(net: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    return {name: p for name, p in net.named_parameters() if p.requires_grad}


def _worker(rank: int, world: int, replicate: int, kind: str, base: str, init_file: str, out: str) -> None:
    import torch.distributed as dist

    torch.cuda.is_available = lambda: False  # CPU processes on a GPU host as well
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world, timeout=timedelta(seconds=120)
    )
    try:
        from cosmos_framework.posttrain.config.build import resolve_pdims

        pdims = resolve_pdims("cpu", parallelism=(replicate, world // replicate))
        torch.manual_seed(100 + rank)
        net = _build(kind, base, pdims)
        initial = {name: _full(p) for name, p in _trainable(net).items()}
        rows = _ROWS // world
        _loss(net, _ids()[rank * rows : (rank + 1) * rows]).backward()
        grads = {name: _full(p.grad) for name, p in _trainable(net).items()}
        if rank == 0:
            torch.save({"initial": initial, "grads": grads}, out)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    ("kind", "world", "replicate"),
    [("hf", 2, 1), ("hf", 4, 2), ("native", 2, 1), ("native", 4, 2)],
    ids=["hf_fsdp_1x2", "hf_hsdp_2x2", "native_fsdp_1x2", "native_hsdp_2x2"],
)
def test_sharded_build_matches_single_process(
    kind: str, world: int, replicate: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if kind == "hf":
        base = tiny_text(tmp_path / "base", dtype=torch.bfloat16)
    else:
        base = tiny_qwen3_vl(tmp_path / "base")
    out = str(tmp_path / "rank0.pt")
    mp.spawn(
        _worker, args=(world, replicate, kind, str(base), str(tmp_path / "rendezvous"), out), nprocs=world, join=True
    )
    result = torch.load(out)

    torch.manual_seed(100)
    reference = _build(kind, str(base), None)
    expected_initial = {name: _full(p) for name, p in _trainable(reference).items()}
    assert result["initial"].keys() == expected_initial.keys()
    for name, tensor in expected_initial.items():
        torch.testing.assert_close(result["initial"][name], tensor, atol=0, rtol=0, msg=name)
    _loss(reference, _ids()).backward()
    for name, parameter in _trainable(reference).items():
        torch.testing.assert_close(result["grads"][name], _full(parameter.grad), atol=2e-4, rtol=2e-2, msg=name)
