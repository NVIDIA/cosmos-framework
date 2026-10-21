# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""In-memory SFT shaping and tiny policy fixtures; no tokenizer/model downloads."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from cosmos_framework.posttrain.data.packer.decoder_only_llm import DecoderOnlyLLMDataPacker
from cosmos_framework.posttrain.data.packer.sft import SFTPolicyInput
from cosmos_framework.posttrain.schema import ObsBundle, SFTTarget, Status, Trajectory, Transition


class MemoryPacker(DecoderOnlyLLMDataPacker):
    samples: dict[str, SFTPolicyInput]
    masking: str
    padding: int

    def __init__(self, *, masking: str = "assistant", padding: int = 0) -> None:
        self.samples = {}
        self.masking = masking
        self.padding = padding

    def add(
        self,
        ids: list[int],
        labels: list[int],
        *,
        weight: float | None = None,
        mm_kwargs: dict[str, Any] | None = None,
    ) -> Trajectory:
        key = str(len(self.samples))
        self.samples[key] = SFTPolicyInput(ids, labels, mm_kwargs)
        metadata = {} if weight is None else {"validation_weight": weight}
        return Trajectory(
            id=key,
            status=Status.COMPLETED,
            transitions=[
                Transition(
                    obs=ObsBundle(),
                    targets=SFTTarget(messages=[{"role": "assistant", "content": key}]),
                    algo_extra=metadata,
                )
            ],
        )

    def validation_semantics_identity(self) -> dict[str, Any]:
        return {"masking": self.masking, "padding": self.padding}

    def sft_process_sample(self, sample: SFTTarget) -> SFTPolicyInput:
        return self.samples[sample.messages[0]["content"]]

    def sft_compute_max_len(self, processed_samples: list[SFTPolicyInput]) -> int:
        return max(len(sample.input_ids) for sample in processed_samples) + self.padding

    def sft_collate_fn(
        self, sub_batch: list[SFTPolicyInput], computed_max_len: int, ignore_label_id: int = -100
    ) -> dict[str, Any]:
        ids = torch.zeros((len(sub_batch), computed_max_len), dtype=torch.long)  # [B,L]
        labels = torch.full_like(ids, ignore_label_id)  # [B,L]
        attention = torch.zeros_like(ids)  # [B,L]
        media: dict[str, list[Any]] = {}
        for index, sample in enumerate(sub_batch):
            size = len(sample.input_ids)
            ids[index, :size] = torch.tensor(sample.input_ids)
            labels[index, :size] = torch.tensor(sample.labels)
            attention[index, :size] = 1
            for key, value in (sample.mm_kwargs or {}).items():
                media.setdefault(key, []).append(value)
        return {"input_ids": ids, "labels": labels, "attention_mask": attention, **media}


class TinyNet(nn.Module):
    embedding: nn.Embedding
    dropout: nn.Dropout
    head: nn.Linear

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.dropout = nn.Dropout(0.0)
        self.head = nn.Linear(8, 16)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **media: Any) -> Any:
        hidden = self.dropout(self.embedding(input_ids))  # [B,L,D]
        if media:
            # Real conditioning: dropping this nested side input changes the answer.
            hidden = hidden + torch.stack([value["features"] for value in media["media"]])[:, None, :]
        return SimpleNamespace(logits=self.head(hidden))  # [B,L,V]


class Policy(nn.Module):
    net: nn.Module
    calls: int
    last_batch: dict[str, Any] | None
    last_logits: torch.Tensor | None

    def __init__(self, net: nn.Module | None = None) -> None:
        super().__init__()
        self.net = net if net is not None else TinyNet()
        self.calls = 0
        self.last_batch = None
        self.last_logits = None

    def forward_policy_batch(self, batch: dict[str, Any]) -> Any:
        self.calls += 1
        self.last_batch = batch
        output = self.net(**batch)
        self.last_logits = output.logits
        if output.logits.requires_grad:
            output.logits.retain_grad()
        return output


def sample_batch(packer: MemoryPacker) -> list[Trajectory]:
    return [
        packer.add([1, 2, 3], [-100, -100, 3]),
        packer.add([4, 5, 6, 7, 8, 9], [-100, 5, -100, 7, 8, 9]),
    ]
