# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Tiny 2-block decoder for FSDP primitive tests (no HF download)."""

import torch.nn as nn


class Block(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, x):
        return self.lin(x)


class TinyDecoder(nn.Module):
    def __init__(self, d=16, n=2):
        super().__init__()
        self.embed = nn.Linear(d, d)
        self.blocks = nn.ModuleList([Block(d) for _ in range(n)])
        self.head = nn.Linear(d, d)

    def forward(self, x):
        x = self.embed(x)
        for b in self.blocks:
            x = b(x)
        return self.head(x)
