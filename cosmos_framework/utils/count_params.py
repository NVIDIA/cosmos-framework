# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from torch import nn


def count_params(model: nn.Module, verbose=False) -> int:
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.0e-6:.2f} M params.")
    return total_params
