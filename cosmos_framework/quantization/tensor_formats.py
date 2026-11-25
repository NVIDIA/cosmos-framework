# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import torch

FP8_DTYPES = {torch.float8_e4m3fn, getattr(torch, "float8_e5m2", torch.float8_e4m3fn)}
SCALE_SUFFIXES = (".weight_scale", ".weight_scale_2", ".input_scale", ".input_scale_2", ".alpha", ".pre_quant_scale")
