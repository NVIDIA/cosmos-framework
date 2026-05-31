# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from cosmos_framework.utils.functional.lr_scheduler import LambdaLinearScheduler
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

LambdaLinearSchedulerConfig: LazyDict = L(LambdaLinearScheduler)(
    warm_up_steps=[1000],
    cycle_lengths=[10000000000000],
    f_start=[1.0e-6],
    f_max=[1.0],
    f_min=[1.0],
)
