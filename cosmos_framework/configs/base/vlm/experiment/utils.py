# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Experiment:
    job_exp: str
    nnode: int
    command_args: List[str]
    job_name: str = None
    init_command: str = ""
    job_group: str = None
    extra_env_vars: Dict[str, str] = field(default_factory=dict)
