# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""roles.teacher — the OPD teacher body (design 2026-08-07 §1). `vllm` holds the engine in-process;
a remote/disaggregated body drops in as a sibling behind the same `TeacherBody` contract."""

from cosmos_framework.posttrain.roles.teacher.base import TeacherBody, TeacherContractError
from cosmos_framework.posttrain.roles.teacher.vllm import VLLMTeacher

__all__ = ["TeacherBody", "TeacherContractError", "VLLMTeacher"]
