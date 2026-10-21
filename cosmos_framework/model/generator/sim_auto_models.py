# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Models for the standard three-stage driving recipe."""

from __future__ import annotations

from typing import Any

from cosmos_framework.model.generator.consistency_model.tf_dcm_joint import TeacherForcingJointDCMModel
from cosmos_framework.model.generator.utils.multiview_negative_prompt import multiview_negative_captions


class SimAutoTeacherForcingDCMModel(TeacherForcingJointDCMModel):
    """Retain endpoint losses and enable matched camera metadata for CFG."""

    def _make_teacher_negative_captions(self, batch_size: int, data_batch: dict[str, Any]) -> list[str]:
        if not self.config.teacher_forcing_dcm_teacher_negative_prompt_keep_metadata:
            return super()._make_teacher_negative_captions(batch_size, data_batch)
        return multiview_negative_captions(
            data_batch,
            batch_size,
            self.config.teacher_forcing_dcm_teacher_negative_prompt,
            video_key=self.input_video_key,
        )
