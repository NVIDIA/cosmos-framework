# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""High-level entry points for VFM sequence packing."""

from cosmos_framework.data.generator.sequence_packing.modality import ModalityData
from cosmos_framework.data.generator.sequence_packing.packers import pack_input_sequence
from cosmos_framework.data.generator.sequence_packing.sequence import (
    PackedSequence,
    SequencePlan,
    build_sequence_plans_from_data_batch,
)

__all__ = [
    "ModalityData",
    "PackedSequence",
    "SequencePlan",
    "build_sequence_plans_from_data_batch",
    "pack_input_sequence",
]
