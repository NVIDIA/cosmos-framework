# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import torch
import wandb

from cosmos_framework.callbacks.every_n import EveryN
from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.trainer import ImaginaireTrainer
from cosmos_framework.data.generator.sequence_packing.runtime import get_padding_stats


class SequencePackingPadding(EveryN):
    """
    Callback that saves lengths to which und and gen sequences are padded. This information will be used
    to compute FLOPs done during training.

    Args:
        every_n (int): Frequency with which callback is run during training.
    """

    def __init__(self, every_n: int = 500):
        super().__init__(every_n=every_n, step_size=1, barrier_after_run=False, run_at_start=True)

    def every_n_impl(
        self,
        trainer: ImaginaireTrainer,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int,
    ) -> None:
        if wandb.run:
            padding_stats = get_padding_stats()
            log_dict = {
                "SequencePackingPadding/max_causal_len_image_batch": padding_stats["MAX_CAUSAL_LEN_IMAGE_BATCH"],
                "SequencePackingPadding/max_full_len_image_batch": padding_stats["MAX_FULL_LEN_IMAGE_BATCH"],
                "SequencePackingPadding/max_causal_len_video_batch": padding_stats["MAX_CAUSAL_LEN_VIDEO_BATCH"],
                "SequencePackingPadding/max_full_len_video_batch": padding_stats["MAX_FULL_LEN_VIDEO_BATCH"],
            }
            modality = "video"
            if "is_image_batch" in output_batch:
                modality = "image" if output_batch["is_image_batch"] else "video"
            if "und_token_length" in output_batch:
                log_dict[f"SequencePackingPadding/und_token_length_{modality}"] = output_batch["und_token_length"]
            if "gen_token_length" in output_batch:
                log_dict[f"SequencePackingPadding/gen_token_length_{modality}"] = output_batch["gen_token_length"]
            if "action_token_length" in output_batch:
                log_dict[f"SequencePackingPadding/action_token_length"] = output_batch["action_token_length"]
            if "sound_token_length" in output_batch:
                log_dict[f"SequencePackingPadding/sound_token_length"] = output_batch["sound_token_length"]
            if "vision_token_length" in output_batch:
                log_dict[f"SequencePackingPadding/vision_token_length"] = output_batch["vision_token_length"]

            wandb.log(
                log_dict,
                step=iteration,
            )
