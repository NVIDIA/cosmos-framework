# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import os
from dataclasses import dataclass

import torch
import torch.distributed.checkpoint
from omegaconf import MISSING, OmegaConf


@dataclass
class ConvertPTToDistCPConfig:
    input_path: str = MISSING
    output_dir: str = MISSING


def main():
    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    cfg = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(ConvertPTToDistCPConfig), OmegaConf.from_cli()))
    checkpoint = torch.load(cfg.input_path, map_location="cpu")
    torch.distributed.checkpoint.save(checkpoint, checkpoint_id=cfg.output_dir)


if __name__ == "__main__":
    main()


"""
python cosmos_framework/model/vfm/tokenizers/dc_ae/convert_pt_to_distcp.py \
    input_path=checkpoints/cosmos_8b_wan22_vae/iter_000020000/model_ema_fp32.pt \
    output_dir=checkpoints/cosmos_8b_wan22_vae/iter_000020000/model_dcp_from_torch_save
"""
