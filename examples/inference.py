# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Inference example."""

from cosmos_framework.inference.common.init import init_output_dir, init_script

init_script()

from pathlib import Path

import safetensors.torch
import torch

from cosmos_framework.inference.args import DEFAULT_CHECKPOINT, OmniSampleOverrides
from cosmos_framework.inference.inference import get_sample_data
from cosmos_framework.inference.model import Cosmos3OmniModel
from cosmos_framework.utils import log
from cosmos_framework.tools.visualize.video import save_img_or_video
from cosmos_framework.configs.base.defaults.compile import CompileConfig


def inference():
    name = "inference"
    output_dir = Path(f"outputs/{name}").absolute()
    init_output_dir(output_dir)

    log.info("Loading model...")
    checkpoint_path = DEFAULT_CHECKPOINT.download()
    model = Cosmos3OmniModel.from_pretrained_dcp(
        Path(checkpoint_path),
        compile_config=CompileConfig(enabled=True),
    ).model

    # Create batch
    sample_args = OmniSampleOverrides(
        name=name,
        output_dir=output_dir,
        prompt="A medium shot of a modern robotics research laboratory with white walls and a gray floor. A robotic arm with a metallic finish is mounted on a clean white workbench, its gripper positioned above a row of small colored objects. A laptop and neatly arranged tools sit beside the robot. A large monitor on the wall behind displays a software interface. The scene is brightly lit by overhead fluorescent lights.",
        num_frames=1,
    ).build_sample(model_config=model.config)
    data_batch = get_sample_data(sample_args, model)

    # Generate samples
    log.info("Generating samples...")
    outputs = model.generate_samples_from_batch(data_batch, seed=[0])

    # Decode
    def decode_vision(vision_latent: torch.Tensor) -> torch.Tensor:
        vision = model.decode(vision_latent)  # Decode to pixel space
        return (1.0 + vision.clamp(-1, 1)) / 2  # [0, 1]

    outputs["vision"] = [decode_vision(vision) for vision in outputs.pop("vision")]
    outputs = {k: torch.cat(v, dim=0) for k, v in outputs.items()}

    # Save outputs
    log.info("Saving outputs...")
    safetensors.torch.save_file(outputs, output_dir / "outputs.safetensors")
    save_img_or_video(outputs["vision"][0], str(output_dir / "vision"), fps=data_batch["fps"][0].item())
    log.success(f"Saved outputs to {output_dir}")


def main():
    inference()


if __name__ == "__main__":
    main()
