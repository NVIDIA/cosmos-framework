# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Inference example using the pipeline."""

from cosmos_framework.inference.common.init import init_output_dir, init_script

init_script()

from pathlib import Path

from cosmos_framework.inference.args import DEFAULT_CHECKPOINT_NAME, OmniSampleOverrides, OmniSetupOverrides
from cosmos_framework.inference.inference import OmniInference, get_sample_data
from cosmos_framework.utils import log


def inference_pipeline():
    name = "inference_pipeline"
    output_dir = Path(f"outputs/{name}").absolute()
    init_output_dir(output_dir)

    log.info("Loading model...")
    setup_args = OmniSetupOverrides(
        checkpoint_path=DEFAULT_CHECKPOINT_NAME,
        output_dir=output_dir,
    ).build_setup()
    pipe = OmniInference.create(setup_args)

    sample_args = OmniSampleOverrides(
        name=name,
        output_dir=output_dir,
        prompt="the quick brown fox is happily jumping over the fence.",
        num_frames=1,
    ).build_sample(model_config=pipe.model_config)
    data_batch = get_sample_data(sample_args, model=pipe.model)

    log.info("Generating samples...")
    pipe.generate_batch([sample_args], data_batch)


def main():
    inference_pipeline()


if __name__ == "__main__":
    main()
