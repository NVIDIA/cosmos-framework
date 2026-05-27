# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from cosmos_framework.inference.common.init import init_script, is_rank0

init_script()

import json
from pathlib import Path
from typing import Annotated

import pydantic
import tyro

from cosmos_framework.inference.args import OmniSetupOverrides
from cosmos_framework.inference.common.args import SampleOutputs, SetupOverrides, tyro_cli
from cosmos_framework.inference.common.init import init_output_dir
from cosmos_framework.utils import log


class InferenceArgs(pydantic.BaseModel):
    input_files: Annotated[list[Path], tyro.conf.arg(aliases=("-i",))]
    """Path to the inference parameter file(s).

    If multiple files are provided, the model will be loaded once and all the samples will be run sequentially.

    Accepts glob patterns (e.g. `inputs/*.json`).
    """

    setup: SetupOverrides = OmniSetupOverrides.model_construct()
    """Setup arguments."""


def inference(args: InferenceArgs):
    from cosmos_framework.inference.common.inference import sync_distributed_errors

    with sync_distributed_errors():
        if args.setup.output_dir is None:
            raise ValueError("'output_dir' is required")
        setup_args = args.setup.build_setup()
        init_output_dir(setup_args.output_dir)
        log.debug(f"{args.__class__.__name__}({args})")
        sample_overrides_list = setup_args.get_sample_overrides_cls().from_files(
            args.input_files, overrides=setup_args.sample_overrides
        )
        log.info(f"Loaded {len(sample_overrides_list)} samples")
        for sample_overrides in sample_overrides_list:
            assert sample_overrides.name
            sample_overrides.output_dir = setup_args.output_dir / sample_overrides.name
            sample_overrides.download(sample_overrides.output_dir / "inputs")

    pipe = setup_args.get_inference_cls().create(setup_args)
    sample_args_list = []
    for overrides in sample_overrides_list:
        try:
            sample_args_list.append(overrides.build_sample(model_config=pipe.model_config))
        except ValueError as e:
            if not setup_args.skip_invalid_samples:
                raise
            msg = f"Skipping sample '{overrides.name}': {e}"
            log.warning(msg)
            overrides.output_dir.mkdir(parents=True, exist_ok=True)
            skip_output = SampleOutputs(args=overrides.model_dump(mode="json"), status="skip", message=msg)
            (overrides.output_dir / "sample_outputs.json").write_text(skip_output.model_dump_json())
    pipe.generate(sample_args_list)

    if setup_args.benchmark and is_rank0():
        benchmark_file = setup_args.output_dir / "benchmark.json"
        benchmark_file.write_text(json.dumps(pipe.get_timer_results(), indent=2, sort_keys=True))
        log.success(f"Saved benchmark to '{benchmark_file}'")


def main():
    args = tyro_cli(
        InferenceArgs,
        description=__doc__,
        config=(
            tyro.conf.OmitArgPrefixes,
            tyro.conf.CascadeSubcommandArgs,
            tyro.conf.OmitSubcommandPrefixes,
        ),
    )
    inference(args)


if __name__ == "__main__":
    main()
