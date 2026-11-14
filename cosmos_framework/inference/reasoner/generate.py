# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Generate reasoner tasks through the same public runtime/config as evaluation."""

import argparse
import json
from pathlib import Path

try:
    import tomllib as toml
except ModuleNotFoundError:  # Python 3.10; also used by the existing reasoner evaluator.
    try:
        import toml
    except ModuleNotFoundError as error:
        raise ImportError(
            "Python 3.10 generation requires the reasoner-eval extra (toml); Python 3.11+ uses tomllib"
        ) from error

from cosmos_framework.inference.reasoner.runtime import CosmosFrameworkRuntime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="TOML with model, generation, vision and tasks sections")
    parser.add_argument("--output", required=True, help="New JSON output path (existing files are not overwritten)")
    args = parser.parse_args()
    config = toml.loads(Path(args.config).read_text())
    model = dict(config["model"])
    model["model_path"] = model.pop("model_name")
    # Reserve the result before expensive loading; an interrupted invocation
    # leaves an empty file rather than overwriting a previous completed result.
    with Path(args.output).open("x") as output:
        runtime = CosmosFrameworkRuntime(**model)
        answers, losses = runtime.generate_tasks(
            config["tasks"], generation_config=config.get("generation"), vision_config=config.get("vision")
        )
        deployment = runtime.adapter_deployment or getattr(runtime, "base_deployment", None)
        json.dump({"answers": answers, "losses": losses, "deployment": deployment}, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
