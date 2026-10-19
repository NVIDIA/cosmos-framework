# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Framework AutoML CLI; GPU dependencies are loaded only by trial commands."""


def main() -> None:
    import sys

    try:
        from automl_core.cli import main as core_main
    except ModuleNotFoundError as exc:
        if exc.name not in {"automl_core", "automl_core.cli"}:
            raise
        raise SystemExit(
            "AutoML requires nvidia-automl-core. Install packages/automl-core from "
            "NVIDIA-TAO/tao-automl (main by default, or your selected development branch) "
            "into this Python environment, then install cosmos-framework[automl,reasoner-eval]. "
            "See docs/automl.md#installation-from-source for source-install commands, "
            "compatible versions, and uv run --no-sync. No published wheel is required."
        ) from exc
    if len(sys.argv) > 1 and sys.argv[1] == "prompts":
        from cosmos_framework.automl.prompts import load

        sys.argv.pop(1)
        core_main(
            loader=load,
            prog="cosmos-automl prompts",
            description="GEPA Auto-Prompter: fixed-checkpoint reasoner evaluation",
        )
    else:
        from cosmos_framework.automl.workload import load

        core_main(
            loader=load,
            description="Cosmos Framework AutoML: bounded reasoner SFT search",
            epilog="For fixed-checkpoint GEPA prompt optimization, use: cosmos-automl prompts --help",
        )


if __name__ == "__main__":
    main()
