# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Register the packaged YAM recipe, then invoke the Cosmos TOML trainer."""

import os
import runpy


def main():
    # Select this worker's GPU before any Cosmos/attention imports.
    import torch

    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    from . import register_domain

    register_domain(os.environ.get("YAM_ACTION_SPACE", "joint"))
    from . import experiment  # noqa: F401  # Register the Hydra experiment.

    runpy.run_module("cosmos_framework.scripts.train", run_name="__main__")


if __name__ == "__main__":
    main()
