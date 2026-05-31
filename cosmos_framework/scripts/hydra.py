# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Hydra CLI."""

import hydra
import omegaconf
from hydra.core.hydra_config import HydraConfig

from cosmos_framework.inference.common.config import CONFIG_DIR


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="base_config")
def main(cfg: omegaconf.DictConfig) -> None:
    hydra_cfg = HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir
    print(f"Config saved to: '{output_dir}/.hydra/config.yaml'")


if __name__ == "__main__":
    main()
