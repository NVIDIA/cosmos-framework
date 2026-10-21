# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Posttrain config system (DESIGN: config system).

- `model.py` — shared dependency-light model backend and parallelism schemas.
- `rl.py` — typed RL schema (RunConfig lifecycle + optimizer; DPOConfig/GRPOConfig/OPDConfig
  algorithm knobs; GridConfig). Dependency-light, OmegaConf-round-trippable.
- `build.py` — lazy imports, general parallelism resolution, optimizer construction, and thin
  backend dispatch. Model-family construction lives in `model/*.py`.
- `config.py` — the shared root machinery: `PosttrainConfig(cosmos3.Config)` base + `build_root`
  (ConfigStore registration + job/trainer defaults). The runnable roots are the thin per-algorithm
  modules `config_dpo.py`, `config_grpo.py`, and `config_opd.py`, each the `--config` target its
  entry loads. OPD inherits the GRPO root and changes only `algorithm` and `rl`.
- `experiment/` — the *config* experiments (which model/data/rl), inheriting an i4 experiment
  via `defaults` + an `rl` section, selected with `experiment=` and overridden via dotlist.
  Runnable entries live under `entrypoints/`; concrete recipes live under `configs/`.
"""
