# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Local recipes retain their checkpoint callback after OmegaConf insertion."""

import pytest
from omegaconf import OmegaConf

from cosmos_framework.data.generator.local_simulation import resume
from cosmos_framework.data.generator.local_simulation.config import make_local_experiment_config
from cosmos_framework.utils.lazy_config import instantiate

pytestmark = [pytest.mark.level(0), pytest.mark.gpus(0)]


@pytest.mark.parametrize("trainer", [None, {}])
def test_defaults_composed_recipe_retains_local_resume_callback(trainer, monkeypatch):
    monkeypatch.setattr(resume, "_LOADED_STATE", None)
    source = OmegaConf.create({"dataloader_train": {}, "job": {}})
    if trainer is not None:
        source["trainer"] = trainer
    original = OmegaConf.to_container(source)
    config = make_local_experiment_config(source, "transfer")
    callback = instantiate(config.trainer.callbacks.local_simulation_resume)
    callback.load_state_dict({"_local_simulation_resume": {"draw_count": 7}})
    assert callback.state_dict() == {"_local_simulation_resume": {"draw_count": 7}}
    assert OmegaConf.to_container(source) == original
