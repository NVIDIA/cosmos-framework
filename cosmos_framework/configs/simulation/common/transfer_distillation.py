# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Registration-free TF-dCM conversion for aligned four-modality Transfer recipes."""

import copy
import json
from importlib.resources import files
from typing import Any

_TF_DCM_OPTIMIZER_KEYS = ["moe_gen", "time_embedder", "vae2llm", "llm2vae"]
_TF_DCM_FIXED_STEP_SAMPLER_CONFIG = {
    "t_list": [1.0, 15 / 16, 5 / 6, 5 / 8],
    "sample_type": "sde",
}
# Match transfer inference's default negative caption, including JSON serialization.
# Store the complete caption in the resolved config for checkpoint reproducibility.
_TF_DCM_TEACHER_NEGATIVE_PROMPT = json.dumps(
    json.loads(files("cosmos_framework.inference").joinpath("defaults/neg_prompts.json").read_text(encoding="utf-8"))
)


def _replace_default(defaults: list[Any], key: str, value: Any) -> list[Any]:
    """Replace one Hydra default while preserving the source experiment order."""
    updated: list[Any] = []
    for default in defaults:
        if isinstance(default, dict) and key in default:
            updated.append({key: copy.deepcopy(value)})
        else:
            updated.append(copy.deepcopy(default))
    return updated


def make_transfer_tf_dcm_experiment(
    base_experiment: dict[str, Any],
    *,
    name: str,
    stage1_checkpoint: str,
    job_group: str | None = None,
    sample_timestep_shift: float | None = 3.0,
) -> dict[str, Any]:
    """Convert an aligned Transfer TF recipe into its TF-dCM stage."""
    experiment = copy.deepcopy(base_experiment)
    experiment["defaults"] = _replace_default(experiment["defaults"], "override /model", "mot_fsdp_tf_dcm")
    experiment["job"]["name"] = name
    if job_group is not None:
        experiment["job"]["group"] = job_group

    experiment["trainer"]["max_iter"] = 10_000
    experiment["trainer"]["logging_iter"] = 10
    # Disable elapsed-time checkpoint saves for this experiment even if the
    # launcher exports COSMOS3_CHECKPOINT_WALL_CLOCK_MINUTES.
    experiment["trainer"]["callbacks"]["wall_clock_checkpoint"] = {
        "interval_minutes": 0.0,
        "keep_last": 0,
    }

    model_config = experiment["model"]["config"]
    model_config["causal_training_strategy"] = "teacher_forcing_dcm"
    model_config["teacher_forcing_target_only_no_text_pass2"] = False
    model_config["teacher_forcing_dcm_defer_fsdp_reshard"] = False
    model_config["teacher_forcing_dcm_total_steps"] = 48
    model_config["teacher_forcing_dcm_skipping_interval_steps"] = 1
    model_config["teacher_forcing_dcm_sample_timestep_shift"] = sample_timestep_shift
    model_config["teacher_forcing_dcm_teacher_guidance"] = 3.0
    model_config["teacher_forcing_dcm_teacher_negative_prompt"] = _TF_DCM_TEACHER_NEGATIVE_PROMPT
    model_config["teacher_forcing_dcm_teacher_negative_prompt_keep_metadata"] = True
    # The frozen teacher loader consumes the DCP model component, while the
    # ordinary student warm start below consumes the checkpoint root.
    model_config["teacher_forcing_dcm_teacher_load_from"] = {
        "load_path": "",
        "credentials": None,
    }
    model_config["teacher_forcing_dcm_teacher_checkpoint_prefix"] = "net_ema"
    model_config["fixed_step_sampler_config"] = copy.deepcopy(_TF_DCM_FIXED_STEP_SAMPLER_CONFIG)
    model_config["rectified_flow_training_config"]["loss_scale"] = 100.0

    experiment["optimizer"] = {
        "lr": 2e-6,
        "betas": [0.0, 0.999],
        "weight_decay": 0.01,
        "keys_to_select": copy.deepcopy(_TF_DCM_OPTIMIZER_KEYS),
    }
    experiment["scheduler"] = {
        "f_max": [1.0],
        "f_min": [1.0],
        "warm_up_steps": [0],
        "cycle_lengths": [10_000],
    }
    experiment["checkpoint"] = {
        **copy.deepcopy(experiment["checkpoint"]),
        "save_iter": 250,
        "load_path": stage1_checkpoint,
        "load_ema_to_reg": True,
        "load_training_state": False,
        "strict_resume": True,
        "keys_to_skip_loading": [],
    }
    return experiment
