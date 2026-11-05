# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Registration-free production Transfer teacher-forcing and TF-dCM profiles."""


from omegaconf import OmegaConf

from cosmos_framework.utils.lazy_config import LazyDict
from cosmos_framework.configs.simulation.common.transfer_distillation import make_transfer_tf_dcm_experiment
from cosmos_framework.configs.simulation.common.transfer_recipes import _make_experiment

STAGE1_EXPERIMENT_NAME: str = "sim_transfer_stage1_tf"
# The Stage-0 teacher checkpoint iteration directory (contains model/); a required input in the released recipe.
TEACHER_CHECKPOINT: str = "${oc.env:COSMOS_SIM_TRANSFER_STAGE0_CHECKPOINT}"
_RESOLUTION_CONFIGS: tuple[tuple[str, int, int, int], ...] = (
    # Resolution, loader batch size, stream sampling weight, maximum RGB frames.
    ("480", 2, 1, 201),
)


def make_transfer_teacher_forcing_experiment() -> LazyDict:
    """Reuse the causal Transfer loss with the corrected teacher's data policy."""
    experiment = LazyDict(
        _make_experiment(
            control_visibility="causal",
            controls_read_strict_past_clean_rgb=True,
            control_dropout_rate=0.0,
            text_dropout_rate=0.1,
            job_name=STAGE1_EXPERIMENT_NAME,
            min_num_frames=9,
            max_num_tokens_after_packing=74_000,
            teacher_forcing_frames_per_chunk=1,
            resolution_configs=_RESOLUTION_CONFIGS,
            job_group="cosmos3_transfer_teacher_forcing",
            model_resolution="480",
            rectified_flow_shift=5,
            encode_exact_durations=(17, 61, 73),
            use_lance_dataloader=False,
            checkpoint_save_iter=500,
            wall_clock_checkpoint_interval_minutes=30.0,
            target_only_no_text_pass2=True,
            activation_checkpointing_mode="full",
        ),
        flags={"allow_objects": True},
    )
    experiment["checkpoint"].update(
        load_path=TEACHER_CHECKPOINT,
        load_training_state=False,
        load_ema_to_reg=True,
    )
    # Keep the latest timed save in addition to the 500-iteration milestones.
    experiment["trainer"]["callbacks"]["wall_clock_checkpoint"]["keep_last"] = 1
    return experiment


STAGE2_EXPERIMENT_NAME: str = "sim_transfer_stage2_tfdcm"
# The Stage-1 checkpoint iteration directory (contains model/); a required input in the released recipe.
_STAGE1_CHECKPOINT_KEY: str = ""
STAGE1_CHECKPOINT: str = "${oc.env:COSMOS_SIM_TRANSFER_STAGE1_CHECKPOINT}"


def make_transfer_teacher_forcing_dcm_experiment() -> LazyDict:
    """Distill Stage-1 EMA weights while retaining its corrected data and replay policy."""
    # The shared factory replaces defaults in plain dictionaries. Preserve lazy
    # interpolation while converting the Stage-1 LazyDict, including its defaults.
    stage1 = OmegaConf.to_container(make_transfer_teacher_forcing_experiment(), resolve=False)
    assert isinstance(stage1, dict)
    experiment = LazyDict(
        make_transfer_tf_dcm_experiment(
            stage1,
            name=STAGE2_EXPERIMENT_NAME,
            stage1_checkpoint=_STAGE1_CHECKPOINT_KEY,
            job_group="cosmos3_transfer_teacher_forcing_dcm",
            # Inherit Stage 1's 480p shift of 5 rather than the factory default of 3.
            sample_timestep_shift=None,
        ),
        flags={"allow_objects": True},
    )
    # Pin the same EMA checkpoint for student initialization and the frozen
    # teacher. Only the student's own Stage-2 training state resumes on restart.
    experiment["checkpoint"]["load_path"] = STAGE1_CHECKPOINT
    model_config = experiment["model"]["config"]
    model_config["teacher_forcing_dcm_teacher_load_from"]["load_path"] = "${checkpoint.load_path}/model"
    # The factory supplies Transfer's negative prompt, CFG 3, and clip metadata
    # for the negative branch. Retain the aligned control/target replay optimization.
    model_config["teacher_forcing_target_only_no_text_pass2"] = True
    model_config["teacher_forcing_dcm_defer_fsdp_reshard"] = True
    # Keep Stage 1's full activation checkpointing and decoder memory bound.
    # The factory disables timed saves; restore them alongside its 250-step saves.
    # Retain only the latest completed timed save; iteration milestones are preserved.
    experiment["trainer"]["callbacks"]["wall_clock_checkpoint"] = {
        "interval_minutes": 30.0,
        "keep_last": 1,
    }
    return experiment
