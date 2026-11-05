# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Registration-free teacher-forcing and TF-dCM recipes for four-modality Transfer2."""

import copy

from cosmos_framework.configs.simulation.common.teacher_forcing_model import _MODEL_VIDEO

_CHECKPOINT: str = ""


_CONTROL_INPUT_TYPE = "edge_blur_depth_seg"
_CONDITIONING_CONFIG = {0: 0.7, 1: 0.2, 2: 0.1}
_GLOBAL_CONTROL_VISIBILITY = "global"
_CAUSAL_CONTROL_VISIBILITY = "causal"
# Probability of removing the complete control item after VAE encoding.
_GLOBAL_CONTROL_DROPOUT_RATE = 0.1
_CAUSAL_CONTROL_DROPOUT_RATE = 0.0
# Probability of replacing the caption with an empty string before tokenization.
_DEFAULT_TEXT_DROPOUT_RATE = 0.1
_ZERO_TEXT_DROPOUT_RATE = 0.0
# Minimum source clip length for the selected framewise Transfer recipe.
_MIN_NUM_FRAMES = 9
_MAX_NUM_TOKENS_AFTER_PACKING = 74_000
_TEACHER_FORCING_FRAMES_PER_CHUNK = 1
_CHECKPOINT_SAVE_ITER = 1_000
_DISABLED_WALL_CLOCK_CHECKPOINT_INTERVAL_MINUTES = 0.0
_ACTIVATION_CHECKPOINTING_MODE = "full"
_TRANSFER_RESOLUTION_CONFIGS: tuple[tuple[str, int, int, int], ...] = (
    # Use the Transfer2 480p batch size and frame ceiling: resolution, batch
    # size, ratio, max frames. The ratio is one because this recipe has one stream.
    ("480", 2, 1, 201),
)


_LEGACY_MIN_NUM_FRAMES = 17
_LEGACY_MAX_NUM_TOKENS_AFTER_PACKING = 208_896
_LEGACY_TEACHER_FORCING_FRAMES_PER_CHUNK = 4
_LEGACY_TRANSFER_RESOLUTION_CONFIGS: tuple[tuple[str, int, int, int], ...] = (("480", 2, 4, 601),)
_LEGACY_TRANSFER_JOB_GROUP = "cosmos3_transfer_teacher_forcing"




def _make_experiment(
    control_visibility: str,
    controls_read_strict_past_clean_rgb: bool,
    control_dropout_rate: float,
    text_dropout_rate: float,
    job_name: str,
    *,
    min_num_frames: int = _LEGACY_MIN_NUM_FRAMES,
    max_num_tokens_after_packing: int = _LEGACY_MAX_NUM_TOKENS_AFTER_PACKING,
    teacher_forcing_frames_per_chunk: int = _LEGACY_TEACHER_FORCING_FRAMES_PER_CHUNK,
    resolution_configs: tuple[tuple[str, int, int, int], ...] = _LEGACY_TRANSFER_RESOLUTION_CONFIGS,
    job_group: str = _LEGACY_TRANSFER_JOB_GROUP,
    model_resolution: str = "480",
    rectified_flow_shift: int | dict[str, int] = 5,
    encode_exact_durations: tuple[int, ...] = (17, 61, 73),
    use_lance_dataloader: bool = False,
    checkpoint_save_iter: int = 1_000,
    wall_clock_checkpoint_interval_minutes: float | None = None,
    target_only_no_text_pass2: bool = False,
    activation_checkpointing_mode: str = "full",
) -> dict[str, object]:
    if control_visibility not in (_GLOBAL_CONTROL_VISIBILITY, _CAUSAL_CONTROL_VISIBILITY):
        raise ValueError(f"Unknown transfer control visibility: {control_visibility!r}")

    model_config = copy.deepcopy(_MODEL_VIDEO)
    model_config["compile"]["enabled"] = False
    model_config["activation_checkpointing"] = {"mode": activation_checkpointing_mode}
    model_config["resolution"] = model_resolution
    model_config["rectified_flow_training_config"]["shift"] = copy.deepcopy(rectified_flow_shift)
    model_config["max_num_tokens_after_packing"] = max_num_tokens_after_packing
    model_config["state_t"] = 300
    model_config["tokenizer"]["encode_exact_durations"] = list(encode_exact_durations)
    model_config["teacher_forcing_detach_clean_kv"] = False
    model_config["teacher_forcing_target_only_no_text_pass2"] = target_only_no_text_pass2
    model_config["teacher_forcing_frames_per_chunk"] = teacher_forcing_frames_per_chunk
    model_config["teacher_forcing_transfer_control_dropout_rate"] = control_dropout_rate
    model_config["teacher_forcing_kv_implementation"] = "singleview_threeway_kv"
    model_config["teacher_forcing_replay_policy"] = {
        "control_visibility": control_visibility,
        "controls_read_strict_past_clean_rgb": controls_read_strict_past_clean_rgb,
        "clean_pass_causality": "frame",
        "multiview_attention_scope": "all_views",
        "decomposed_temporal_window_seconds": None,
    }

    trainer_callbacks: dict[str, dict[str, object]] = {
        "grad_clip": {"clip_norm": 1.0},
        "manual_gc": {"every_n": 200},
        "compile_tokenizer": {
            "enabled": True,
            "warmup_resolutions": [resolution for resolution, _, _, _ in resolution_configs],
        },
    }
    if wall_clock_checkpoint_interval_minutes is not None:
        # An explicit value overrides COSMOS3_CHECKPOINT_WALL_CLOCK_MINUTES from the launcher.
        trainer_callbacks["wall_clock_checkpoint"] = {
            "interval_minutes": wall_clock_checkpoint_interval_minutes,
            "keep_last": 0,
        }

    return {
        "defaults": [
            {"override /data_train": None},
            {"override /data_val": None},
            {"override /model": "mot_fsdp_causal"},
            {"override /optimizer": "fusedadamw"},
            {"override /scheduler": "lambdalinear"},
            {"override /tokenizer": "wan2pt2_tokenizer"},
            {"override /vlm_config": "cosmos3_nano_reasoner_vlm_gcp_0517"},
            {"override /checkpoint": "local"},
            {"override /callbacks": ["basic", "optimization", "job_monitor"]},
            "_self_",
        ],
        "job": {
            "project": "cosmos3_simulation",
            "group": job_group,
            "name": job_name,
        },
        "trainer": {
            "max_iter": 100_000,
            "logging_iter": 200,
            "callbacks": trainer_callbacks,
        },
        "model": {"config": {**model_config, "causal_training_strategy": "teacher_forcing"}},
        "optimizer": {
            "lr": 1e-5,
            "betas": [0.9, 0.999],
            "weight_decay": 0.01,
            "keys_to_select": ["moe_gen", "time_embedder", "vae2llm", "llm2vae"],
        },
        "scheduler": {"f_max": [1.0], "f_min": [1.0], "warm_up_steps": [0], "cycle_lengths": [100_000]},
        "checkpoint": {
            "save_iter": checkpoint_save_iter,
            "load_path": _CHECKPOINT,
            "load_training_state": False,
            "strict_resume": True,
        },
        # The released recipe ships no training data: the trainer refuses to start until the
        # experiment supplies a loader. That loader owns the data policy passed in above:
        # text_dropout_rate, min_num_frames, the resolution_configs streams and
        # teacher_forcing_frames_per_chunk.
        "dataloader_train": {"_target_": "???"},
    }


