# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Stage-3 Transfer SF-DMD with aligned direct resizing and generated RGB history."""

from omegaconf import OmegaConf

from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict
from cosmos_framework.callbacks.wall_clock_checkpoint import WallClockCheckpoint
from cosmos_framework.configs.simulation.common.transfer_distillation import (
    _TF_DCM_TEACHER_NEGATIVE_PROMPT,
)
from cosmos_framework.configs.simulation.common.transfer_profiles import (
    TEACHER_CHECKPOINT,
    make_transfer_teacher_forcing_experiment,
)
from cosmos_framework.configs.simulation.common.transfer_self_forcing import (
    make_transfer_sf_dmd_experiment,
)
from cosmos_framework.data.generator.indexed_transfer_webdataset_loader import IndexedTransferWebDataset

EXPERIMENT_NAME: str = "sim_transfer_stage3_sfdmd"
# Required inputs in the released recipe: the Stage-0 and Stage-2 checkpoint iteration directories
# (filesystem paths that contain model/) and, for an indexed WebDataset, its length-index manifest.
# Credentials default to None: the manifest and WebDataset media then use boto3's standard credential
# chain, and checkpoints are read from the filesystem (set the load_from credentials for s3:// paths).
STAGE2_CHECKPOINT: str = "${oc.env:COSMOS_SIM_TRANSFER_STAGE2_CHECKPOINT}"
_CHECKPOINT_CREDENTIAL: str | None = None
_WEBDATASET_LENGTH_INDEX_MANIFEST: str = "${oc.env:COSMOS_SIM_TRANSFER_LENGTH_INDEX_MANIFEST}"


def make_transfer_self_forcing_dmd_experiment() -> LazyDict:
    """Bootstrap the causal student and bidirectional scorers on the corrected data."""
    # Reuse Stage 1's architecture and control policy rather than inheriting
    # Stage 2's dCM teacher or optimizer. SF-DMD has bidirectional scorers and
    # uses the established population-weighted length schedule with a reference
    # table built from Stage 1's WebDataset media and exact caption source.
    base = OmegaConf.to_container(make_transfer_teacher_forcing_experiment(), resolve=False)
    assert isinstance(base, dict)
    experiment = LazyDict(
        make_transfer_sf_dmd_experiment(
            base,
            name=EXPERIMENT_NAME,
            student_checkpoint={
                "load_path": f"{STAGE2_CHECKPOINT}/model",
                "credentials": _CHECKPOINT_CREDENTIAL,
            },
            teacher_checkpoint={
                "load_path": f"{TEACHER_CHECKPOINT}/model",
                "credentials": _CHECKPOINT_CREDENTIAL,
            },
            stream_configs=(("480", 1, 1),),
            job_group="cosmos3_transfer_self_forcing_dmd",
            length_index_manifest=_WEBDATASET_LENGTH_INDEX_MANIFEST,
        ),
        flags={"allow_objects": True},
    )
    # Keep the deterministic bucket schedule and one resumable worker pool.
    # The source-specific index supplies the exact WebDataset caption window
    # and media byte ranges; Lance row IDs and bucket counts are not reused.
    experiment["dataloader_train"]["dataset"] = L(IndexedTransferWebDataset)(
        manifest_path=_WEBDATASET_LENGTH_INDEX_MANIFEST,
        index_creds_path=_CHECKPOINT_CREDENTIAL,
        index_cache_dir=None,
        creds_path=None,
        dataset_name="transfer",
        caption_key="caption",
        tokenizer_config="${model.config.vlm_config.tokenizer}",
        modality_weights={"edge": 1.0, "blur": 1.0, "depth": 1.0, "seg": 1.0},
        conditioning_config={0: 0.7, 1: 0.2, 2: 0.1},
        resolution="480",
        seed=42,
        max_attempts=100,
        min_num_frames=9,
        cfg_dropout_rate=0.0,
        dense_caption_probability=0.07,
        resize_on_read=True,
    )
    model = experiment["model"]["config"]
    # Match the Stage-2 teacher guidance deliberately: the legacy SF-DMD
    # factory uses CFG 6. Both branches retain identical controls and metadata.
    model["teacher_guidance"] = 3.0
    model["teacher_negative_prompt"] = _TF_DCM_TEACHER_NEGATIVE_PROMPT
    model["teacher_negative_prompt_keep_metadata"] = True
    # The loader aligns complete caption windows before tokenization. Refuse
    # the legacy shortest-rank truncation, which silently shortens rollouts.
    model["sf_require_equal_rollout_lengths"] = True
    model["sf_replayed_training"] = True
    model["sf_replay_block_size"] = 1
    model["sf_replay_include_conditioned_frames"] = False
    model["sf_keep_student_fsdp_unsharded_during_rollout"] = True
    model["sf_defer_teacher_fsdp_reshard"] = True
    # Keep the corrected Stage-2 topology. Transfer self-forcing requires CP=1.
    model["parallelism"]["data_parallel_shard_degree"] = 8
    model["activation_checkpointing"]["mode"] = "full"
    # Stage a pinned CPU snapshot before the background writer persists the
    # student, EMA, fake-score network, and both optimizer/scheduler states.
    experiment["checkpoint"]["dcp_async_mode_enabled"] = True
    # The SF-DMD factory replaces trainer callbacks. Restore timed saves in
    # addition to its 250-iteration milestones, retaining one completed timed save.
    # Distillation defaults do not register this callback, so supply its target.
    experiment["trainer"]["callbacks"]["wall_clock_checkpoint"] = L(WallClockCheckpoint)(
        interval_minutes=30.0,
        keep_last=1,
    )
    return experiment
