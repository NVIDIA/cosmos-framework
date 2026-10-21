# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
import os
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

from cosmos_framework.inference.common.multiview_captions import (
    JOINT_CAM_LIDAR_TASK,
    PER_VIEW_CAPTION_TASK_KEY,
    PER_VIEW_CAPTIONS_KEY,
    exported_caption_task,
    exported_checkpoint_metadata,
    exports_per_view_captions,
    per_view_caption_export_fields,
    per_view_caption_tasks,
    separate_view_text_caption_streams,
)

# A config.json as imaginaire4's export_model writes it for the camera+LiDAR production
# checkpoint. Kept verbatim rather than paraphrased: these two keys are a cross-repo contract,
# and a fixture that invents its own shape passes while the real artifact reads as merged.
# Point COSMOS_EXPORTED_CHECKPOINT at an exported checkpoint directory to pin these keys
# against a real artifact. Not defaulted to a path: exports live wherever they were written.
_EXPORTED_CHECKPOINT_ENV = "COSMOS_EXPORTED_CHECKPOINT"

_EXPORTED_CONFIG = {
    "model_type": "cosmos3_omni",
    PER_VIEW_CAPTIONS_KEY: True,
    PER_VIEW_CAPTION_TASK_KEY: "av_joint_camera_lidar_transfer",
    "model": {"config": {}},
}


def test_reads_both_keys_off_an_exported_config() -> None:
    config = SimpleNamespace(**_EXPORTED_CONFIG)
    assert exports_per_view_captions(config) is True
    assert exported_caption_task(config) == "av_joint_camera_lidar_transfer"


def test_reads_them_off_a_plain_mapping_too() -> None:
    """`config.json` is read as a dict in some paths and as an object in others."""
    assert exports_per_view_captions(_EXPORTED_CONFIG) is True
    assert exported_caption_task(_EXPORTED_CONFIG) == "av_joint_camera_lidar_transfer"


def test_absent_keys_read_as_not_per_view() -> None:
    """Every checkpoint exported before the keys existed lands here.

    Safe only because it is paired with a refusal: a per-view checkpoint that does not say so is
    served as merged text, which is wrong and silent. Absence is not evidence of merged captions.
    """
    config = SimpleNamespace(model_type="cosmos3_omni")
    assert exports_per_view_captions(config) is False
    assert exported_caption_task(config) is None


def test_per_view_without_a_recorded_task() -> None:
    """The exporter leaves the task unset when it cannot determine it, or when streams disagree."""
    config = SimpleNamespace(**{PER_VIEW_CAPTIONS_KEY: True})
    assert exports_per_view_captions(config) is True
    assert exported_caption_task(config) is None


def test_empty_task_string_is_not_a_task() -> None:
    config = SimpleNamespace(**{PER_VIEW_CAPTIONS_KEY: True, PER_VIEW_CAPTION_TASK_KEY: ""})
    assert exported_caption_task(config) is None


def test_tolerates_a_missing_config() -> None:
    assert exports_per_view_captions(None) is False
    assert exported_caption_task(None) is None


def test_matches_a_real_exported_artifact_when_one_is_available() -> None:
    """Pins the contract against an artifact imaginaire4 actually produced, not a fixture.

    Skipped where the artifact is not mounted; this is the check that would have caught a key
    renamed on the exporter side.
    """
    exported_dir = os.environ.get(_EXPORTED_CHECKPOINT_ENV)
    if not exported_dir:
        return
    artifact = Path(exported_dir) / "config.json"
    if not artifact.is_file():
        return

    config = json.loads(artifact.read_text())
    assert exports_per_view_captions(config) is True
    assert exported_caption_task(config) == "av_joint_camera_lidar_transfer"


def test_reads_the_keys_out_of_a_checkpoint_directory(tmp_path: Path) -> None:
    """The load path has no other source: the wrapper config keeps only `model`.

    `_create` builds `Cosmos3OmniConfig(model=config_dict["model"])` for an exported checkpoint,
    so by the time a model object exists every other top-level key has been dropped. Reading the
    file is what makes the recorded layout visible at all.
    """
    (tmp_path / "config.json").write_text(json.dumps(_EXPORTED_CONFIG))

    metadata = exported_checkpoint_metadata(tmp_path)
    assert exports_per_view_captions(metadata) is True
    assert exported_caption_task(metadata) == "av_joint_camera_lidar_transfer"


def test_the_model_section_alone_loses_the_keys() -> None:
    """The exact regression: this is what the loaded wrapper carries, and it declares nothing.

    Reading the wrapper instead of the file reported "not per-view" for the camera+LiDAR
    production export, so the refusal never fired and cosmos-framework sampled it with one
    merged prompt -- the silent wrong-conditioning outcome the refusal exists to prevent.
    """
    wrapper_config = SimpleNamespace(model=_EXPORTED_CONFIG["model"])
    assert exports_per_view_captions(wrapper_config) is False
    assert exported_caption_task(wrapper_config) is None


def test_a_checkpoint_directory_without_a_config_reads_as_nothing(tmp_path: Path) -> None:
    """A DCP directory or an S3 prefix has no config.json, and that is not an error."""
    assert exported_checkpoint_metadata(tmp_path) is None
    assert exported_checkpoint_metadata(tmp_path / "does-not-exist") is None
    assert exported_checkpoint_metadata(None) is None
    assert exported_checkpoint_metadata("") is None


def test_an_unparseable_config_reads_as_nothing(tmp_path: Path) -> None:
    """Never let a malformed sidecar take down model loading; the caller's default is safe."""
    (tmp_path / "config.json").write_text("{not json")
    assert exported_checkpoint_metadata(tmp_path) is None

    (tmp_path / "config.json").write_text("[1, 2, 3]")
    assert exported_checkpoint_metadata(tmp_path) is None


def test_a_wrapper_without_a_config_declares_nothing() -> None:
    """The model wrapper is stubbed in inference_test as a bare object carrying only `.model`.

    Reading this metadata must not depend on a config being present: it is opportunistic, and a
    wrapper with nothing to declare is not evidence either way. Paired with the refusal, the
    conservative answer is the safe one.
    """
    assert exports_per_view_captions(SimpleNamespace(model=object())) is False
    assert exported_caption_task(SimpleNamespace(model=object())) is None


# ---------------------------------------------------------------------------
# Writer half (export_model): derive the layout from the training dataloaders.
# ---------------------------------------------------------------------------


def _training_config(**streams: dict) -> SimpleNamespace:
    """A training config shaped like the real one: `dataloaders.<name>.dataloader.dataset`.

    Fixtures that nest the flags anywhere else pass while the production config yields nothing.
    """
    return SimpleNamespace(
        dataloader_train={"dataloaders": {name: {"dataloader": {"dataset": node}} for name, node in streams.items()}}
    )


def test_derives_the_joint_task_from_the_dataset_node() -> None:
    config = _training_config(joint_480={"separate_view_text_tokenization": True, "joint_cam_lidar_transfer": True})
    assert per_view_caption_tasks(config) == [JOINT_CAM_LIDAR_TASK]


def test_deduplicates_the_480p_and_720p_streams_of_one_recipe() -> None:
    node = {"separate_view_text_tokenization": True, "joint_cam_lidar_transfer": True}
    assert per_view_caption_tasks(_training_config(s480=dict(node), s720=dict(node))) == [JOINT_CAM_LIDAR_TASK]


def test_ignores_joint_streams_that_are_not_per_view() -> None:
    config = _training_config(joint={"separate_view_text_tokenization": False, "joint_cam_lidar_transfer": True})
    assert per_view_caption_tasks(config) == []
    assert separate_view_text_caption_streams(config) == []


def test_yields_no_task_for_a_non_joint_per_view_stream() -> None:
    """The plain multiview task also depends on unserialized inputs; nothing beats a guess."""
    config = _training_config(mv={"separate_view_text_tokenization": True, "joint_cam_lidar_transfer": False})
    assert separate_view_text_caption_streams(config) != []
    assert per_view_caption_tasks(config) == []


def test_empty_for_a_config_with_no_training_dataloaders() -> None:
    for config in (SimpleNamespace(), None, {}):
        assert per_view_caption_tasks(config) == []
        assert separate_view_text_caption_streams(config) == []


def test_finds_flags_inside_local_simulation_transform_overrides() -> None:
    """Framework's local-simulation recipes set the flags inside `transform_overrides`."""
    overrides = {"separate_view_text_tokenization": True, "joint_cam_lidar_transfer": True}
    config = _training_config(joint_480={"bundle_path": "${oc.env:UNSET_BUNDLE}", "transform_overrides": overrides})
    assert separate_view_text_caption_streams(config) == [
        "dataloaders.joint_480.dataloader.dataset.transform_overrides"
    ]
    assert per_view_caption_tasks(config) == [JOINT_CAM_LIDAR_TASK]


def test_reads_omegaconf_nodes_without_resolving_interpolations() -> None:
    """Resolving would make one boolean depend on every env var in the dataloader config."""
    node = {"bundle_path": "${oc.env:UNSET_BUNDLE}", "separate_view_text_tokenization": True}
    config = OmegaConf.create({"dataloader_train": {"dataloaders": {"mv": {"dataloader": {"dataset": node}}}}})
    assert separate_view_text_caption_streams(config) == ["dataloaders.mv.dataloader.dataset"]


def test_export_fields_record_layout_and_single_task() -> None:
    node = {"separate_view_text_tokenization": True, "joint_cam_lidar_transfer": True}
    fields, streams, tasks = per_view_caption_export_fields(_training_config(s480=node))
    assert fields == {PER_VIEW_CAPTIONS_KEY: True, PER_VIEW_CAPTION_TASK_KEY: JOINT_CAM_LIDAR_TASK}
    assert streams and tasks == [JOINT_CAM_LIDAR_TASK]
    # Round trip through the reader half.
    assert exports_per_view_captions(fields) is True
    assert exported_caption_task(fields) == JOINT_CAM_LIDAR_TASK


def test_export_fields_record_merged_layout_explicitly() -> None:
    fields, streams, tasks = per_view_caption_export_fields(
        _training_config(rgb={"separate_view_text_tokenization": False})
    )
    assert fields == {PER_VIEW_CAPTIONS_KEY: False}
    assert streams == [] and tasks == []


def test_export_fields_leave_the_task_unset_when_underivable() -> None:
    fields, _, tasks = per_view_caption_export_fields(
        _training_config(mv={"separate_view_text_tokenization": True, "joint_cam_lidar_transfer": False})
    )
    assert fields == {PER_VIEW_CAPTIONS_KEY: True}
    assert tasks == []
    assert exported_caption_task(fields) is None
