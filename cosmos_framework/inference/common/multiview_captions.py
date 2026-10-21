# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""A checkpoint's multiview caption layout: recorded at export, read back at inference.

A checkpoint trained with ``separate_view_text_tokenization`` packs one caption per camera, and
the weights only read text that way. The training dataloaders do not travel with an exported
checkpoint, so ``export_model`` derives the layout from them (the writer half below) and records
it in ``config.json`` under the keys below; inference reads those fields back (the reader half).
"""

import json
from collections.abc import Generator, Mapping
from pathlib import Path
from typing import Any

__all__ = [
    "JOINT_CAM_LIDAR_TASK",
    "JOINT_CAM_LIDAR_TRANSFER_KEY",
    "PER_VIEW_CAPTION_TASK_KEY",
    "PER_VIEW_CAPTIONS_KEY",
    "SEPARATE_VIEW_TEXT_KEY",
    "exported_caption_task",
    "exported_checkpoint_metadata",
    "exports_per_view_captions",
    "find_bool_flags",
    "has_training_dataloaders",
    "per_view_caption_export_fields",
    "per_view_caption_tasks",
    "separate_view_text_caption_streams",
]

# Top-level keys in an exported checkpoint's config.json. Preserve their spelling
# for compatibility with published model metadata.
PER_VIEW_CAPTIONS_KEY = "per_view_captions"
PER_VIEW_CAPTION_TASK_KEY = "per_view_caption_task"

# The dataset-node flag that makes a stream tokenize one caption per camera.
SEPARATE_VIEW_TEXT_KEY = "separate_view_text_tokenization"

# Set on the same dataset node as SEPARATE_VIEW_TEXT_KEY. The tokenizer task itself is not in the
# config -- the text-tokenizer augmentor derives it when the dataset is built -- but this input to
# that decision is, which is what makes the task recoverable at export time.
JOINT_CAM_LIDAR_TRANSFER_KEY = "joint_cam_lidar_transfer"

JOINT_CAM_LIDAR_TASK = "av_joint_camera_lidar_transfer"


def exported_checkpoint_metadata(checkpoint_dir: Any) -> dict[str, Any] | None:
    """Load an exported checkpoint's own ``config.json``, or ``None`` if there is none to read.

    The keys above are top-level in that file, and the loaded model object is not a reliable
    place to look for them: on the exported-checkpoint path the wrapper config is rebuilt from
    the file's ``model`` section alone (``Cosmos3OmniConfig(model=...)``), which keeps no other
    top-level key. Reading the file is the only way to see what the exporter recorded.

    Tolerant by construction -- a missing, unreadable or non-object file is not an error here,
    because plenty of valid checkpoint paths (a DCP directory, an S3 prefix) have no such file.
    Callers pair the ``None`` with the same conservative default as a missing key.
    """
    if not checkpoint_dir:
        return None
    try:
        config_file = Path(checkpoint_dir) / "config.json"
        if not config_file.is_file():
            return None
        loaded = json.loads(config_file.read_text())
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def exports_per_view_captions(config: Any) -> bool:
    """Whether this checkpoint packs one caption per camera.

    False for anything that does not say so, which includes every checkpoint exported before the
    key existed. That default is safe only because it is paired with a refusal: a checkpoint that
    really is per-view and does not say so is served as merged text, which is wrong but silent,
    so the absence of the key must never be read as positive evidence of merged captions.
    """
    return bool(_get(config, PER_VIEW_CAPTIONS_KEY, default=False))


def exported_caption_task(config: Any) -> str | None:
    """The tokenizer task the per-view captions were trained under, if the export recorded it.

    Names the system prompt. ``None`` means the exporter could not determine it -- either the
    checkpoint predates the key, or its streams disagreed -- and the caller has to decide rather
    than assume the plain multiview prompt.
    """
    task = _get(config, PER_VIEW_CAPTION_TASK_KEY, default=None)
    return task if isinstance(task, str) and task else None


def _get(config: Any, key: str, *, default: Any) -> Any:
    """Read *key* off a config that may be an object, a mapping, or absent."""
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


# ---------------------------------------------------------------------------
# Writer half: derive the layout from a training config at export time.
# ---------------------------------------------------------------------------


def _as_plain_container(node: Any) -> Any:
    """An OmegaConf node as plain dicts and lists; anything else unchanged.

    Unresolved on purpose: the dataloader nodes carry interpolations (environment variables,
    references into the model config), so resolving them would make reading one boolean depend on
    the whole config still being resolvable.
    """
    try:
        from omegaconf import OmegaConf
    except ImportError:
        return node
    if not OmegaConf.is_config(node):
        return node
    try:
        return OmegaConf.to_container(node, resolve=False, throw_on_missing=False)
    except Exception:  # A config node this cannot read cannot answer the question either.
        return None


def find_bool_flags(node: Any, key: str, path: str = "") -> Generator[tuple[str, bool], None, None]:
    """Every boolean stored under *key* anywhere inside *node*, with the path that reached it."""
    node = _as_plain_container(node)
    if isinstance(node, Mapping):
        for node_key, value in node.items():
            child_path = f"{path}.{node_key}" if path else str(node_key)
            if node_key == key and isinstance(value, bool):
                yield path, value
            else:
                yield from find_bool_flags(value, key, child_path)
    elif isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            yield from find_bool_flags(value, key, f"{path}[{index}]")


def _find_mappings_with_flag(node: Any, key: str) -> Generator[Mapping[str, Any], None, None]:
    """Every mapping inside *node* that sets *key* to True."""
    node = _as_plain_container(node)
    if isinstance(node, Mapping):
        if node.get(key) is True:
            yield node
        for value in node.values():
            yield from _find_mappings_with_flag(value, key)
    elif isinstance(node, (list, tuple)):
        for value in node:
            yield from _find_mappings_with_flag(value, key)


def _dataloader_train(config: Any) -> Any:
    dataloader_train = getattr(config, "dataloader_train", None)
    if dataloader_train is None and isinstance(config, Mapping):
        dataloader_train = config.get("dataloader_train")
    return dataloader_train


def has_training_dataloaders(config: Any) -> bool:
    """Whether *config* carries training dataloaders at all, i.e. whether the layout is knowable.

    A model-only config (e.g. a prepared student config) has none, and an empty answer from the
    searches below is then "unknown", not "merged captions".
    """
    return _dataloader_train(config) is not None


def per_view_caption_tasks(config: Any) -> list[str]:
    """The tokenizer tasks of the training streams that pack one caption per camera.

    The task names the system prompt those captions were trained under, and getting it wrong is
    silent: the captions stay well formed and only their opening turn differs from training.

    Only the joint camera+LiDAR task is derived. The plain multiview task additionally depends on
    ``use_transfer_text``, which is not serialized, so a stream that is not joint yields nothing
    here rather than a guess: an absent task tells the caller to decide, a wrong one does not.

    Sorted and de-duplicated. More than one entry means the streams disagree, which is surfaced
    rather than resolved.
    """
    dataloader_train = _dataloader_train(config)
    if dataloader_train is None:
        return []
    tasks = {
        JOINT_CAM_LIDAR_TASK
        for node in _find_mappings_with_flag(dataloader_train, SEPARATE_VIEW_TEXT_KEY)
        if node.get(JOINT_CAM_LIDAR_TRANSFER_KEY) is True
    }
    return sorted(tasks)


def separate_view_text_caption_streams(config: Any) -> list[str]:
    """The training streams behind this checkpoint that tokenized one caption per camera.

    Searched rather than read from a fixed path: the flag rides on whichever multiview dataset node
    an experiment builds (directly as a keyword, or inside a local-simulation dataset's
    ``transform_overrides``), and experiments differ in how many streams they declare.

    One stream is enough to answer for the checkpoint; streams disagreeing is normal in a mixture
    where only the multiview builder sets the flag. Empty for anything this cannot read: no config,
    no training dataloaders, or no such flag anywhere in them.
    """
    dataloader_train = _dataloader_train(config)
    if dataloader_train is None:
        return []
    return [path for path, enabled in find_bool_flags(dataloader_train, SEPARATE_VIEW_TEXT_KEY) if enabled]


def per_view_caption_export_fields(training_config: Any) -> tuple[dict[str, Any], list[str], list[str]]:
    """The top-level ``config.json`` fields ``export_model`` records for *training_config*.

    Returns ``(fields, streams, tasks)``. ``fields`` always carries :data:`PER_VIEW_CAPTIONS_KEY`;
    it carries :data:`PER_VIEW_CAPTION_TASK_KEY` only when the per-view streams agree on exactly one
    derivable task. ``streams`` and ``tasks`` are returned for logging: more than one task means the
    streams disagree and the task is deliberately left unrecorded.
    """
    streams = separate_view_text_caption_streams(training_config)
    fields: dict[str, Any] = {PER_VIEW_CAPTIONS_KEY: bool(streams)}
    tasks: list[str] = []
    if streams:
        tasks = per_view_caption_tasks(training_config)
        if len(tasks) == 1:
            fields[PER_VIEW_CAPTION_TASK_KEY] = tasks[0]
    return fields, streams, tasks
