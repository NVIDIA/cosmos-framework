# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Source contract for the shared native action update and recovery path.

Family model/processor identities are deliberately separate: changing training code
must refuse exact resume without preventing inference from a portable artifact.
This explicit dependency inventory covers the single-device action SFT path, not
every trainer/backend in the repository. Runtime dependencies remain bound by the
family's pinned model identity; this contract binds the local Python implementation.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

_FORMAT = "cosmos-native-action-training-sources-v1"
_POSTTRAIN_ROOT = Path(__file__).resolve().parents[1]

# Keep paths package-relative so an identical wheel can resume from a different
# installation directory. Hash whole participating modules conservatively, including
# this inventory/validator; docs, recipes and unrelated backends are not traversed.
TRAINING_SOURCE_FILES = (
    "algorithm/flow_bc.py",  # masked native task loss and supervised-count reduction
    "algorithm/primitives/objective.py",  # backward and accumulation normalization
    "checkpoint/adapter.py",  # strict adapter restore / optimizer inventory
    "checkpoint/io.py",  # native control and DCP component loading
    "checkpoint/manager.py",  # model/optimizer/data restoration dispatch
    "checkpoint/provenance.py",  # stable generation across joint state restoration
    "checkpoint/rng.py",  # Python, NumPy, Torch and CUDA RNG capture/restore
    "checkpoint/store.py",  # selected local durable checkpoint store
    "config/action_sft.py",  # supported driver/update configuration
    "config/build.py",  # native optimizer partition and scheduler construction
    "config/gr00t_official.py",  # explicit official-head optimizer/schedule policy
    "control/checkpointing.py",  # joint trainer/data restore selection
    "data/datasets/flow_sft.py",  # content-bound row lookup and task data decoding
    "data/flow_media.py",  # shared media resolution, decoding and content checks
    "entrypoints/action_sft.py",  # batch/step/evaluation clocks and resume admission
    "model/checkpoint.py",  # complete trainable inventory and exact-resume codec
    "roles/env/cursor.py",  # epoch, shuffle, consumed index and strict cursor restore
    "roles/trainer/base.py",  # update controls, pristine optimizer and RNG restoration
    "roles/trainer/finite.py",  # loss / gradient finite checks before mutation
    "roles/trainer/llm.py",  # backward, clip, optimizer, scheduler and state restore
    "utils/action_evaluation_inputs.py",  # bounded deterministic evaluation preparation and cleanup
    "utils/gr00t_evaluation_backbone.py",  # one-batch raw backbone evaluation reuse and cleanup
    "utils/action_training_identity.py",
    "utils/determinism.py",
    "utils/dist.py",  # count/metric normalization and converged failures
    "utils/entry.py",  # explicit resume-source and run-attempt selection
    "utils/identity.py",  # exact identity canonicalization/comparison
    "utils/rank_state.py",  # rank-owned RNG/data state selection
    "utils/scheduler.py",  # scheduler construction identity
)


def action_training_source_identity() -> dict:
    """Hash required installed sources without importing them or changing RNG/state."""
    files = {}
    for relative in TRAINING_SOURCE_FILES:
        try:
            files[relative] = hashlib.sha256((_POSTTRAIN_ROOT / relative).read_bytes()).hexdigest()
        except OSError as error:
            raise RuntimeError(
                f"cannot bind native action training source {relative}; install the complete "
                "qualified Cosmos source/wheel before starting or resuming training"
            ) from error
    return {"format": _FORMAT, "files": files}


def require_matching_training_source(saved_contract: object, current: Mapping) -> None:
    """Refuse legacy or changed execution before restoring any mutable training state.

    The checkpoint codec subsequently compares the complete training contract. This
    earlier check provides an actionable error before expensive family model loading.
    It must never be used for portable/inference reload, which intentionally ignores
    the training-only contract.
    """
    instruction = (
        "Resume with the original qualified source build and runtime. Cross-build exact "
        "resume requires separate qualification; do not edit checkpoint identities. "
        "Portable inference reload remains supported independently."
    )
    saved = saved_contract.get("implementation") if isinstance(saved_contract, Mapping) else None
    if not isinstance(saved, Mapping) or saved.get("format") != _FORMAT:
        raise ValueError(f"native action checkpoint lacks the required training source contract. {instruction}")
    files = saved.get("files")
    if (
        set(saved) != {"format", "files"}
        or not isinstance(files, Mapping)
        or any(
            not isinstance(name, str)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            for name, digest in files.items()
        )
    ):
        raise ValueError(f"native action checkpoint training source contract is malformed. {instruction}")
    expected = current["files"]
    changed = sorted(name for name in set(files) | set(expected) if files.get(name) != expected.get(name))
    if saved != current:
        raise ValueError(
            f"native action checkpoint training implementation differs: {', '.join(changed) or 'format'}. {instruction}"
        )
