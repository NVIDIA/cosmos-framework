# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The exported Bimanual action contract, shared by preflight and inference."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import torch

# Schema-5 sidecars contain post-normalization values in the canonical 59D
# layout. Slots without semantics for a row's domain must remain exactly zero;
# otherwise the model receives conditioning that no training sample could have
# produced. This applies equally to uniform and mixed-domain sidecars.
_AGIBOT_VALID_ACTION_RANGES: tuple[tuple[int, int], ...] = ((0, 18), (33, 34), (34, 43), (58, 59))
_DOMAIN_VALID_ACTION_RANGES: dict[str, tuple[tuple[int, int], ...]] = {
    "camera_pose": ((0, 9),),
    "hand_pose": ((0, 33), (34, 58)),
    # Released checkpoints retain the AgiBot GEAR names. Framework-local
    # exports use the embodiment-c aliases. Training routes all five names to
    # domain 15 with the same head/right-wrist/right-gripper/left-wrist/
    # left-gripper unified-v1 semantics.
    "agibotworld": _AGIBOT_VALID_ACTION_RANGES,
    "agibot_gear_gripper": _AGIBOT_VALID_ACTION_RANGES,
    "agibot_gear_gripper_ext": _AGIBOT_VALID_ACTION_RANGES,
    "embodiment_c_gripper": _AGIBOT_VALID_ACTION_RANGES,
    "embodiment_c_gripper_ext": _AGIBOT_VALID_ACTION_RANGES,
    "behavior1k_lerobot": ((0, 18), (33, 34), (34, 43), (58, 59)),
    "abc_yam": ((9, 18), (33, 34), (34, 43), (58, 59)),
    "molmoact2_yam": ((9, 18), (33, 34), (34, 43), (58, 59)),
    "xdof_yam": ((9, 18), (33, 34), (34, 43), (58, 59)),
}


@cache
def _invalid_action_slots(domain_name: str, action_dim: int) -> tuple[int, ...]:
    ranges = _DOMAIN_VALID_ACTION_RANGES.get(domain_name)
    if ranges is None:
        raise ValueError(f"No Bimanual validity mask is registered for action domain {domain_name!r}")
    valid = [False] * action_dim
    for start, stop in ranges:
        if start < 0 or start >= stop or stop > action_dim:
            raise ValueError(f"Invalid Bimanual validity range {(start, stop)} for {domain_name!r}")
        valid[start:stop] = [True] * (stop - start)
    return tuple(index for index, is_valid in enumerate(valid) if not is_valid)


def _expand_domain_names(names: Any, *, action_count: int, fallback: str | None) -> list[str]:
    """Resolve scalar or framewise domain routing."""
    if names is None:
        names = fallback
    if isinstance(names, str):
        names = [names] * action_count
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError("domain_names must contain strings")
    if len(names) != action_count:
        raise ValueError("domain_names must be one name or contain one name per action row")
    return names


def build_bimanual_action_metadata() -> dict[str, Any]:
    """Build the canonical normalized-unified action contract for an export."""
    domains = {
        "abc_yam": 16,
        "embodiment_c_gripper": 15,
        "embodiment_c_gripper_ext": 15,
        "agibotworld": 15,
        "behavior1k_lerobot": 22,
        "camera_pose": 2,
        "hand_pose": 3,
        "molmoact2_yam": 16,
        "xdof_yam": 16,
    }
    fields = (
        ("ego_pose", 0, 9),
        ("right_wrist_pose", 9, 9),
        ("right_fingertips", 18, 15),
        ("right_gripper_width_m", 33, 1),
        ("left_wrist_pose", 34, 9),
        ("left_fingertips", 43, 15),
        ("left_gripper_width_m", 58, 1),
    )
    return {
        "cosmos3_nano_sim_bimanual": {
            "conditioning": {
                "embodiments": {
                    name: {"domain_id": domain_id, "input_action_dim": 59} for name, domain_id in domains.items()
                },
                "input_contract": {
                    "action_dim": 59,
                    "action_space": "normalized_unified_v1",
                    "domain_routing": "scalar_or_per_action_row",
                    "model_mode": "forward_dynamics",
                    "runtime_normalization": False,
                    "validity": "masked_to_zero_after_normalization_by_source",
                },
                "layout": {
                    "fields": [{"name": name, "offset": offset, "size": size} for name, offset, size in fields],
                    "id": "unified_v1",
                    "pose_convention": "backward_chunk_anchored_16f",
                    "rotation_representation": "rot6d_columns",
                },
                "model_action_dim": 64,
                "num_embodiment_domains": 32,
                "schema_version": 5,
            }
        }
    }


def bimanual_action_metadata_for_model_config(model_config: dict[str, Any]) -> dict[str, Any] | None:
    """Return metadata when the model declares the Bimanual action contract.

    The action schema and normalization settings survive experiment renames and
    config serialization, unlike a CLI experiment string.  A declared unified
    schema must match the complete public contract or export fails closed.
    """
    if model_config.get("action_schema", "native") != "unified_v1":
        return None
    expected = {
        "action_gen": True,
        "action_global_asinh": True,
        "action_metric_gripper": True,
        "max_action_dim": 64,
        "num_embodiment_domains": 32,
    }
    mismatches = {name: model_config.get(name) for name, value in expected.items() if model_config.get(name) != value}
    if mismatches:
        raise ValueError(f"Unified Bimanual export model config does not match its action contract: {mismatches}")
    return build_bimanual_action_metadata()


@dataclass(frozen=True)
class BimanualActionContract:
    action_dim: int
    model_action_dim: int
    domain_ids: dict[str, int]

    def validate(self, payload: Any, *, action_count: int, domain_name: str) -> tuple[list[list[float]], list[int]]:
        if not isinstance(payload, dict) or payload.get("action_space") != "normalized_unified_v1":
            raise ValueError(
                "This checkpoint requires a normalized_unified_v1 action sidecar object; "
                "legacy native action arrays are incompatible."
            )
        actions = payload.get("action")
        if not isinstance(actions, list) or len(actions) != action_count:
            raise ValueError(f"Expected exactly {action_count} action rows")
        for index, row in enumerate(actions):
            if not isinstance(row, list) or len(row) != self.action_dim:
                raise ValueError(f"Action row {index} must have {self.action_dim} values")
            if any(type(value) not in (int, float) or not math.isfinite(value) for value in row):
                raise ValueError(f"Action row {index} contains a non-finite or non-numeric value")
        names = _expand_domain_names(payload.get("domain_names"), action_count=action_count, fallback=domain_name)
        if any(not isinstance(name, str) or name not in self.domain_ids for name in names):
            raise ValueError(f"Unsupported action domain; checkpoint supports {sorted(self.domain_ids)}")
        for row_index, (row, name) in enumerate(zip(actions, names, strict=True)):
            invalid_slots = _invalid_action_slots(name, self.action_dim)
            if nonzero := [slot for slot in invalid_slots if row[slot] != 0.0]:
                raise ValueError(
                    f"Action row {row_index} for domain {name!r} has nonzero values in invalid unified-v1 "
                    f"slots {nonzero[:8]}"
                )
        return actions, [self.domain_ids[name] for name in names]


def recover_bimanual_camera_poses(action_path: Path) -> torch.Tensor:
    """Invert normalized unified-v1 camera actions into an OpenCV c2w trajectory.

    The Stage 3 Bimanual action profile uses global-asinh normalization and the
    backward chunk-anchored 16-frame pose convention. Ego-pose slots remain
    valid across AgiBot robot rows and inserted camera rows, so the full
    trajectory must be reconstructed without restarting at domain boundaries.
    """
    from cosmos_framework.data.generator.action.utils.action_processing import make_camera_global_asinh_normalizer
    from cosmos_framework.data.generator.action.utils.pose_utils import pose_rel_to_abs

    payload = json.loads(Path(action_path).read_text())
    if not isinstance(payload, dict) or payload.get("action_space") != "normalized_unified_v1":
        raise ValueError("Camera recovery requires normalized_unified_v1 action JSON")
    rows = payload.get("action")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Camera recovery requires a nonempty action sequence")
    actions = torch.as_tensor(rows, dtype=torch.float32)
    if actions.ndim != 2 or actions.shape[1] != 59 or not bool(torch.isfinite(actions).all()):
        raise ValueError("Camera recovery requires finite [T,59] actions")
    names = _expand_domain_names(payload.get("domain_names"), action_count=len(rows), fallback=None)
    if not set(names) <= {"agibotworld", "camera_pose"}:
        raise ValueError("Camera recovery supports only AgiBot/camera domains with valid ego-pose slots")

    normalizer = make_camera_global_asinh_normalizer(pose_convention="backward_chunk_anchored_16f")
    raw_camera = normalizer.denormalize_action(actions[:, :9])
    poses = pose_rel_to_abs(
        raw_camera.numpy(),
        rotation_format="rot6d",
        pose_convention="backward_chunk_anchored_16f",
    )
    return torch.from_numpy(poses)


def load_bimanual_frustum_geometry(path: Path, action_path: Path | None) -> dict[str, Any]:
    """Load explicit frustum geometry or derive its poses from Bimanual actions."""
    geometry = json.loads(Path(path).read_text())
    if not isinstance(geometry, dict):
        raise ValueError("kv_cache_frustum_path must contain a camera-geometry JSON object")
    if "pose_source" not in geometry:
        return geometry
    if "c2w" in geometry:
        raise ValueError("Specify explicit c2w or pose_source, not both")
    if geometry["pose_source"] != "bimanual_unified_v1_chunk16":
        raise ValueError("Unsupported frustum pose_source; expected bimanual_unified_v1_chunk16")
    if action_path is None:
        raise ValueError("Action-derived frustum poses require action_path")
    geometry["c2w"] = recover_bimanual_camera_poses(Path(action_path))
    return geometry


def load_bimanual_action_contract(checkpoint_path: Path) -> BimanualActionContract | None:
    """Read the action schema from the selected export, without loading weights.

    Native ``export_model`` checkpoints record the contract in ``config.json``;
    Diffusers checkpoints record it in ``transformer/config.json``. Accept both
    layouts. A Diffusers root config may omit ``conditioning`` while its
    transformer config carries the contract; all other metadata must agree.

    Schema-3 exports and checkpoints without metadata retain their native path.
    Never substitute the framework's native-width registry for a unified export.
    """
    metadata_sources: list[tuple[Path, dict[str, Any]]] = []
    for relative_path in (Path("transformer/config.json"), Path("config.json")):
        path = Path(checkpoint_path) / relative_path
        if not path.is_file():
            continue
        metadata = json.loads(path.read_text()).get("cosmos3_nano_sim_bimanual")
        if metadata is None:
            continue
        if not isinstance(metadata, dict):
            raise ValueError(f"Invalid Bimanual action metadata in {relative_path}")
        metadata_sources.append((relative_path, metadata))
    if not metadata_sources:
        return None
    source_path, metadata = metadata_sources[0]
    schema = metadata.get("conditioning")
    shared_metadata = {key: value for key, value in metadata.items() if key != "conditioning"}
    for other_path, other_metadata in metadata_sources[1:]:
        other_shared_metadata = {key: value for key, value in other_metadata.items() if key != "conditioning"}
        other_schema = other_metadata.get("conditioning")
        if other_shared_metadata != shared_metadata or (schema is not None and other_schema not in (None, schema)):
            raise ValueError(f"Conflicting Bimanual action metadata in {source_path} and {other_path}")
        if schema is None and other_schema is not None:
            schema = other_schema
    if schema is None:
        return None
    if schema.get("schema_version") is None:
        # Sim-Transfer exports reuse this metadata block for their own conditioning
        # (`mode: "control_video"` plus control hints) and declare no action schema,
        # so they keep their native path. Admit other non-action families here
        # explicitly; a versionless block of any other shape is an incomplete
        # Bimanual export and must still fail loudly.
        if schema.get("mode") == "control_video":
            return None
        raise ValueError("Bimanual action conditioning is missing schema_version")
    if schema.get("schema_version") == 3:
        # The 2026-09-23 export declares its native action layout explicitly.
        # It still needs the existing 29D AgiBot / JSON-prompt path, not the
        # schema-5 normalized-unified adapter. Do not guess for other schemas.
        agibot = schema.get("embodiments", {}).get("agibotworld", {})
        if (
            schema.get("mode") != "action"
            or schema.get("model_action_dim") != 64
            or schema.get("num_embodiment_domains") != 32
            or schema.get("padding") != {"stage": "after_normalization", "value": 0.0}
            or agibot.get("raw_action_dim") != 29
            or agibot.get("domain_id") != 15
            or agibot.get("layout", {}).get("id") != "agibot_backward_framewise_rot6d_v1"
            or agibot.get("normalizer", {}).get("method") != "quantile_rot"
        ):
            raise ValueError("Unsupported legacy Bimanual action contract; expected the schema-3 native export")
        return None
    contract = schema.get("input_contract", {})
    if (
        schema.get("schema_version") != 5
        or contract.get("action_space") != "normalized_unified_v1"
        or contract.get("model_mode") != "forward_dynamics"
        or contract.get("runtime_normalization") is not False
        or contract.get("domain_routing") != "scalar_or_per_action_row"
        or schema.get("layout", {}).get("id") != "unified_v1"
        or contract.get("action_dim") != 59
        or schema.get("model_action_dim") != 64
    ):
        raise ValueError("Unsupported Bimanual action contract; use a checkpoint matching docs/inference_bimanual.md")
    domains = schema.get("embodiments", {})
    domain_ids = {}
    domain_count = schema.get("num_embodiment_domains")
    if type(domain_count) is not int or domain_count <= 0:
        raise ValueError("Invalid number of Bimanual embodiment domains")
    for name, item in domains.items():
        domain_id = item.get("domain_id")
        if item.get("input_action_dim") != 59 or type(domain_id) is not int or not 0 <= domain_id < domain_count:
            raise ValueError(f"Invalid Bimanual embodiment definition: {name}")
        domain_ids[name] = domain_id
    if not domain_ids:
        raise ValueError("Bimanual checkpoint declares no action domains")
    if unknown_domains := set(domain_ids) - set(_DOMAIN_VALID_ACTION_RANGES):
        raise ValueError(f"Bimanual checkpoint domains have no schema-5 validity mask: {sorted(unknown_domains)}")
    return BimanualActionContract(action_dim=59, model_action_dim=64, domain_ids=domain_ids)
