# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1


"""Native local-bundle Bimanual semantics; no remote index access."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, cast

import torch

from cosmos_framework.data.generator.action.action_binary import get_action_binary_spec_name
from cosmos_framework.data.generator.action.lance.lance_action_dataset import (
    BaseLanceActionDataset,
    LanceActionSampleRequest,
    MaterializedActionWindow,
)
from cosmos_framework.data.generator.action.utils.action_processing import ActionNormalizer
from cosmos_framework.data.generator.action.utils.domain_utils import get_domain_id
from cosmos_framework.data.generator.lance.lance_action_source import LanceActionDatasetSpec
from cosmos_framework.data.generator.local_simulation.core import LocalByteRangeBackend as ByteRangeBackend

CANONICAL_SOURCE_SAMPLE_ID_KEY: str = "__canonical_source_sample_id__"


CANONICAL_DATASET_ID_KEY: str = "__canonical_dataset_id__"


_EMPTY_CAPTION_PLACEHOLDER: str = "__canonical_empty_caption__"


CANONICAL_CAMERA_MASK_NATIVE_ACTION_KEY: str = "_canonical_camera_mask_native_action"


CANONICAL_CAMERA_MASK_TRANSLATION_SCALE_KEY: str = "_canonical_camera_mask_translation_scale"


@dataclass(frozen=True)
class _CameraMaskedNativeNormalizer:
    """Normalize robot rows and apply the camera source's model-space scale."""

    domain_ids: torch.Tensor  # [T]
    native_normalizer: ActionNormalizer | None
    camera_translation_scale: float

    def _camera_rows(self, action: torch.Tensor) -> torch.Tensor:
        if action.ndim != 2:
            raise ValueError(f"Camera-masked native action must be 2D, got {tuple(action.shape)}.")
        if action.shape[0] != self.domain_ids.numel():
            raise ValueError(
                f"Action/domain length mismatch: {action.shape[0]} action rows and {self.domain_ids.numel()} domain IDs."
            )
        return self.domain_ids.to(device=action.device).eq(get_domain_id("camera_pose")).unsqueeze(-1)

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Normalize native rows and scale camera-row translations."""
        camera_rows = self._camera_rows(action)
        native_action = (
            action.clone() if self.native_normalizer is None else self.native_normalizer.normalize_action(action)
        )
        camera_action = action.clone()
        camera_action[..., :3] *= self.camera_translation_scale
        return torch.where(camera_rows, camera_action, native_action)

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Invert native normalization and camera-row translation scaling."""
        camera_rows = self._camera_rows(action)
        native_action = (
            action.clone() if self.native_normalizer is None else self.native_normalizer.denormalize_action(action)
        )
        camera_action = action.clone()
        camera_action[..., :3] /= self.camera_translation_scale
        return torch.where(camera_rows, camera_action, native_action)


@dataclass
class CanonicalLanceActionDatasetSpec(LanceActionDatasetSpec):
    """Select materialized samples and describe their runtime projection.

    ``source_datasets`` keeps the source-family contract expected by the
    reused dataset-specific adapter. ``action_source_datasets_by_id`` lists
    the ACT2 layouts physically allowed for each canonical dataset ID. A
    canonical dataset can contain more than one layout (for example standard
    and extended Embodiment C), so the adapter resolves the unique layout from
    the materialized frame count and action-asset byte size.

    ``view_stream_id_by_name`` may describe every view available to the
    dataset family. Runtime validation and reads are restricted to the
    adapter's selected ``view_names``, so an ego-view recipe does not require
    or read wrist-camera assets merely because the spec also declares them.

    ``source_caption_type`` selects one exact ``captions[].caption_type``
    without falling back to another type. ``task`` is the development default
    and the non-strict overlay fallback. Production configs can instead pin a
    Lance caption sidecar through ``caption_overlay_path`` and select its key
    and caption columns. Strict overlays must cover every selected sample and
    never fall back to an embedded caption.

    ``caption_overlay_source_lance_version`` optionally maps rematerialized RGB
    rows back to the sample IDs used by the caption corpus. Dataset, source-window,
    episode and time boundaries must match exactly; missing matches are errors.

    ``validate_assets_during_index_scan`` can defer asset validation to the
    existing worker-side source-row projection. This is reserved for pinned
    table versions whose large sequential scan cannot decode an otherwise
    readable complex ``assets`` column.

    ``sample_exclusion_manifest_path`` optionally rejects exact
    ``(dataset_id, source_sample_id)`` pairs while materializing the shared row
    index. Its content digest participates in sampler/checkpoint identity, so a
    changed manifest cannot silently reuse an older cached index.

    ``sample_exclusion_audit_lance_version`` preserves the original action audit
    when a reviewed RGB-only rematerialization retains the same source sample keys.
    """

    dataset_ids: tuple[str, ...] = ()
    action_source_datasets_by_id: dict[str, list[str]] = field(default_factory=dict)
    view_stream_id_by_name: dict[str, str] = field(default_factory=dict)
    source_caption_type: str = "task"
    caption_overlay_path: str | None = None
    caption_overlay_key: str = "sample_id"
    caption_overlay_column: str = "caption"
    caption_overlay_lance_version: int | None = None
    caption_overlay_source_lance_version: int | None = None
    caption_overlay_strict: bool = True
    validate_assets_during_index_scan: bool = True
    sample_exclusion_manifest_path: str | None = None
    sample_exclusion_manifest_sha256: str | None = None
    sample_exclusion_audit_lance_version: int | None = None
    lance_version: int | None = None
    expected_sample_count: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.source_index_uri != self.lance_uri:
            raise ValueError("Canonical Action samples use the same URI for lance_uri and source_index_uri.")
        if not self.dataset_ids or any(not dataset_id.strip() for dataset_id in self.dataset_ids):
            raise ValueError("Canonical Action dataset_ids must contain non-empty values.")
        if len(set(self.dataset_ids)) != len(self.dataset_ids):
            raise ValueError(f"Canonical Action dataset_ids must not contain duplicates: {self.dataset_ids!r}.")
        if set(self.action_source_datasets_by_id) != set(self.dataset_ids):
            raise ValueError(
                "action_source_datasets_by_id keys must exactly match dataset_ids; "
                f"got keys={tuple(self.action_source_datasets_by_id)!r}, dataset_ids={self.dataset_ids!r}."
            )
        for dataset_id, candidates in self.action_source_datasets_by_id.items():
            if not candidates or any(not candidate.strip() for candidate in candidates):
                raise ValueError(f"Canonical dataset {dataset_id!r} must select non-empty ACT2 source aliases.")
            source_families = [get_action_binary_spec_name(candidate) for candidate in candidates]
            if len(set(source_families)) != len(source_families):
                raise ValueError(
                    f"Canonical dataset {dataset_id!r} has duplicate ACT2 layouts {source_families!r}; "
                    "one candidate per fixed-stride layout is sufficient."
                )
        if not self.view_stream_id_by_name:
            raise ValueError("view_stream_id_by_name must select at least one canonical RGB stream.")
        if any(not name.strip() or not stream_id.strip() for name, stream_id in self.view_stream_id_by_name.items()):
            raise ValueError("view_stream_id_by_name keys and values must not be empty.")
        if len(set(self.view_stream_id_by_name.values())) != len(self.view_stream_id_by_name):
            raise ValueError("view_stream_id_by_name must map logical views to distinct canonical streams.")
        if not isinstance(self.source_caption_type, str) or not self.source_caption_type.strip():
            raise ValueError("source_caption_type must not be empty.")
        if not isinstance(self.caption_overlay_key, str) or not self.caption_overlay_key.strip():
            raise ValueError("caption_overlay_key must be a non-empty string.")
        if not isinstance(self.caption_overlay_column, str) or not self.caption_overlay_column.strip():
            raise ValueError("caption_overlay_column must be a non-empty string.")
        if not isinstance(self.caption_overlay_strict, bool):
            raise ValueError("caption_overlay_strict must be a bool.")
        if self.caption_overlay_source_lance_version is not None and (
            self.caption_overlay_path is None
            or self.lance_version is None
            or isinstance(self.caption_overlay_source_lance_version, bool)
            or not isinstance(self.caption_overlay_source_lance_version, int)
            or self.caption_overlay_source_lance_version < 1
        ):
            raise ValueError(
                "caption_overlay_source_lance_version requires a caption overlay and positive source version."
            )
        if not isinstance(self.validate_assets_during_index_scan, bool):
            raise ValueError("validate_assets_during_index_scan must be a bool.")
        if (self.sample_exclusion_manifest_path is None) != (self.sample_exclusion_manifest_sha256 is None):
            raise ValueError(
                "sample_exclusion_manifest_path and sample_exclusion_manifest_sha256 must be set together."
            )
        if self.sample_exclusion_manifest_path is not None and self.lance_version is None:
            raise ValueError("sample_exclusion_manifest_path requires a pinned lance_version.")
        if self.sample_exclusion_audit_lance_version is not None and (
            self.sample_exclusion_manifest_path is None
            or isinstance(self.sample_exclusion_audit_lance_version, bool)
            or self.sample_exclusion_audit_lance_version < 1
        ):
            raise ValueError(
                "sample_exclusion_audit_lance_version requires an exclusion manifest and a positive version."
            )
        if self.sample_exclusion_manifest_path is not None and (
            not isinstance(self.sample_exclusion_manifest_path, str) or not self.sample_exclusion_manifest_path.strip()
        ):
            raise ValueError("sample_exclusion_manifest_path must be a non-empty string.")
        if self.sample_exclusion_manifest_sha256 is not None and (
            not isinstance(self.sample_exclusion_manifest_sha256, str)
            or len(self.sample_exclusion_manifest_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.sample_exclusion_manifest_sha256)
        ):
            raise ValueError("sample_exclusion_manifest_sha256 must be a lowercase SHA-256 hex digest.")
        if self.caption_overlay_key == self.caption_overlay_column:
            raise ValueError("Caption overlay key and caption columns must be different.")
        if self.caption_overlay_path is None:
            if self.caption_overlay_lance_version is not None:
                raise ValueError("caption_overlay_lance_version requires caption_overlay_path.")
        else:
            if not isinstance(self.caption_overlay_path, str) or not self.caption_overlay_path.strip():
                raise ValueError("caption_overlay_path must be a non-empty string.")
            if self.caption_overlay_lance_version is None:
                raise ValueError("Caption overlays must pin caption_overlay_lance_version.")
        if self.caption_overlay_lance_version is not None and (
            not isinstance(self.caption_overlay_lance_version, int)
            or isinstance(self.caption_overlay_lance_version, bool)
            or self.caption_overlay_lance_version < 1
        ):
            raise ValueError(
                f"caption_overlay_lance_version must be a positive integer, got {self.caption_overlay_lance_version!r}."
            )
        if self.lance_version is not None and (
            not isinstance(self.lance_version, int) or isinstance(self.lance_version, bool) or self.lance_version < 1
        ):
            raise ValueError(f"lance_version must be a positive integer, got {self.lance_version!r}.")
        if self.expected_sample_count is not None and self.expected_sample_count < 1:
            raise ValueError(f"expected_sample_count must be positive, got {self.expected_sample_count}.")


def _long_stationary_runs(score: torch.Tensor, *, threshold: float, minimum_frames: int) -> torch.Tensor:
    """Return rows in contiguous numerical-low-movement runs."""
    stationary = torch.isfinite(score) & (score <= threshold)
    padded = torch.cat(
        (
            torch.zeros(1, dtype=torch.bool, device=score.device),
            stationary,
            torch.zeros(1, dtype=torch.bool, device=score.device),
        )
    )
    transitions = padded[1:].to(torch.int8) - padded[:-1].to(torch.int8)
    starts = torch.nonzero(transitions == 1, as_tuple=False).flatten()
    ends = torch.nonzero(transitions == -1, as_tuple=False).flatten()
    result = torch.zeros_like(stationary)
    for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
        if end - start >= minimum_frames:
            result[start:end] = True
    return result


class CanonicalLanceDatasetMixin:
    """Project canonical sample rows for an existing Action Lance adapter."""

    def __init__(
        self,
        *args: Any,
        camera_mask: bool = False,
        translation_scale: float = 1.0,
        camera_mask_threshold: float | None = None,
        camera_mask_min_frames: int = 11,
        camera_mask_gripper_threshold: float = 1e-4,
        camera_mask_gripper_guard_frames: int = 15,
        robot_multiview: bool = False,
        **kwargs: Any,
    ) -> None:
        if not math.isfinite(translation_scale) or translation_scale == 0:
            raise ValueError(f"translation_scale must be non-zero and finite, got {translation_scale}.")
        if camera_mask and (
            camera_mask_threshold is None or not math.isfinite(camera_mask_threshold) or camera_mask_threshold < 0
        ):
            raise ValueError("camera_mask requires a finite non-negative camera_mask_threshold.")
        if camera_mask_min_frames < 1:
            raise ValueError(f"camera_mask_min_frames must be positive, got {camera_mask_min_frames}.")
        if not math.isfinite(camera_mask_gripper_threshold) or camera_mask_gripper_threshold < 0:
            raise ValueError(
                f"camera_mask_gripper_threshold must be finite and non-negative, got {camera_mask_gripper_threshold}."
            )
        if camera_mask_gripper_guard_frames < 0:
            raise ValueError(
                f"camera_mask_gripper_guard_frames must be non-negative, got {camera_mask_gripper_guard_frames}."
            )
        self.camera_mask: bool = bool(camera_mask)
        self.translation_scale: float = float(translation_scale)
        self.camera_mask_threshold: float | None = camera_mask_threshold
        self.camera_mask_min_frames: int = int(camera_mask_min_frames)
        self.camera_mask_gripper_threshold: float = float(camera_mask_gripper_threshold)
        self.camera_mask_gripper_guard_frames: int = int(camera_mask_gripper_guard_frames)
        self.robot_multiview: bool = robot_multiview
        super().__init__(*args, **kwargs)
        if robot_multiview:
            runtime = cast(BaseLanceActionDataset, self)
            if runtime.skip_video_loading:
                raise ValueError("robot_multiview requires video loading.")
            self.view_names = tuple(self.canonical_dataset_spec.view_stream_id_by_name)
            if len(self.view_names) != 3:
                raise ValueError("robot_multiview requires ordered head, left-arm and right-arm RGB streams.")

    def _build_training_item_from_request(
        self,
        request: LanceActionSampleRequest,
        source_rows: dict[str, dict[str, Any]],
        window: MaterializedActionWindow,
    ) -> dict[str, Any]:
        """Keep native action construction, exposing wrist pixels separately only on opt-in."""
        item = cast(BaseLanceActionDataset, super())._build_training_item_from_request(request, source_rows, window)
        if self.robot_multiview:
            runtime = cast(BaseLanceActionDataset, self)
            expected_frames = runtime.chunk_length + 1
            # Reuse the single-view head, including its source-specific preprocessing.
            # Dataset adapters own source-specific preprocessing; other wrist
            # views retain their native aspect ratio before the 256-tier transform.
            # Wrist views are never mosaicked or treated as controls.
            videos = [item["video"]]
            for name in self.view_names[1:]:
                video = runtime._video_view_tchw(window, name, expected_frames)
                videos.append(video.permute(1, 0, 2, 3).contiguous())
            item["_robot_multiview_videos"] = videos
        return item

    def _camera_mask_movement_score(self, action: torch.Tensor) -> torch.Tensor:
        """Return the source-specific numerical movement score for each action row."""
        raise NotImplementedError

    def _camera_mask_source_action(self, item: dict[str, Any], action: torch.Tensor) -> torch.Tensor:
        """Return source-native rows used only to classify camera frames."""
        del item
        return action

    def _camera_mask_gripper_change(
        self,
        action: torch.Tensor,
        *,
        initial_grippers: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return source-schema gripper transitions that must guard camera labels."""
        del initial_grippers
        return torch.zeros(action.shape[0], dtype=torch.bool, device=action.device)

    def _camera_mask_initial_grippers(
        self,
        item: dict[str, Any],
        action: torch.Tensor,
    ) -> torch.Tensor | None:
        """Return the observation immediately before the first emitted gripper row."""
        del item, action
        return None

    def _camera_mask_camera_pose(self, item: dict[str, Any], action: torch.Tensor) -> torch.Tensor:
        """Read the 9D camera pose already produced by the parent Action adapter."""
        del item
        if action.shape[-1] < 9:
            raise ValueError(f"camera_mask requires at least 9 action dimensions, got {action.shape[-1]}.")
        return action[:, :9].clone()

    def _camera_mask_rows(
        self,
        action: torch.Tensor,
        *,
        initial_grippers: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Find long stationary runs after symmetrically guarding gripper changes."""
        score = self._camera_mask_movement_score(action)
        if score.shape != (action.shape[0],):
            raise ValueError(
                f"camera-mask movement score must have shape {(action.shape[0],)}, got {tuple(score.shape)}."
            )
        guarded_gripper_change = self._camera_mask_gripper_change(
            action,
            initial_grippers=initial_grippers,
        )
        if guarded_gripper_change.shape != score.shape:
            raise ValueError("camera-mask gripper-change rows must match the action length.")
        guard_frames = self.camera_mask_gripper_guard_frames
        if guard_frames and bool(guarded_gripper_change.any()):
            guarded_indices = torch.nonzero(guarded_gripper_change, as_tuple=False).flatten()
            offsets = torch.arange(-guard_frames, guard_frames + 1, device=action.device)
            guarded_indices = (guarded_indices[:, None] + offsets[None, :]).flatten()
            guarded_indices = guarded_indices[(guarded_indices >= 0) & (guarded_indices < action.shape[0])]
            guarded_gripper_change = torch.zeros_like(guarded_gripper_change)
            guarded_gripper_change[guarded_indices] = True
        guarded_score = torch.where(guarded_gripper_change, torch.full_like(score, torch.inf), score)
        assert self.camera_mask_threshold is not None
        return _long_stationary_runs(
            guarded_score,
            threshold=self.camera_mask_threshold,
            minimum_frames=self.camera_mask_min_frames,
        )

    @staticmethod
    def _camera_mask_domain_ids(item: dict[str, Any], camera_mask: torch.Tensor) -> torch.Tensor:
        """Expand a sample domain into framewise IDs and relabel camera rows."""
        domain_id = item.get("domain_id")
        if not isinstance(domain_id, torch.Tensor) or domain_id.ndim not in (0, 1):
            raise ValueError("camera_mask requires scalar or framewise domain_id.")
        domain_ids = domain_id.expand(camera_mask.shape[0]).clone() if domain_id.ndim == 0 else domain_id.clone()
        if domain_ids.shape != camera_mask.shape:
            raise ValueError("Framewise domain_id must match the action length.")
        domain_ids[camera_mask.to(device=domain_ids.device)] = get_domain_id("camera_pose")
        return domain_ids

    def materialize_local_sample(
        self,
        request: LanceActionSampleRequest,
        source_rows: dict[str, dict[str, Any]],
        backend: ByteRangeBackend,
    ) -> dict[str, Any]:
        """Retain camera masking when the parent materializes captured local assets."""
        item = cast(BaseLanceActionDataset, super()).materialize_local_sample(request, source_rows, backend)
        return self._apply_camera_mask(item)

    def _apply_camera_mask(self, item: dict[str, Any]) -> dict[str, Any]:
        """Apply the same framewise mask to remote and local native Action items."""
        if not self.camera_mask:
            return item
        action = item.get("action")
        if not isinstance(action, torch.Tensor) or action.ndim != 2:
            raise ValueError("camera_mask requires a two-dimensional action tensor.")
        mask_action = self._camera_mask_source_action(item, action)
        if not isinstance(mask_action, torch.Tensor) or mask_action.ndim != 2:
            raise ValueError("camera_mask source action must be a two-dimensional tensor.")
        if mask_action.shape[0] != action.shape[0]:
            raise ValueError(
                "camera_mask source action must match the output action length, got "
                f"{mask_action.shape[0]} and {action.shape[0]}."
            )
        initial_grippers = self._camera_mask_initial_grippers(item, mask_action)
        if initial_grippers is None:
            camera_mask = self._camera_mask_rows(mask_action)
        else:
            camera_mask = self._camera_mask_rows(mask_action, initial_grippers=initial_grippers)
        camera_pose = self._camera_mask_camera_pose(item, action)
        if camera_pose.shape != (action.shape[0], 9):
            raise ValueError(f"camera pose must have shape {(action.shape[0], 9)}, got {tuple(camera_pose.shape)}.")
        masked_action = action
        if bool(camera_mask.any()):
            masked_action = action.clone()
            masked_action[camera_mask] = 0
            masked_action[camera_mask, :9] = camera_pose[camera_mask]
            domain_ids = self._camera_mask_domain_ids(item, camera_mask)
        else:
            domain_ids = item["domain_id"]
        return {
            **item,
            "action": masked_action,
            "domain_id": domain_ids,
            CANONICAL_CAMERA_MASK_NATIVE_ACTION_KEY: masked_action,
            CANONICAL_CAMERA_MASK_TRANSLATION_SCALE_KEY: self.translation_scale,
        }

    def get_action_normalizer(self, sample: dict[str, Any] | None = None) -> ActionNormalizer | None:
        """Use native normalization on robot rows and identity on camera rows."""
        native_normalizer = super().get_action_normalizer(sample)
        if not self.camera_mask or sample is None:
            return native_normalizer
        domain_ids = sample.get("domain_id")
        if not isinstance(domain_ids, torch.Tensor) or domain_ids.ndim == 0:
            return native_normalizer
        if domain_ids.ndim != 1:
            raise ValueError("camera_mask requires scalar or framewise domain_id.")
        return _CameraMaskedNativeNormalizer(
            domain_ids=domain_ids,
            native_normalizer=native_normalizer,
            camera_translation_scale=self.translation_scale,
        )

    @property
    def canonical_dataset_spec(self) -> CanonicalLanceActionDatasetSpec:
        """Return the canonical spec with a useful configuration error."""
        spec = self.lance_dataset_spec
        if not isinstance(spec, CanonicalLanceActionDatasetSpec):
            raise TypeError(
                f"{type(self).__name__} requires CanonicalLanceActionDatasetSpec, got {type(spec).__name__}."
            )
        return spec

    def _load_resolved_item(
        self,
        request: LanceActionSampleRequest,
        source_rows: dict[str, dict[str, Any]] | None = None,
        object_backend: ByteRangeBackend | None = None,
    ) -> dict[str, Any]:
        """Preserve empty embedded text and treat sidecar captions as final text."""
        sidecar_caption = request.caption if self.canonical_dataset_spec.caption_overlay_path is not None else None
        caption_is_empty = not bool(request.caption)
        resolved_request = replace(request, caption=_EMPTY_CAPTION_PLACEHOLDER) if caption_is_empty else request
        dataset = cast(BaseLanceActionDataset, self)
        item = (
            BaseLanceActionDataset._load_resolved_item(dataset, resolved_request)
            if source_rows is None and object_backend is None
            else BaseLanceActionDataset._load_resolved_item(dataset, resolved_request, source_rows, object_backend)
        )
        if "ai_caption" in item:
            if sidecar_caption is not None:
                # The external annotation is already the final T2V/I2V text;
                # do not apply legacy task-name or family-specific formatting.
                item["ai_caption"] = sidecar_caption
            elif caption_is_empty:
                item["ai_caption"] = ""
        return item
