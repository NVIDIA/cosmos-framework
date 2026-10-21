# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""DPO preference dataset for Cosmos3 OmniMoT flow-matching DPO — the DATA LAYER.

Per DESIGN §1.3 the concrete dataset (storage/decode/snap/shuffle) lives in
`data/datasets/`, separate from the `roles/env/` source node that wraps it: this file
produces samples and knows nothing about `Trajectory`/roles/placement;
`roles/env/preference.py::DPOPreferenceEnv` adds the pipeline-facing Env.

Reads `(winner, loser)` preference pairs from a JSONL file (produced by the PhysReward
Best-of-N / red-tint pair-gen pipelines) and yields each pair as a
`(prompt, winner_dict, loser_dict)` triple for `DatasetPreferenceEnv`. Supports both
`media="video"` (T2V; key `video`, `[C,T,H,W]`) and `media="image"` (T2I; key `images`,
single-frame `[C,H,W]`). The per-side dicts are in the exact batch format
`Cosmos3FlowPolicy.encode_and_condition` consumes (the media as a one-item vision list
under its routing key, `ai_caption`, `image_size`, `conditioning_fps`), so
`data.packer.preference.unpack_preferences` produces a ready-to-encode batch.

Behavior is ported (not imported) from cosmos-rl's `DPOPreferenceDataset`
(`cosmos_rl/tools/dataset/wfm/dpo_preference_dataset.py`): score-gap filtering,
canonical {VIDEO,IMAGE}_RES_SIZE_INFO bucket-snap (contain-fit resize + reflection pad,
mirroring the cosmos3 `ResizeLargestSideAspectPreserving` + `ReflectionPadding`
augmentors), and classifier-free-guidance caption dropout. Two DPO-critical
invariants are preserved:
  - **symmetric cfg-dropout**: the dropout decision is made ONCE per pair and the
    (possibly emptied) caption is shared by winner and loser, so their text tokens
    stay identical (the DPO log-ratio is only defined for a fixed prompt);
  - **identical shapes**: bucket-snap lands both sides in the same canonical (H,W); a
    residual cross-bin mismatch raises rather than silently broadcasting.

The {VIDEO,IMAGE}_RES_SIZE_INFO tables are imported from i4 (`projects.cosmos3.cosmos3`) —
the canonical resolution tables the loaded checkpoint was trained on; i4, not cosmos-rl.

Data format (one JSON object per line):
    {"prompt": "...", "winner_video": "a.mp4", "loser_video": "b.mp4",
     "winner_score": -0.38, "loser_score": -0.79, "score_gap": 0.41, "fps": 24}
Paths may be videos (mp4 via decord), images (png/jpg via PIL, `media="image"`), or
`.pt` tensors (a dict with uint8 `"frames"`: `[C,T,H,W]` video / `[C,H,W]` image).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from typing import Any

import torch
import torch.nn.functional as F

from cosmos_framework.data.generator.utils import IMAGE_RES_SIZE_INFO, VIDEO_RES_SIZE_INFO

# Canonical aspect-ratio binning, mirroring
# cosmos_framework.data.generator.local_datasets.helper.get_aspect_ratio (`<` = [lo,hi)).
_AR_BINS: tuple[tuple[float, str], ...] = (
    (0.65, "9,16"),  # ratio < 0.65 -> 9:16
    (0.88, "3,4"),  # ratio < 0.88 -> 3:4
    (1.16, "1,1"),  # ratio < 1.16 -> 1:1
    (1.55, "4,3"),  # ratio < 1.55 -> 4:3
)


def _canonical_sha256_digest(value: Any, key: str) -> str:
    """Normalize a declared SHA-256 alias or fail closed on ambiguous metadata."""
    if not isinstance(value, str):
        raise TypeError(f"preference manifest field {key!r} must be a SHA-256 string")
    normalized = value.strip().lower()
    if normalized.startswith("sha256:"):
        normalized = normalized[len("sha256:") :]
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError(f"preference manifest field {key!r} must be 64 hexadecimal SHA-256 characters, got {value!r}")
    return normalized


def _aspect_ratio_bin(width: int, height: int) -> str:
    """Aspect-ratio bucket key for `VIDEO_RES_SIZE_INFO` lookup ("16,9" = catch-all)."""
    ratio = width / height
    for hi, key in _AR_BINS:
        if ratio < hi:
            return key
    return "16,9"


def _snap_to_bucket(
    frames: torch.Tensor, resolution: str, table: dict = VIDEO_RES_SIZE_INFO
) -> tuple[torch.Tensor, torch.Tensor]:
    """Snap `[C,T,H,W]` uint8 frames to the canonical bucket of `table` (VIDEO/IMAGE
    RES_SIZE_INFO): contain-fit bilinear resize (half-up rounding) + reflection pad
    (bottom/right), matching the cosmos3 resize+pad augmentor pair. Returns
    `(frames, image_size)` where `image_size = [target_h, target_w, new_h, new_w]`."""
    if frames.dtype != torch.uint8:
        raise ValueError(f"_snap_to_bucket expected uint8 frames, got {frames.dtype}.")
    _, _, height, width = frames.shape
    ar_bin = _aspect_ratio_bin(width, height)
    target_w, target_h = table[resolution][ar_bin]

    scale = min(target_w / width, target_h / height)  # contain-fit
    new_h = max(int(scale * height + 0.5), 1)
    new_w = max(int(scale * width + 0.5), 1)

    frames_f = (frames.float() / 255.0).permute(1, 0, 2, 3)  # [T,C,H,W]
    if (new_h, new_w) != (height, width):
        frames_f = F.interpolate(frames_f, size=(new_h, new_w), mode="bilinear", align_corners=False, antialias=True)

    pad_right, pad_bottom = target_w - new_w, target_h - new_h
    if pad_right < 0 or pad_bottom < 0:
        raise RuntimeError(f"contain-fit produced ({new_h},{new_w}) > target ({target_h},{target_w}).")
    if pad_right or pad_bottom:
        pad_spec = (0, pad_right, 0, pad_bottom)  # (left,right,top,bottom)
        # Reflection needs per-side pad < dim; fall back to replicate ("edge") otherwise.
        mode = "replicate" if (pad_right >= new_w or pad_bottom >= new_h) else "reflect"
        frames_f = F.pad(frames_f, pad_spec, mode=mode)

    out = (frames_f * 255.0).clamp(0, 255).round().to(torch.uint8).permute(1, 0, 2, 3)  # [C,T,target_h,target_w]
    image_size = torch.tensor([target_h, target_w, new_h, new_w], dtype=torch.float)
    return out, image_size


class DPOPreferenceDataset:
    """Map-style dataset of `(winner, loser)` preference video pairs for flow-matching
    DPO. `__getitem__` returns `(prompt, winner_dict, loser_dict)` for
    `DatasetPreferenceEnv`. Each side dict is encode-ready (video as a one-item vision
    list + `ai_caption` + `image_size` + `conditioning_fps`)."""

    def __init__(
        self,
        jsonl_path: str,
        resolution: str | None = "256",
        num_frames: int = 121,
        caption_key: str = "prompt",
        winner_video_key: str = "winner_video",
        loser_video_key: str = "loser_video",
        min_score_gap: float = 0.0,
        min_wm_gap: float = 0.0,
        min_align_gap: float = 0.0,
        cfg_dropout_rate: float = 0.1,
        default_fps: int = 24,
        seed: int = 0,
        media: str = "video",
    ) -> None:
        if media not in ("video", "image"):
            raise ValueError(f"DPOPreferenceDataset: media must be 'video' or 'image', got {media!r}.")
        self.media = media
        # Image vs video: different resolution table + the OmniMoT routing key. Images
        # are single-frame ([C,H,W], augmented to a T=1 latent by the model).
        self._table = IMAGE_RES_SIZE_INFO if media == "image" else VIDEO_RES_SIZE_INFO
        self._media_key = "images" if media == "image" else "video"
        if resolution is not None and resolution not in self._table:
            raise ValueError(
                f"DPOPreferenceDataset: unknown {media} resolution '{resolution}'. "
                f"Must be one of {sorted(self._table.keys())} or None."
            )
        self.resolution = resolution
        self.num_frames = num_frames
        self.caption_key = caption_key
        self.winner_video_key = winner_video_key
        self.loser_video_key = loser_video_key
        self.cfg_dropout_rate = cfg_dropout_rate
        self.default_fps = default_fps
        self._rng = random.Random(seed)
        self._jsonl_dir = os.path.dirname(os.path.abspath(jsonl_path)) if jsonl_path else os.getcwd()

        self.entries: list[dict[str, Any]] = self._read_entries(
            jsonl_path, min_score_gap=min_score_gap, min_wm_gap=min_wm_gap, min_align_gap=min_align_gap
        )
        if not self.entries:
            raise RuntimeError(
                f"DPOPreferenceDataset: no entries passed the filters "
                f"(min_score_gap={min_score_gap}, min_wm_gap={min_wm_gap}, "
                f"min_align_gap={min_align_gap}) in '{jsonl_path}'."
            )

    def _read_entries(
        self, jsonl_path: str, *, min_score_gap: float, min_wm_gap: float, min_align_gap: float
    ) -> list[dict[str, Any]]:
        """Entry source — overridable (SyntheticPreferenceDataset fabricates entries
        in memory instead of reading a jsonl)."""
        entries: list[dict[str, Any]] = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if (
                    entry.get("score_gap", float("inf")) >= min_score_gap
                    and entry.get("wm_gap", float("inf")) >= min_wm_gap
                    and entry.get("align_gap", float("inf")) >= min_align_gap
                ):
                    entries.append(entry)
        return entries

    def __len__(self) -> int:
        return len(self.entries)

    def _cursor_manifest_entry(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Return the semantic row representation used for cursor identity."""
        return entry

    def cursor_fingerprint(self) -> str:
        """Stable identity for the ordered source and cursor-relevant preprocessing.

        The digest covers the filtered JSONL rows rather than launch-directory paths or
        mutable filesystem timestamps. Rows may carry immutable media/manifest digests;
        when they do, those fields are covered automatically. The cursor fingerprint is
        a semantic manifest identity, not an implicit bytewise scan of every media file.
        """
        cached = getattr(self, "_cursor_fingerprint", None)
        if cached is not None:
            return str(cached)
        digest = hashlib.sha256()
        config = {
            "class": f"{type(self).__module__}.{type(self).__qualname__}",
            "media": self.media,
            "resolution": self.resolution,
            "num_frames": self.num_frames,
            "caption_key": self.caption_key,
            "winner_video_key": self.winner_video_key,
            "loser_video_key": self.loser_video_key,
            "cfg_dropout_rate": self.cfg_dropout_rate,
            "default_fps": self.default_fps,
        }
        digest.update(json.dumps(config, sort_keys=True, separators=(",", ":")).encode())
        for entry in self.entries:
            digest.update(b"\n")
            manifest_entry = self._cursor_manifest_entry(entry)
            digest.update(json.dumps(manifest_entry, sort_keys=True, separators=(",", ":"), default=str).encode())
        self._cursor_fingerprint = digest.hexdigest()
        return str(self._cursor_fingerprint)

    def membership_ids(self) -> tuple[str, ...]:
        """Order-independent semantic pair identities used only for leakage rejection.

        Cursor identity intentionally includes preprocessing and row order. Membership identity has
        the opposite job: recognize the same effective prompt/winner/loser pair after JSON
        whitespace, key ordering, row ordering, path aliases, or CFG/preprocessing knobs change.
        This launch-time audit is deliberately metadata-only: it never decodes or hashes the media
        payload. The normalized locator is always an alias, even when a manifest-provided immutable
        digest exists; every available winner/loser alias combination becomes an identity. This
        catches the same path when only one manifest declares a digest and content clones at
        different locators when both manifests declare matching digests.
        """
        identities: list[str] = []
        for entry in self.entries:
            winner_identities = self._membership_media_identities(entry, "winner", self.winner_video_key)
            loser_identities = self._membership_media_identities(entry, "loser", self.loser_video_key)
            for winner_identity in winner_identities:
                for loser_identity in loser_identities:
                    payload = {
                        "prompt": entry.get(self.caption_key, ""),
                        "winner": winner_identity,
                        "loser": loser_identity,
                    }
                    encoded = json.dumps(
                        payload,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    ).encode()
                    identities.append(hashlib.sha256(encoded).hexdigest())
        return tuple(sorted(identities))

    def _membership_media_identities(
        self,
        entry: dict[str, Any],
        side: str,
        media_key: str,
    ) -> tuple[str, ...]:
        """Return the canonical locator plus every declared digest without media I/O."""
        resolved = os.path.realpath(self._resolve_media_path(entry[media_key]))
        identities = [f"locator:{resolved}"]
        digest_keys = (
            f"{side}_sha256",
            f"{side}_digest",
            f"{media_key}_sha256",
            f"{media_key}_digest",
        )
        for key in digest_keys:
            value = entry.get(key)
            if value not in (None, ""):
                identity = f"digest:{_canonical_sha256_digest(value, key)}"
                if identity not in identities:
                    identities.append(identity)
        return tuple(identities)

    def cursor_state_dict(self) -> dict[str, Any]:
        """Return dataset-local stochastic state consumed by ``__getitem__``."""
        return {"version": 1, "rng": self._rng.getstate()}

    def load_cursor_state_dict(self, state: dict[str, Any]) -> None:
        """Restore dataset-local stochastic state from :meth:`cursor_state_dict`."""
        version = state.get("version")
        if type(version) is not int:
            raise ValueError("DPOPreferenceDataset cursor version must be an integer")
        if version != 1:
            raise ValueError(f"unsupported DPOPreferenceDataset cursor version {version}")
        if "rng" not in state:
            raise ValueError("DPOPreferenceDataset cursor is missing rng state")
        self._rng.setstate(state["rng"])

    def __getitem__(self, idx: int) -> tuple[str, dict[str, Any], dict[str, Any]]:
        entry = self.entries[idx]
        # Symmetric cfg-dropout: ONE decision per pair, shared caption (keeps the
        # winner/loser text tokens identical — required for a well-defined DPO ratio).
        prompt = entry.get(self.caption_key, "")
        if self._rng.random() < self.cfg_dropout_rate:
            prompt = ""

        winner_video, image_size = self._load_media(entry[self.winner_video_key])
        loser_video, _ = self._load_media(entry[self.loser_video_key])
        if winner_video.shape != loser_video.shape:
            raise ValueError(
                f"DPO winner/loser shape mismatch at idx={idx}: winner={tuple(winner_video.shape)} "
                f"loser={tuple(loser_video.shape)}. Both sides must share shape — land in the same "
                f"{self.media} RES_SIZE_INFO aspect-ratio bin or pre-crop to matching dims."
            )
        if image_size is None:
            h, w = winner_video.shape[2], winner_video.shape[3]
            image_size = torch.tensor([h, w, h, w], dtype=torch.float)
        fps = float(entry.get("fps", self.default_fps))
        return (
            prompt,
            self._side(winner_video, prompt, image_size, fps),
            self._side(loser_video, prompt, image_size, fps),
        )

    def _side(self, media: torch.Tensor, prompt: str, image_size: torch.Tensor, fps: float) -> dict[str, Any]:
        """One side of a pair in `encode_and_condition` batch format: the media tensor as
        a one-item vision list under the routing key (`video` [C,T,H,W] / `images`
        [C,H,W]), raw caption (tokenized downstream), image_size, fps."""
        return {
            self._media_key: [media],
            "ai_caption": prompt,
            "image_size": image_size,
            "conditioning_fps": torch.tensor([fps], dtype=torch.bfloat16),
        }

    def _load_media(self, path: str) -> tuple[torch.Tensor, torch.Tensor | None]:
        frames = self._read_frames(path)
        # Images are single-frame: normalize to [C,1,H,W] for snapping, return [C,H,W].
        if self.media == "image":
            if frames.dim() == 3:  # [C,H,W] -> [C,1,H,W]
                frames = frames.unsqueeze(1)
            if self.resolution is None:
                return frames.squeeze(1), None
            snapped, image_size = _snap_to_bucket(frames, self.resolution, table=self._table)
            return snapped.squeeze(1), image_size  # [C,H,W]
        if self.resolution is None:
            return frames, None
        return _snap_to_bucket(frames, self.resolution, table=self._table)

    def _resolve_media_path(self, path: str | os.PathLike[str]) -> str:
        """Resolve a row-relative media path against its JSONL, never the launch CWD."""
        expanded = os.path.expanduser(os.fspath(path))
        if os.path.isabs(expanded):
            return os.path.normpath(expanded)
        return os.path.normpath(os.path.join(self._jsonl_dir, expanded))

    def _read_frames(self, path: str) -> torch.Tensor:
        """Raw frame source ([C,H,W] image / [C,T,H,W] video, uint8) — overridable
        (SyntheticPreferenceDataset generates frames instead of reading files)."""
        path = self._resolve_media_path(path)
        if path.endswith(".pt"):
            return torch.load(path, weights_only=True, map_location="cpu")["frames"]  # uint8
        if self.media == "image":
            return self._load_image(path)
        return self._load_video_decord(path)

    def _load_image(self, path: str) -> torch.Tensor:
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Pillow is required to load image files (pip install pillow).") from exc
        import numpy as np

        arr = torch.from_numpy(np.array(Image.open(path).convert("RGB")))  # [H,W,C]
        return arr.permute(2, 0, 1).contiguous()  # [C,H,W] uint8

    def _load_video_decord(self, path: str) -> torch.Tensor:
        try:
            import decord

            decord.bridge.set_bridge("torch")
        except ImportError as exc:
            raise ImportError("decord is required to load MP4 videos (pip install decord).") from exc
        vr = decord.VideoReader(path)
        total = len(vr)
        if total <= self.num_frames:
            indices = list(range(total))
        else:
            step = total / self.num_frames
            indices = [int(i * step) for i in range(self.num_frames)]
        return vr.get_batch(indices).permute(3, 0, 1, 2)  # [T,H,W,C] -> [C,T,H,W]
