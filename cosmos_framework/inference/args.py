# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
import math
import os
import re
from functools import cache
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal, Self, Sequence, TypeVar, cast, override

import pydantic
import pynvml
from typing_extensions import assert_never
from tyro.conf import Suppress

from cosmos_framework.data.generator.action.utils.pose_utils import PoseConvention
from cosmos_framework.inference.common.args import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    ArgsBase,
    CfgpSize,
    CheckpointConfig,
    CheckpointType,
    ConfigFileType,
    OverridesBase,
    ResolvedFilePath,
    ResolvedFilePathOrUrl,
    SampleArgs,
    SampleOverrides,
    SetupArgs,
    SetupOverrides,
    StrEnum,
    Training,
    _deep_merge,
    download_file,
)
from cosmos_framework.inference.common.config import CONFIG_DIR, PACKAGE_DIR, deserialize_config_dict
from cosmos_framework.utils import log
from cosmos_framework.utils.checkpoint_db import CheckpointDirHf
from cosmos_framework.utils.flags import SMOKE, TRAINING
from cosmos_framework.utils.generator.multiview import is_causal_multiview_config, safe_multiview_camera_name

CameraActionNormalization = Literal["scale", "global_asinh"]
CameraNumFrames = Annotated[int, pydantic.Field(ge=2)]
ContentCropBox = tuple[
    pydantic.NonNegativeInt,
    pydantic.NonNegativeInt,
    pydantic.PositiveInt,
    pydantic.PositiveInt,
]

if TYPE_CHECKING:
    from cosmos_framework.configs.base.defaults.model_config import OmniMoTModelConfig
    from cosmos_framework.inference.common.inference import Inference


def _load_transfer_prompt_path(path: str | Path) -> str:
    """Load a transfer prompt from a ``.json`` or plain ``.txt`` file."""
    resolved = Path(path)
    text = resolved.read_text()
    if resolved.suffix.lower() == ".json":
        return json.dumps(json.loads(text))
    return text.strip()


def _load_transfer_negative_prompt_file(path: str | Path) -> str:
    """Load a JSON negative caption file for transfer inference."""
    candidate = Path(path)
    if not candidate.is_file():
        defaults_path = PACKAGE_DIR / "defaults" / candidate.name
        if defaults_path.is_file():
            candidate = defaults_path
        else:
            raise FileNotFoundError(f"Missing negative prompt file: {path} (also checked {defaults_path})")
    return json.dumps(json.loads(candidate.read_text()))


@cache
def _load_modality_defaults(model_mode: str) -> dict[str, Any]:
    default_file = PACKAGE_DIR / f"defaults/{model_mode}/sample_args.json"
    if not default_file.exists():
        raise FileNotFoundError(f"Missing modality defaults: {default_file}")
    data = json.loads(default_file.read_text())
    neg_file = data.pop("negative_prompt_file", None)
    if neg_file is not None:
        neg_path = PACKAGE_DIR / "defaults" / neg_file
        if not neg_path.exists():
            raise FileNotFoundError(f"Missing negative prompt file: {neg_path}")
        data["negative_prompt"] = json.dumps(json.loads(neg_path.read_text()))
    return data


Guidance = Annotated[float, pydantic.Field(ge=0, le=7)]
GuidanceInterval = tuple[pydantic.NonNegativeFloat, pydantic.NonNegativeFloat]
PromptUpsamplerProbability = Annotated[float, pydantic.Field(ge=0, le=1)]
ControlGuidance = Annotated[float, pydantic.Field(ge=0, le=10)]


class SamplingArgs(ArgsBase):
    num_steps: pydantic.PositiveInt
    guidance: Guidance
    guidance_interval: GuidanceInterval | None
    normalize_cfg: bool
    shift: float
    sigma_max: float


class SamplingOverrides(OverridesBase):
    """Sampling arguments for 'OmniMoTModel.generate_samples'."""

    num_steps: pydantic.PositiveInt | None = None
    """Number of diffusion steps. Distilled checkpoints use their fixed schedule instead."""
    guidance: Training[Guidance | None] = None
    """Guidance scale for the diffusion model."""
    guidance_interval: Training[GuidanceInterval | None] = None
    """Guidance interval for the diffusion model."""
    normalize_cfg: Training[bool | None] = None
    """If True, normalize the CFG output."""
    shift: Training[float | None] = None
    """Shift in the UniPC sampler. Ignored when sampler='edm'."""
    sigma_max: Training[float | None] = None
    """Maximum sigma for the EDM sampler. Ignored when sampler='unipc'."""

    def _build_sampling(
        self, model_config: "OmniMoTModelConfig", sample_meta: "SampleMeta", *, num_steps_configured: bool = True
    ):
        if sample_meta.model_mode.is_reasoner:
            # Diffusion sampling fields are unused by the reasoner but required by
            # OmniSampleArgs validation; fill in inert sentinels.
            if self.num_steps is None:
                self.num_steps = 1
            if self.guidance is None:
                self.guidance = 0.0
            if self.normalize_cfg is None:
                self.normalize_cfg = False
            if self.shift is None:
                self.shift = 0.0
            if self.sigma_max is None:
                self.sigma_max = 0.0
            return
        fixed_sampler = getattr(model_config, "fixed_step_sampler_config", None)
        if fixed_sampler is not None:
            schedule = fixed_sampler.t_list
            # FixedStepSampler appends the terminal zero only when it is absent.
            steps = len(schedule) - int(bool(schedule) and schedule[-1] == 0.0)
            if steps < 1:
                raise ValueError("fixed_step_sampler_config.t_list must contain a nonzero sigma")
            # Warn only about a value someone chose; the built-in modality default (35) is not a
            # request, so every default run of a distilled checkpoint would otherwise warn.
            if num_steps_configured and self.num_steps is not None and self.num_steps != steps:
                log.warning(
                    f"num_steps={self.num_steps} is ignored for this distilled checkpoint; "
                    f"using its fixed {steps}-step sampling schedule."
                )
            self.num_steps = steps
        assert self.num_steps is not None
        if SMOKE:
            self.num_steps = min(self.num_steps, 1)


InferenceResolution = Literal["256", "480", "720", "768"]
if TRAINING:
    Resolution = Literal["256", "480", "704", "720", "768"]
else:
    Resolution = InferenceResolution
AspectRatio = Literal["1,1", "4,3", "3,4", "16,9", "9,16"]

# Resolutions that only support image generation (num_frames == 1). Video
# generation at these resolutions is rejected by ``_build_vision_data`` because
# the model wasn't trained on temporal data above 720p and ``MAX_NUM_FRAMES``
# has no entry for them.
IMAGE_ONLY_RESOLUTIONS: frozenset[str] = frozenset({"1080"})

MIN_NUM_FRAMES = 24
MAX_NUM_FRAMES: dict[Resolution, int] = {
    "256": 400,
    "480": 300,
    "704": 200,
    "720": 200,
    "768": 200,
}


ModelSize = Literal["0.6B", "2B", "8B", "30B-A3B", "32B", "235B-A22B"]


class ModelMode(StrEnum):
    TEXT2IMAGE = "text2image"
    TEXT2VIDEO = "text2video"
    IMAGE2IMAGE = "image2image"
    IMAGE2VIDEO = "image2video"
    VIDEO2VIDEO = "video2video"
    AUDIO_IMAGE2VIDEO = "audio_image2video"

    # Action
    FORWARD_DYNAMICS = "forward_dynamics"
    INVERSE_DYNAMICS = "inverse_dynamics"
    WAM = "wam"

    REASONER = "reasoner"

    @property
    def is_action(self) -> bool:
        return self in ACTION_MODEL_MODES

    @property
    def is_reasoner(self) -> bool:
        return self in REASONER_MODEL_MODES

    @property
    def is_sound_condition(self) -> bool:
        return self in SOUND_CONDITION_MODEL_MODES


# Image-output modes: ``num_frames`` defaults to 1 and the output is saved as a still image.
_IMAGE_OUTPUT_MODES: frozenset[ModelMode] = frozenset({ModelMode.TEXT2IMAGE, ModelMode.IMAGE2IMAGE})

# Modes that produce action tensors and require a model with ``action_gen=True``.
ACTION_MODEL_MODES: frozenset[ModelMode] = frozenset(
    {ModelMode.FORWARD_DYNAMICS, ModelMode.INVERSE_DYNAMICS, ModelMode.WAM}
)

REASONER_MODEL_MODES: frozenset[ModelMode] = frozenset({ModelMode.REASONER})

# Modes that condition generation on a real input audio clip (require a model
# with ``sound_gen=True`` and a ``sound_path``).
SOUND_CONDITION_MODEL_MODES: frozenset[ModelMode] = frozenset({ModelMode.AUDIO_IMAGE2VIDEO})


def is_reasoner_only(sample_overrides: Sequence["OmniSampleOverrides"]) -> bool:
    """Return whether every requested sample uses the reasoner-only path."""
    return bool(sample_overrides) and all(sample.sample_meta.model_mode.is_reasoner for sample in sample_overrides)


def reasoner_only_overrides() -> list[str]:
    """Config overrides that drop generation-side modules for reasoner-only runs.

    Reasoner decoding runs entirely in the understanding tower, so none of these
    modules are reachable. ``dcp.load`` is pull-based, so not building them also
    means their tensors are never read from the checkpoint.

    The three flags are deliberately orthogonal and each defaults to the
    generation-enabled value: dropping any one entry from this list degrades to
    the previous behaviour for that module alone, leaving the others in effect.
    """
    return [
        # The generation VAE (Wan2.2 for Edge): ~1.3 GiB resident, and its weights
        # are downloaded lazily at tokenizer construction, so skipping it also
        # avoids a ~2.6 GiB download on a cold cache.
        "model.config.load_vision_tokenizer=false",
        # VFM-level generation heads: time_embedder, proj_in/proj_out, action_*, sound_*.
        # All three modality flags must fall together: ``Cosmos3VFMNetworkConfig``
        # asserts that action and sound generation each imply vision generation, and
        # Edge checkpoints ship with ``action_gen=true``.
        "model.config.vision_gen=false",
        "model.config.action_gen=false",
        "model.config.sound_gen=false",
        # The MoT generation tower (every ``*_moe_gen`` module): ~2.0 GiB resident.
        "model.config.vlm_config.model_instance.config.include_gen_pathway=false",
    ]


class VisionMode(StrEnum):
    IMAGE = "image"
    VIDEO = "video"

    @classmethod
    def from_model_mode(cls, model_mode: ModelMode) -> Self:
        return cls.IMAGE if model_mode in _IMAGE_OUTPUT_MODES else cls.VIDEO


class ConditionVisionMode(StrEnum):
    IMAGE = "image"
    VIDEO = "video"


class NegativeMetadataMode(StrEnum):
    NONE = "none"
    SAME = "same"
    INVERSE = "inverse"


class TransferHintKey(StrEnum):
    EDGE = "edge"
    BLUR = "blur"
    DEPTH = "depth"
    SEG = "seg"
    WSM = "wsm"
    LIDAR = "lidar"


class PresetEdgeThreshold(StrEnum):
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class PresetBlurStrength(StrEnum):
    NONE = "none"
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class TransferArgs(ArgsBase):
    """Resolved transfer inference arguments for a single control hint."""

    control_path: ResolvedFilePathOrUrl | None = None
    weight: float = 1.0
    """Strength of this control signal in the weighted multi-control attention aggregation."""

    @pydantic.field_validator("weight", mode="before")
    @classmethod
    def _coerce_none_weight(cls, v: float | None) -> float:
        # ``TransferOverrides.weight`` defaults to None ("unset") and is resolved into
        # this required float via ``_build`` (which does not strip None). An unset weight
        # resolves to the neutral 1.0 → equal weighting / single-control parity.
        return 1.0 if v is None else v


class LidarTransferArgs(TransferArgs):
    """HD-map control and optional measured target prefix for LiDAR inference."""

    condition_path: ResolvedFilePathOrUrl | None = None
    num_conditional_sweeps: pydantic.PositiveInt = 1


class EdgeTransferArgs(TransferArgs):
    preset_edge_threshold: PresetEdgeThreshold = PresetEdgeThreshold.MEDIUM


class BlurTransferArgs(TransferArgs):
    preset_blur_strength: PresetBlurStrength = PresetBlurStrength.MEDIUM


class TransferOverrides(OverridesBase):
    """Transfer inference overrides for a single control hint (all optional)."""

    control_path: ResolvedFilePathOrUrl | None = None
    """Path or URL to pre-computed control input."""

    weight: float | None = None
    """Override the control weight for multi-control weighted attention aggregation."""

    def download(self, output_dir: Path):
        if self.control_path is not None:
            self.control_path = download_file(self.control_path, output_dir, "transfer_control")


class LidarTransferOverrides(TransferOverrides):
    condition_path: ResolvedFilePathOrUrl | None = None
    """Measured LiDAR prefix: a metric 3-channel .pt clip or prepared lidar_rangemap .tar.
    Joint inference only. Must start at the same instant as the camera/control clips.
    """
    num_conditional_sweeps: pydantic.PositiveInt | None = None
    """Number of clean target sweeps to read from condition_path (default: 1)."""

    def download(self, output_dir: Path) -> None:
        super().download(output_dir)
        if self.condition_path is not None:
            self.condition_path = download_file(self.condition_path, output_dir, "lidar_condition")


class EdgeTransferOverrides(TransferOverrides):
    preset_edge_threshold: PresetEdgeThreshold | None = None
    """Edge detection threshold preset."""


class BlurTransferOverrides(TransferOverrides):
    preset_blur_strength: PresetBlurStrength | None = None
    """Blur strength preset."""


class SampleMeta(pydantic.BaseModel):
    model_mode: ModelMode
    vision_mode: VisionMode
    condition_vision_mode: ConditionVisionMode | None


RESOLUTION_ADAPTER = pydantic.TypeAdapter(Resolution)
ASPECT_RATIO_ADAPTER = pydantic.TypeAdapter(AspectRatio)

DEFAULT_CONDITION_FRAME_INDEXES_VISION: dict[ConditionVisionMode, list[int]] = {
    ConditionVisionMode.IMAGE: [0],
    ConditionVisionMode.VIDEO: [0, 1],
}


class TextDataArgs(ArgsBase):
    prompt: str

    negative_prompt: str | None
    per_view_negative_prompt: str | None = None

    duration_template: str | None
    resolution_template: str | None
    negative_metadata_mode: NegativeMetadataMode
    inverse_duration_template: str
    inverse_resolution_template: str
    negative_prompt_keep_metadata: bool


class TextDataOverrides(OverridesBase):
    prompt_path: ResolvedFilePath | None = None
    """Path to a .txt file containing the prompt. Only one of 'prompt' or 'prompt_path' should be provided."""
    prompt: str | None = None
    """Text prompt for generation. Only one of 'prompt' or 'prompt_path' should be provided."""

    negative_prompt: str | None = None
    """Negative prompt - describing what you don't want in the generated video."""
    per_view_negative_prompt: str | None = None
    """Explicit shared negative caption for checkpoints that tokenize each camera caption separately."""

    duration_template: Training[str | None] = None
    """Template string for appending duration/fps to prompt. Use {duration} and {fps} placeholders."""
    resolution_template: Training[str | None] = None
    """Template string for appending resolution to prompt. Use {height} and {width} placeholders."""
    negative_metadata_mode: Training[NegativeMetadataMode | None] = None
    """Negative prompt metadata mode: 'none', 'same', or 'inverse'."""
    inverse_duration_template: Training[str | None] = None
    """Inverse template for duration/fps metadata in the negative prompt."""
    inverse_resolution_template: Training[str | None] = None
    """Inverse template for resolution metadata in the negative prompt."""
    negative_prompt_keep_metadata: Training[bool | None] = None
    """Compatibility flag. If True and mode is 'none', mode is promoted to 'same'."""

    def _build_text_data(self, model_config: "OmniMoTModelConfig", sample_meta: SampleMeta):
        if self.prompt is not None:
            pass
        elif self.prompt_path is not None:
            transfer_self = cast("_TransferDataBase", self)
            if transfer_self.transfer_hints:
                self.prompt = _load_transfer_prompt_path(self.prompt_path)
            else:
                self.prompt = self.prompt_path.read_text().strip()
        else:
            self.prompt = ""

        if sample_meta.model_mode.is_reasoner:
            # Negative-prompt / metadata-template fields are unused by the reasoner
            # but required by OmniSampleArgs validation; fill in inert sentinels.
            if self.negative_metadata_mode is None:
                self.negative_metadata_mode = NegativeMetadataMode.NONE
            if self.inverse_duration_template is None:
                self.inverse_duration_template = ""
            if self.inverse_resolution_template is None:
                self.inverse_resolution_template = ""
            if self.negative_prompt_keep_metadata is None:
                self.negative_prompt_keep_metadata = False
            return

        if self.negative_prompt_keep_metadata and self.negative_metadata_mode == NegativeMetadataMode.NONE:
            self.negative_metadata_mode = NegativeMetadataMode.SAME


Fps = Annotated[int, pydantic.Field(ge=1)]
VideoSaveQuality = Annotated[int, pydantic.Field(ge=0, le=10)]
ImageSaveQuality = Annotated[int, pydantic.Field(ge=0, le=100)]
VideoSaveCrf = Annotated[int, pydantic.Field(ge=0, le=51)]

# The full-quality video is saved as ``vision.mp4`` (qscale ``video_save_quality``). With
# ``save_preview`` a smaller, easy-to-download ``vision_preview.mp4`` is also emitted alongside it:
# same resolution, but H.264 CRF (constant rate factor) encoded, which is far more byte-efficient
# than qscale at a matched perceptual quality.
DEFAULT_SAVE_PREVIEW = False
DEFAULT_PREVIEW_CRF = 28
DEFAULT_PREVIEW_PRESET = "medium"


class _VisionDataBase:
    @property
    def condition_vision_mode(self) -> ConditionVisionMode | None:
        self = cast(VisionDataOverrides, self)
        if self.vision_path is not None:
            vision_ext = Path(self.vision_path).suffix.lower()
            if vision_ext in IMAGE_EXTENSIONS:
                return ConditionVisionMode.IMAGE
            elif vision_ext in VIDEO_EXTENSIONS:
                return ConditionVisionMode.VIDEO
            else:
                raise ValueError(f"Invalid vision extension: {vision_ext}")
        else:
            return None


class VisionDataArgs(ArgsBase, _VisionDataBase):
    vision_path: ResolvedFilePath | None
    condition_frame_indexes_vision: list[int]
    condition_video_keep: Literal["first", "last"]

    resolution: Resolution | None
    aspect_ratio: AspectRatio | None
    fps: pydantic.PositiveInt
    output_fps: pydantic.PositiveInt | None = None
    num_frames: pydantic.PositiveInt
    video_save_quality: VideoSaveQuality
    image_save_quality: ImageSaveQuality
    save_preview: bool
    preview_crf: VideoSaveCrf
    preview_preset: str

    @property
    def duration(self) -> float:
        return self.num_frames / self.fps

    @property
    def video_output_fps(self) -> int:
        """Playback clock only; ``fps`` and ``duration`` retain their model-clock meanings."""
        return self.output_fps if self.output_fps is not None else self.fps

    @property
    def vision_size(self) -> tuple[int, int]:
        """Vision size (width, height) in pixels.

        Per the VisionDataOverrides.aspect_ratio docstring, ``None`` means
        "default to 16:9 for all modes except image_edit". This property is
        only reached by non-image_edit code paths (image_edit branches off in
        cosmos3/inference.py:_get_image_edit_sample_data before this is
        consulted), so the documented legacy default applies here. Callers
        that want native aspect-ratio preservation (e.g. transfer inference)
        autodetect via cosmos_framework.inference.vision.read_and_resize_media before reaching
        this property and never observe the fallback.
        """
        from cosmos_framework.data.generator.utils import IMAGE_RES_SIZE_INFO, VIDEO_RES_SIZE_INFO

        assert self.resolution
        aspect_ratio: AspectRatio = self.aspect_ratio or "16,9"
        if self.num_frames == 1:
            return IMAGE_RES_SIZE_INFO[self.resolution][aspect_ratio]
        else:
            return VIDEO_RES_SIZE_INFO[self.resolution][aspect_ratio]

    @property
    def vision_extension(self) -> str:
        return ".jpg" if self.num_frames == 1 else ".mp4"


class VisionDataOverrides(OverridesBase, _VisionDataBase):
    # Vision condition fields
    vision_path: ResolvedFilePathOrUrl | None = None
    """Path or URL to conditioning image/video."""
    condition_frame_indexes_vision: Training[list[int] | None] = None
    """Latent frame indices to condition on. Defaults to [0] for image, [0, 1] for video."""
    condition_video_keep: Training[Literal["first", "last"] | None] = None
    """Whether to take the first or last ``max_frames`` of the conditioning
    video when it is longer than needed. Defaults to ``"first"``. No effect
    on image conditioning."""

    # Vision fields
    resolution: Resolution | None = None
    """Vision resolution tier, not a pixel dimension. Resolved against
    ``aspect_ratio`` by ``vision_size`` -- ``"256"`` at the default ``16,9``
    yields 320x192. See docs/inference.md for the full table.

    Defaults to model config resolution.

    Accepting a value here does not mean the loaded checkpoint can generate it.
    Per the published model cards:

    * Cosmos3-Edge: 256/480, images and video.
    * Cosmos3-Nano: 256/480/720, images and video.
    * Cosmos3-Super: 256/480/720; 768 additionally for images (the shipped tier
      of the Super-Text2Image variants).

    These ranges are not enforced per checkpoint: a value outside them still
    runs, at a tier the model was never trained on.
    """
    aspect_ratio: AspectRatio | None = None
    """Vision aspect ratio. When None, image_edit preserves the input image's native
    aspect ratio; all other modes default to 16:9."""
    fps: Fps | None = None
    """Model conditioning frames per second. Recommended range [10, 30]; quality may be degraded outside
    of this range. Released Auto camera models require 30 (also their default); use output_fps for playback."""
    output_fps: pydantic.PositiveInt | None = None
    """Saved video playback FPS for Auto RGB-only generation; defaults to fps. Does not resample frames
    or change model conditioning. Joint RGB/LiDAR currently requires output_fps == fps. Unsupported
    for other models and LiDAR-only generation."""
    num_frames: pydantic.PositiveInt | None = None
    """Number of vision frames.

    Generic single-view recommendations by resolution: 256p: [24, 400],
    480p: [24, 300], 720p/768p: [24, 200]. Follow the checkpoint's published
    generation settings. Multiview Auto uses its own sensor-aligned horizons.
    Cosmos3-Edge is documented as 50-150 video frames.
    """
    video_save_quality: VideoSaveQuality | None = None
    """Quality of the saved video (0-10)."""
    image_save_quality: Training[ImageSaveQuality | None] = None
    """Quality of the saved image (0-100)."""
    save_preview: Training[bool | None] = None
    """Whether to also emit a smaller, easy-to-download ``vision_preview.mp4`` alongside the
    full-quality video. The preview is the same resolution but H.264 CRF-encoded (see ``preview_crf``).
    No effect on image (single-frame) outputs. Defaults to False."""
    preview_crf: Training[VideoSaveCrf | None] = None
    """H.264 Constant Rate Factor for ``vision_preview.mp4`` (0-51, lower = higher quality / larger
    file). CRF is far more byte-efficient than the qscale ``video_save_quality`` path. Defaults to 28;
    raise it (e.g. 30-32) for a smaller preview, lower it for higher quality."""
    preview_preset: Training[str | None] = None
    """x264 speed/efficiency preset for ``vision_preview.mp4`` (e.g. 'fast', 'medium', 'slow').
    Slower presets give smaller files at the same CRF. Defaults to 'medium'."""

    @override
    def download(self, output_dir: Path):
        super().download(output_dir)
        self.vision_path = download_file(self.vision_path, output_dir, "vision")

    def _fill_save_compression_defaults(self, *, enable_preview: bool):
        """Populate the preview-video compression knobs (required on ``VisionDataArgs``).

        These have no per-modality JSON defaults, so they are filled here. The full-quality
        ``vision.mp4`` uses ``video_save_quality``; when ``save_preview`` is enabled a smaller
        CRF-encoded ``vision_preview.mp4`` is also emitted (same resolution) for quick download.
        """
        if self.save_preview is None:
            self.save_preview = DEFAULT_SAVE_PREVIEW and enable_preview
        if self.preview_crf is None:
            self.preview_crf = DEFAULT_PREVIEW_CRF
        if self.preview_preset is None:
            self.preview_preset = DEFAULT_PREVIEW_PRESET

    def _build_vision_data(self, model_config: "OmniMoTModelConfig", sample_meta: SampleMeta):
        """Finalize and validate in-place."""
        if self.vision_path and "://" in self.vision_path:
            raise ValueError("Must call `download()` before building vision data")

        # Reasoner mode treats ``vision_path`` as a PIL image source; resolution/fps/num_frames are unused.
        if sample_meta.model_mode.is_reasoner:
            self.condition_frame_indexes_vision = self.condition_frame_indexes_vision or []
            self.condition_video_keep = self.condition_video_keep or "first"
            self.num_frames = self.num_frames or 1
            # Vision-output fields are unused by the reasoner but required by
            # OmniSampleArgs validation; fill in inert sentinels.
            if self.fps is None:
                self.fps = 1
            if self.video_save_quality is None:
                self.video_save_quality = 0
            if self.image_save_quality is None:
                self.image_save_quality = 0
            self._fill_save_compression_defaults(enable_preview=False)
            return

        if self.condition_frame_indexes_vision is None:
            if sample_meta.condition_vision_mode:
                self.condition_frame_indexes_vision = DEFAULT_CONDITION_FRAME_INDEXES_VISION[
                    sample_meta.condition_vision_mode
                ]
            else:
                self.condition_frame_indexes_vision = []

        if self.condition_video_keep is None:
            self.condition_video_keep = "first"

        # Image edit defaults to input image size
        if sample_meta.model_mode != ModelMode.IMAGE2IMAGE:
            if self.resolution is None:
                self.resolution = RESOLUTION_ADAPTER.validate_python(model_config.resolution)

        # Image-output modes always emit a single frame; infer it so callers don't
        # have to set ``num_frames=1`` in every text2image / image2image preset.
        if self.num_frames is None and sample_meta.vision_mode == VisionMode.IMAGE:
            self.num_frames = 1
        assert self.num_frames is not None
        if self.fps is not None and (self.fps < 10 or self.fps > 30):
            log.warning(f"FPS {self.fps} is outside the recommended range [10, 30]. Quality may be degraded.")
        if self.num_frames > 1:
            assert self.resolution is not None
            if self.resolution in IMAGE_ONLY_RESOLUTIONS:
                raise ValueError(
                    f"Resolution {self.resolution!r} only supports image generation (num_frames=1). "
                    f"For video, use one of: {sorted(MAX_NUM_FRAMES)}"
                )
            # Bimanual rollout length follows the supplied action trajectory,
            # not the generic single-view video duration recommendation.
            has_bimanual_contract = getattr(model_config, "_bimanual_action_contract", None) is not None or (
                getattr(model_config, "action_gen", False) is True
                and getattr(model_config, "action_schema", "native") == "unified_v1"
                and getattr(model_config, "action_global_asinh", False) is True
                and getattr(model_config, "action_metric_gripper", False) is True
                and getattr(model_config, "max_action_dim", None) == 64
                and getattr(model_config, "num_embodiment_domains", None) == 32
            )
            is_bimanual_rollout = has_bimanual_contract
            # Auto multiview uses model-specific horizons, including the 297-frame
            # example at 720p and longer causal rollouts. Its sampler validates
            # sensor alignment and input coverage, independently of this generic
            # single-view recommendation.
            is_multiview_rollout = getattr(self, "multiview", None) is not None
            if (
                not is_bimanual_rollout
                and not is_multiview_rollout
                and (self.num_frames < MIN_NUM_FRAMES or self.num_frames > MAX_NUM_FRAMES[self.resolution])
            ):
                log.warning(
                    f"Number of frames {self.num_frames} is outside the recommended range [{MIN_NUM_FRAMES}, {MAX_NUM_FRAMES[self.resolution]}]. Quality may be degraded."
                )
        if SMOKE:
            self.num_frames = min(self.num_frames, 2)
        temporal_compression_factor = model_config.tokenizer.temporal_compression_factor
        self.num_frames = (
            math.ceil((self.num_frames - 1) / temporal_compression_factor) * temporal_compression_factor + 1
        )
        # Preview only applies to multi-frame (video) outputs; it is additionally gated on
        # frame count at save time, so single-frame (image) modes never emit one.
        self._fill_save_compression_defaults(enable_preview=self.num_frames > 1)


class SoundDataArgs(ArgsBase):
    enable_sound: bool = False
    sound_path: ResolvedFilePath | None = None


class SoundDataOverrides(OverridesBase):
    """Sound data overrides."""

    enable_sound: Training[bool | None] = None
    """Enable joint video+sound generation (t2vs mode). Requires a checkpoint with sound modules."""
    sound_path: ResolvedFilePathOrUrl | None = None
    """Path or URL to a conditioning audio clip (e.g. .wav/.mp3/.flac). Required for
    audio_image2video; the clip is encoded by the AVAE and used as a clean condition."""

    @override
    def download(self, output_dir: Path):
        super().download(output_dir)
        self.sound_path = download_file(self.sound_path, output_dir, "sound")

    def _build_sound_data(self, model_config: "OmniMoTModelConfig", sample_meta: SampleMeta):
        if sample_meta.model_mode.is_sound_condition:
            if self.sound_path is None:
                raise ValueError(
                    f"model_mode={sample_meta.model_mode.value} requires a `sound_path` (a conditioning audio clip)"
                )
            self.enable_sound = True
        if self.enable_sound is None:
            self.enable_sound = False
        if self.enable_sound and not model_config.sound_gen:
            raise ValueError(
                "enable_sound=True requires a model with a sound tokenizer "
                "(model.config.sound_gen=True), but the loaded checkpoint has no sound tokenizer"
            )


class ActionDataArgs(ArgsBase):
    action_path: ResolvedFilePath | None = None
    kv_cache_frustum_path: ResolvedFilePath | None = None
    domain_name: str = ""
    image_size: pydantic.PositiveInt = 256
    content_crop_box: ContentCropBox | None = None
    action_chunk_size: pydantic.PositiveInt = 16
    raw_action_dim: int | None = None
    view_point: str | None = None
    format_prompt_as_json: bool | None = None


class InteractiveDataArgs(ArgsBase):
    """Resolved causal / autoregressive inference arguments.

    Houses the fields that only apply to the causal AR pipeline: per-frame
    autoregressive generation and camera-trajectory action conditioning for
    forward-dynamics models.
    """

    autoregressive: bool = False
    save_latents: bool = False
    action_condition: bool = False
    sigma_diffusion_forcing: float | None = None
    camera_trajectory: str | None = None
    camera_translation_scale: float = 1.0
    camera_action_normalization: CameraActionNormalization = "global_asinh"
    camera_num_frames: CameraNumFrames | None = None
    camera_pose_convention: PoseConvention = "backward_chunk_anchored_16f"


class InteractiveDataOverrides(OverridesBase):
    """Causal / autoregressive inference overrides."""

    autoregressive: bool | None = None
    """If True, use autoregressive per-frame generation.
    Defaults to ``model.config.video_temporal_causal``."""
    save_latents: bool | None = None
    """Save causal multiview RGB/joint latents alongside the generated videos. Defaults to False."""
    action_condition: bool | None = None
    """If True, enable action-conditioned autoregressive generation. Defaults to False."""
    sigma_diffusion_forcing: float | None = None
    """Post-shift sigma used to seed clean context frames into the KV cache under
    diffusion_forcing AR inference. Overrides model.config.sigma_diffusion_forcing when set.
    Ignored when causal_training_strategy != 'diffusion_forcing'."""
    camera_trajectory: str | None = None
    """Camera trajectory string parsed by ``parse_camera_string``
    (e.g. 'w-30,right-15,w-15'). Mutually exclusive with ``action_path``;
    only valid for ``model_mode='forward_dynamics'``."""
    camera_translation_scale: float | None = None
    """Translation scale used to normalize generated camera-trajectory actions.
    Set this to the training camera dataset's ``translation_scale``; defaults to 10.0."""
    camera_action_normalization: CameraActionNormalization | None = None
    """How generated camera actions are normalized. ``scale`` applies the pose
    scale normalizer with ``camera_translation_scale``; ``global_asinh`` applies the
    global asinh normalizer and requires ``camera_translation_scale=1.0``.
    Defaults to ``scale``."""
    camera_num_frames: CameraNumFrames | None = None
    """Number of camera poses/actions to consume, independently of tokenizer
    frame-count alignment."""
    camera_pose_convention: PoseConvention | None = None
    """Relative-pose convention used to encode ``camera_trajectory``.
    Defaults to ``backward_framewise``."""

    def _build_interactive_data(self, model_config: "OmniMoTModelConfig", sample_meta: SampleMeta) -> None:
        del sample_meta

        # A causal checkpoint drives autoregressive generation unless the caller
        # said otherwise, which is what lets a distilled Self-Forcing artifact run
        # without repeating --autoregressive on every invocation.
        if self.autoregressive is None:
            self.autoregressive = bool(getattr(model_config, "video_temporal_causal", False))
        self.save_latents = bool(self.save_latents)

        if self.action_condition is None:
            self.action_condition = False
        if self.camera_translation_scale is None:
            self.camera_translation_scale = 1.0
        if self.camera_action_normalization is None:
            self.camera_action_normalization = "global_asinh"
        if self.camera_pose_convention is None:
            self.camera_pose_convention = "backward_chunk_anchored_16f"

        if self.camera_trajectory is None:
            return

        if getattr(self, "action_path", None) is not None:
            raise ValueError("'camera_trajectory' and 'action_path' are mutually exclusive")
        if not self.autoregressive:
            raise ValueError("'camera_trajectory' requires autoregressive inference")


class ActionDataOverrides(OverridesBase):
    """Action data overrides."""

    action_path: ResolvedFilePathOrUrl | None = None
    """Path to action JSON file. Required for forward_dynamics mode."""
    kv_cache_frustum_path: ResolvedFilePathOrUrl | None = None
    """Camera-geometry JSON enabling frustum-IoU KV-history retrieval.

    Use explicit ``c2w`` poses, or ``pose_source=bimanual_unified_v1_chunk16``
    to recover the trajectory from a compatible AgiBot unified-v1 sidecar.
    Omission preserves the checkpoint's ordinary KV-history behavior.
    """
    domain_name: str | None = None
    """Action domain name passed to get_domain_id()."""
    image_size: pydantic.PositiveInt | None = None
    """Spatial preprocessing resolution tier for recorded-action inference.

    Input aspect ratio selects a canvas within this tier. Larger inputs are
    downscaled to fit; smaller inputs are padded without upscaling. Padding is
    removed in latent space before generation, so this is not an output-height
    guarantee. See docs/inference_bimanual.md for the Bimanual example.
    """
    content_crop_box: ContentCropBox | None = None
    """Optional ``(left, top, right, bottom)`` crop selecting real image content.

    Applied before action-inference resizing and reflection padding. Use this
    only when ``vision_path`` contains an outer canvas or transport padding;
    canonical inputs should point directly at the unpadded source image.
    """
    action_chunk_size: pydantic.PositiveInt | None = None
    """Number of action steps to predict."""
    raw_action_dim: pydantic.PositiveInt | None = None
    """Dimension of the raw action data. Required when action_path is not provided."""
    view_point: str | None = None
    """Viewpoint description for the action prompt."""
    format_prompt_as_json: bool | None = None
    """Format legacy action prompts as JSON; unified Bimanual checkpoints require plain text.

    When unset, inference selects the format from the checkpoint's action contract.
    """

    @override
    def download(self, output_dir: Path):
        super().download(output_dir)
        self.action_path = download_file(self.action_path, output_dir, "action")
        self.kv_cache_frustum_path = download_file(self.kv_cache_frustum_path, output_dir, "kv_cache_frustum")

    def _build_action_data(self, model_config: "OmniMoTModelConfig", sample_meta: SampleMeta):
        if self.domain_name is None:
            self.domain_name = ""
        # image_size and action_chunk_size are required non-null on OmniSampleArgs, so
        # they must be defaulted here even for a multiview sample, which never reads
        # them: multiview resolution comes from `vision_size` (resolution/aspect_ratio),
        # and the "image_size" key the multiview batch carries is a height/width tensor,
        # not this field.
        if self.image_size is None:
            self.image_size = 256
        if self.action_chunk_size is None:
            self.action_chunk_size = 16
        # Multiview samples leave view_point unset. Supplying a default here would
        # violate _validate_multiview_options, which rejects that single-view field.
        if self.view_point is None and self.multiview is None:
            self.view_point = "ego_view"

        mode = sample_meta.model_mode
        if not mode.is_action:
            return
        if not model_config.action_gen:
            raise ValueError(
                f"model_mode={mode.value!r} requires a model with action support "
                "(model.config.action_gen=True), but the loaded checkpoint has action_gen=False"
            )
        match mode:
            case ModelMode.FORWARD_DYNAMICS:
                # camera_trajectory derives the action tensor instead of reading one.
                if self.action_path is None and getattr(self, "camera_trajectory", None) is None:
                    raise ValueError(f"'action_path' or 'camera_trajectory' is required for model_mode={mode.value!r}")
            case ModelMode.INVERSE_DYNAMICS | ModelMode.WAM:
                pass
            case _:
                assert_never(mode)

        if self.action_path and "://" in self.action_path:
            raise ValueError("Must call `download()` before building action data")


_MADS_MULTIVIEW_MEDIA_DIRS: frozenset[str] = frozenset({"controls", "rgb"})
"""Per-clip media directories inside an evaluation bundle, as written by copy_mads_mv_val_assets."""
_MADS_MULTIVIEW_MEDIA_RE = re.compile(r"^(?P<sample_id>.+)_c(?P<camera_index>\d+)\.[^.]+$")
_MADS_CLIP_MEDIA_RE = re.compile(r"^(?P<sample_id>.+)\.[^.]+$")


class MultiviewCameraArgs(ArgsBase):
    """One camera entry for AV multiview inference."""

    camera_key: str
    vision_path: ResolvedFilePath | None = None
    control_path: ResolvedFilePath | None = None
    caption_path: ResolvedFilePath | None = None
    """Local caption JSON for this camera, resolved when load_caption_from_data=True."""


class MultiviewCameraOverrides(OverridesBase):
    """One camera entry for AV multiview inference."""

    camera_key: str
    vision_path: ResolvedFilePathOrUrl | None = None
    """Optional per-camera image/video condition. All views must either provide this or omit it."""
    control_path: ResolvedFilePathOrUrl | None = None
    """Optional per-camera precomputed transfer control video. All views must either provide this or omit it."""
    caption_path: ResolvedFilePathOrUrl | None = None
    """Optional caption JSON for this camera. When load_caption_from_data=True and this is omitted,
    it is inferred from this camera's own media, or taken from multiview.caption_path if that is set."""


def _infer_multiview_caption_path(
    *,
    view: MultiviewCameraOverrides,
    caption_root: str | None,
) -> str:
    """Locate one camera's caption JSON from the media that camera reads.

    Per camera, not per sample: every kept view supplies its own text in training, so inferring a
    single path from ``views[0]`` and handing it to all of them conditions ten cameras on the
    eleventh's caption -- which reads as a plausible video, not as an error.
    """
    source_path = view.vision_path or view.control_path
    if source_path is None:
        raise ValueError(
            f"Cannot infer a caption path for camera {view.camera_key!r} because it has neither vision_path "
            "nor control_path. Set its caption_path, or multiview.caption_path, explicitly."
        )

    if caption_root:
        filename = Path(source_path).name
        match = _MADS_MULTIVIEW_MEDIA_RE.match(filename)
        if match is not None:
            return f"{caption_root.rstrip('/')}/{match.group('sample_id')}_c{match.group('camera_index')}.json"
        clip_match = _MADS_CLIP_MEDIA_RE.match(filename)
        if clip_match is not None:
            # Clip-level MADS assets (hdmap/lidar rangemap tars) have no _cN suffix.
            # Front-wide captions are the lidar-transfer training text source.
            return f"{caption_root.rstrip('/')}/{clip_match.group('sample_id')}_c0.json"
        raise ValueError(
            f"Cannot infer the caption path for camera {view.camera_key!r} from "
            f"{source_path!r}; expected a file name ending in '_c<index>.<ext>' "
            "or a clip-level '<sample_id>.<ext>'."
        )

    # Evaluation bundles keep captions beside the media they describe, as
    # 'samples/<clip>/{controls,rgb}/c<N>.mp4' next to 'samples/<clip>/captions/raw/c<N>.json',
    # so the caption is derivable from the view alone and needs no root.
    media_path = Path(source_path)
    if media_path.parent.name in _MADS_MULTIVIEW_MEDIA_DIRS:
        bundle_caption = media_path.parent.parent / "captions" / "raw" / f"{media_path.stem}.json"
        if bundle_caption.is_file():
            return str(bundle_caption)
    raise ValueError(
        f"Cannot infer the caption path for camera {view.camera_key!r} from {source_path!r}: it is not "
        "inside an evaluation bundle. Set its caption_path or multiview.caption_path, or "
        "multiview.caption_root for flat '<clip>_c<index>.json' caption directories."
    )


class _MultiviewBase:
    @property
    def camera_keys(self) -> list[str]:
        self = cast("MultiviewArgs | MultiviewOverrides", self)
        return [view.camera_key for view in self.views]

    @property
    def has_condition_vision(self) -> bool:
        self = cast("MultiviewArgs | MultiviewOverrides", self)
        return any(view.vision_path is not None for view in self.views)

    @property
    def has_control(self) -> bool:
        self = cast("MultiviewArgs | MultiviewOverrides", self)
        return any(view.control_path is not None for view in self.views)

    @property
    def condition_vision_mode(self) -> ConditionVisionMode | None:
        self = cast("MultiviewArgs | MultiviewOverrides", self)
        paths = [view.vision_path for view in self.views if view.vision_path is not None]
        if not paths:
            return None
        suffixes = {Path(path).suffix.lower() for path in paths}
        if all(suffix in IMAGE_EXTENSIONS for suffix in suffixes):
            return ConditionVisionMode.IMAGE
        if all(suffix in VIDEO_EXTENSIONS for suffix in suffixes):
            if getattr(self, "condition_video_as_image", False):
                return ConditionVisionMode.IMAGE
            return ConditionVisionMode.VIDEO
        raise ValueError(
            f"Multiview condition paths must be all images or all videos, got suffixes {sorted(suffixes)}."
        )


class MultiviewArgs(ArgsBase, _MultiviewBase):
    views: list[MultiviewCameraArgs]
    """Ordered camera entries. This order is used for camera-major tensor packing and output saving."""
    condition_video_as_image: bool = False
    """If True, treat per-view video vision_path files as single-frame image conditions."""
    load_caption_from_data: bool = False
    """If True, read structured caption chunks from caption_path and run chunkwise multiview generation."""
    caption_path: ResolvedFilePath | None = None
    """Local caption JSON path used when load_caption_from_data=True."""
    caption_root: str | None = None
    """Root used to infer caption_path before downloads; retained for provenance only."""
    use_first_chunk_only: bool = True
    """If True, generate only from caption chunk zero instead of stitching all chunks."""
    first_caption_paragraph_only: bool = False
    """Trim each loaded caption before adding camera headers, matching short-window LiDAR training."""


class MultiviewOverrides(OverridesBase, _MultiviewBase):
    views: list[MultiviewCameraOverrides] = pydantic.Field(min_length=1)
    """Ordered camera entries. This order is used for camera-major tensor packing and output saving."""
    condition_video_as_image: bool = False
    """If True, treat per-view video vision_path files as single-frame image conditions."""
    load_caption_from_data: bool = False
    """If True, read structured caption chunks from caption_path and run chunkwise multiview generation."""
    caption_path: ResolvedFilePathOrUrl | None = None
    """Optional shared caption JSON path. Prefer caption_path on each view for camera-local text."""
    caption_root: str | None = None
    """Root for flat '<clip>_c<index>.json' captions. Without a root or explicit caption_path,
    infer 'captions/raw/<media-stem>.json' beside a view's 'rgb/' or 'controls/' directory."""
    use_first_chunk_only: bool = True
    """If True, generate only from caption chunk zero instead of stitching all chunks."""
    first_caption_paragraph_only: bool = False
    """Trim each loaded caption to its opening paragraph before adding camera headers.

    Matches training recipes that set ``first_caption_paragraph_only`` (Transfer-Auto stage 3 and
    LiDAR transfer), so inference conditions on the same caption text the checkpoint was trained on.
    Only applies when ``load_caption_from_data=True``."""

    @pydantic.model_validator(mode="after")
    def _validate_views(self) -> Self:
        if not self.views:
            raise ValueError("multiview.views must contain at least one camera.")
        camera_keys = [view.camera_key for view in self.views]
        if len(set(camera_keys)) != len(camera_keys):
            raise ValueError(f"multiview.views camera_key values must be unique, got {camera_keys}.")
        has_condition = [view.vision_path is not None for view in self.views]
        if any(has_condition) and not all(has_condition):
            raise ValueError("All multiview.views entries must provide vision_path, or none of them should.")
        has_control = [view.control_path is not None for view in self.views]
        if any(has_control) and not all(has_control):
            raise ValueError("All multiview.views entries must provide control_path, or none of them should.")
        _ = self.condition_vision_mode
        return self

    @override
    def download(self, output_dir: Path) -> None:
        super().download(output_dir)
        if self.load_caption_from_data:
            # Each camera reads its own caption file, as in training, where every kept view
            # supplies its own text. A sample-level caption_path overrides that with one shared
            # source, which is what lidar transfer wants: front-wide text drives every view.
            for view in self.views:
                if view.caption_path is None:
                    view.caption_path = self.caption_path or _infer_multiview_caption_path(
                        view=view,
                        caption_root=self.caption_root,
                    )
            if self.caption_path is not None:
                self.caption_path = download_file(self.caption_path, output_dir, "multiview_caption")
        for view_index, view in enumerate(self.views):
            safe_camera_key = safe_multiview_camera_name(view.camera_key, view_index)
            view.vision_path = download_file(view.vision_path, output_dir, f"multiview_{safe_camera_key}_vision")
            view.control_path = download_file(view.control_path, output_dir, f"multiview_{safe_camera_key}_control")
            view.caption_path = download_file(view.caption_path, output_dir, f"multiview_{safe_camera_key}_caption")


class MultiviewDataArgs(ArgsBase):
    multiview: MultiviewArgs | None = None


class MultiviewDataOverrides(OverridesBase):
    multiview: Training[MultiviewOverrides | None] = None
    """Optional AV multiview inference payload. When set, inference builds camera-major multiview batches."""

    @override
    def download(self, output_dir: Path) -> None:
        super().download(output_dir)
        if self.multiview is not None:
            self.multiview.download(output_dir)

    def _build_multiview_data(self, *, sample_meta: SampleMeta, model_config: "OmniMoTModelConfig") -> None:
        self = cast("SampleDataOverrides", self)
        if self.multiview is None:
            return
        self._validate_multiview_options(sample_meta=sample_meta, model_config=model_config)

    def _validate_multiview_options(self, *, sample_meta: SampleMeta, model_config: "OmniMoTModelConfig") -> None:
        self = cast("SampleDataOverrides", self)
        assert self.multiview is not None
        errors: list[str] = []
        # A causal multiview checkpoint samples its multiview inputs autoregressively only (build_sample
        # turns AR on for them unless the user set it). Which inputs it takes, per-camera WSM transfer
        # and joint camera+LiDAR, is for check_causal_multiview_options to say, so a sample without
        # controls hears what it lacks rather than that AR is unsupported. On any other checkpoint,
        # multiview runs the one-shot sampler.
        causal_multiview = is_causal_multiview_config(model_config)

        if sample_meta.model_mode.is_action or sample_meta.model_mode.is_reasoner:
            errors.append("action and reasoner model modes")
        if sample_meta.vision_mode != VisionMode.VIDEO:
            errors.append("image output modes; set num_frames > 1")
        if self.vision_path is not None:
            errors.append("top-level vision_path; use multiview.views[*].vision_path")
        if self.enable_sound:
            errors.append("enable_sound=True")
        # Native upsampling would rewrite the shared caption for every view.
        # Reject it consistently instead of allowing only text-to-video samples
        # to reach that path while image-conditioned samples fail later.
        if self.prompt_upsampling:
            errors.append("prompt_upsampling=True")
        if self.autoregressive and not causal_multiview:
            errors.append("autoregressive=True")
        if self.action_condition:
            errors.append("action_condition=True")
        if self.action_path is not None:
            errors.append("action_path")
        if self.raw_action_dim is not None:
            errors.append("raw_action_dim")
        if self.domain_name not in (None, ""):
            errors.append("domain_name")
        if self.view_point is not None:
            errors.append("view_point")
        if getattr(self, "camera_trajectory", None) is not None:
            errors.append("camera_trajectory")
        if self.multiview.load_caption_from_data:
            # Each camera reads its own caption file, so the sample-level caption_path being unset
            # says nothing: download() leaves it None whenever the paths were inferred per camera
            # or came from an evaluation bundle. What a run cannot do without is a path per camera.
            uncaptioned = [view.camera_key for view in self.multiview.views if view.caption_path is None]
            if uncaptioned:
                errors.append(
                    f"load_caption_from_data=True but camera(s) {uncaptioned} have no caption_path; call "
                    "download(), which fills each camera's caption_path from multiview.caption_path, "
                    "multiview.caption_root or the evaluation bundle beside its media, or set them"
                )

        expected_input_mode = {
            None: "text",
            ConditionVisionMode.IMAGE: "image",
            ConditionVisionMode.VIDEO: "video",
        }[self.multiview.condition_vision_mode]
        actual_input_mode = sample_meta.model_mode.value.split("2", maxsplit=1)[0]
        if actual_input_mode != expected_input_mode:
            errors.append(
                "model_mode is inconsistent with multiview conditioning: "
                f"model_mode={sample_meta.model_mode.value!r} expects {actual_input_mode!r} input, "
                f"but multiview.views imply {expected_input_mode!r} input"
            )

        # Joint camera+LiDAR legitimately carries two hints (wsm + lidar) alongside per-view
        # control paths; _validate_lidar_transfer checks that combination instead.
        if self.is_joint_cam_lidar_transfer:
            pass
        elif not self.multiview.has_control and self.transfer_hints:
            errors.append("transfer hints without multiview.views[*].control_path")
        elif self.multiview.has_control:
            if not self.transfer_hints:
                errors.append("per-view control_path without an explicit transfer hint such as wsm={}")
            if len(self.transfer_hints) > 1:
                errors.append("more than one transfer hint")
            for hint_key, config in self.transfer_hints.items():
                if config.control_path is not None:
                    errors.append(
                        "Multiview transfer inference uses per-view control_path; "
                        f"do not also set {hint_key.value}.control_path"
                    )

        if errors:
            raise ValueError("Multiview inference does not support the following option(s):\n- " + "\n- ".join(errors))


_ReasonerTemperature = Annotated[float, pydantic.Field(gt=0, le=100)]
_ReasonerTopP = Annotated[float, pydantic.Field(gt=0, le=1)]
_ReasonerRepetitionPenalty = Annotated[float, pydantic.Field(gt=0)]


def _mapping_get(node: Any, key: str) -> Any:
    """``.get`` that tolerates None and non-mapping nodes (dict / DictConfig / LazyDict)."""
    getter = getattr(node, "get", None)
    return getter(key) if callable(getter) else None


def _reasoner_vision_capable(model_config: "OmniMoTModelConfig") -> bool:
    """Whether the checkpoint's reasoner LM can encode image/video prompts.

    Qwen-family reasoners carry the ViT inside the checkpoint, gated by
    ``include_visual``; a falsy value (e.g. an old ``--no-vit`` export, or a
    text-only LLM backbone) means the tower was never built, so a vision
    prompt would only fail mid-batch at the ``hasattr(causal_lm, "visual")``
    gate. Edge's Nemotron reasoner sets ``include_visual=None`` by design and
    loads its SigLIP2 tower lazily (``_ensure_vision_tower``), so it is always
    vision-capable. Unknown families are treated as capable (never block).
    """
    vlm_config = model_config.vlm_config
    model_instance = getattr(vlm_config, "model_instance", None)
    target = str(_mapping_get(model_instance, "_target_") or "")
    model_name = getattr(vlm_config, "model_name", "") or ""
    if "Nemotron3DenseVLTextForCausalLM" in target or "Cosmos3-Edge" in model_name:
        return True
    if "Qwen" not in target and not model_name.startswith("Qwen/"):
        return True
    include_visual = _mapping_get(_mapping_get(model_instance, "config"), "include_visual")
    return bool(include_visual)


class ReasonerDataArgs(ArgsBase):
    """Resolved reasoner (VLM) text-generation arguments. All fields are ``| None`` so
    non-reasoner samples (which never populate these) pass ``OmniSampleArgs`` validation;
    runtime values for reasoner mode come from ``defaults/reasoner/sample_args.json``."""

    max_new_tokens: pydantic.PositiveInt | None = None
    do_sample: bool | None = None
    temperature: _ReasonerTemperature | None = None
    top_k: pydantic.PositiveInt | None = None
    top_p: _ReasonerTopP | None = None
    repetition_penalty: _ReasonerRepetitionPenalty | None = None
    presence_penalty: float | None = None
    video_fps: pydantic.PositiveFloat | None = None


class ReasonerDataOverrides(OverridesBase):
    """Reasoner overrides for ``model_mode='reasoner'``. ``vision_path`` (if set) is
    used as the conditioning image or video; the VLM processor handles preprocessing."""

    max_new_tokens: pydantic.PositiveInt | None = None
    """Maximum number of new tokens to generate per prompt."""
    do_sample: bool | None = None
    """If True, sample from the logits; otherwise greedy decode."""
    temperature: _ReasonerTemperature | None = None
    """Sampling temperature. Ignored when ``do_sample`` is False."""
    top_k: pydantic.PositiveInt | None = None
    """Top-k logit truncation. Ignored when ``do_sample`` is False."""
    top_p: _ReasonerTopP | None = None
    """Nucleus-sampling threshold ``0 < top_p <= 1``. Ignored when ``do_sample`` is False."""
    repetition_penalty: _ReasonerRepetitionPenalty | None = None
    """CTRL/HF-style multiplicative repetition penalty (>0). ``1.0`` is identity."""
    presence_penalty: float | None = None
    """Additive presence penalty (any sign). ``0.0`` is identity."""
    video_fps: pydantic.PositiveFloat | None = None
    """Frames per second to sample from a video vision_path. None -> decoder default (2.0)."""

    def _build_reasoner_data(self, model_config: "OmniMoTModelConfig", sample_meta: SampleMeta):
        if not sample_meta.model_mode.is_reasoner:
            return
        self = cast("SampleDataOverrides", self)
        if not self.prompt.strip():
            raise ValueError("Reasoner inference requires a non-empty 'prompt'.")
        if self.vision_path is not None and not _reasoner_vision_capable(model_config):
            raise ValueError(
                f"Reasoner sample {getattr(self, 'name', None)!r} conditions on vision input "
                f"'{self.vision_path}', but the loaded checkpoint's reasoner LM has no visual tower "
                "(model config include_visual is false — e.g. a --no-vit export or a text-only LLM "
                "backbone). Use a text-only prompt, or re-export the checkpoint with the default "
                "--vit / use a full checkpoint for image/video reasoner prompts."
            )


class _TransferDataBase:
    @property
    def transfer_hints(self) -> dict[TransferHintKey, TransferOverrides | TransferArgs]:
        # Iteration order is `TransferHintKey` enum order, not JSON-key order — keep this
        # deterministic so the model sees a stable [ctrl_1, ..., ctrl_N] sequence.
        return {key: getattr(self, key.value) for key in TransferHintKey if getattr(self, key.value) is not None}

    @property
    def is_multiview_transfer(self) -> bool:
        """True for a single-hint transfer driven by per-view controls."""
        multiview = getattr(self, "multiview", None)
        return multiview is not None and multiview.has_control and len(self.transfer_hints) == 1

    @property
    def is_joint_cam_lidar_transfer(self) -> bool:
        """True when WSM, LiDAR and camera views are all present.

        Joint generation denoises the camera and LiDAR streams together, so it takes a different
        path through validation, sampling defaults and batch assembly than either modality alone.
        """
        return (
            TransferHintKey.LIDAR in self.transfer_hints
            and TransferHintKey.WSM in self.transfer_hints
            and getattr(self, "multiview", None) is not None
        )


class TransferDataArgs(ArgsBase, _TransferDataBase):
    control_guidance: ControlGuidance = 1.0
    control_guidance_interval: GuidanceInterval | None = None
    edge: EdgeTransferArgs | None = None
    blur: BlurTransferArgs | None = None
    depth: TransferArgs | None = None
    seg: TransferArgs | None = None
    wsm: TransferArgs | None = None
    lidar: LidarTransferArgs | None = None
    negative_prompt_file: str | None = None
    """JSON negative caption file for transfer specs (absolute path or filename under ``defaults/``)."""
    num_video_frames_per_chunk: pydantic.PositiveInt | None = None
    num_conditional_frames: pydantic.NonNegativeInt | None = None
    max_frames: pydantic.PositiveInt | None = None
    show_control_condition: bool | None = None
    show_input: bool | None = None
    num_first_chunk_conditional_frames: pydantic.NonNegativeInt | None = None
    share_vision_temporal_positions: bool | None = None
    emphasize_control_in_prompt: bool | None = None
    transfer_resize_mode: Literal["center_crop", "direct_resize"] | None = None
    transfer_output_size: Literal["model", "original"] | None = None


class TransferDataOverrides(OverridesBase, _TransferDataBase):
    """Transfer inference overrides — activated when at least one control hint is set."""

    control_guidance: Training[ControlGuidance | None] = None
    """Control-CFG scale for transfer. The control map stays in the main forward; 1.0 disables
    the extra comparison forward that drops control-map vision items. Values > 1.0 blend
    velocities from with-maps vs without-maps forwards on the generated clip."""
    control_guidance_interval: Training[GuidanceInterval | None] = None
    """Timestep interval [lo, hi] (0–1000) in which control-CFG is applied; None applies
    on every step."""
    edge: EdgeTransferOverrides | None = None
    """Edge (Canny) control-hint overrides; set to activate the edge transfer hint."""
    blur: BlurTransferOverrides | None = None
    """Blur control-hint overrides; set to activate the blur transfer hint."""
    depth: TransferOverrides | None = None
    """Depth control-hint overrides; set to activate the depth transfer hint."""
    seg: TransferOverrides | None = None
    """Segmentation control-hint overrides; set to activate the segmentation transfer hint."""
    wsm: TransferOverrides | None = None
    """World-surface-map (WSM) control-hint overrides; set to activate the WSM transfer hint."""
    lidar: LidarTransferOverrides | None = None
    """LiDAR rangemap control (``.pt``, prepared ``.tar``, or rangeview directory).

    LiDAR-only: set ``lidar.control_path`` and leave ``wsm``, ``multiview`` and ``vision_path``
    unset. Joint camera+LiDAR: pair it with ``wsm: {}`` and one or more multiview views, each
    carrying its own ``control_path``. ``lidar.condition_path`` and ``lidar.num_conditional_sweeps``
    supply measured target sweeps as a clean conditioning prefix."""
    negative_prompt_file: str | None = None
    """JSON negative caption file for transfer specs (absolute path or filename under ``defaults/``)."""
    num_video_frames_per_chunk: pydantic.PositiveInt | None = None
    """Number of video frames generated per autoregressive chunk."""
    num_conditional_frames: pydantic.NonNegativeInt | None = None
    """Number of conditioning frames carried into each generated chunk."""
    max_frames: pydantic.PositiveInt | None = None
    """Maximum number of frames to generate across all chunks."""
    show_control_condition: bool | None = None
    """If set, include the control condition (hint map) in the saved output."""
    show_input: bool | None = None
    """If set, include the input video alongside the generated output."""
    num_first_chunk_conditional_frames: pydantic.NonNegativeInt | None = None
    """Number of conditioning frames for the first chunk (defaults to ``num_conditional_frames``)."""
    share_vision_temporal_positions: bool | None = None
    """Share vision temporal position ids across autoregressive chunks."""
    emphasize_control_in_prompt: bool | None = None
    """If True (default), auto-append a one-sentence directive to the user prompt that
    names the active control modality (e.g. "Follow the edge control video precisely.
    ..."). Set False for clean baselines / ablations. The system prompt is unchanged."""
    transfer_resize_mode: Literal["center_crop", "direct_resize"] | None = None
    """Spatial preprocessing for bidirectional, Stage 1 TF, Stage 2 TF-dCM, or Stage 3 SF-DMD Transfer.
    The default center_crop preserves legacy checkpoints. direct_resize matches the teacher
    fine-tuned without spatial padding: full-frame RGB bicubic, depth bilinear,
    segmentation nearest, with at most 3% aspect distortion at 480 or 720."""
    transfer_output_size: Literal["model", "original"] | None = None
    """Save generated RGB at the model bucket size (default) or restore the original
    source dimensions. original requires direct_resize; the model-size RGB is
    retained separately for inspection; evaluation uses the selected output size.
    MP4 encoding adds one bottom/right pixel
    when the corresponding source dimension is odd; output metadata records it."""

    @pydantic.model_validator(mode="after")
    def _validate_transfer_hints(self) -> Self:
        hint_field_names = {k.value for k in TransferHintKey}
        # ``control_guidance`` is concretized to its no-op default (1.0) by
        # ``_build_transfer_data`` for *every* sample, transfer or not, so the
        # default value must round-trip without a hint. Only an *explicit*
        # non-default ``control_guidance`` signals an intended-but-incomplete
        # transfer config, so it is checked separately below.
        exempt = hint_field_names | {"control_guidance"}
        transfer_only = [
            name
            for name in TransferDataOverrides.__annotations__
            if name in type(self).model_fields and name not in exempt
        ]
        configured = any(getattr(self, f) is not None for f in transfer_only)
        cg = self.control_guidance
        cg_configured = cg is not None and cg != 1.0
        if (configured or cg_configured) and not self.transfer_hints:
            raise ValueError(
                f"transfer inference requires at least one control hint ({', '.join(k.value for k in TransferHintKey)})"
            )
        return self

    @override
    def download(self, output_dir: Path):
        super().download(output_dir)
        for config in self.transfer_hints.values():
            assert isinstance(config, TransferOverrides)
            config.download(output_dir)

    _TRANSFER_SAMPLE_DEFAULTS: ClassVar[dict[str, Any]] = {
        "num_video_frames_per_chunk": 93,
        "num_conditional_frames": 1,
        "max_frames": 5000,
        "show_control_condition": False,
        "show_input": False,
        "num_first_chunk_conditional_frames": 0,
        "share_vision_temporal_positions": True,
        "emphasize_control_in_prompt": True,
        "transfer_resize_mode": "center_crop",
        "transfer_output_size": "model",
    }
    _SF_DMD_TRANSFER_MAX_FRAMES: ClassVar[dict[str, int]] = {"480": 201, "720": 81}
    _SF_DMD_TRANSFER_RECTIFIED_FLOW_SHIFT: ClassVar[dict[str, float]] = {"256": 3.0, "480": 5.0, "720": 10.0}
    _TRANSFER_HINT_DEFAULTS: ClassVar[dict[TransferHintKey, dict[str, Any]]] = {
        TransferHintKey.EDGE: {"preset_edge_threshold": PresetEdgeThreshold.MEDIUM},
        TransferHintKey.BLUR: {"preset_blur_strength": PresetBlurStrength.MEDIUM},
        # The final _build() dumps nested overrides with None intact, so the args
        # field default alone cannot resolve an omitted conditional-sweep count.
        TransferHintKey.LIDAR: {"num_conditional_sweeps": 1},
    }
    # Tuned guidance / control_guidance per transfer task. Applied when the input JSON omits
    # these fields (generic video2video sampling defaults otherwise apply).
    _TRANSFER_DEFAULTS: ClassVar[dict[TransferHintKey, dict[str, Any]]] = {
        TransferHintKey.EDGE: {"guidance": 3.0, "control_guidance": 1.5, "shift": 10.0},
        TransferHintKey.BLUR: {"guidance": 3.0, "control_guidance": 1.5, "shift": 10.0},
        TransferHintKey.DEPTH: {"guidance": 3.0, "control_guidance": 1.5, "shift": 10.0},
        TransferHintKey.SEG: {"guidance": 3.0, "control_guidance": 2.0, "shift": 10.0},
        TransferHintKey.WSM: {
            "guidance": 1.0,
            "control_guidance": 3.0,
            "shift": 10.0,
            "num_frames": 101,
            "fps": 10,
            "num_video_frames_per_chunk": 101,
        },
        # LiDAR-only keeps its 10 Hz / 96-sweep training recipe rather than inheriting
        # video2video's frame rate.
        TransferHintKey.LIDAR: {
            "num_frames": 96,
            "fps": 10,
            "num_video_frames_per_chunk": 96,
            "guidance": 3.0,
            "control_guidance": 1.0,
        },
    }
    # Joint WSM + LiDAR runs on the camera clock instead: ``num_frames`` stays the camera count
    # (multiview default 93) and the sweep count is derived from it as ``round(num_frames * 10/30)``.
    # Training is 11-view 480p over 200 camera frames / 67 sweeps.
    _JOINT_CAM_LIDAR_DEFAULTS: ClassVar[dict[str, Any]] = {
        "fps": 30,
        "resolution": "480",
        "control_guidance": 1.0,
    }

    def _build_transfer_data(
        self,
        model_config: "OmniMoTModelConfig",
        sample_meta: SampleMeta,
        *,
        user_fields: frozenset[str] | None = None,
    ):
        self = cast("SampleDataOverrides", self)
        # ``control_guidance`` is a required float in ``TransferDataArgs``.
        # Keep it concrete even for non-transfer samples so ``OmniSampleArgs``
        # validation never sees ``None``.
        if self.control_guidance is None:
            self.control_guidance = 1.0
        hints = self.transfer_hints
        if not hints:
            return

        if self.negative_prompt is None and self.negative_prompt_file is not None:
            self.negative_prompt = _load_transfer_negative_prompt_file(self.negative_prompt_file)

        # A multiview caption run takes its chunk length from the span each caption describes, so
        # the generic 93-frame transfer chunk must not stand in for it: a 300-frame caption would
        # generate less than a third of what its text covers. Left None for inference to derive,
        # which is unambiguous because a value the sample set is non-None here.
        skipped_defaults = (
            {"num_video_frames_per_chunk"}
            if self.multiview is not None and self.multiview.load_caption_from_data
            else frozenset()
        )
        for field, default in self._TRANSFER_SAMPLE_DEFAULTS.items():
            if field not in skipped_defaults and getattr(self, field) is None:
                setattr(self, field, default)

        if self.transfer_output_size == "original" and self.transfer_resize_mode != "direct_resize":
            raise ValueError("transfer_output_size='original' requires transfer_resize_mode='direct_resize'")
        if self.transfer_resize_mode == "direct_resize":
            video_temporal_causal = bool(getattr(model_config, "video_temporal_causal", False))
            if self.autoregressive or video_temporal_causal:
                strategy = getattr(model_config, "causal_training_strategy", None)
                fixed_step_sampler = getattr(model_config, "fixed_step_sampler_config", None)
                is_self_forcing = getattr(model_config, "simulation_mode", None) == "self_forcing"
                # Self-forcing checkpoints use their distilled sampler even if
                # their config retains an inherited teacher-forcing strategy.
                if is_self_forcing:
                    supported_causal_stage = fixed_step_sampler is not None
                else:
                    # Stage-3 student exports omit training-only simulation_mode;
                    # they retain strategy='none' and the distilled fixed sampler.
                    supported_causal_stage = (strategy == "teacher_forcing" and fixed_step_sampler is None) or (
                        strategy in {"teacher_forcing_dcm", "none"} and fixed_step_sampler is not None
                    )
                if not (
                    self.autoregressive
                    and video_temporal_causal
                    and supported_causal_stage
                    and getattr(model_config, "teacher_forcing_frames_per_chunk", None) == 1
                ):
                    raise ValueError(
                        "Direct Transfer resizing requires a bidirectional teacher or an autoregressive "
                        "Stage 1 teacher_forcing checkpoint without a fixed sampler or Stage 2 teacher_forcing_dcm "
                        "checkpoint with a fixed sampler or Stage 3 SF-DMD checkpoint with a fixed sampler, "
                        "with one latent frame per chunk"
                    )
                if (
                    len(hints) != 1
                    or self.num_first_chunk_conditional_frames != 0
                    or getattr(self, "control_guidance", None) != 1.0
                    or not self.share_vision_temporal_positions
                ):
                    raise ValueError(
                        "Direct causal Transfer requires exactly one control hint, "
                        "num_first_chunk_conditional_frames=0, control_guidance=1.0, "
                        "and share_vision_temporal_positions=true"
                    )
            if self.multiview is not None or any(
                key not in {TransferHintKey.EDGE, TransferHintKey.BLUR, TransferHintKey.DEPTH, TransferHintKey.SEG}
                for key in hints
            ):
                raise ValueError("Direct Transfer resizing supports single-view edge, blur, depth, and seg only")
            if self.resolution not in {"480", "720"}:
                raise ValueError("Direct Transfer resizing supports the teacher's trained 480 and 720 tiers only")

        for hint_key, config in hints.items():
            for field, default in self._TRANSFER_HINT_DEFAULTS.get(hint_key, {}).items():
                if getattr(config, field) is None:
                    setattr(config, field, default)

            # A multiview transfer sample carries its control videos per view, in
            # multiview.views[].control_path, not at the top level -- so neither
            # vision_path nor the hint's control_path is set for it.
            multiview_has_control = self.multiview is not None and self.multiview.has_control
            if self.vision_path is None and config.control_path is None and not multiview_has_control:
                raise ValueError(
                    f"transfer inference requires 'vision_path' or a pre-computed 'control_path' (hint: {hint_key})"
                )

        # Multiview WSM uses the multiview preset and base field defaults. Applying
        # single-view WSM defaults would change num_frames from 93 to 101 and
        # control_guidance from 1.0 to 3.0, enabling an additional control-CFG pass.
        # Keep the single-view defaults scoped to single-view samples.
        if self.is_joint_cam_lidar_transfer:
            for field, value in self._JOINT_CAM_LIDAR_DEFAULTS.items():
                if user_fields is None or field not in user_fields:
                    setattr(self, field, value)
            return
        if len(hints) == 1 and self.multiview is None and not self.autoregressive:
            hint_key = next(iter(hints))
            for field, value in self._TRANSFER_DEFAULTS[hint_key].items():
                if user_fields is None or field not in user_fields:
                    setattr(self, field, value)

    def _apply_sf_dmd_transfer_inference_contract(
        self,
        *,
        model_config: "OmniMoTModelConfig",
        user_fields: frozenset[str],
    ) -> None:
        """Apply the resolution contract recorded by the SF-DMD Transfer checkpoint."""
        self = cast("SampleDataOverrides", self)
        video_temporal_causal = bool(getattr(model_config, "video_temporal_causal", False))
        is_self_forcing = getattr(model_config, "simulation_mode", None) == "self_forcing"
        is_distilled = getattr(model_config, "fixed_step_sampler_config", None) is not None
        if not video_temporal_causal or not (is_self_forcing or is_distilled):
            return
        if int(getattr(model_config, "teacher_forcing_frames_per_chunk", 1) or 1) != 1:
            return
        replay_policy = getattr(model_config, "teacher_forcing_replay_policy", None)
        if (
            getattr(replay_policy, "control_visibility", None) != "causal"
            or not getattr(replay_policy, "controls_read_strict_past_clean_rgb", False)
            or getattr(replay_policy, "clean_pass_causality", None) != "frame"
        ):
            return

        resolution = self.resolution
        # The selected 480-only checkpoint must not inherit the legacy 720 preset.
        # Earlier mixed-resolution students record resolution=720 and retain both tiers.
        if str(getattr(model_config, "resolution", "")) == "480" and resolution != "480":
            raise ValueError("This SF-DMD Transfer checkpoint supports only resolution='480'.")
        if not self.transfer_hints or not self.autoregressive:
            return
        frame_cap = self._SF_DMD_TRANSFER_MAX_FRAMES.get(resolution)
        if frame_cap is not None and "max_frames" not in user_fields:
            self.max_frames = min(self.max_frames or frame_cap, frame_cap)

        if "shift" not in user_fields:
            resolution_shift = self._SF_DMD_TRANSFER_RECTIFIED_FLOW_SHIFT.get(resolution)
            if resolution_shift is not None:
                self.shift = resolution_shift


class _SampleDataBase:
    @property
    def resolved_model_mode(self) -> ModelMode:
        """Return ``model_mode`` if set, else infer the VFM modality from ``vision_path`` and ``num_frames``."""
        self = cast(SampleDataOverrides, self)
        if self.model_mode is not None:
            return self.model_mode
        input_mode = self.condition_vision_mode.value if self.condition_vision_mode else "text"
        output_mode = VisionMode.IMAGE.value if self.num_frames == 1 else VisionMode.VIDEO.value
        return ModelMode(f"{input_mode}2{output_mode}")

    @property
    def sample_meta(self) -> SampleMeta:
        self = cast(SampleDataOverrides, self)
        mode = self.resolved_model_mode
        condition_vision_mode = self.condition_vision_mode
        # Multiview carries its own conditioning mode. Its per-view first frames live in
        # multiview.views[].vision_path, not in the top-level vision_path that the
        # single-view path reads, so the top-level mode stays None for a multiview i2v
        # sample. Leaving it None makes _get_multiview_condition_pixel_frames return
        # [None, ...] and an image2video sample then generates text2video output --
        # byte-identical to the t2v run, with no error anywhere. Measured: parity job
        # 2041279, all 11 views identical to the t2v run.
        if self.multiview is not None:
            condition_vision_mode = self.multiview.condition_vision_mode
        return SampleMeta(
            model_mode=mode,
            vision_mode=VisionMode.from_model_mode(mode),
            condition_vision_mode=condition_vision_mode,
        )


class SampleDataArgs(
    _SampleDataBase,
    TextDataArgs,
    VisionDataArgs,
    SoundDataArgs,
    ActionDataArgs,
    InteractiveDataArgs,
    ReasonerDataArgs,
    TransferDataArgs,
    MultiviewDataArgs,
):
    model_mode: ModelMode


class SampleDataOverrides(
    _SampleDataBase,
    TextDataOverrides,
    VisionDataOverrides,
    SoundDataOverrides,
    ActionDataOverrides,
    InteractiveDataOverrides,
    ReasonerDataOverrides,
    TransferDataOverrides,
    MultiviewDataOverrides,
):
    """Sample data arguments for 'OmniMoTModel.generate_samples'."""

    model_mode: ModelMode | None = None
    """Generation modality. When omitted, the VFM modality is inferred from ``vision_path`` and
    ``num_frames``; action modes must be set explicitly."""


class PromptUpsamplingArgs(ArgsBase):
    native_prompt_upsampling: bool = False
    """If True, use the native prompt upsampler."""
    prompt_upsampler_max_tokens: pydantic.PositiveInt
    """Maximum tokens generated by the prompt upsampler."""
    prompt_upsampler_temperature: pydantic.NonNegativeFloat
    """Native prompt upsampler sampling temperature."""
    prompt_upsampler_top_p: PromptUpsamplerProbability
    """Native prompt upsampler nucleus sampling probability."""
    prompt_upsampler_top_k: pydantic.NonNegativeInt
    """Native prompt upsampler top-k sampling limit."""
    prompt_upsampler_repetition_penalty: pydantic.PositiveFloat
    """Native prompt upsampler CTRL/HF-style multiplicative repetition penalty.

    Applied to logits at vocab positions already seen in each caption's
    history (prompt + everything generated so far).  ``>1.0`` discourages
    verbatim repetition, ``<1.0`` encourages it, ``1.0`` is identity and
    adds zero overhead to the reasoner AR loop.  Constrained ``> 0`` so
    the CTRL formula (``logit /= penalty`` for positive logits, ``logit *
    penalty`` for negative) stays well-defined.
    """
    prompt_upsampler_presence_penalty: float
    """Native prompt upsampler OpenAI-style additive presence penalty.

    Subtracted once from every logit at a vocab position already seen in
    each caption's history (binary presence, not frequency).  ``>0``
    discourages reuse, ``<0`` encourages it, ``0`` is identity.
    Unconstrained sign: negative values are valid for legitimate "favor
    repetition" use cases.
    """
    prompt_upsampler_seed: int | None = None
    """Optional integer seed for the native prompt upsampler's sampling RNG.

    When set (and ``prompt_upsampler_temperature > 0`` so the AR loop
    actually samples), a device-local ``torch.Generator`` is seeded once
    inside ``_impl_generate_reasoner_text`` and threaded into every
    ``torch.multinomial`` draw, making the upsampled caption a
    deterministic function of ``seed``, the prompt, and the penalty
    masks.  ``None`` (default) consumes the device's default RNG,
    preserving the pre-seed behavior.  Greedy decoding (temperature 0)
    never reads the generator, so the value has no effect in that case.
    Under multi-rank inference, callers that need cross-rank agreement
    on sampled upsampler tokens must pass the same seed on every rank.
    """


class PromptUpsamplingOverrides(OverridesBase):
    prompt_upsampling: Training[bool | None] = None
    """If True, replace the prompt with a dense JSON prompt."""
    prompt_upsampler_max_tokens: pydantic.PositiveInt | None = None
    """Maximum tokens generated by the prompt upsampler."""
    prompt_upsampler_temperature: pydantic.NonNegativeFloat | None = None
    """Native prompt upsampler sampling temperature."""
    prompt_upsampler_top_p: PromptUpsamplerProbability | None = None
    """Native prompt upsampler nucleus sampling probability."""
    prompt_upsampler_top_k: pydantic.NonNegativeInt | None = None
    """Native prompt upsampler top-k sampling limit."""
    prompt_upsampler_repetition_penalty: pydantic.PositiveFloat | None = None
    """Native prompt upsampler CTRL/HF-style multiplicative repetition penalty (>0)."""
    prompt_upsampler_presence_penalty: float | None = None
    """Native prompt upsampler OpenAI-style additive presence penalty (any sign)."""
    prompt_upsampler_seed: int | None = None
    """Optional integer seed for the native prompt upsampler's sampling RNG."""

    def _build_prompt_upsampling(self, *, model_config: "OmniMoTModelConfig") -> None:
        if self.prompt_upsampler_max_tokens is None:
            self.prompt_upsampler_max_tokens = 20000
        if self.prompt_upsampler_temperature is None:
            self.prompt_upsampler_temperature = 0.7
        if self.prompt_upsampler_top_p is None:
            self.prompt_upsampler_top_p = 0.8
        if self.prompt_upsampler_top_k is None:
            self.prompt_upsampler_top_k = 20
        if self.prompt_upsampler_repetition_penalty is None:
            self.prompt_upsampler_repetition_penalty = 1.0
        if self.prompt_upsampler_presence_penalty is None:
            self.prompt_upsampler_presence_penalty = 1.5
        if self.prompt_upsampler_seed is None:
            self.prompt_upsampler_seed = 3407


# The --experiment-overrides that bake the rolling-KV profile into a causal multiview export, with
# the values of the Cosmos3-Nano-Sim-Auto recipe. The rolling sampler reads them from the exported
# model config; nothing at inference time can supply them. Every refusal that asks for a re-export
# quotes this one string.
CAUSAL_MULTIVIEW_ROLLING_EXPORT_OVERRIDES = (
    "model.config.rolling_kv_cache_chunks=2 model.config.rolling_kv_sink_chunks=1 "
    "model.config.rolling_prompt_policy=segmented"
)


def check_auto_fps_options(sample: "OmniSampleOverrides | OmniSampleArgs", config: Any) -> None:
    """Validate effective arguments for CLI and API callers, including already-built samples."""
    camera_fps = getattr(config, "_auto_camera_fps", None)
    if camera_fps is None or sample.multiview is None:
        if sample.output_fps is not None:
            raise ValueError("output_fps is only supported for identified Auto camera models with multiview inputs.")
        return
    if sample.fps != camera_fps:
        raise ValueError(
            f"This Auto checkpoint requires model conditioning fps={camera_fps}, got fps={sample.fps}. "
            f"For RGB-only playback at {sample.fps} FPS, omit --fps and use --output-fps {sample.fps} "
            "(JSON: output_fps). Keep the synchronized input frames unchanged."
        )
    if sample.output_fps is not None and sample.output_fps != camera_fps:
        if sample.is_joint_cam_lidar_transfer:
            raise ValueError(
                "Joint RGB/LiDAR retiming is not supported: output_fps must equal fps=30 to preserve "
                "camera/sweep synchronization. Use different output_fps only for RGB-only Auto generation."
            )
        if sample.enable_sound or sample.sound_path is not None:
            raise ValueError("Auto output_fps retiming supports RGB-only generation without audio.")


def check_causal_multiview_lidar_prefix_options(sample: "OmniSampleOverrides", config: Any) -> None:
    """Check the causal joint LiDAR-prefix options before transfer defaults fill them in.

    The rolling sampler holds at most one clean initial target sweep. An explicit sweep count
    therefore needs a prefix to read it from, which only the request (not the defaults) can tell.
    """
    if sample.multiview is None or not is_causal_multiview_config(config) or not sample.is_joint_cam_lidar_transfer:
        return
    lidar = sample.lidar
    if lidar is not None and (lidar.condition_path is not None or lidar.num_conditional_sweeps is not None):
        if lidar.num_conditional_sweeps not in (None, 1):
            raise ValueError("Causal multiview inference supports only one initial LiDAR sweep.")
        if lidar.condition_path is None:
            raise ValueError("lidar.num_conditional_sweeps requires lidar.condition_path.")


def check_causal_multiview_options(sample: "OmniSampleOverrides", config: Any) -> None:
    """Check the options consumed by the causal RGB/joint multiview runtime."""
    causal_multiview = sample.multiview is not None and is_causal_multiview_config(config)
    if not causal_multiview:
        if sample.save_latents:
            raise ValueError("Latent saving requires causal multiview inference.")
        return
    if not sample.autoregressive:
        raise ValueError("Causal multiview inference requires autoregressive=True.")
    if not (sample.is_multiview_transfer or sample.is_joint_cam_lidar_transfer):
        raise ValueError("Causal multiview inference requires per-camera transfer controls.")
    if sample.control_guidance != 1.0:
        raise ValueError("Causal multiview inference requires control_guidance=1.0.")
    if getattr(config, "fixed_step_sampler_config", None) is not None and sample.guidance != 1.0:
        log.warning(
            f"Sample '{sample.name}': overriding guidance={sample.guidance} with guidance=1.0 "
            "for distilled causal multiview inference; guidance is already distilled into the student."
        )
        sample.guidance = 1.0
    if sample.is_joint_cam_lidar_transfer:
        if getattr(config, "lidar_tokenizer", None) is None:
            raise ValueError("Joint generation requires a checkpoint with a LiDAR tokenizer.")
        source_fps = TransferDataOverrides._JOINT_CAM_LIDAR_DEFAULTS["fps"]
        if sample.fps != source_fps:
            raise ValueError(
                f"Causal joint RGB/LiDAR inference requires fps={source_fps} to match the trained source-frame clock."
            )
        if sample.normalize_cfg:
            raise ValueError("Joint inference requires normalize_cfg=False.")


class OmniSampleArgs(
    SampleArgs,
    SamplingArgs,
    SampleDataArgs,
    PromptUpsamplingArgs,
): ...


class OmniSampleOverrides(
    SampleOverrides,
    SamplingOverrides,
    SampleDataOverrides,
    PromptUpsamplingOverrides,
):
    defaults_file: ResolvedFilePath | None = None
    """Path to a JSON file of per-modality default sample fields. Overrides the built-in defaults."""

    _VLM_MODEL_SIZE: ClassVar[dict[str, ModelSize]] = {
        "Qwen/Qwen3-0.6B": "0.6B",
        "Qwen/Qwen3-VL-2B-Instruct": "2B",
        "Qwen/Qwen3-VL-8B-Instruct": "8B",
        "Qwen/Qwen3-VL-32B-Instruct": "32B",
        "Qwen/Qwen3-VL-30B-A3B-Instruct": "30B-A3B",
        "Qwen/Qwen3-VL-235B-A22B-Instruct": "235B-A22B",
        "nvidia/Cosmos3-Edge-Reasoner": "2B",
        "nvidia/Cosmos3-Nano-Reasoner": "8B",
    }

    _NUM_FRAMES_DEFAULTS: ClassVar[dict[str, int]] = {
        # Cosmos3-Edge defaults to a shorter 121-frame clip for video
        # generation; every other model keeps the per-modality JSON default
        # (189). Keyed by ``vlm_config.model_name`` so only Edge is affected.
        "nvidia/Cosmos3-Edge-Reasoner": 121,
    }

    _RESOLUTION_SHIFT_DEFAULTS: ClassVar[dict[(ModelSize, Resolution), float]] = {
        # 2B rows mirror Cosmos3-Edge's training shift (Cosmos3-Edge.yaml
        # rectified_flow_training_config.shift: 256->3, 480->5, 720->10).
        ("2B", "256"): 3.0,
        ("2B", "480"): 5.0,
        ("2B", "720"): 10.0,
        ("8B", "256"): 3.0,
        ("8B", "480"): 5.0,
        ("8B", "720"): 10.0,
        ("8B", "768"): 10.0,
        ("32B", "256"): 5.0,
        ("32B", "480"): 5.0,
        ("32B", "720"): 5.0,
        ("32B", "768"): 5.0,
    }

    def _validate_lidar_transfer(self) -> None:
        """Reject LiDAR hint combinations that later parsing would silently mis-read.

        Three mutually exclusive shapes are legal:

        * LiDAR-only -- ``lidar.control_path``, with no ``wsm``, ``multiview`` or ``vision_path``.
        * WSM-only -- not a LiDAR sample; ``_validate_multiview_options`` handles it.
        * Joint camera+LiDAR -- ``wsm`` plus ``lidar.control_path`` plus one or more camera views,
          each carrying its own ``control_path``.

        Anything else is rejected here rather than surfacing later as a shape mismatch deep in
        batch assembly.
        """
        self = cast("SampleDataOverrides", self)
        hints = self.transfer_hints
        if TransferHintKey.LIDAR not in hints:
            return
        extra_hints = set(hints) - {TransferHintKey.LIDAR, TransferHintKey.WSM}
        if extra_hints:
            raise ValueError("lidar transfer cannot be combined with other control hints")
        if self.vision_path is not None:
            raise ValueError("lidar transfer uses lidar.control_path; vision_path is not supported")
        lidar = self.lidar
        assert lidar is not None
        if lidar.control_path is None:
            raise ValueError("lidar transfer requires lidar.control_path")

        has_wsm = TransferHintKey.WSM in hints
        has_multiview = self.multiview is not None
        if has_wsm and has_multiview:
            assert self.multiview is not None
            if not self.multiview.views:
                raise ValueError("joint camera+LiDAR transfer requires at least one multiview view")
            has_vision_paths = [view.vision_path is not None for view in self.multiview.views]
            if any(has_vision_paths) and not all(has_vision_paths):
                raise ValueError("joint camera+LiDAR transfer does not support partial-view conditioning")
            for view_index, view in enumerate(self.multiview.views):
                if view.control_path is None:
                    raise ValueError(
                        "joint camera+LiDAR transfer requires control_path for every view "
                        f"(missing at views[{view_index}])"
                    )
            if hints[TransferHintKey.WSM].control_path is not None:
                raise ValueError(
                    "joint camera+LiDAR transfer uses the view control_path; do not also set wsm.control_path"
                )
            return
        if has_multiview:
            raise ValueError("lidar transfer does not support multiview; set lidar.control_path")
        if has_wsm:
            raise ValueError("lidar + wsm requires multiview with at least one camera view")
        # LiDAR-only generation has no measured-prefix path here, so refuse the prefix rather
        # than drop it in silence.
        if lidar.condition_path is not None:
            raise ValueError("lidar.condition_path is only supported for joint camera+LiDAR transfer")

    @override
    def build_sample(self, *, model_config: Any) -> OmniSampleArgs:
        model_config = cast("OmniMoTModelConfig", model_config)
        sample_meta = self.sample_meta
        self._validate_lidar_transfer()

        # Apply per-modality defaults from JSON config files.
        # User-provided values take precedence over JSON defaults.
        if self.defaults_file is not None:
            defaults = json.loads(self.defaults_file.read_text())
        else:
            defaults_key = sample_meta.model_mode
            if self.multiview is not None:
                defaults_key = "multiview"
            defaults = _load_modality_defaults(defaults_key)
        overrides = self.model_dump(exclude_none=True)
        shift_configured = "shift" in overrides or defaults.get("shift") is not None
        user_fields = frozenset(overrides)
        merged_data = _deep_merge(defaults, overrides)
        # The built-in video2video recipe has a scalar shift of 10. The new
        # directly resized Stage 1 teacher uses the resolution defaults below
        # (5 at 480, 10 at 720); explicit sample/custom-defaults choices still win.
        if (
            merged_data.get("transfer_resize_mode") == "direct_resize"
            and getattr(model_config, "causal_training_strategy", None) == "teacher_forcing"
            and self.defaults_file is None
            and "shift" not in overrides
        ):
            shift_configured = False
        # The built-in video defaults select 16:9 for generation. Direct Transfer
        # instead derives the least-distorted bucket from decoded source pixels;
        # only an explicit sample/defaults-file aspect choice should override it.
        if (
            merged_data.get("transfer_resize_mode") == "direct_resize"
            and "aspect_ratio" not in overrides
            and (self.defaults_file is None or defaults.get("aspect_ratio") is None)
        ):
            merged_data["aspect_ratio"] = None
        merged_data = {k: v for k, v in merged_data.items() if k in type(self).model_fields}
        merged = type(self).model_validate(merged_data)

        self.__dict__.update(merged.__dict__)
        self.model_mode = sample_meta.model_mode

        # Fields chosen by the sample, the CLI or a custom defaults file (not the built-in presets).
        configured = (user_fields | frozenset(defaults)) if self.defaults_file is not None else user_fields
        causal_multiview = self.multiview is not None and is_causal_multiview_config(model_config)
        if causal_multiview:
            # Match the model's sensor clock and distilled sampling settings instead of the
            # bidirectional multiview preset. Explicit sample/CLI/custom-defaults choices still
            # take precedence.
            if "fps" not in configured:
                self.fps = int(model_config.diffusion_expert_config.base_fps)
            if "guidance" not in configured and getattr(model_config, "fixed_step_sampler_config", None) is not None:
                self.guidance = 1.0

        # Auto's released camera clock is fixed, but the shared interface remains variable-FPS
        # for other models. Only omitted FPS gets defaulted; sample/CLI/custom-defaults values
        # must pass validation rather than being silently replaced.
        auto_fps = getattr(model_config, "_auto_camera_fps", None)
        if auto_fps is not None and self.multiview is not None:
            if "fps" not in configured or self.fps is None:
                self.fps = auto_fps
            if self.output_fps is None:
                self.output_fps = self.fps
        check_auto_fps_options(self, model_config)

        # Model-specific num_frames default for video generation (e.g.
        # Cosmos3-Edge -> 121). Applies only when the user did not request a
        # frame count and the mode produces a plain video; image, action, and
        # reasoner modes keep their own num_frames handling (reasoner reports
        # VIDEO vision_mode but uses num_frames as an inert 1).
        if (
            "num_frames" not in user_fields
            and sample_meta.vision_mode == VisionMode.VIDEO
            and not sample_meta.model_mode.is_action
            and not sample_meta.model_mode.is_reasoner
        ):
            num_frames_default = self._NUM_FRAMES_DEFAULTS.get(model_config.vlm_config.model_name)
            if num_frames_default is not None:
                self.num_frames = num_frames_default

        self._build_sample()
        self._build_sampling(
            model_config=model_config, sample_meta=sample_meta, num_steps_configured="num_steps" in configured
        )
        self._build_text_data(model_config=model_config, sample_meta=sample_meta)
        self._build_vision_data(model_config=model_config, sample_meta=sample_meta)

        self._build_prompt_upsampling(model_config=model_config)

        self._build_action_data(model_config=model_config, sample_meta=sample_meta)

        # Transfer defaults must follow the checkpoint's causal contract even though
        # the generic per-modality JSON presets explicitly default AR to False. A causal
        # multiview checkpoint is trained for AR generation only (the one-shot multiview
        # sampler runs it without error but off-distribution), so its multiview samples
        # follow the same contract; an explicit False is refused below.
        if "autoregressive" not in user_fields and (
            causal_multiview or (self.transfer_hints and self.multiview is None and model_config.video_temporal_causal)
        ):
            self.autoregressive = True
        self._build_interactive_data(model_config=model_config, sample_meta=sample_meta)
        if self.kv_cache_frustum_path is not None and (
            not self.autoregressive or sample_meta.model_mode not in (ModelMode.IMAGE2VIDEO, ModelMode.FORWARD_DYNAMICS)
        ):
            raise ValueError("kv_cache_frustum_path requires autoregressive image2video or forward_dynamics")

        self._build_sound_data(model_config=model_config, sample_meta=sample_meta)

        self._build_reasoner_data(model_config=model_config, sample_meta=sample_meta)

        # Before the transfer defaults populate lidar.num_conditional_sweeps.
        check_causal_multiview_lidar_prefix_options(self, model_config)
        self._build_transfer_data(model_config=model_config, sample_meta=sample_meta, user_fields=user_fields)

        self._apply_sf_dmd_transfer_inference_contract(model_config=model_config, user_fields=user_fields)

        self._build_multiview_data(sample_meta=sample_meta, model_config=model_config)
        # After the transfer defaults, so control_guidance and the joint clock are resolved.
        check_causal_multiview_options(self, model_config)
        check_auto_fps_options(self, model_config)
        if auto_fps is not None and self.multiview is not None:
            log.info(f"Model conditioning: {self.fps} FPS | Output playback: {self.output_fps} FPS")

        if not shift_configured and not sample_meta.model_mode.is_reasoner:
            # Unregistered backbone names (custom fine-tunes) simply skip shift
            # defaulting rather than KeyError-ing.
            model_size = self._VLM_MODEL_SIZE.get(model_config.vlm_config.model_name)
            key = (model_size, self.resolution)
            if key in self._RESOLUTION_SHIFT_DEFAULTS:
                self.shift = self._RESOLUTION_SHIFT_DEFAULTS[key]

        # Engage the in-model (V4.2-template) native upsampler exactly
        # when the user opted into upsampling AND the dense
        # (endpoint-backed) path did not successfully apply it.
        # ``prompt_upsampling_applied`` is:
        #   * ``True``  — dense path applied, native should skip.
        #   * ``False`` — dense path declined (no endpoint configured),
        #                  native should take over.
        #   * ``None``  — dense path never ran.  Two cases produce this:
        #       (a) release builds, where the dense dispatcher and the
        #           ``prompt_upsampling_applied`` field itself are both
        #           stripped, so the attribute does not exist on the
        #           instance — ``getattr(..., None)`` papers over the
        #           AttributeError;
        #       (b) ``prompt_upsampling`` was not requested, in which
        #           case the leading ``is True`` short-circuits anyway.
        # Both ``False`` and ``None`` should engage native upsampling
        # when the user opted in, so the gate is ``not applied`` rather
        # than ``applied is False`` (which would miss case (a)).
        prompt_upsampling_applied = getattr(self, "prompt_upsampling_applied", None)
        native_prompt_upsampling = self.prompt_upsampling is True and not prompt_upsampling_applied
        return self._build(OmniSampleArgs, native_prompt_upsampling=native_prompt_upsampling)


_MODEL_MEMORY_FACTOR: int = int(1e9) * 2 * 2  # 1B params/tower * 2 bytes/param (bfloat16) * 2 towers
MODEL_MEMORY_BYTES_BY_SIZE: dict[ModelSize, int] = {
    "0.6B": round(0.6 * _MODEL_MEMORY_FACTOR),
    "2B": 2 * _MODEL_MEMORY_FACTOR,
    "8B": 8 * _MODEL_MEMORY_FACTOR,
    "30B-A3B": 30 * _MODEL_MEMORY_FACTOR,
    "32B": 32 * _MODEL_MEMORY_FACTOR,
    "235B-A22B": 235 * _MODEL_MEMORY_FACTOR,
}

_CHECKPOINTS: dict[str, CheckpointConfig] = {
    "Cosmos3-Nano": CheckpointConfig(
        model_memory_bytes=MODEL_MEMORY_BYTES_BY_SIZE["8B"],
        config_file=str(CONFIG_DIR / "model/Cosmos3-Nano.yaml"),
        s3_uri="s3://bucket1/cosmos3_vfm/cosmos3_ga_midtraining/cosmos3_ga_16bm8b_v2_midtrain/checkpoints/iter_000006000/",
        hf=CheckpointDirHf(
            repository="nvidia/Cosmos3-Nano",
            revision="main",
        ),
    ),
    "Cosmos3-Edge": CheckpointConfig(
        model_memory_bytes=MODEL_MEMORY_BYTES_BY_SIZE["2B"],
        config_file=str(CONFIG_DIR / "model/Cosmos3-Edge.yaml"),
        s3_uri="s3://bucket1/cosmos3_vfm/cosmos3_ga_midtraining/cosmos3_ga_4bm2b_v1_midtrain_0630a/checkpoints/iter_000028000/",
        hf=CheckpointDirHf(
            repository="nvidia/Cosmos3-Edge",
            revision="main",
        ),
    ),
    "Cosmos3-Super": CheckpointConfig(
        model_memory_bytes=MODEL_MEMORY_BYTES_BY_SIZE["32B"],
        config_file=str(CONFIG_DIR / "model/Cosmos3-Super.yaml"),
        s3_uri="s3://bucket1/cosmos3_vfm/cosmos3_ga_midtraining/cosmos3_ga_64bm32b_v3_midtrain/checkpoints/iter_000001800/",
        hf=CheckpointDirHf(
            repository="nvidia/Cosmos3-Super",
            revision="main",
        ),
    ),
    # Task-specialized Super variants published as diffusers HF checkpoints.
    # s3_uri is unused for HF-backed checkpoints (kept for parity with the
    # registry schema); the architecture lives in each model YAML.
    "Cosmos3-Super-Image2Video": CheckpointConfig(
        model_memory_bytes=MODEL_MEMORY_BYTES_BY_SIZE["32B"],
        config_file=str(CONFIG_DIR / "model/Cosmos3-Super.yaml"),
        s3_uri="s3://bucket1/cosmos3_vfm/cosmos3_ga_image2video/",
        hf=CheckpointDirHf(
            repository="nvidia/Cosmos3-Super-Image2Video",
            revision="main",
        ),
        # Self-contained checkpoint: use its bundled processor instead of
        # downloading the base Cosmos3-Super repo just for the tokenizer.
        vlm_processor_from_checkpoint=True,
    ),
    "Cosmos3-Super-Text2Image": CheckpointConfig(
        model_memory_bytes=MODEL_MEMORY_BYTES_BY_SIZE["32B"],
        config_file=str(CONFIG_DIR / "model/Cosmos3-Super.yaml"),
        s3_uri="s3://bucket1/cosmos3_vfm/cosmos3_ga_text2image/",
        hf=CheckpointDirHf(
            repository="nvidia/Cosmos3-Super-Text2Image",
            revision="main",
        ),
        # Self-contained checkpoint: use its bundled processor instead of
        # downloading the base Cosmos3-Super repo just for the tokenizer.
        vlm_processor_from_checkpoint=True,
    ),
    "Cosmos3-Super-Text2Image-4Step": CheckpointConfig(
        model_memory_bytes=MODEL_MEMORY_BYTES_BY_SIZE["32B"],
        config_file=str(CONFIG_DIR / "model/Cosmos3-Super.yaml"),
        s3_uri="s3://bucket1/cosmos3_vfm/cosmos3_ga_text2image_4step/",
        hf=CheckpointDirHf(
            repository="nvidia/Cosmos3-Super-Text2Image-4Step",
            revision="main",
        ),
        experiment_overrides=(
            "model.config.resolution='768'",
            "model.config.action_gen=false",
            "model.config.sound_gen=false",
            "model.config.sound_dim=null",
            "model.config.sound_tokenizer=null",
            "model.config.rectified_flow_training_config.shift.720=5",
            "model.config.rectified_flow_training_config.shift.768=5",
            "model.config.tokenizer.encode_chunk_frames.768=8",
            "model.config.tokenizer.encode_exact_durations=null",
            "model.config.fixed_step_sampler_config.t_list=[1.0,0.9375,0.8333333333333334,0.625]",
            "model.config.fixed_step_sampler_config.sample_type=sde",
        ),
    ),
    "Cosmos3-Super-Image2Video-4Step": CheckpointConfig(
        model_memory_bytes=MODEL_MEMORY_BYTES_BY_SIZE["32B"],
        config_file=str(CONFIG_DIR / "model/Cosmos3-Super.yaml"),
        s3_uri="s3://bucket1/cosmos3_vfm/cosmos3_ga_image2video_4step/",
        hf=CheckpointDirHf(
            repository="nvidia/Cosmos3-Super-Image2Video-4Step",
            revision="main",
        ),
        experiment_overrides=(
            "model.config.resolution='480'",
            "model.config.action_gen=false",
            "model.config.sound_gen=false",
            "model.config.sound_dim=null",
            "model.config.sound_tokenizer=null",
            "model.config.diffusion_expert_config.base_fps=16",
            "model.config.fixed_step_sampler_config.t_list=[1.0,0.9375,0.8333333333333334,0.625]",
            "model.config.fixed_step_sampler_config.sample_type=sde",
        ),
    ),
}
DEFAULT_CHECKPOINT_NAME = "Cosmos3-Nano"
DEFAULT_CHECKPOINT = _CHECKPOINTS[DEFAULT_CHECKPOINT_NAME]

MAX_CP_SIZE = 32
CpSize = Annotated[int, pydantic.Field(ge=1, le=MAX_CP_SIZE)]


_OmniSetupT = TypeVar("_OmniSetupT", bound="OmniSetupArgs | OmniSetupOverrides")


def _validate_condition_stream_encode(setup: _OmniSetupT) -> _OmniSetupT:
    """Validate the streaming topology both before and after automatic resolution."""
    if not setup.condition_stream_encode:
        return setup
    allowed_sizes = (0, 1) if isinstance(setup, SetupOverrides) else (1,)
    if any(size not in allowed_sizes for size in (setup.dp_shard_size, setup.cp_size, setup.cfgp_size)):
        raise ValueError("condition_stream_encode requires dp_shard_size=1, cp_size=1, and cfgp_size=1")
    return setup


class OmniSetupArgs(SetupArgs):
    variant: Suppress[Literal["omni"]] = "omni"
    """Discriminator."""

    # pyrefly: ignore[bad-override]
    sample_overrides: OmniSampleOverrides

    sampler: Literal["unipc", "edm"]

    condition_stream_encode: bool = False
    """Encode causal Transfer controls incrementally with the cached video tokenizer."""

    kv_cache_inference_size: pydantic.NonNegativeInt | None = None
    """Causal history window in latent frames; 0 selects full history, None preserves the checkpoint."""
    attention_sink_size: pydantic.NonNegativeInt | None = None
    """Number of initial latent frames retained by the causal cache; None preserves the checkpoint."""

    _validate_condition_stream_encode = pydantic.model_validator(mode="after")(_validate_condition_stream_encode)

    # Override defaults
    cp_size: CpSize

    @override
    @classmethod
    def get_sample_overrides_cls(cls) -> type[SampleOverrides]:
        return OmniSampleOverrides

    @override
    @classmethod
    def get_sample_args_cls(cls) -> type[SampleArgs]:
        return OmniSampleArgs

    @override
    @classmethod
    def get_inference_cls(cls) -> type["Inference"]:
        from cosmos_framework.inference.inference import OmniInference

        return OmniInference

    @pydantic.model_validator(mode="after")
    def _validate_parallelism(self) -> Self:
        world_size = int(os.environ.get("WORLD_SIZE", "0"))

        if world_size:
            if self.dp_shard_size * self.dp_replicate_size > world_size:
                raise ValueError(
                    f"dp_shard_size({self.dp_shard_size}) * dp_replicate_size({self.dp_replicate_size}) must be <= WORLD_SIZE({world_size})"
                )
            if world_size % (self.dp_shard_size * self.dp_replicate_size) != 0:
                raise ValueError(
                    f"dp_shard_size({self.dp_shard_size}) * dp_replicate_size({self.dp_replicate_size}) must divide WORLD_SIZE({world_size})"
                )

        if world_size:
            if self.cp_size * self.cfgp_size > world_size:
                raise ValueError(
                    f"cp_size({self.cp_size}) * cfgp_size({self.cfgp_size}) must be <= WORLD_SIZE({world_size})"
                )
            if world_size % (self.cp_size * self.cfgp_size) != 0:
                raise ValueError(
                    f"cp_size({self.cp_size}) * cfgp_size({self.cfgp_size}) must divide WORLD_SIZE({world_size})"
                )

        return self


class OmniSetupOverrides(SetupOverrides):
    variant: Suppress[Literal["omni"]] = "omni"
    """Discriminator."""

    CHECKPOINTS: ClassVar[dict[str, CheckpointConfig]] = _CHECKPOINTS

    sample_overrides: OmniSampleOverrides = OmniSampleOverrides()

    model_size: Training[ModelSize | None] = None

    sampler: Literal["unipc", "edm"] = "unipc"

    condition_stream_encode: bool = False
    """Encode causal Transfer controls incrementally; requires one unsharded GPU per worker."""

    kv_cache_inference_size: pydantic.NonNegativeInt | None = None
    """Causal history window in latent frames; 0 selects full history, None preserves the checkpoint."""
    attention_sink_size: pydantic.NonNegativeInt | None = None
    """Number of initial latent frames retained by the causal cache; None preserves the checkpoint."""

    _validate_condition_stream_encode = pydantic.model_validator(mode="after")(_validate_condition_stream_encode)

    # Override defaults
    dp_replicate_size: pydantic.NonNegativeInt = 0
    dp_shard_size: pydantic.NonNegativeInt = 0
    cp_size: CpSize | Literal[0] = 0
    cfgp_size: CfgpSize | Literal[0] = 0

    use_cuda_graphs: bool = False

    compiled_region: Literal["all", "language"] = "all"
    # Unsupported
    tp_size: Suppress[pydantic.NonNegativeInt] = 1

    def _apply_sf_dmd_transfer_setup_contract(self) -> None:
        """Pin the SF-DMD Transfer runtime described by the exported student config.

        This checkpoint has two sane runtimes: the static-shape CUDA-graph AR path
        (``torch.compile`` + CUDA graphs, the default) and plain eager attention.
        ``use_torch_compile`` selects between them: with its default ``True`` the
        CUDA graphs are switched on as well (compile without graphs retraces every
        frame here), ``--no-use-torch-compile`` gives the eager runtime.  Asking for
        CUDA graphs while disabling compile is a contradiction and raises.  The
        compiled region is pinned to the decoder blocks (``"language"``): compiling
        the VFM heads as well measured no gain on this checkpoint while lengthening the
        warm-up and changing the numerics floor.  The diffusion cache and CP/CFGP
        parallelism are always disabled.
        """
        if not self._is_sf_dmd_transfer_hf_checkpoint():
            return
        if self.use_cuda_graphs and not self.use_torch_compile:
            raise ValueError(
                "SF-DMD Transfer CUDA graphs require torch.compile; pass --use-torch-compile together with "
                "--use-cuda-graphs (or --no-use-torch-compile for the eager runtime)."
            )
        use_torch_compile = bool(self.use_torch_compile)
        use_cuda_graphs = use_torch_compile
        runtime_requirements: tuple[tuple[str, bool | int | str, bool | int | str], ...] = (
            ("use_cuda_graphs", self.use_cuda_graphs, use_cuda_graphs),
            ("compiled_region", self.compiled_region, "language"),
            ("diffusion_cache", self.diffusion_cache, False),
            ("cp_size", self.cp_size, 1),
            ("cfgp_size", self.cfgp_size, 1),
        )
        transitions = [
            f"{field}={current!r}->{required!r}"
            for field, current, required in runtime_requirements
            if current != required
        ]
        if transitions:
            log.warning("Applying the SF-DMD Transfer runtime contract; overriding " + ", ".join(transitions) + ".")
        self.use_torch_compile = use_torch_compile
        self.use_cuda_graphs = use_cuda_graphs
        self.compiled_region = "language"
        self.diffusion_cache = False
        self.cp_size = 1
        self.cfgp_size = 1

    def _hf_checkpoint_model_config(self) -> dict[str, Any] | None:
        """Read the exported ``model.config`` of an HF checkpoint; None for any other checkpoint type."""
        if self.checkpoint_type != CheckpointType.HF:
            return None
        if self.config_file_type != ConfigFileType.MODULE:
            model_dict = deserialize_config_dict(Path(self.config_file))
        else:
            if self.checkpoint_hf is None:
                checkpoint_dir = Path(self.checkpoint_path)
            else:
                metadata = self.checkpoint_hf.model_copy(update={"include": ("config.json",), "exclude": ()})
                checkpoint_dir = Path(metadata.download())
            model_dict = json.loads((checkpoint_dir / "config.json").read_text())
        return model_dict.get("model", {}).get("config", {})

    def _is_sf_dmd_transfer_hf_checkpoint(self) -> bool:
        """Recognize the student inference contract without a private experiment name."""
        config = self._hf_checkpoint_model_config()
        if config is None:
            return False
        replay_policy = config.get("teacher_forcing_replay_policy")
        if not isinstance(replay_policy, dict):
            return False
        return (
            config.get("fixed_step_sampler_config") is not None
            and config.get("video_temporal_causal") is True
            and config.get("teacher_forcing_frames_per_chunk") == 1
            and replay_policy.get("control_visibility") == "causal"
            and replay_policy.get("controls_read_strict_past_clean_rgb") is True
            and replay_policy.get("clean_pass_causality") == "frame"
        )

    def _apply_causal_multiview_setup_contract(self, world_size: int) -> None:
        """Pin the eager rolling runtime of a causal multiview checkpoint (Cosmos3-Nano-Sim-Auto).

        Rolling replay runs eagerly with serial CFG, and the diffusion cache does not apply to its
        AR loop, so ``torch.compile``, CUDA graphs, CFG parallelism and the cache are switched off.
        Context parallelism is the only multi-GPU axis this runtime uses, so an unset ``cp_size``
        spans every rank instead of following the latency preset, which would spend half of them on
        CFG parallelism.

        The runtime implements rolling KV with segmented prompts only. A checkpoint exported without
        that profile, a request for the framewise causal cache, or a separate VAE decode GPU is
        refused here rather than by the guards of the sampler and the causal frontend, which run
        only after the model has loaded.
        """
        config = self._hf_checkpoint_model_config()
        if config is None or not is_causal_multiview_config(SimpleNamespace(**config)):
            return
        rolling_kv_cache_chunks = config.get("rolling_kv_cache_chunks")
        rolling_prompt_policy = config.get("rolling_prompt_policy")
        if rolling_kv_cache_chunks is None or rolling_prompt_policy != "segmented":
            raise ValueError(
                "This causal multiview checkpoint was exported without the rolling KV profile "
                f"(rolling_kv_cache_chunks={rolling_kv_cache_chunks!r}, rolling_prompt_policy={rolling_prompt_policy!r}); "
                "its inference runtime requires rolling KV with segmented prompts. Re-export it with the "
                "rolling overrides, e.g. for Cosmos3-Nano-Sim-Auto: --experiment-overrides "
                f"{CAUSAL_MULTIVIEW_ROLLING_EXPORT_OVERRIDES}."
            )
        framewise_cache_flags = [
            f"--{field.replace('_', '-')}"
            for field in ("kv_cache_inference_size", "attention_sink_size")
            if getattr(self, field) is not None
        ]
        if framewise_cache_flags:
            raise ValueError(
                f"{' and '.join(framewise_cache_flags)} configure the framewise causal cache, which rolling "
                "causal multiview inference does not use; its cache is set by rolling_kv_cache_chunks and "
                "rolling_kv_sink_chunks in the exported config. Drop the flag(s), or re-export the checkpoint "
                "with different rolling overrides."
            )
        if self.use_separate_pipeline_vision_decode_gpu:
            # The causal frontend refuses it as well, but only at the first sample, after the weights
            # have loaded and the VAE has moved to its own GPU.
            raise ValueError(
                "Causal multiview inference does not support a separate VAE decode GPU; drop "
                "--use-separate-pipeline-vision-decode-gpu."
            )

        cp_size = self.cp_size or max(1, min(MAX_CP_SIZE, world_size))
        runtime_requirements: tuple[tuple[str, bool | int, bool | int], ...] = (
            ("use_torch_compile", self.use_torch_compile, False),
            ("use_cuda_graphs", self.use_cuda_graphs, False),
            ("diffusion_cache", self.diffusion_cache, False),
            ("cp_size", self.cp_size, cp_size),
            ("cfgp_size", self.cfgp_size, 1),
        )
        transitions = [
            f"{field}={current!r}->{required!r}"
            for field, current, required in runtime_requirements
            if current != required
        ]
        if transitions:
            log.warning("Applying the causal multiview runtime contract; overriding " + ", ".join(transitions) + ".")
        self.use_torch_compile = False
        self.use_cuda_graphs = False
        self.diffusion_cache = False
        self.cp_size = cp_size
        self.cfgp_size = 1

    def _build_model_parallelism(self, world_size: int, device_memory_bytes: int):
        if not self.dp_shard_size:
            # Shard the model across every rank by default (full FSDP). world_size == 0
            # means we're not under torchrun (single process) -> a single shard.
            self.dp_shard_size = max(1, world_size)
        if not self.dp_replicate_size:
            self.dp_replicate_size = max(1, world_size // self.dp_shard_size)

    def _build_context_parallelism(self, world_size: int):
        if not self.cfgp_size:
            match self.parallelism_preset:
                case "throughput":
                    self.cfgp_size = 1
                case "latency":
                    self.cfgp_size = max(1, min(2, world_size))
                case _:
                    assert_never(self.parallelism_preset)
        if not self.cp_size:
            match self.parallelism_preset:
                case "throughput":
                    self.cp_size = 1
                case "latency":
                    self.cp_size = max(1, min(MAX_CP_SIZE, world_size // self.cfgp_size))
                case _:
                    assert_never(self.parallelism_preset)

    @override
    def _build_parallelism(self, world_size: int | None, local_world_size: int | None, device_memory_bytes: int | None):
        if world_size is None:
            world_size = int(os.environ.get("WORLD_SIZE", "0"))
        if local_world_size is None:
            local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
        if device_memory_bytes is None:
            device_memory_bytes = _get_device_memory_bytes()

        if self.model_memory_bytes is None and self.model_size is not None:
            self.model_memory_bytes = MODEL_MEMORY_BYTES_BY_SIZE[self.model_size]
        self._build_model_parallelism(world_size=world_size, device_memory_bytes=device_memory_bytes)
        self._build_context_parallelism(world_size=world_size)
        if self.cfgp_size > 1 and self.use_batched_cfg:
            log.warning("CFG parallelism takes precedence over batched CFG; disabling batched CFG.")
            self.use_batched_cfg = False

    @override
    def build_setup(
        self, world_size: int | None = None, local_world_size: int | None = None, device_memory_bytes: int | None = None
    ) -> OmniSetupArgs:
        if world_size is None:
            world_size = int(os.environ.get("WORLD_SIZE", "0"))
        self._build_setup()
        self._build_checkpoint(checkpoints=self.CHECKPOINTS)
        self._apply_sf_dmd_transfer_setup_contract()
        self._apply_causal_multiview_setup_contract(world_size=world_size)
        for field in ("kv_cache_inference_size", "attention_sink_size"):
            value = getattr(self, field)
            if value is None:
                continue
            key = f"model.config.{field}="
            resolved_value = "null" if field == "kv_cache_inference_size" and value == 0 else str(value)
            self.experiment_overrides = [item for item in self.experiment_overrides if not item.startswith(key)]
            self.experiment_overrides.append(key + resolved_value)
        self._build_parallelism(
            world_size=world_size, local_world_size=local_world_size, device_memory_bytes=device_memory_bytes
        )
        return self._build(OmniSetupArgs)


# Reserved: the memory-based shard-size heuristic. No longer used as the default
# (we now shard across every rank), but kept for future opt-in / reference.
def _get_dp_shard_size(
    model_memory_bytes: int, device_memory_bytes: int, device_memory_utilization: float = 0.75
) -> int:
    return math.ceil(model_memory_bytes / device_memory_bytes / device_memory_utilization)


@cache
def _get_device_memory_bytes() -> int:
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = _get_nvml_device_memory_info(handle)
        return info.total
    except Exception:
        # Fallback for unified memory architectures (e.g., GB10) where
        # nvmlDeviceGetMemoryInfo is not supported.
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.get_device_properties(0).total_memory)
        return 128 * 1024**3  # Default 128GB
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _get_nvml_device_memory_info(handle: Any) -> Any:
    try:
        return pynvml.nvmlDeviceGetMemoryInfo_v2(handle)
    except AttributeError:
        return pynvml.nvmlDeviceGetMemoryInfo(handle)
    except pynvml.NVMLError_NotSupported:
        return pynvml.nvmlDeviceGetMemoryInfo(handle)
