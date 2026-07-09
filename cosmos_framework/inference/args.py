# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
import math
import os
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal, Self, cast, override

import pydantic
import pynvml
from typing_extensions import assert_never
from tyro.conf import Suppress

from cosmos_framework.inference.common.args import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    ArgsBase,
    CfgpSize,
    CheckpointConfig,
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
from cosmos_framework.inference.common.config import CONFIG_DIR, PACKAGE_DIR
from cosmos_framework.utils import log
from cosmos_framework.utils.checkpoint_db import CheckpointDirHf
from cosmos_framework.utils.flags import SMOKE, TRAINING

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

    num_steps: Training[pydantic.PositiveInt | None] = None
    """Number of steps for the diffusion model."""
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

    def _build_sampling(self, model_config: "OmniMoTModelConfig", sample_meta: "SampleMeta"):
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
        assert self.num_steps is not None
        if SMOKE:
            self.num_steps = min(self.num_steps, 1)


InferenceResolution = Literal["256", "480", "720", "768", "1080"]
if TRAINING:
    Resolution = Literal["256", "480", "704", "720", "768", "1080"]
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
    POLICY = "policy"

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
    {ModelMode.FORWARD_DYNAMICS, ModelMode.INVERSE_DYNAMICS, ModelMode.POLICY}
)

REASONER_MODEL_MODES: frozenset[ModelMode] = frozenset({ModelMode.REASONER})

# Modes that condition generation on a real input audio clip (require a model
# with ``sound_gen=True`` and a ``sound_path``).
SOUND_CONDITION_MODEL_MODES: frozenset[ModelMode] = frozenset({ModelMode.AUDIO_IMAGE2VIDEO})


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
    num_frames: pydantic.PositiveInt
    video_save_quality: VideoSaveQuality
    image_save_quality: ImageSaveQuality

    @property
    def duration(self) -> float:
        return self.num_frames / self.fps

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
    """Vision resolution.
    
    Defaults to model config resolution.
    """
    aspect_ratio: AspectRatio | None = None
    """Vision aspect ratio. When None, image_edit preserves the input image's native
    aspect ratio; all other modes default to 16:9."""
    fps: Fps | None = None
    """Vision frames per second. Recommended range [10, 30]; quality may be degraded outside of this range."""
    num_frames: pydantic.PositiveInt | None = None
    """Number of vision frames.

    Range by resolution: 256p: [24, 400], 480p: [24, 300], 720p/768p: [24, 200].
    Image-only resolutions (e.g. 1080p) require num_frames=1.
    """
    video_save_quality: Training[VideoSaveQuality | None] = None
    """Quality of the saved video (0-10)."""
    image_save_quality: Training[ImageSaveQuality | None] = None
    """Quality of the saved image (0-100)."""

    @override
    def download(self, output_dir: Path):
        super().download(output_dir)
        self.vision_path = download_file(self.vision_path, output_dir, "vision")

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
            if self.num_frames < MIN_NUM_FRAMES or self.num_frames > MAX_NUM_FRAMES[self.resolution]:
                log.warning(
                    f"Number of frames {self.num_frames} is outside the recommended range [{MIN_NUM_FRAMES}, {MAX_NUM_FRAMES[self.resolution]}]. Quality may be degraded."
                )
        if SMOKE:
            self.num_frames = min(self.num_frames, 2)
        temporal_compression_factor = model_config.tokenizer.temporal_compression_factor
        self.num_frames = (
            math.ceil((self.num_frames - 1) / temporal_compression_factor) * temporal_compression_factor + 1
        )


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
                    f"model_mode={sample_meta.model_mode.value} requires a `sound_path` "
                    "(a conditioning audio clip)"
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
    domain_name: str = ""
    image_size: pydantic.PositiveInt = 256
    action_chunk_size: pydantic.PositiveInt = 16
    raw_action_dim: int | None = None
    view_point: str | None = None


class ActionDataOverrides(OverridesBase):
    """Action data overrides."""

    action_path: Training[ResolvedFilePathOrUrl | None] = None
    """Path to action JSON file. Required for forward_dynamics mode."""
    domain_name: Training[str | None] = None
    """Action domain name passed to get_domain_id()."""
    image_size: Training[pydantic.PositiveInt | None] = None
    """Target image height in pixels (aspect-ratio-preserving resize)."""
    action_chunk_size: Training[pydantic.PositiveInt | None] = None
    """Number of action steps to predict."""
    raw_action_dim: Training[pydantic.PositiveInt | None] = None
    """Dimension of the raw action data. Required when action_path is not provided."""
    view_point: Training[str | None] = None
    """Viewpoint description for the action prompt."""

    @override
    def download(self, output_dir: Path):
        super().download(output_dir)
        self.action_path = download_file(self.action_path, output_dir, "action")

    def _build_action_data(self, model_config: "OmniMoTModelConfig", sample_meta: SampleMeta):
        if self.domain_name is None:
            self.domain_name = ""
        if self.image_size is None:
            self.image_size = 256
        if self.action_chunk_size is None:
            self.action_chunk_size = 16
        if self.view_point is None:
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
                if self.action_path is None:
                    raise ValueError(f"'action_path' is required for model_mode={mode.value!r}")
            case ModelMode.INVERSE_DYNAMICS | ModelMode.POLICY:
                pass
            case _:
                assert_never(mode)

        if self.action_path and "://" in self.action_path:
            raise ValueError("Must call `download()` before building action data")


_ReasonerTemperature = Annotated[float, pydantic.Field(gt=0, le=100)]
_ReasonerTopP = Annotated[float, pydantic.Field(gt=0, le=1)]
_ReasonerRepetitionPenalty = Annotated[float, pydantic.Field(gt=0)]


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


class ReasonerDataOverrides(OverridesBase):
    """Reasoner overrides for ``model_mode='reasoner'``. ``vision_path`` (if set) is
    used as the conditioning image; the VLM processor handles preprocessing."""

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

    def _build_reasoner_data(self, model_config: "OmniMoTModelConfig", sample_meta: SampleMeta):
        if not sample_meta.model_mode.is_reasoner:
            return
        self = cast("SampleDataOverrides", self)
        if not self.prompt.strip():
            raise ValueError("Reasoner inference requires a non-empty 'prompt'.")


class _TransferDataBase:
    @property
    def transfer_hints(self) -> dict[TransferHintKey, TransferOverrides | TransferArgs]:
        # Iteration order is `TransferHintKey` enum order, not JSON-key order — keep this
        # deterministic so the model sees a stable [ctrl_1, ..., ctrl_N] sequence.
        return {key: getattr(self, key.value) for key in TransferHintKey if getattr(self, key.value) is not None}


class TransferDataArgs(ArgsBase, _TransferDataBase):
    control_guidance: ControlGuidance = 1.0
    control_guidance_interval: GuidanceInterval | None = None
    edge: EdgeTransferArgs | None = None
    blur: BlurTransferArgs | None = None
    depth: TransferArgs | None = None
    seg: TransferArgs | None = None
    wsm: TransferArgs | None = None
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
    }
    _TRANSFER_HINT_DEFAULTS: ClassVar[dict[TransferHintKey, dict[str, Any]]] = {
        TransferHintKey.EDGE: {"preset_edge_threshold": PresetEdgeThreshold.MEDIUM},
        TransferHintKey.BLUR: {"preset_blur_strength": PresetBlurStrength.MEDIUM},
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

        for field, default in self._TRANSFER_SAMPLE_DEFAULTS.items():
            if getattr(self, field) is None:
                setattr(self, field, default)

        for hint_key, config in hints.items():
            for field, default in self._TRANSFER_HINT_DEFAULTS.get(hint_key, {}).items():
                if getattr(config, field) is None:
                    setattr(config, field, default)

            if self.vision_path is None and config.control_path is None:
                raise ValueError(
                    f"transfer inference requires 'vision_path' or a pre-computed 'control_path' (hint: {hint_key})"
                )

        if len(hints) == 1:
            hint_key = next(iter(hints))
            for field, value in self._TRANSFER_DEFAULTS[hint_key].items():
                if user_fields is None or field not in user_fields:
                    setattr(self, field, value)


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
        return SampleMeta(
            model_mode=mode,
            vision_mode=VisionMode.from_model_mode(mode),
            condition_vision_mode=self.condition_vision_mode,
        )


class SampleDataArgs(
    _SampleDataBase,
    TextDataArgs,
    VisionDataArgs,
    SoundDataArgs,
    ActionDataArgs,
    ReasonerDataArgs,
    TransferDataArgs,
):
    model_mode: ModelMode


class SampleDataOverrides(
    _SampleDataBase,
    TextDataOverrides,
    VisionDataOverrides,
    SoundDataOverrides,
    ActionDataOverrides,
    ReasonerDataOverrides,
    TransferDataOverrides,
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
    }

    _RESOLUTION_SHIFT_DEFAULTS: ClassVar[dict[(ModelSize, Resolution), float]] = {
        ("8B", "256"): 3.0,
        ("8B", "480"): 5.0,
        ("8B", "720"): 10.0,
        ("8B", "768"): 10.0,
        ("32B", "256"): 5.0,
        ("32B", "480"): 5.0,
        ("32B", "720"): 5.0,
        ("32B", "768"): 5.0,
    }

    @override
    def build_sample(self, *, model_config: Any) -> OmniSampleArgs:
        model_config = cast("OmniMoTModelConfig", model_config)
        sample_meta = self.sample_meta

        # Apply per-modality defaults from JSON config files.
        # User-provided values take precedence over JSON defaults.
        if self.defaults_file is not None:
            defaults = json.loads(self.defaults_file.read_text())
        else:
            defaults = _load_modality_defaults(sample_meta.model_mode)
        overrides = self.model_dump(exclude_none=True)
        shift_configured = "shift" in overrides or defaults.get("shift") is not None
        user_fields = frozenset(overrides)
        merged_data = _deep_merge(defaults, overrides)
        merged_data = {k: v for k, v in merged_data.items() if k in type(self).model_fields}
        merged = type(self).model_validate(merged_data)

        self.__dict__.update(merged.__dict__)
        self.model_mode = sample_meta.model_mode

        self._build_sample()
        self._build_sampling(model_config=model_config, sample_meta=sample_meta)
        self._build_text_data(model_config=model_config, sample_meta=sample_meta)
        self._build_vision_data(model_config=model_config, sample_meta=sample_meta)

        self._build_prompt_upsampling(model_config=model_config)

        self._build_action_data(model_config=model_config, sample_meta=sample_meta)

        self._build_sound_data(model_config=model_config, sample_meta=sample_meta)

        self._build_reasoner_data(model_config=model_config, sample_meta=sample_meta)

        self._build_transfer_data(
            model_config=model_config, sample_meta=sample_meta, user_fields=user_fields
        )

        if not shift_configured and not sample_meta.model_mode.is_reasoner:
            model_size = self._VLM_MODEL_SIZE[model_config.vlm_config.model_name]
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
}
DEFAULT_CHECKPOINT_NAME = "Cosmos3-Nano"
DEFAULT_CHECKPOINT = _CHECKPOINTS[DEFAULT_CHECKPOINT_NAME]

MAX_CP_SIZE = 32
CpSize = Annotated[int, pydantic.Field(ge=1, le=MAX_CP_SIZE)]


class OmniSetupArgs(SetupArgs):
    variant: Suppress[Literal["omni"]] = "omni"
    """Discriminator."""

    # pyrefly: ignore[bad-override]
    sample_overrides: OmniSampleOverrides

    sampler: Literal["unipc", "edm"]

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

    # Override defaults
    dp_replicate_size: pydantic.NonNegativeInt = 0
    dp_shard_size: pydantic.NonNegativeInt = 0
    cp_size: CpSize | Literal[0] = 0
    cfgp_size: CfgpSize | Literal[0] = 0

    use_cuda_graphs: bool = False

    compiled_region: Literal["all", "language"] = "all"
    # Unsupported
    tp_size: Suppress[pydantic.NonNegativeInt] = 1

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

    @override
    def build_setup(
        self, world_size: int | None = None, local_world_size: int | None = None, device_memory_bytes: int | None = None
    ) -> OmniSetupArgs:
        self._build_setup()
        self._build_checkpoint(checkpoints=self.CHECKPOINTS)
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
