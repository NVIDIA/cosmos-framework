# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import hashlib
import json
import pickle
from collections.abc import Callable, Generator, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Sequence, TypeVar, cast, override

import attrs
import cattrs
import cattrs.preconf.json
import safetensors.torch
import torch
from PIL import Image
from torch.utils._pytree import tree_map_only
from torch.utils.data import Dataset
from typing_extensions import Self

from cosmos_framework.inference.args import (
    ModelMode,
    NegativeMetadataMode,
    OmniSampleArgs,
    OmniSetupArgs,
)
from cosmos_framework.inference.common.args import (
    CheckpointType,
    ConfigFileType,
    ParallelismArgs,
    SampleArgs,
    SampleOutput,
    SampleOutputs,
    SetupArgs,
)
from cosmos_framework.inference.common.inference import Inference, sync_distributed_errors
from cosmos_framework.inference.common.init import get_rank, get_world_size
from cosmos_framework.inference.model import Cosmos3OmniConfig, Cosmos3OmniModel
from cosmos_framework.inference.vision import (
    build_conditioned_video_batch,
    build_image_edit_batch,
    load_conditioning_image,
    load_conditioning_video,
    load_prompt_upsampling_image,
    pil_to_conditioning_frames,
    resize_pil_image,
)
from cosmos_framework.utils import log
from cosmos_framework.tools.visualize.video import save_img_or_video
from cosmos_framework.configs.base.defaults.compile import CompileConfig
from cosmos_framework.configs.base.defaults.parallelism import ParallelismConfig
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
from cosmos_framework.model.generator.reasoner.qwen3_vl.utils import _SYSTEM_PROMPT_IMAGE_EDITING
from cosmos_framework.model.generator.upsampler.prompts import is_upsampled_prompt

if TYPE_CHECKING:
    from cosmos_framework.configs.base.defaults.model_config import OmniMoTModelConfig

UpsampleTask = Literal["t2i", "t2v", "i2v"]


_BatchItem = TypeVar("_BatchItem")


def _iter_packed_batches(
    items: Iterable[_BatchItem],
    get_sample_args: Callable[[_BatchItem], OmniSampleArgs],
    model: OmniMoTModel,
    max_model_len: int | None,
    max_num_seqs: int | None,
) -> Generator[list[_BatchItem]]:
    """Greedily pack a stream of items into batches under a single budget.

    Walks ``items`` once, in order, and accumulates them into the current
    batch. Before adding an item, the helper checks whether doing so would
    exceed the configured budget; if it would (and the current batch is
    non-empty), the current batch is emitted and a fresh one is started with
    the item. If a single item exceeds the budget on its own, it still gets
    emitted in a batch by itself (no item is ever dropped). Order is preserved.

    The helper is item-agnostic: it never inspects ``items`` beyond passing
    each one to ``get_sample_args`` to get the ``OmniSampleArgs`` used for
    token counting. Callers can therefore batch indices, ``(args, data)``
    pairs, or any other value type, and the yielded batches keep the same
    element type.

    Exactly one of ``max_model_len`` and ``max_num_seqs`` must be provided —
    that selects which budget is enforced:

    - ``max_model_len``: token budget. Each item's token count is computed
      via ``_compute_num_tokens_for_sample(sa, model)``.
    - ``max_num_seqs``: sequence-count budget. Each item counts as one slot.

    Streaming: at any time only the current in-flight batch is held in
    memory; ``items`` is consumed lazily.

    Args:
        items: Iterable of arbitrary items (typed as ``_BatchItem``) to pack.
            Consumed exactly once, in order.
        get_sample_args: Callable mapping each item to its ``OmniSampleArgs``.
            Used for token counting and for the per-item invariants below.
        model: The model, forwarded to ``_compute_num_tokens_for_sample`` for
            token-budget mode. Unused (but still required) for sequence-count
            mode.
        max_model_len: Token budget per batch. Mutually exclusive with
            ``max_num_seqs``; exactly one must be set (the other ``None``).
        max_num_seqs: Sequence-count budget per batch. Mutually exclusive
            with ``max_model_len``; exactly one must be set (the other
            ``None``).

    Yields:
        ``list[_BatchItem]``: a non-empty batch of items, in input order.
        The yielded list is owned by the caller — the helper drops its
        reference after yielding and starts fresh, so callers are free to
        mutate or store it.

    Raises:
        AssertionError: If neither or both of ``max_model_len`` /
            ``max_num_seqs`` are provided, if ``get_sample_args`` returns a
            non-``OmniSampleArgs``, or if any item's ``num_outputs != 1``.
            Callers must seed-expand multi-output samples (e.g. via
            ``_finalize_sample_args_list``) before passing them in.
    """
    assert max_model_len is not None or max_num_seqs is not None, "Either max_model_len or max_num_seqs must be set"
    assert max_model_len is None or max_num_seqs is None, "Either max_model_len or max_num_seqs must be set, not both"

    cur: list[_BatchItem] = []
    running_tokens = 0
    running_seqs = 0

    for item in items:
        sa = get_sample_args(item)
        assert isinstance(sa, OmniSampleArgs)
        assert sa.num_outputs == 1, "num_outputs must be 1"

        if max_model_len is not None:
            num_tokens = _compute_num_tokens_for_sample(sa, model)
            if cur and running_tokens + num_tokens > max_model_len:
                yield cur
                cur = []
                running_tokens = 0
            running_tokens += num_tokens
        else:
            if cur and running_seqs + 1 > max_num_seqs:  # type: ignore[operator]
                yield cur
                cur = []
                running_seqs = 0
            running_seqs += 1

        cur.append(item)

    if cur:
        yield cur


def _fallback_seed(sample_args: OmniSampleArgs) -> int:
    """Stable per-sample fallback seed used when ``sample_args.seed is None``.

    We derive the seed deterministically from the sample's identity
    (``name`` + ``output_dir``) instead of calling ``random.randint``, because
    each rank of a CP / CFG-parallel replica must agree on the seed: the
    sampler uses it to draw the initial noise, and divergent noise across
    ranks corrupts the collective denoising loop.  Independent calls to
    Python's global ``random`` module (which is what the previous code did)
    return different values on different ranks unless the user has separately
    seeded it identically everywhere, which is easy to forget.

    The returned int fits in 31 bits so it's safe to pass to any downstream
    API expecting a non-negative ``int32`` seed.
    """
    identity = f"{sample_args.name}|{sample_args.output_dir}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(identity).digest()[:4], "big") & 0x7FFFFFFF


def _compute_num_tokens_for_sample(sample_args: OmniSampleArgs, model: OmniMoTModel) -> int:
    """Estimate the number of tokens for a single inference sample.

    Follows the counting logic in
    ``JointDataLoader._compute_num_tokens_per_sample`` (vision + text + EOS).
    """
    w, h = sample_args.vision_size
    T = sample_args.num_frames

    spatial_cf = cast(int, model.tokenizer_vision_gen.spatial_compression_factor)
    temporal_cf = cast(int, model.tokenizer_vision_gen.temporal_compression_factor)
    patch_spatial: int = model.config.diffusion_expert_config.patch_spatial

    vae_spatial_downsample = spatial_cf * patch_spatial
    vae_temporal_downsample = temporal_cf

    latent_h = h // vae_spatial_downsample
    latent_w = w // vae_spatial_downsample
    latent_t = 1 + (T - 1) // vae_temporal_downsample
    num_vision_tokens = latent_h * latent_w * latent_t


    # small compared to vision tokens, so we can ignore them for now.

    return num_vision_tokens


def _infer_native_prompt_upsampling_tasks(
    data_batch: dict[str, Any],
    sample_args_list: Sequence[OmniSampleArgs],
    model: OmniMoTModel,
) -> list[UpsampleTask | None]:
    """Return the per-sample native V4.2 prompt-upsample task list.

    Each entry corresponds positionally to ``sample_args_list[i]`` and is
    either the canonical V4.2 task (``"t2v"`` / ``"t2i"`` / ``"i2v"``) when
    native upsampling should run for that sample, or ``None`` when it should
    be skipped.  Per-sample reasons for ``None`` include:

    * Action-only samples (``model_mode.is_action``) — the V4.2 templates
      are vision-caption only.
    * Samples that did not opt in via ``native_prompt_upsampling=True``
      (see ``OmniSampleOverrides.build_sample`` for how this flag is
      derived from ``prompt_upsampling`` × ``prompt_upsampling_applied``).
    * Samples whose prompt already looks like upsampler output (fenced or
      bare JSON; see
      :func:`projects.cosmos3.vfm.upsampler.prompts.is_upsampled_prompt`).
    * Batch-wide modality ambiguity — ``data_batch`` tensors are stacked
      along the batch dim, so the image/video keys are inherently
      batch-uniform; if both keys are present (mixed batch) or neither is
      (no vision input), every sample is marked ``None``.
    * Image-editing samples (image input *plus* per-sample vision
      conditioning) — not yet supported by the V4.2 upsampler.
    * I2V samples when ``data_batch`` lacks the VLM-ready
      ``_prompt_upsampling_images`` side channel.
    * I2V samples when the reasoner LM was loaded with
      ``include_visual=False`` (the ViT is required to encode the
      conditioning frame for the V4.2 i2v template).

    Mixed batches — i.e. some samples returning a task and others ``None``,
    or different samples returning different tasks — are *allowed* by this
    function; the caller is responsible for routing them to the model.
    """
    n_samples = len(sample_args_list)

    # Batch-wide modality presence: ``data_batch`` tensors are stacked
    # along the batch dim, so the image/video keys are inherently
    # batch-uniform.  If both keys are present (mixed modality) or
    # neither is (no vision input), every sample is ambiguous.
    has_image = model.input_image_key in data_batch
    has_video = model.input_video_key in data_batch
    if has_image == has_video:
        return [None] * n_samples

    plans = data_batch.get("sequence_plan") or [None] * n_samples
    has_upsampling_images = "_prompt_upsampling_images" in data_batch
    # The reasoner's ViT is constructed only when the MoT wrapper was
    # built with ``include_visual=True`` (see ``Qwen3VLTextForCausalLM.__init__``
    # and friends in ``unified_mot``).  Without it, the i2v branch in
    # ``_impl_generate_reasoner_text`` raises ``ValueError`` at the
    # ``hasattr(causal_lm, "visual")`` gate — fall back to no-op
    # upsampling so the (already-captioned) prompt flows to diffusion
    # unchanged.  Walk the ``model.net.language_model.visual`` chain
    # defensively with ``getattr(..., None)`` so test mocks that don't
    # set up the full reasoner-LM attribute chain (e.g.
    # ``tests/test_eval_model.py``'s mock for the
    # ``EvalModel → OmniInference.generate_batch`` boundary) gracefully
    # land on ``has_visual_tower = False`` instead of raising
    # ``AttributeError`` here.
    language_model = getattr(getattr(model, "net", None), "language_model", None)
    has_visual_tower = hasattr(language_model, "visual")

    results: list[UpsampleTask | None] = []
    for i, sample_args in enumerate(sample_args_list):
        if sample_args.model_mode.is_action:
            results.append(None)
            continue
        if not sample_args.native_prompt_upsampling:
            results.append(None)
            continue
        # Content-based per-sample check: skip native prompt upsampling
        # when the prompt already looks like the V4.2 upsampler's output
        # (fenced ``json`` payload or bare JSON object).  Re-running
        # upsampling on a JSON-shaped prompt would corrupt it.
        if is_upsampled_prompt(sample_args.prompt):
            results.append(None)
            continue
        plan = plans[i] if i < len(plans) else None
        sample_has_conditioning = bool(getattr(plan, "condition_frame_indexes_vision", []))
        if has_image:
            # Image editing (image input + vision conditioning) is not
            # yet supported by the V4.2 upsampler templates.
            results.append(None if sample_has_conditioning else "t2i")
            continue
        # Video path.
        if not sample_has_conditioning:
            results.append("t2v")
            continue
        # I2V: requires both the VLM-ready side-channel images and a
        # reasoner with a visual tower.
        if not has_upsampling_images:
            raise ValueError(
                "I2V prompt upsampling requires '_prompt_upsampling_images' with one VLM-ready image per caption."
            )
        if not has_visual_tower:
            raise ValueError(
                "I2V prompt upsampling requires the reasoner LM to have a visual tower (include_visual=True)"
            )
        results.append("i2v")

    return results


def _format_prompt_with_template(
    prompt: str,
    *,
    fps: int,
    num_frames: int,
    duration_template: str | None,
    resolution_template: str | None,
    h: int,
    w: int,
    force_duration_template: bool = False,
) -> str:
    """Append duration/fps and resolution metadata to a prompt."""
    prompt = prompt.strip()
    if duration_template is not None and (num_frames > 1 or force_duration_template):
        duration = num_frames / fps
        dur_text = duration_template.format(duration=duration, fps=fps)
        prompt = prompt.rstrip(".") + ". " + dur_text

    prompt = prompt.strip()
    if resolution_template is not None:
        res_text = resolution_template.format(height=h, width=w)
        prompt = prompt.rstrip(".") + ". " + res_text

    return prompt


def _parse_json_object_prompt(prompt: str) -> dict | None:
    """Return the parsed dict iff ``prompt`` is a JSON object string; else ``None``.

    JSON arrays / numbers / strings / nulls are NOT considered "JSON-object
    prompts" and return ``None`` so they continue down the plain-text path.
    """
    try:
        obj = json.loads(prompt)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def _format_json_prompt_with_template(
    prompt_obj: dict,
    *,
    fps: int,
    num_frames: int,
    aspect_ratio: str | None,
    h: int,
    w: int,
    include_temporal_metadata: bool,
) -> str:
    """JSON-prompt counterpart to ``_format_prompt_with_template``.

    Injects structured metadata fields directly into the parsed prompt object,
    matching the training-time augmentors so the tokenizer sees the exact
    schema the model was trained on:

        - ``ResolutionTextInfo``        -> ``resolution: {"H": int, "W": int}``, ``aspect_ratio: str``
        - ``DurationFPSTextTimeStamps`` -> ``duration: "<int>s"``, ``fps: float`` for video samples only

    Always overwrites existing values for these keys, mirroring the augmentors'
    ``dict.update(...)`` semantics: the actual generation specs are the source
    of truth, regardless of what the input prompt may have specified.
    """
    metadata: dict[str, Any] = {}
    if include_temporal_metadata:
        duration_seconds = int(num_frames / fps) if fps > 0 else 0
        metadata.update(
            {
                "duration": f"{duration_seconds}s",
                "fps": float(fps),
            }
        )
    else:
        prompt_obj.pop("duration", None)
        prompt_obj.pop("fps", None)
    metadata["resolution"] = {"H": int(h), "W": int(w)}
    if aspect_ratio is not None:
        metadata["aspect_ratio"] = aspect_ratio

    prompt_obj.update(metadata)
    log.debug(f"Injected JSON-prompt metadata fields: {sorted(metadata.keys())}")

    return json.dumps(prompt_obj)


def _get_prompt_sample_data(sample_args: OmniSampleArgs, model: OmniMoTModel, *, h: int, w: int, device: Any) -> dict:
    duration_template = sample_args.duration_template
    inverse_duration_template = sample_args.inverse_duration_template
    prompt_obj = _parse_json_object_prompt(sample_args.prompt)
    prompt_is_json = prompt_obj is not None

    if prompt_is_json:
        assert prompt_obj is not None  # type-narrowing
        prompt = _format_json_prompt_with_template(
            prompt_obj,
            fps=sample_args.fps,
            num_frames=sample_args.num_frames,
            aspect_ratio=sample_args.aspect_ratio,
            h=h,
            w=w,
            include_temporal_metadata=sample_args.num_frames > 1,
        )
    elif not sample_args.native_prompt_upsampling:
        prompt = _format_prompt_with_template(
            sample_args.prompt,
            fps=sample_args.fps,
            num_frames=sample_args.num_frames,
            duration_template=duration_template,
            resolution_template=sample_args.resolution_template,
            h=h,
            w=w,
        )
    else:
        # If native prompt upsampling is enabled, the duration and resolution
        # metadata are added into the upsampled JSON prompt directly.
        prompt = sample_args.prompt.strip()
    out = {
        model.input_caption_key: [prompt] * sample_args.num_outputs,
    }

    negative_prompt = sample_args.negative_prompt
    if sample_args.negative_metadata_mode == NegativeMetadataMode.SAME:
        negative_prompt = (
            _format_prompt_with_template(
                negative_prompt if negative_prompt is not None else "",
                fps=sample_args.fps,
                num_frames=sample_args.num_frames,
                duration_template=duration_template,
                resolution_template=sample_args.resolution_template,
                h=h,
                w=w,
            )
            .lstrip(".")
            .strip()
        )
    elif sample_args.negative_metadata_mode == NegativeMetadataMode.INVERSE:
        negative_prompt = (
            _format_prompt_with_template(
                negative_prompt if negative_prompt is not None else "",
                fps=sample_args.fps,
                num_frames=sample_args.num_frames,
                duration_template=inverse_duration_template,
                resolution_template=sample_args.inverse_resolution_template,
                h=h,
                w=w,
                force_duration_template=True,
            )
            .lstrip(".")
            .strip()
        )

    if negative_prompt:
        neg_key = "neg_" + model.input_caption_key
        out[neg_key] = [negative_prompt] * sample_args.num_outputs

    return out


def _get_reasoner_sample_data(sample_args: OmniSampleArgs, model: OmniMoTModel) -> dict[str, Any]:
    """Sample batch for reasoner text generation: prompt + optional conditioning image."""
    image: Image.Image | None = None
    if sample_args.vision_path is not None:
        image = Image.open(sample_args.vision_path).convert("RGB")
    return {
        model.input_caption_key: [sample_args.prompt],
        "reasoner_images": [image],
    }


def _get_image_edit_sample_data(
    sample_args: OmniSampleArgs,
    model: OmniMoTModel,
    *,
    device: Any,
) -> dict:
    """Create a sample batch for image-edit generation."""
    assert sample_args.vision_path is not None
    if sample_args.resolution and sample_args.aspect_ratio:
        w, h = sample_args.vision_size
        conditioning_frames = load_conditioning_image(Path(sample_args.vision_path), target_h=h, target_w=w)
    else:
        pil_img = Image.open(sample_args.vision_path).convert("RGB")
        pil_img = resize_pil_image(pil_img, max_size=512, padding_constant=32)
        conditioning_frames, h, w = pil_to_conditioning_frames(pil_img)

    conditioning_frames = conditioning_frames.to(device=device)
    batch = build_image_edit_batch(conditioning_frames, h=h, w=w, batch_size=sample_args.num_outputs)
    batch["system_prompt"] = _SYSTEM_PROMPT_IMAGE_EDITING
    batch |= _get_prompt_sample_data(sample_args, model, h=h, w=w, device=device)
    return batch


def get_sample_data(
    sample_args: OmniSampleArgs,
    model: OmniMoTModel,
    *,
    device: Any = "cuda",
) -> dict:
    """Create a sample batch for generation."""
    if sample_args.model_mode.is_reasoner:
        return _get_reasoner_sample_data(sample_args, model)

    if sample_args.transfer_hints:
        return {}

    if sample_args.model_mode.is_action:
        from cosmos_framework.inference.action import get_action_sample_data

        assert sample_args.vision_path is not None
        return get_action_sample_data(
            model_config=model,
            batch_size=sample_args.num_outputs,
            prompt=sample_args.prompt,
            vision_path=sample_args.vision_path,
            model_mode=sample_args.model_mode,
            action_path=sample_args.action_path,
            domain_name=sample_args.domain_name,
            view_point=sample_args.view_point,
            resolution=str(sample_args.image_size),
            action_chunk_size=sample_args.action_chunk_size,
            max_action_dim=model.config.max_action_dim,
            fps=sample_args.fps,
            device=device,
        )


    if sample_args.model_mode == ModelMode.IMAGE2IMAGE:
        return _get_image_edit_sample_data(sample_args, model, device=device)

    w, h = sample_args.vision_size
    if sample_args.num_frames == 1:
        input_vision_key = model.input_image_key
    else:
        input_vision_key = model.input_video_key

    with torch.device(device):
        prompt_upsampling_image: Image.Image | None = None
        match sample_args.condition_vision_mode:
            case "image":
                assert sample_args.vision_path is not None
                vision_path = Path(sample_args.vision_path)
                conditioning_frames = load_conditioning_image(vision_path, target_h=h, target_w=w)
                prompt_upsampling_image = load_prompt_upsampling_image(vision_path, target_h=h, target_w=w)
            case "video":
                assert sample_args.vision_path is not None
                assert sample_args.condition_frame_indexes_vision is not None
                num_condition_latent_frames = max(sample_args.condition_frame_indexes_vision) + 1
                max_frames = model.tokenizer_vision_gen.get_pixel_num_frames(num_condition_latent_frames)
                conditioning_frames = load_conditioning_video(
                    Path(sample_args.vision_path),
                    target_h=h,
                    target_w=w,
                    max_frames=max_frames,
                    keep=sample_args.condition_video_keep or "first",
                )
            case _:
                conditioning_frames = None

        if conditioning_frames is not None:
            assert sample_args.condition_frame_indexes_vision is not None
            conditioned = build_conditioned_video_batch(
                conditioning_frames,
                condition_frames_vision=sample_args.condition_frame_indexes_vision,
                w=w,
                h=h,
                num_frames=sample_args.num_frames,
                fps=sample_args.fps,
                batch_size=sample_args.num_outputs,
            )
            # Keep the list form (rather than ``torch.cat``ing into a single
            # tensor) so this branch has the *same Python type* as the
            # unconditioned branch below.  ``_merge_data_batches`` picks its
            # branch from ``values[0]``: when one batch is a Tensor and
            # another a list it would silently iterate the tensor by dim 0
            # via ``for item in v``, producing wrong shapes.  Emitting a list
            # on both paths eliminates the inconsistency at the source.
            video_tensor = [t.to(device=device) for t in conditioned["video"]]  # list of [1,3,T,H,W]
            sequence_plan = conditioned["sequence_plan"]
        else:
            video_tensor = [
                torch.zeros(1, 3, sample_args.num_frames, h, w) for _ in range(sample_args.num_outputs)
            ]  # list of [1,3,T,H,W]
            sequence_plan = None

        out: dict = {
            input_vision_key: video_tensor,
            "image_size": [
                torch.tensor([[h, w, h, w]], dtype=torch.float32) for _ in range(sample_args.num_outputs)
            ],  # list of [1,4]
            "t5_text_embeddings": torch.randn(sample_args.num_outputs, 512, 1024, dtype=torch.bfloat16),  # [B,512,1024]
            "fps": torch.full((sample_args.num_outputs,), float(sample_args.fps)),  # [B]
            "conditioning_fps": torch.full((sample_args.num_outputs,), float(sample_args.fps)),  # [B]
            "num_frames": torch.full((sample_args.num_outputs,), sample_args.num_frames),  # [B]
            "is_preprocessed": True,
        }
        if sequence_plan is not None:
            out["sequence_plan"] = sequence_plan

        out |= _get_prompt_sample_data(sample_args, model, w=w, h=h, device=device)
        if prompt_upsampling_image is not None and sequence_plan is not None:
            out["_prompt_upsampling_images"] = [prompt_upsampling_image.copy() for _ in out[model.input_caption_key]]

        if sample_args.enable_sound:
            from cosmos_framework.inference.sound import (
                create_placeholder_audio,
                get_audio_tokenizer_info,
                inject_sound_into_batch,
                load_conditioning_audio,
            )

            audio_info = get_audio_tokenizer_info(model)
            if not audio_info.has_sound:
                raise ValueError("enable_sound=True but model has no sound tokenizer")

            condition_sound = sample_args.sound_path is not None
            if condition_sound:
                num_samples = int(sample_args.num_frames / sample_args.fps * audio_info.sample_rate)
                audio = load_conditioning_audio(
                    Path(sample_args.sound_path),
                    sample_rate=audio_info.sample_rate,
                    audio_channels=getattr(audio_info.tokenizer, "audio_channels", 2),
                    num_samples=num_samples,
                )
            else:
                audio = create_placeholder_audio(
                    num_frames=sample_args.num_frames,
                    conditioning_fps=sample_args.fps,
                    audio_info=audio_info,
                )
            inject_sound_into_batch(out, audio, model, condition_sound=condition_sound)

        return out


def _merge_data_batches(batches: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge per-sample data dicts into a single batched dict.

    Values that are lists are concatenated. Tensors with a batch dimension are
    concatenated along dim 0. Scalar/bool values are taken from the first
    batch (and must be equal across all batches).

    **Aliasing contract.** For a single-batch input, the returned dict is
    ``batches[0]`` itself (no copy) — this avoids an unnecessary
    ``torch.cat`` / list rebuild on the hot path. For a multi-batch input,
    the returned dict is freshly allocated (list values via list-comp,
    tensor values via ``torch.cat``).

    The singleton fast path is safe to consume because the in-tree
    producers already hand this function dicts that the caller fully owns:

    - ``create_batches_from_dataset`` shallow-copies the per-sample dict
      once per seed-expanded sibling in ``_expanded_samples``, so the
      ``batches[0]`` returned here aliases only that sibling's copy — a
      caller-side top-level mutation cannot leak into sibling samples.
    - ``_finalize_data_batch`` shallow-copies its input before applying
      any rename / unbind, so the ``generate_batch`` path is also safe.

    If you add a new producer that hands shared dict references in, copy
    at the producer (matching ``_expanded_samples``) rather than removing
    this fast path — concatenating a length-1 list / single tensor on
    every singleton batch is meaningful overhead on the hot path.

    Args:
        batches (list[dict[str, Any]]): List of data batches to merge. Must be
            non-empty.

    Returns:
        dict[str, Any]: Merged data batch.

    Raises:
        ValueError: If the batches have different keys, if values for the same
            key have inconsistent Python types across batches, or if
            scalar/bool values are not equal across all batches.
    """
    if len(batches) == 1:
        return batches[0]

    # First ensure that all batches have the same keys.
    reference_keys = set(batches[0].keys())
    for i, batch in enumerate(batches[1:], start=1):
        if set(batch.keys()) != reference_keys:
            raise ValueError(f"Batch {i} keys {set(batch.keys())} differ from batch 0 keys {reference_keys}")

    # Then merge the batches.
    merged: dict[str, Any] = {}
    keys = batches[0].keys()
    for key in keys:
        values = [b[key] for b in batches if key in b]
        first = values[0]
        # Without this guard, mixing ``list`` and ``Tensor`` values for the
        # same key silently produces wrong shapes: the ``isinstance(first, list)``
        # branch below uses ``for item in v`` which is also valid on Tensors
        # (it iterates dim 0), so subsequent Tensor values get unpacked into
        # the merged list as slices rather than triggering a clear error.
        if not all(isinstance(v, type(first)) for v in values):
            raise ValueError(
                f"Inconsistent value types for key '{key}': "
                f"{[type(v).__name__ for v in values]}. "
                "Normalize the type at the source (e.g. always emit a list[Tensor]) "
                "before calling _merge_data_batches."
            )
        if isinstance(first, list):
            merged[key] = [item for v in values for item in v]
        elif isinstance(first, torch.Tensor):
            if first.ndim <= 0:
                raise ValueError("Tensor must have at least one (batch) dimension")
            merged[key] = torch.cat(values, dim=0)
        else:
            if not all(v == values[0] for v in values):
                raise ValueError(f"Key {key} values are not the same: {values}")
            merged[key] = first
    return merged


def _finalize_sample_args_list(sample_args_list: Sequence[OmniSampleArgs]) -> list[OmniSampleArgs]:
    """Validate and seed-expand a list of sample args.

    Behavior is per-sample, so adding samples to (or removing them from) the
    input list never changes how the *other* samples are handled:

    - ``num_outputs == 1`` samples are passed through unchanged — the
      original ``sample_args`` reference (and in particular its
      ``output_dir``, which may be ``None`` for e.g. padding samples) is
      preserved.
    - ``num_outputs > 1`` samples are expanded into ``num_outputs`` fresh
      deep-copies; each copy gets ``num_outputs = 1``, a unique
      ``output_dir = original / "{i}"``, and a per-replica seed
      (``original_seed + i`` if a base seed was provided, else ``None``).

    Args:
        sample_args_list: Input samples; may freely mix ``num_outputs == 1``
            and ``num_outputs > 1`` entries.

    Returns:
        New list of ``OmniSampleArgs`` in input order.  Single-output entries
        are the original references; multi-output entries are fresh
        deep-copies.

    Raises:
        ValueError: If any sample has ``num_outputs > 1`` but no
            ``output_dir`` (we can't form per-output subdirectories without
            one).
    """
    finalized_sample_args_list: list[OmniSampleArgs] = []
    for sample_args in sample_args_list:
        if sample_args.num_outputs == 1:
            finalized_sample_args_list.append(sample_args)
            continue

        seed = sample_args.seed
        num_outputs = sample_args.num_outputs
        output_dir = sample_args.output_dir

        if output_dir is None:
            raise ValueError(
                f"num_outputs={num_outputs} requires output_dir to be set "
                f"(sample name={sample_args.name!r}); cannot create "
                "per-output subdirectories"
            )

        for i in range(num_outputs):
            sample_args_i = sample_args.model_copy(deep=True)
            sample_args_i.seed = (seed + i) if seed is not None else None
            sample_args_i.num_outputs = 1
            sample_args_i.output_dir = output_dir / f"{i}"
            finalized_sample_args_list.append(sample_args_i)

    return finalized_sample_args_list


def create_batches_from_dataset(
    samples: Iterable[tuple[OmniSampleArgs, dict[str, Any]]],
    model: OmniMoTModel,
    *,
    max_num_seqs: int | None = None,
    max_model_len: int | None = None,
) -> Generator[tuple[list[OmniSampleArgs], dict[str, Any]]]:
    """Create batches from pre-loaded (sample_args, data_batch) pairs.

    Reuses the same token-count / sample-count batching logic as
    ``OmniInference.create_batches``, but works with dataset iterators that
    already provide data. Samples with ``num_outputs > 1`` are multi-seed
    expanded via ``_finalize_sample_args_list``; callers that want only a
    subset of samples expanded should set ``num_outputs`` accordingly before
    yielding each sample.

    Args:
        samples: Iterable of ``(OmniSampleArgs, data_batch)`` pairs.
        model: The model, used for token counting and seed expansion.
        max_num_seqs: Maximum number of sequences per batch.
        max_model_len: Maximum total tokens per batch.
            Exactly one of ``max_num_seqs`` or ``max_model_len`` must be set.

    Yields:
        ``(sample_args_list, merged_data_batch, per_sample_data_batches)`` tuples.
        ``per_sample_data_batches`` is the list of individual data dicts before
        merging, useful when callers need per-sample post-processing.
    """

    # Tensor keys whose non-batch dims may differ across samples and must be
    # promoted to a length-1 ``list[Tensor]`` so ``_merge_data_batches`` can
    # flatten them via list-extension instead of failing in ``torch.cat``.
    # Also, include domain_id so that it is produced as a list[Tensor].
    _VARIABLE_SHAPE_TENSOR_KEYS = {"video", "action", "domain_id"}

    def _prepare_for_merge(db: dict[str, Any]) -> dict[str, Any]:
        """Reshape a per-sample data dict so ``_merge_data_batches`` can combine it.

        Returns a shallow copy of ``db`` in which select keys are wrapped in
        length-1 lists so ``_merge_data_batches`` routes them through its
        list-concatenation branch (flattening one list per sample into a
        single batch list) instead of its tensor-``torch.cat`` branch.  All
        other values are passed through by reference.

        - **``"video"`` / ``"action"``** (see ``_VARIABLE_SHAPE_TENSOR_KEYS``).
          Input is expected to be a ``[1, *variable_dims]`` tensor — the
          shape produced by ``_collate_sample``'s ``unsqueeze(0)`` — and is
          converted to a length-1 ``list[Tensor]`` of shape
          ``[*variable_dims]``.  Without this, samples that share a chunk
          but have different non-batch dims (e.g. videos at 544x736 vs
          640x640, or actions of length 148 vs 104 from variable-length
          clips in the camera-480 dataset) would fail ``torch.cat``.
          Downstream consumers (``pack_action``, video tokenization) read
          these as per-sample lists anyway, so the list form also matches
          the downstream contract.
        - **``"domain_id"``**.  Each sample carries its own scalar domain
          id that must remain individually addressable after merging (not
          concatenated into a single batch tensor) so downstream code can
          dispatch per sample.  Input is expected to be a 0-D tensor and is
          wrapped in a length-1 ``list[Tensor]`` *preserving the 0-D
          shape*.  Routing a 0-D tensor through ``torch.cat`` directly
          would otherwise hit ``_merge_data_batches``'s ``ndim <= 0``
          guard.

        Args:
            db: Single sample's data dict, typically straight out of
                ``_collate_sample``.

        Returns:
            A new dict with the rewrites applied; the original ``db`` is
            not mutated.

        Raises:
            ValueError: If a value at one of the special-cased keys has an
                unexpected Python type or tensor dimensionality —
                specifically, a non-``Tensor`` ``video`` / ``action`` /
                ``domain_id``, or a ``domain_id`` whose ``ndim != 0``.
        """
        updated_db: dict[str, Any] = {}
        for key, value in db.items():
            if key in _VARIABLE_SHAPE_TENSOR_KEYS:
                if not isinstance(value, torch.Tensor):
                    raise ValueError(f"Expected {key} to be a tensor, got {type(value)}")
                updated_db[key] = [value.squeeze(0)]
            else:
                updated_db[key] = value
        return updated_db

    def _expanded_samples() -> Generator[tuple[OmniSampleArgs, dict[str, Any]]]:
        # Lazily normalize and seed-expand each input sample, yielding
        # (sample_args, data_batch) pairs ready for budget-batching.
        #
        # Each seed-expanded sibling gets its own *shallow copy* of
        # ``updated_db``.  Without this, all siblings of a single source
        # sample would share the exact same dict reference, and singleton
        # merge batches (the common case here) would alias straight back
        # to that dict — so any caller-side top-level key reassignment on
        # ``merged_batch`` (e.g. ``merged_batch["domain_id"] = [...]``)
        # would leak into the next sibling.  The shallow copy isolates
        # top-level mutations per sibling at negligible cost; tensor /
        # list values are still shared between siblings, which is fine
        # because they're treated as read-only inputs.
        for sa, db in samples:
            updated_db = _prepare_for_merge(db)
            expanded = _finalize_sample_args_list([sa])
            for exp_sa in expanded:
                yield exp_sa, dict(updated_db)

    for batch in _iter_packed_batches(
        items=_expanded_samples(),
        get_sample_args=lambda pair: pair[0],
        model=model,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
    ):
        chunk_args = [pair[0] for pair in batch]
        chunk_data = [pair[1] for pair in batch]
        yield chunk_args, _merge_data_batches(chunk_data)


def _finalize_data_batch(data_batch: dict[str, Any], batch_size: int, model: OmniMoTModel) -> dict[str, Any]:
    """Return a finalized + validated copy of *data_batch*.

    All mutations (key renames, tensor → list unbind) are applied to a fresh
    shallow copy so the caller's input dict is never modified.  This keeps
    the "no aliasing" responsibility localized at the single place where the
    mutation happens, instead of forcing every producer of a data dict (e.g.
    seed-expansion fan-out in ``create_batches_from_dataset``, dummy padding
    batches in ``create_batches``) to defensively copy before handing the
    dict to ``generate_batch``.

    Only the top-level dict structure is copied; tensor / list values inside
    are shared with the input (which is safe because every mutation here is a
    top-level key rename or value reassignment, never an in-place op on the
    value itself).

    Args:
        data_batch: Input data dict.  Not modified.
        batch_size: Expected number of samples in the batch (used for
            validation against the caption-list length).
        model: Model used to resolve the canonical key names.

    Returns:
        New dict with renames + unbinds applied.

    Raises:
        ValueError: If both old and new variants of a renamed key are
            present, or if the post-finalize caption-list length doesn't
            match ``batch_size``.
    """
    data_batch = dict(data_batch)

    for old_key, new_key in [
        ("video", model.input_video_key),
        ("images", model.input_image_key),
        ("ai_caption", model.input_caption_key),
    ]:
        if old_key in data_batch and new_key != old_key:
            if new_key in data_batch:
                raise ValueError(f"Conflicting keys: '{old_key}' and '{new_key}'")
            data_batch[new_key] = data_batch.pop(old_key)

    # Unstack variable length tensors
    _multi_item_keys = {
        "text_token_ids",
        "action",
        model.input_video_key,
        model.input_image_key,
    }
    for key in _multi_item_keys:
        if key in data_batch and isinstance(data_batch[key], torch.Tensor):
            if key == model.input_image_key:
                data_batch[key] = [
                    t.unsqueeze(0).squeeze(2) for t in torch.unbind(data_batch[key])
                ]  # list of [1,C,H,W]
            elif key == model.input_video_key:
                if data_batch.get("is_preprocessed", False):
                    data_batch[key] = [t.unsqueeze(0) for t in torch.unbind(data_batch[key])]  # list of [1,C,T,H,W]
                else:
                    data_batch[key] = list(torch.unbind(data_batch[key]))
            else:
                data_batch[key] = [[t] for t in torch.unbind(data_batch[key])]

    # Validate
    if len(data_batch[model.input_caption_key]) != batch_size:
        raise ValueError(
            f"Data batch length ({len(data_batch[model.input_caption_key])}) does not match batch size ({batch_size})"
        )

    return data_batch


class SampleDataset(Dataset):
    """PyTorch map-style dataset over inference sample args.

    Each item is a ``(SampleArgs, data_dict)`` tuple where the data dict is
    lazily prepared on access via ``__getitem__``.
    """

    def __init__(self, sample_args_list: Sequence[SampleArgs], model: OmniMoTModel) -> None:
        self._sample_args_list = list(sample_args_list)
        self._model = model

    def __len__(self) -> int:
        return len(self._sample_args_list)

    def __getitem__(self, idx: int) -> tuple[SampleArgs, dict[str, Any]]:
        sample_args = self._sample_args_list[idx]
        assert isinstance(sample_args, OmniSampleArgs)
        assert sample_args.output_dir is not None
        data_batch = sample_args.get_data(device="cuda")
        if not data_batch:
            data_batch = get_sample_data(sample_args=sample_args, model=self._model)
        return sample_args, data_batch


@dataclass
class OmniInference(Inference):
    # pyrefly: ignore[bad-override]
    model: OmniMoTModel
    vae_decode_stream: torch.cuda.Stream | None = None

    @property
    def model_config(self) -> "OmniMoTModelConfig":
        return self.model.config

    @classmethod
    def _get_parallelism_config(cls, setup_args: ParallelismArgs) -> ParallelismConfig:
        return ParallelismConfig(
            enable_inference_mode=True,
            data_parallel_shard_degree=setup_args.dp_shard_size,
            context_parallel_shard_degree=setup_args.cp_size,
            cfg_parallel_shard_degree=setup_args.cfgp_size,
        )

    @classmethod
    def _get_compile_config(cls, setup_args: ParallelismArgs) -> CompileConfig:
        return CompileConfig(
            # Translate the flat ``OmniSetupOverrides.use_torch_compile`` public
            # surface into R1's nested ``CompileConfig.enabled`` knob.
            enabled=setup_args.use_torch_compile,
            use_cuda_graphs=setup_args.use_cuda_graphs
            and setup_args.dp_shard_size * setup_args.cp_size * setup_args.cfgp_size == 1,
            compiled_region=setup_args.compiled_region,
            compile_dynamic=setup_args.compile_dynamic,
        )

    @override
    @classmethod
    def _create(cls, setup_args: SetupArgs, **kwargs: Any) -> Self:
        assert isinstance(setup_args, OmniSetupArgs)
        assert setup_args.output_dir is not None

        sampler_override = setup_args.sampler
        parallelism_config = cls._get_parallelism_config(setup_args)
        compile_config = cls._get_compile_config(setup_args)
        if setup_args.checkpoint_type == CheckpointType.DCP and setup_args.config_file_type == ConfigFileType.MODULE:
            from cosmos_framework.inference.common.config import save_config
            from cosmos_framework.utils.generator.model_loader import load_model_from_checkpoint

            if not setup_args.experiment:
                raise ValueError("'experiment' is required")
            if not setup_args.config_file:
                raise ValueError("'config_file' is required")

            Cosmos3OmniModel.before_load_model()
            model, config = load_model_from_checkpoint(
                experiment_name=setup_args.experiment,
                config_file=setup_args.config_file,
                checkpoint_path=setup_args.checkpoint_path,
                credential_path=setup_args.credential_path or None,
                parallelism_config=attrs.asdict(parallelism_config),
                compile_config=attrs.asdict(compile_config),
                load_ema_to_reg=setup_args.use_ema_weights,
                experiment_opts=[
                    *setup_args.experiment_overrides,
                    f"model.config.rectified_flow_inference_config.scheduler_type={sampler_override}",
                ],
                use_cache_checkpoint=setup_args.checkpoint_cache_dir is not None,
                cache_checkpoint_rootdir=str(setup_args.checkpoint_cache_dir or ""),
            )
            model = cast("OmniMoTModel", model)
            Cosmos3OmniModel.after_load_model(model)
            save_config(config, setup_args.output_dir)
        else:
            checkpoint_path = setup_args.download_checkpoint()
            if setup_args.config_file_type == ConfigFileType.MODULE:
                config = None
            else:
                model_dict = setup_args.load_model_config_dict()
                if setup_args.vlm_processor_from_checkpoint:
                    # Source the VLM processor from the loaded checkpoint's own
                    # bundled files instead of the repository hardcoded in the
                    # model config. Drops the redundant base-model download.
                    tokenizer_cfg = model_dict["config"]["vlm_config"]["tokenizer"]
                    tokenizer_cfg.pop("repository", None)
                    tokenizer_cfg.pop("revision", None)
                    tokenizer_cfg.pop("subdir", None)
                    tokenizer_cfg["tokenizer_type"] = str(checkpoint_path)
                # AVAE source: the configured ``avae_path`` when set, else the loaded
                # checkpoint's bundled ``sound_tokenizer/``. The inference-only
                # ``from_checkpoint`` key (default False) forces bundled; pop it so it
                # never reaches AVAEInterface.
                sound_cfg = model_dict["config"].get("sound_tokenizer")
                if sound_cfg is not None:
                    from_checkpoint = sound_cfg.pop("from_checkpoint", False)
                    sound_tokenizer_dir = Path(checkpoint_path) / "sound_tokenizer"
                    if sound_tokenizer_dir.is_dir() and (from_checkpoint or not sound_cfg.get("avae_path")):
                        from cosmos_framework.inference.common.checkpoints import (
                            _AVAE_LEGACY_CKPT_NAME,
                            _materialize_avae_ckpt,
                        )

                        _materialize_avae_ckpt(str(sound_tokenizer_dir))
                        sound_cfg["bucket_name"] = ""
                        sound_cfg["avae_path"] = str(sound_tokenizer_dir / _AVAE_LEGACY_CKPT_NAME)
                config = Cosmos3OmniConfig(model=model_dict)
            model = Cosmos3OmniModel.from_pretrained_dcp(
                checkpoint_path,
                config=config,
                parallelism_config=parallelism_config,
                compile_config=compile_config,
            ).model
            if model.config.rectified_flow_inference_config.scheduler_type != sampler_override:
                model.config.rectified_flow_inference_config.scheduler_type = sampler_override
                model.set_up_scheduler_and_sampler()
                log.debug(f"Sampler overridden to: {sampler_override}")

        vae_decode_stream: torch.cuda.Stream | None = None
        if setup_args.use_separate_pipeline_vision_decode_gpu:
            # The CP/CFGP ranks are partitioned into replica-local groups of size
            # cp_size * cfgp_size. Only the first rank in each group owns separate-VAE
            # decode work. For example, with cp_size=2 and cfgp_size=1, ranks [0,1]
            # form one replica and only rank 0 returns True here.
            replica_size = setup_args.cp_size * setup_args.cfgp_size

            is_vae_output_rank = (replica_size <= 1) or (get_rank() % replica_size == 0)

            vae_device_index = setup_args.cp_size * setup_args.cfgp_size
            if torch.cuda.device_count() <= vae_device_index:
                raise RuntimeError(
                    "--use-separate-pipeline-vision-decode-gpu requires a spare visible local GPU on the "
                    "same node as the decode-owning rank, but the configured local decode GPU index "
                    f"{vae_device_index} is unavailable with only {torch.cuda.device_count()} visible local GPUs."
                )
            if is_vae_output_rank:
                vae_device = torch.device("cuda", vae_device_index)
                inference_device = torch.device("cuda", torch.cuda.current_device())
                vae_decode_stream = torch.cuda.Stream(device=vae_device)
                vae = model.tokenizer_vision_gen.model
                vae.device = str(vae_device)
                vae.model = vae.model.to(device=vae_device)
                vae.scale = tree_map_only(torch.Tensor, lambda tensor: tensor.to(device=vae_device), vae.scale)

                original_encode = model.encode
                original_decode = model.decode

                def encode_on_vae(state: torch.Tensor) -> torch.Tensor:
                    return original_encode(state.to(device=vae_device, non_blocking=True)).to(
                        device=inference_device, non_blocking=True
                    )

                def decode_on_vae(latent: torch.Tensor) -> torch.Tensor:
                    return original_decode(latent.to(device=vae_device, non_blocking=True))

                model.encode = encode_on_vae
                model.decode = decode_on_vae
                log.info(
                    f"Configured vision VAE on device '{vae_device}' while inference remains on '{inference_device}'",
                    rank0_only=False,
                )

        return cls(setup_args=setup_args, model=model, vae_decode_stream=vae_decode_stream, **kwargs)

    @classmethod
    def save_data(
        cls,
        data: dict[str, Any],
        *,
        output_dir: Path,
        output_name: str,
        truncate_action_dim: bool = True,
    ) -> list[Path]:
        """Save data to disk in multiple formats.

        Tensors are saved as ``<output_name>.safetensors``, non-tensor values as
        ``<output_name>.pickle``. If ``truncate_action_dim`` is True and both ``action``
        and ``raw_action_dim`` are present in ``data``, the action tensor's last dimension
        is truncated to ``raw_action_dim`` before saving.

        Returns a list of paths to all files written.
        """
        files: list[Path] = []
        data_tensors: dict[str, torch.Tensor] = {}
        data_pickle: dict[str, Any] = {}
        for k, v in data.items():
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                for i, x in enumerate(v):
                    data_tensors[f"{k}[{i}]"] = x
            elif isinstance(v, torch.Tensor):
                data_tensors[k] = v
            else:
                data_pickle[k] = v

        # Truncate `action` tensor's last dimension to `raw_action_dim` if available;
        # otherwise use the full action tensor as-is.
        if truncate_action_dim and "action" in data_tensors and "raw_action_dim" in data_tensors:
            raw_action_dim = data_tensors["raw_action_dim"][0]
            action = data_tensors["action"][..., :raw_action_dim]
            data_tensors["action"] = action
            log.debug(f"Truncated 'action' tensor to shape={action.shape}")

        if data_tensors:
            tensors_file = output_dir / f"{output_name}.safetensors"
            safetensors.torch.save_file(
                {k: v.detach().cpu().contiguous() for k, v in data_tensors.items()}, tensors_file
            )
            files.append(tensors_file)

        if data_pickle:
            pickle_file = output_dir / f"{output_name}.pickle"
            with pickle_file.open("wb") as f:
                pickle.dump(data_pickle, f)
            files.append(pickle_file)

        return files

    @override
    def create_batches(
        self, sample_args_list: Sequence[SampleArgs]
    ) -> Generator[tuple[list[SampleArgs], dict[str, Any]]]:
        assert isinstance(self.setup_args, OmniSetupArgs)
        max_model_len = self.setup_args.max_model_len
        max_num_seqs = self.setup_args.max_num_seqs

        sample_args_list = _finalize_sample_args_list(cast(Sequence[OmniSampleArgs], sample_args_list))
        dataset = SampleDataset(sample_args_list, self.model)

        # Mod-shard the dataset indices across replicas.
        sampler_indices = list(range(self.replica_id, len(dataset), self.num_replicas))

        # --- Phase 1: pre-compute batch boundaries (cheap, no data prep) ---
        batch_position_lists = list(
            _iter_packed_batches(
                items=range(len(sampler_indices)),
                get_sample_args=lambda pos: sample_args_list[sampler_indices[pos]],
                model=self.model,
                max_model_len=max_model_len,
                max_num_seqs=max_num_seqs,
            )
        )

        num_local_batches = len(batch_position_lists)

        log.debug(f"Number of local batches: {num_local_batches}", rank0_only=False)

        # --- Phase 2: synchronize batch count across replicas ---
        # All ranks within a replica share the same replica_id and therefore
        # the same local batch count, so a global MAX all-reduce is sufficient
        # to align all replicas.
        if torch.distributed.is_initialized() and self.num_replicas > 1:
            count_tensor = torch.tensor([num_local_batches], dtype=torch.long, device="cuda")
            torch.distributed.all_reduce(count_tensor, op=torch.distributed.ReduceOp.MAX)
            global_max_batches = int(count_tensor.item())
        else:
            global_max_batches = num_local_batches

        log.debug(f"Number of global batches: {global_max_batches}")
        log.debug(f"Number of padding batches: {global_max_batches - num_local_batches}", rank0_only=False)

        # --- Phase 3: yield real batches (lazily prepare data) ---
        batches_yielded = 0

        for batch_positions in batch_position_lists:
            chunk_args: list[SampleArgs] = []
            chunk_data: list[dict[str, Any]] = []

            for pos in batch_positions:
                sample_idx = sampler_indices[pos]
                sample_args, data_batch = dataset[sample_idx]

                if self.setup_args.debug and self.should_process_sample(sample_args):
                    assert sample_args.output_dir is not None
                    sample_args.output_dir.mkdir(parents=True, exist_ok=True)
                    self.save_data(
                        data_batch,
                        output_dir=sample_args.output_dir,
                        output_name="sample_data",
                    )

                chunk_args.append(sample_args)
                chunk_data.append(data_batch)

            yield chunk_args, _merge_data_batches(chunk_data)
            batches_yielded += 1

        assert batches_yielded == num_local_batches

        # --- Phase 4: pad with dummy batches so every replica calls
        #     generate_batch the same number of times (prevents collective
        #     deadlocks in context-parallel / CFG-parallel communication).
        # Minimal-cost padding sample: the dummy batch only exists to keep the
        # generate_batch call count aligned across replicas, and its output is
        # discarded (output_dir=None). Force num_steps=1 / guidance=1.0 so it never
        # raises the per-iteration align_num_steps MAX (which would make the dummy
        # *and* real samples on peer ranks pad up). The per-step alignment still
        # pads this dummy up to MAX(real samples), so collective alignment holds;
        # we just stop inflating that MAX with the (arbitrary) global sample[0].
        dummy_sa = sample_args_list[0].model_copy(
            update={"output_dir": None, "name": "padding", "num_steps": 1, "guidance": 1.0}
        )
        dummy_data = dataset[0][1]
        while batches_yielded < global_max_batches:
            yield [dummy_sa], dummy_data
            batches_yielded += 1
        assert batches_yielded == global_max_batches

    @torch.no_grad()
    @override
    def generate_batch(
        self, sample_args_list: Sequence[SampleArgs], data_batch: dict[str, Any], *, warmup: bool = False
    ) -> list[SampleOutputs]:
        assert all(isinstance(sa, OmniSampleArgs) for sa in sample_args_list)

        transfer_flags = [bool(sa.transfer_hints) for sa in sample_args_list]
        if any(transfer_flags):
            assert all(transfer_flags), "Cannot mix transfer and non-transfer samples in a batch"
            assert len(sample_args_list) == 1, "Batching is not supported for transfer inference"
            return self._generate_transfer_batch(sample_args_list[0], warmup=warmup)

        reasoner_flags = [cast(OmniSampleArgs, sa).model_mode.is_reasoner for sa in sample_args_list]
        if any(reasoner_flags):
            assert all(reasoner_flags), "Cannot mix reasoner and non-reasoner samples in a batch"
            return self._generate_reasoner_batch(sample_args_list, data_batch, warmup=warmup)

        # Process inputs
        try:
            with sync_distributed_errors():
                for sample_args in sample_args_list:
                    if self.should_process_sample(sample_args) and not warmup:
                        log.debug(f"{sample_args.__class__.__name__}({sample_args})")
                        assert sample_args.output_dir is not None
                        sample_args.output_dir.mkdir(parents=True, exist_ok=True)
                        sample_args_file = sample_args.output_dir / "sample_args.json"
                        sample_args_file.write_text(sample_args.model_dump_json())
                        log.info(f"Saved sample args to '{sample_args_file}'", rank0_only=False)

                assert all(sa.num_outputs == 1 for sa in sample_args_list), "num_outputs must be 1"
                data_batch = _finalize_data_batch(
                    data_batch=data_batch, batch_size=len(sample_args_list), model=self.model
                )
        except Exception as e:
            return [
                self._handle_sample_exception(args, e)
                for args in sample_args_list
                if self.should_process_sample(args) and not warmup
            ]

        # Generate samples
        #
        # Can't catch exceptions here. This code contains collective operations
        # that will hang if any rank fails. If a rank fails, we must restart
        # the entire distributed environment.
        #
        # Use the first sample's sampling parameters for the whole batch.
        # All samples in a batch share guidance, num_steps, shift, etc.
        def _getattr(sample_args_list: Sequence[OmniSampleArgs], attr: str) -> Any:
            attr_values = [getattr(sa, attr) for sa in sample_args_list]
            if all(v == attr_values[0] for v in attr_values):
                return attr_values[0]
            else:
                raise ValueError(f"Attribute '{attr}' is not the same for all samples: {attr_values}")

        is_distilled = self.model.config.fixed_step_sampler_config is not None
        if is_distilled:
            sampler = self.model.fixed_step_sampler
            guidance = 1.0
        else:
            sampler = None
            guidance = _getattr(sample_args_list, "guidance")

        should_decode_outputs = self.should_process_sample(sample_args_list[0])

        def decode_vision(vision_latent: torch.Tensor) -> torch.Tensor:
            """
            Handles decoding of vision latents, either on the inference device or on a separate VAE device if configured.
            """
            if not should_decode_outputs:
                tokenizer_vision_gen = self.model.tokenizer_vision_gen
                return vision_latent.new_zeros(
                    (
                        vision_latent.shape[0],
                        3,
                        tokenizer_vision_gen.get_pixel_num_frames(int(vision_latent.shape[2])),
                        int(vision_latent.shape[3]) * tokenizer_vision_gen.spatial_compression_factor,
                        int(vision_latent.shape[4]) * tokenizer_vision_gen.spatial_compression_factor,
                    )
                )
            if self.vae_decode_stream is None:
                # We are not using a separate GPU for VAE decoding, so decode directly on the inference device
                vision = self.model.decode(vision_latent)  # [B,C,T,H,W]
                return ((1.0 + vision) / 2).clamp(0, 1)  # [B,C,T,H,W]
            # We are using a separate GPU for VAE decoding, so we need to issue decode on the VAE device
            vision_ready = torch.cuda.Event()
            torch.cuda.current_stream(device=vision_latent.device).record_event(vision_ready)
            self.vae_decode_stream.wait_event(vision_ready)
            with torch.cuda.stream(self.vae_decode_stream):
                vision = self.model.decode(vision_latent)  # [B,C,T,H,W]
                return ((1.0 + vision) / 2).clamp(0, 1)  # [B,C,T,H,W]

        # Use a deterministic fallback (rather than ``random.randint``) when
        # the caller didn't supply a seed: every rank in a CP / CFG-parallel
        # replica must compute the same seed for a given sample, otherwise the
        # initial sampling noise diverges across ranks and the parallel
        # denoising loop produces corrupt outputs.
        seed = [sa.seed if sa.seed is not None else _fallback_seed(cast(OmniSampleArgs, sa)) for sa in sample_args_list]
        outputs: dict[str, Any] | None = None


        if outputs is None:
            assert all(sa.num_outputs == 1 for sa in sample_args_list), "num_outputs must be 1"
            n_sample = sum(cast(OmniSampleArgs, sa).num_outputs for sa in sample_args_list)
            neg_key = "neg_" + self.model.input_caption_key

            omni_sample_args_list = cast(Sequence[OmniSampleArgs], sample_args_list)

            # ``_infer_native_prompt_upsampling_tasks`` owns the full
            # per-sample decision: opt-in flag, already-upsampled content
            # check, modality routing, and reasoner capability gating.
            # It returns a per-sample list of ``UpsampleTask | None``;
            # ``None`` entries mean "skip native upsampling for this
            # sample".  ``generate_samples_from_batch`` (via
            # ``_maybe_apply_prompt_upsampling``) consumes the list
            # directly and dispatches per-task group, so mixed
            # opted-in / opted-out and mixed-task batches all flow
            # through without a caller-side collapse.
            resolved_upsample_tasks = _infer_native_prompt_upsampling_tasks(
                data_batch,
                omni_sample_args_list,
                self.model,
            )
            distinct_upsample_tasks = set(resolved_upsample_tasks)
            if len(distinct_upsample_tasks) > 1:
                raise ValueError(
                    "[prompt-upsampling] mixed-task batch: per-sample tasks "
                    f"{resolved_upsample_tasks!r} contain multiple distinct V4.2 "
                    f"tasks {sorted(distinct_upsample_tasks, key=lambda x: x or '')}, but "
                    "`generate_samples_from_batch` carries a single "
                    "`upsample_task` knob. Split the batch by task at the "
                    "caller, or refactor `_maybe_apply_prompt_upsampling` to "
                    "dispatch per-sample."
                )
            upsample_task = next(iter(distinct_upsample_tasks))

            # FSDP collective-sequence alignment (throughput-style inference where
            # ranks hold different samples). Each per-step model forward issues a
            # param all-gather over the FSDP-shard (dp_shard) group, so if dp_shard
            # peers disagree on ``num_steps`` that group's collective stream
            # desyncs and deadlocks NCCL at the watchdog timeout (observed: rank0
            # wedged at step 31/50 the instant its dp_shard peer finished 35).
            #
            # all_reduce(MAX) the local num_steps over the *dp_shard group* and
            # pass it as ``align_num_steps``; ranks below the max pad with
            # discarded dummy steps in generate_samples_from_batch. Scope = the
            # dp_shard group (not world), because that keeps the reduction within a
            # single modality: modality must already be homogeneous within any
            # per-forward collective group (else the forward itself desyncs), and
            # reasoner-only batches take an early return below and never reach this
            # collective — a world reduction would deadlock against them. The
            # per-step CP / CFGP collectives are also covered: cp/cfgp groups
            # always sit inside one data-parallel replica (replica_id =
            # rank // (cp*cfgp)), so when dp_shard and the replica block (cp*cfgp)
            # nest, every cp/cfgp peer lands in a dp_shard group with the same MAX.
            # The nesting precondition is asserted just below.
            local_num_steps = _getattr(sample_args_list, "num_steps")
            align_num_steps = local_num_steps
            parallel_dims = getattr(self.model, "parallel_dims", None)
            if (
                parallel_dims is not None
                and parallel_dims.dp_shard_mesh is not None
                and torch.distributed.is_initialized()
                and parallel_dims.dp_shard_mesh.size() > 1
            ):
                # Non-nesting CP/CFGP overlays (neither dp_shard nor the cp*cfgp
                # replica block divides the other) let a cp/cfgp group straddle two
                # dp_shard groups with different maxima, which a dp_shard-scoped
                # reduction cannot align. Both presets nest (throughput: cp=cfgp=1;
                # latency: single replica), so this only guards hand-built layouts.
                replica_block = parallel_dims.cp * parallel_dims.cfgp
                dp_shard_sz = parallel_dims.dp_shard
                if replica_block > 1 and dp_shard_sz % replica_block != 0 and replica_block % dp_shard_sz != 0:
                    raise NotImplementedError(
                        "num_steps collective alignment requires dp_shard "
                        f"({dp_shard_sz}) and cp*cfgp ({replica_block}) to nest "
                        "(one must divide the other). Non-nesting CP/CFGP overlays "
                        "with divergent per-sample num_steps are unsupported."
                    )
                _steps_t = torch.tensor(
                    [local_num_steps], device=self.model.tensor_kwargs["device"], dtype=torch.int32
                )
                torch.distributed.all_reduce(
                    _steps_t, op=torch.distributed.ReduceOp.MAX, group=parallel_dims.dp_shard_mesh.get_group()
                )
                align_num_steps = int(_steps_t.item())

            with self._get_timer(f"{self.model.__class__.__name__}.generate_samples_from_batch"):
                outputs = self.model.generate_samples_from_batch(
                    data_batch,
                    sampler=sampler,
                    guidance=guidance,
                    guidance_interval=_getattr(sample_args_list, "guidance_interval"),
                    seed=seed,
                    num_steps=local_num_steps,
                    align_num_steps=align_num_steps,
                    shift=_getattr(sample_args_list, "shift"),
                    sigma_max=_getattr(sample_args_list, "sigma_max"),
                    has_negative_prompt=neg_key in data_batch,
                    n_sample=n_sample,
                    normalize_cfg=_getattr(sample_args_list, "normalize_cfg"),
                    upsample_task=upsample_task,
                    upsample_max_new_tokens=_getattr(omni_sample_args_list, "prompt_upsampler_max_tokens"),
                    upsample_temperature=_getattr(omni_sample_args_list, "prompt_upsampler_temperature"),
                    upsample_top_k=_getattr(omni_sample_args_list, "prompt_upsampler_top_k"),
                    upsample_top_p=_getattr(omni_sample_args_list, "prompt_upsampler_top_p"),
                    upsample_repetition_penalty=_getattr(omni_sample_args_list, "prompt_upsampler_repetition_penalty"),
                    upsample_presence_penalty=_getattr(omni_sample_args_list, "prompt_upsampler_presence_penalty"),
                    upsample_seed=_getattr(omni_sample_args_list, "prompt_upsampler_seed"),
                )

            with self._get_timer(f"{self.model.__class__.__name__}.decode"):
                output_vision = outputs.pop("vision")
                decoded_vision = [decode_vision(vision) for vision in output_vision]
                outputs["vision"] = [cast(torch.Tensor, vision) for vision in decoded_vision]
                if self.vae_decode_stream is not None:
                    # If we are using a separate GPU for VAE decoding, wait for results to be ready
                    torch.cuda.current_stream(device=outputs["vision"][0].device).wait_stream(self.vae_decode_stream)
        for k, v in outputs.items():
            if len(v) != len(sample_args_list):
                raise ValueError(f"Output key '{k}' has length {len(v)} but expected {len(sample_args_list)}")

        if "sound" in outputs:
            with self._get_timer(f"{self.model.__class__.__name__}.decode_sound"):
                outputs["sound"] = [self.model.decode_sound(sound) for sound in outputs.pop("sound")]

        if warmup:
            return []

        # Save outputs
        sample_outputs: list[SampleOutputs] = []
        try:
            with sync_distributed_errors():
                for sample_idx, sample_args in enumerate(sample_args_list):
                    if self.should_process_sample(sample_args):
                        assert isinstance(sample_args, OmniSampleArgs)
                        assert sample_args.output_dir is not None
                        assert sample_args.num_outputs == 1
                        output = {k: v[sample_idx].squeeze(0) for k, v in outputs.items()}
                        vision_cthw = output.pop("vision")

                        # Run guardrails
                        self._run_text_guardrail(
                            str(sample_args.output_dir), data_batch[self.model.input_caption_key][sample_idx]
                        )
                        vision_cthw = self._run_video_guardrail(str(sample_args.output_dir), vision_cthw)
                        output["vision"] = vision_cthw

                        content: dict[str, Any] = {}
                        files: list[Path] = []

                        # Save debug
                        if self.setup_args.debug:
                            files.extend(
                                self.save_data(output, output_dir=sample_args.output_dir, output_name="output")
                            )

                        # Save vision
                        if vision_cthw.shape[1] == 1:
                            quality = sample_args.image_save_quality
                        else:
                            quality = sample_args.video_save_quality
                        vision_file = sample_args.output_dir / f"vision{sample_args.vision_extension}"
                        output_fps = sample_args.fps
                        save_img_or_video(
                            vision_cthw, str(vision_file.with_suffix("")), fps=output_fps, quality=quality
                        )
                        assert vision_file.is_file(), vision_file
                        files.append(vision_file)

                        if "sound" in output:
                            from cosmos_framework.inference.sound import (
                                get_audio_tokenizer_info,
                                mux_audio_into_video,
                            )

                            audio_info = get_audio_tokenizer_info(self.model)
                            mux_audio_into_video(vision_file, output["sound"], audio_info.sample_rate)

                        if "action" in output:
                            pred_action = output["action"]
                            if "raw_action_dim" in data_batch:
                                raw_action_dim = int(data_batch["raw_action_dim"][sample_idx].item())
                                assert pred_action.shape[-1] >= raw_action_dim, (
                                    f"invalid raw_action_dim={raw_action_dim} for action with shape {pred_action.shape}"
                                )
                                pred_action = pred_action[..., :raw_action_dim]
                            content["action"] = pred_action.detach().cpu().tolist()

                        sample_output = SampleOutputs(
                            args=sample_args.model_dump(mode="json"),
                            outputs=[SampleOutput(content=content, files=files)],
                        )
                        sample_outputs_file = sample_args.output_dir / "sample_outputs.json"
                        sample_outputs_file.write_text(sample_output.model_dump_json())
                        log.success(f"Saved sample outputs to '{sample_outputs_file}'", rank0_only=False)

                        sample_outputs.append(sample_output)

        except Exception as e:
            return [
                self._handle_sample_exception(sample_args, e)
                for sample_args in sample_args_list
                if self.should_process_sample(sample_args)
            ]

        return sample_outputs

    @torch.no_grad()
    def _generate_transfer_batch(self, sample_args: OmniSampleArgs, *, warmup: bool = False) -> list[SampleOutputs]:
        """Handle transfer inference using the autoregressive generate_transfer_sample path."""
        from cosmos_framework.inference.transfer import generate_transfer_sample

        try:
            with sync_distributed_errors():
                if self.should_process_sample(sample_args) and not warmup:
                    log.debug(f"{sample_args.__class__.__name__}({sample_args})")
                    assert sample_args.output_dir is not None
                    sample_args.output_dir.mkdir(parents=True, exist_ok=True)
                    sample_args_file = sample_args.output_dir / "sample_args.json"
                    sample_args_file.write_text(sample_args.model_dump_json())
                    log.info(f"Saved sample args to '{sample_args_file}'", rank0_only=False)
        except Exception as e:
            if self.should_process_sample(sample_args) and not warmup:
                return [self._handle_sample_exception(sample_args, e)]
            return []

        transfer_output = generate_transfer_sample(sample_args=sample_args, model=self.model)

        if warmup:
            return []

        sample_outputs: list[SampleOutputs] = []
        try:
            with sync_distributed_errors():
                if self.should_process_sample(sample_args):
                    assert sample_args.output_dir is not None
                    content: dict[str, Any] = {}
                    files: list[Path] = []

                    vision_cthw = ((1.0 + transfer_output.output_video.squeeze(0)) / 2).clamp(0, 1)

                    if vision_cthw.shape[1] == 1:
                        quality = sample_args.image_save_quality
                    else:
                        quality = sample_args.video_save_quality
                    vision_file = sample_args.output_dir / f"vision{sample_args.vision_extension}"
                    output_fps = transfer_output.fps
                    save_img_or_video(vision_cthw, str(vision_file.with_suffix("")), fps=output_fps, quality=quality)
                    assert vision_file.is_file(), vision_file
                    files.append(vision_file)

                    for hint_key, control_video in transfer_output.control_videos.items():
                        control_cthw = ((1.0 + control_video.squeeze(0)) / 2).clamp(0, 1)
                        control_file = sample_args.output_dir / f"control_{hint_key}{sample_args.vision_extension}"
                        save_img_or_video(
                            control_cthw, str(control_file.with_suffix("")), fps=output_fps, quality=quality
                        )
                        files.append(control_file)
                        log.info(f"Saved control video to '{control_file}'", rank0_only=False)

                    sample_output = SampleOutputs(
                        args=sample_args.model_dump(mode="json"),
                        outputs=[SampleOutput(content=content, files=files)],
                    )
                    sample_outputs_file = sample_args.output_dir / "sample_outputs.json"
                    sample_outputs_file.write_text(sample_output.model_dump_json())
                    log.success(f"Saved transfer outputs to '{sample_outputs_file}'", rank0_only=False)

                    sample_outputs.append(sample_output)

        except Exception as e:
            return [self._handle_sample_exception(sample_args, e)] if self.should_process_sample(sample_args) else []

        return sample_outputs

    @torch.no_grad()
    def _generate_reasoner_batch(
        self,
        sample_args_list: Sequence[SampleArgs],
        data_batch: dict[str, Any],
        *,
        warmup: bool = False,
    ) -> list[SampleOutputs]:
        """Reasoner AR text generation. Each prompt writes ``reasoner_text.txt`` and
        ``SampleOutput.content["reasoner_text"]``. Mixing image-conditioned and
        text-only samples in one batch is rejected."""
        sample_args_list = cast(list[OmniSampleArgs], sample_args_list)

        prompts: list[str] = data_batch[self.model.input_caption_key]
        raw_images: list[Image.Image | None] = data_batch["reasoner_images"]
        n_set = sum(img is not None for img in raw_images)
        if 0 < n_set < len(raw_images):
            raise ValueError(
                "Reasoner batch mixes image-conditioned and text-only samples "
                f"({n_set}/{len(raw_images)} have vision_path). Split into separate batches."
            )
        images: list[Image.Image] | None = cast(list[Image.Image], raw_images) if n_set == len(raw_images) else None

        try:
            with sync_distributed_errors():
                for sa, prompt in zip(sample_args_list, prompts):
                    if self.should_process_sample(sa) and not warmup:
                        log.debug(f"{sa.__class__.__name__}({sa})")
                        assert sa.output_dir is not None
                        sa.output_dir.mkdir(parents=True, exist_ok=True)
                        (sa.output_dir / "sample_args.json").write_text(sa.model_dump_json())
                        self._run_text_guardrail(str(sa.output_dir), prompt)
        except Exception as e:
            return [
                self._handle_sample_exception(sa, e)
                for sa in sample_args_list
                if self.should_process_sample(sa) and not warmup
            ]

        # Collective call: every rank must enter so FSDP unshard/reshard and the
        # cross-rank early-exit reduction stay in lockstep. Not wrapped in try/except.
        with self._get_timer(f"{self.model.__class__.__name__}.generate_reasoner_text"):
            texts = self.model.generate_reasoner_text(
                prompts,
                max_new_tokens=sample_args_list[0].max_new_tokens,
                images=images,
                do_sample=sample_args_list[0].do_sample,
                temperature=sample_args_list[0].temperature,
                top_k=sample_args_list[0].top_k,
                top_p=sample_args_list[0].top_p,
                repetition_penalty=sample_args_list[0].repetition_penalty,
                presence_penalty=sample_args_list[0].presence_penalty,
                seed=sample_args_list[0].seed,
            )

        if warmup:
            return []

        sample_outputs: list[SampleOutputs] = []
        try:
            with sync_distributed_errors():
                for sa, text in zip(sample_args_list, texts):
                    if not self.should_process_sample(sa):
                        continue
                    assert sa.output_dir is not None
                    self._run_text_guardrail(str(sa.output_dir), text)
                    txt_path = sa.output_dir / "reasoner_text.txt"
                    txt_path.write_text(text)
                    sample_output = SampleOutputs(
                        args=sa.model_dump(mode="json"),
                        outputs=[SampleOutput(content={"reasoner_text": text}, files=[txt_path])],
                    )
                    (sa.output_dir / "sample_outputs.json").write_text(sample_output.model_dump_json())
                    log.success(f"Saved reasoner outputs to '{sa.output_dir}'", rank0_only=False)
                    sample_outputs.append(sample_output)
        except Exception as e:
            return [self._handle_sample_exception(sa, e) for sa in sample_args_list if self.should_process_sample(sa)]

        return sample_outputs

    @property
    def replica_size(self) -> int:
        """
        The ranks are divided into computation replicas. The replica size is
        the product of the context parallelism and CFG parallelism sizes.
        """
        if not hasattr(self.model, "parallel_dims") or self.model.parallel_dims is None:
            return 1
        else:
            return self.model.parallel_dims.cp_size * self.model.parallel_dims.cfgp_size

    @property
    def num_replicas(self) -> int:
        assert get_world_size() % self.replica_size == 0
        return get_world_size() // self.replica_size

    @property
    def replica_id(self) -> int:
        return get_rank() // self.replica_size

    @property
    def index_in_replica(self) -> int:
        return get_rank() % self.replica_size

    def should_process_sample(self, sample_args: SampleArgs) -> bool:
        """Whether the sample should be processed by the current rank."""
        return sample_args.output_dir is not None and self.index_in_replica == 0


_data_converter = cattrs.preconf.json.make_converter()


# torch.Tensor
@_data_converter.register_unstructure_hook
def _unstructure_torch_tensor(obj: torch.Tensor) -> Any:
    return {
        "shape": obj.shape,
        "dtype": str(obj.dtype),
        "device": str(obj.device),
        "values": obj.detach().flatten()[:5].cpu().tolist(),
    }
