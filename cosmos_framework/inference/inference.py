# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import hashlib
import json
import os
import pickle
import traceback
from collections.abc import Callable, Generator, Iterable, MutableMapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Sequence, TypeVar, cast, override

import attrs
import cattrs
import cattrs.preconf.json
import safetensors.torch
import torch
from PIL import Image
from qwen_vl_utils.vision_process import smart_nframes
from torch.utils._pytree import tree_map_only
from torch.utils.data import Dataset
from typing_extensions import Self

from cosmos_framework.configs.base.defaults.compile import CompileConfig
from cosmos_framework.configs.base.defaults.parallelism import ParallelismConfig
from cosmos_framework.configs.base.defaults.quantization import QuantizationConfig
from cosmos_framework.inference.args import (
    ModelMode,
    MultiviewCameraArgs,
    NegativeMetadataMode,
    OmniSampleArgs,
    OmniSetupArgs,
    check_auto_fps_options,
)
from cosmos_framework.inference.common.args import (
    VIDEO_EXTENSIONS,
    CheckpointType,
    ConfigFileType,
    ParallelismArgs,
    SampleArgs,
    SampleOutput,
    SampleOutputs,
    SetupArgs,
)
from cosmos_framework.inference.common.auto_fps import auto_camera_fps
from cosmos_framework.inference.common.inference import (
    Inference,
    PreparedBatch,
    _download_on_rank0,
    sync_distributed_errors,
)
from cosmos_framework.inference.common.init import get_rank, get_world_size
from cosmos_framework.inference.common.multiview_captions import (
    exported_caption_task,
    exported_checkpoint_metadata,
    exports_per_view_captions,
)
from cosmos_framework.inference.model import Cosmos3OmniConfig, Cosmos3OmniModel
from cosmos_framework.inference.transfer_output import restore_transfer_output_size
from cosmos_framework.inference.vision import (
    build_conditioned_video_batch,
    build_image_edit_batch,
    decode_video_thwc_uint8,
    load_conditioning_image,
    load_conditioning_video,
    load_prompt_upsampling_image,
    pil_to_conditioning_frames,
    resize_pil_image,
)
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
from cosmos_framework.model.generator.reasoner.qwen3_vl.utils import _SYSTEM_PROMPT_IMAGE_EDITING
from cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 import Wan2pt2VAEInterface
from cosmos_framework.model.generator.upsampler.prompts import is_upsampled_prompt
from cosmos_framework.tools.visualize.video import save_img_or_video
from cosmos_framework.utils import log
from cosmos_framework.utils.checkpoint_db import CheckpointDirHf
from cosmos_framework.utils.generator.host_threads import host_threads_per_local_rank, torch_host_threads
from cosmos_framework.utils.generator.multiview import is_causal_multiview_config

if TYPE_CHECKING:
    from cosmos_framework.configs.base.defaults.model_config import OmniMoTModelConfig
    from cosmos_framework.inference.bimanual_session import BimanualSession

UpsampleTask = Literal["t2i", "t2v", "i2v"]


def _save_transfer_vision(
    vision_cthw: torch.Tensor,
    vision_file: Path,
    *,
    fps: float,
    sample_args: OmniSampleArgs,
    encoder_threads: int | None = None,
) -> list[Path]:
    """Save a ``[C, T, H, W]`` vision tensor (image or video) and return all files written.

    The full-quality output is ``vision.mp4`` (qscale ``video_save_quality``). For multi-frame video,
    when ``save_preview`` is enabled, a same-resolution but CRF-encoded ``*_preview.mp4`` is also
    written — far smaller and quick to download. Regular single-frame image outputs use
    ``image_save_quality``. Image-edit outputs are saved losslessly as PNG so benchmark metrics are not
    affected by JPEG compression.
    """
    is_video = vision_cthw.shape[1] > 1
    if not is_video and sample_args.model_mode == ModelMode.IMAGE2IMAGE:
        vision_file = vision_file.with_suffix(".png")
        vision_chw = vision_cthw.squeeze(1).clamp(0, 1)  # [C,H,W]
        vision_hwc = (vision_chw.permute(1, 2, 0).cpu().float().numpy() * 255).astype("uint8")  # [H,W,C]
        Image.fromarray(vision_hwc, mode="RGB").save(vision_file)
        return [vision_file]

    base = str(vision_file.with_suffix(""))
    written: list[Path] = [vision_file]

    quality = sample_args.video_save_quality if is_video else sample_args.image_save_quality
    ffmpeg_params = None if not is_video or encoder_threads is None else ["-threads", str(encoder_threads)]
    save_img_or_video(
        vision_cthw,
        base,
        fps=fps,
        quality=quality,
        ffmpeg_params=ffmpeg_params,
    )

    if is_video and getattr(sample_args, "save_preview", False):
        preview_file = vision_file.with_name(f"{vision_file.stem}_preview{vision_file.suffix}")
        save_img_or_video(
            vision_cthw,
            str(preview_file.with_suffix("")),
            fps=fps,
            quality=None,
            crf=sample_args.preview_crf,
            preset=sample_args.preview_preset,
            ffmpeg_params=ffmpeg_params,
        )
        written.append(preview_file)

    return written


def _save_transfer_rgb(
    video: torch.Tensor,  # [C,T,H_model,W_panels], floating-point pixels in [0,1]
    *,
    original_hw: tuple[int, int],
    model_hw: tuple[int, int],
    fps: float,
    sample_args: OmniSampleArgs,
    content: dict[str, Any],
) -> list[Path]:
    """Save Transfer RGB, retaining native pixels alongside an optional source-size export."""
    assert sample_args.output_dir is not None
    files: list[Path] = []
    if sample_args.transfer_output_size == "original":
        native_file = sample_args.output_dir / f"vision_model{sample_args.vision_extension}"
        native_written = _save_transfer_vision(video, native_file, fps=fps, sample_args=sample_args)
        files.extend(native_written)
        content["native_vision_file"] = str(native_written[0])
        video = restore_transfer_output_size(  # [C,T,H_source,W_source*panels]
            video, original_hw=original_hw, model_hw=model_hw
        )
    if sample_args.transfer_resize_mode == "direct_resize":
        scale_x = model_hw[1] / original_hw[1]
        scale_y = model_hw[0] / original_hw[0]
        resized_hw = list(video.shape[-2:])
        # The shared MP4 writer pads an odd height/width by one bottom/right
        # pixel for yuv420p. Record that encoded canvas separately from resizing;
        # still images keep their exact dimensions, including lossless PNG edits.
        encoded_hw = [dimension + dimension % 2 for dimension in resized_hw] if video.shape[1] > 1 else resized_hw
        content["transfer_geometry"] = {
            "resize_mode": "direct_resize",
            "source_hw": list(original_hw),
            "model_hw": list(model_hw),
            "scale_xy": [scale_x, scale_y],
            "aspect_distortion": max(scale_x, scale_y) / min(scale_x, scale_y) - 1.0,
            "output_size": sample_args.transfer_output_size,
            "resized_hw": resized_hw,
            "output_hw": encoded_hw,
            "encoder_padding_hw": [encoded - resized for encoded, resized in zip(encoded_hw, resized_hw, strict=True)],
        }
    vision_file = sample_args.output_dir / f"vision{sample_args.vision_extension}"
    files.extend(_save_transfer_vision(video, vision_file, fps=fps, sample_args=sample_args))
    assert all(path.is_file() for path in files), files
    return files


_BatchItem = TypeVar("_BatchItem")

_PROCESSOR_HF_INCLUDE = (
    "*.json",
    "*.jinja",
    "merges.txt",
    "vocab.json",
)

# Model default for ``model.config.attention_io_layout`` (see ``OmniMoTModelConfig``).
_DEFAULT_ATTENTION_IO_LAYOUT = "sequence_sharded"


def _configure_vae_compile(model: OmniMoTModel, setup_args: OmniSetupArgs) -> None:
    """Opt into the existing Wan encoder AOT and steady-state decoder compilers."""
    if not (setup_args.use_vae_encode_compile or setup_args.use_vae_decode_compile):
        return
    tokenizer = model.tokenizer_vision_gen
    if not isinstance(tokenizer, Wan2pt2VAEInterface):
        raise ValueError(
            "--use-vae-encode-compile and --use-vae-decode-compile require the Wan 2.2 4x16x16 vision tokenizer."
        )
    if setup_args.use_vae_encode_compile and setup_args.use_separate_pipeline_vision_decode_gpu:
        raise ValueError(
            "--use-vae-encode-compile binds the encoder to this rank's GPU and cannot be combined with "
            "--use-separate-pipeline-vision-decode-gpu."
        )
    if setup_args.use_vae_decode_compile:
        enable_compile = getattr(tokenizer.active_decoder, "enable_decoder_compile", None)
        if not callable(enable_compile):
            raise ValueError("--use-vae-decode-compile is unsupported by the active vision decoder.")
        enable_compile()
    if setup_args.use_vae_encode_compile:
        resolution = setup_args.sample_overrides.resolution or model.config.resolution
        aspect_ratio = setup_args.sample_overrides.aspect_ratio or "16,9"
        tokenizer.compile_encode([resolution], output_dir=str(setup_args.output_dir), aspect_ratio=aspect_ratio)


def _unbatch_transfer_debug_tensor(
    tensor: torch.Tensor, *, name: str
) -> torch.Tensor:  # tensor: [B,C,T,H,W], returns: [C,T,H,W]
    """Validate and remove one Transfer debug tensor's singleton batch dimension."""
    if tensor.ndim != 5 or tensor.shape[0] != 1:
        raise ValueError(f"Transfer debug tensor {name!r} must have shape [1,C,T,H,W]; got {tuple(tensor.shape)}")
    return tensor[0]  # [C,T,H,W]


def _checkpoint_has_processor_files(checkpoint_path: Path) -> bool:
    """True when the checkpoint dir bundles VLM processor/tokenizer files at its root."""
    return any((checkpoint_path / name).is_file() for name in ("tokenizer_config.json", "preprocessor_config.json"))


# Files ``Qwen2Tokenizer.from_pretrained`` reads from a local directory: its
# ``vocab_files_names`` (vocab.json + merges.txt, both opened unconditionally in
# ``__init__``) plus tokenizer_config.json, which carries the added special
# tokens (<|vision_start|> etc.) the SFT data pipeline depends on.
_QWEN2_TOKENIZER_FILES = ("tokenizer_config.json", "vocab.json", "merges.txt")

# ``tokenizer_type`` substrings that ``build_processor`` maps to Qwen3VLProcessor
# in hub mode — the same processor its local-directory mode defaults to, so a
# rewrite to a bundled dir preserves the dispatch for these nodes.
_QWEN3VL_TOKENIZER_TYPES = ("Qwen/Qwen3-VL", "Siglip2-Qwen3-1.7B")

# Repos served by the native Cosmos3-Edge (Nemotron bridge) processor; covers
# nvidia/Cosmos3-Edge and nvidia/Cosmos3-Edge-Reasoner (mirrors the substring
# dispatch in ``build_processor``).
_EDGE_REPO_SUBSTRING = "nvidia/Cosmos3-Edge"


def _tokenizer_node_target(tokenizer_cfg: MutableMapping[str, Any]) -> str:
    """Last component of the node's ``_target_`` (e.g. ``build_processor_lazy``)."""
    return str(tokenizer_cfg.get("_target_", "")).rsplit(".", 1)[-1]


def _dir_serves_qwen3vl_autoprocessor(path: Path) -> bool:
    """True when ``path`` holds the files Qwen3VLProcessor's AutoProcessor load needs.

    That is the image/video preprocessor params, the tokenizer config, and the
    tokenizer data (fast ``tokenizer.json`` or the slow vocab/merges pair).
    """
    if not all((path / name).is_file() for name in ("preprocessor_config.json", "tokenizer_config.json")):
        return False
    return (path / "tokenizer.json").is_file() or all((path / name).is_file() for name in ("vocab.json", "merges.txt"))


def _bundled_processor_serves_node(tokenizer_cfg: MutableMapping[str, Any], checkpoint_path: Path) -> bool:
    """True when checkpoint-bundled processor files can serve this ``vlm_config.tokenizer`` node.

    Gates the local-first override (``_point_tokenizer_node_at_dir``) on the
    bundle being dispatchable for the node's shape: Edge-family nodes need the
    native-snapshot markers, Qwen3-VL-family nodes the AutoProcessor file set,
    and Qwen2 nodes take a local dir only with ``config_variant="hf"``. Unknown
    shapes return False (keep the configured hub behavior).
    """
    if not isinstance(tokenizer_cfg, MutableMapping):
        return False
    target = _tokenizer_node_target(tokenizer_cfg)
    if target == "build_processor_lazy":
        from cosmos_framework.data.generator.processors.cosmos3_edge_processing import is_cosmos3_edge_native_snapshot

        repository = tokenizer_cfg.get("repository")
        name = repository or tokenizer_cfg.get("tokenizer_type")
        if not isinstance(name, str) or not name:
            return False
        if _EDGE_REPO_SUBSTRING in name:
            return is_cosmos3_edge_native_snapshot(str(checkpoint_path))
        if repository or any(qwen_type in name for qwen_type in _QWEN3VL_TOKENIZER_TYPES):
            # Local-artifact (repository) nodes and Qwen3-VL tokenizer types both
            # dispatch to Qwen3VLProcessor in dir mode; an edge-native bundle
            # would instead route to the Nemotron bridge, so reject it.
            return _dir_serves_qwen3vl_autoprocessor(checkpoint_path) and not is_cosmos3_edge_native_snapshot(
                str(checkpoint_path)
            )
        return False
    if target == "create_qwen2_tokenizer_with_download":
        if tokenizer_cfg.get("config_variant") != "hf" or not tokenizer_cfg.get("pretrained_model_name"):
            return False
        return all((checkpoint_path / name).is_file() for name in _QWEN2_TOKENIZER_FILES)
    return False


def _point_tokenizer_node_at_dir(tokenizer_cfg: MutableMapping[str, Any], processor_path: Path | str) -> bool:
    """Rewrite ``vlm_config.tokenizer`` in place to source its processor from a local dir.

    Dispatches on the node's shape so the mutated kwargs match the target's
    signature (``build_processor_lazy`` takes ``tokenizer_type``; the Qwen2 node
    takes ``pretrained_model_name``). Returns False — node left completely
    untouched — for any other shape, so callers keep the configured hub behavior.
    """
    if not isinstance(tokenizer_cfg, MutableMapping):
        return False
    target = _tokenizer_node_target(tokenizer_cfg)
    if target == "build_processor_lazy" and (tokenizer_cfg.get("repository") or tokenizer_cfg.get("tokenizer_type")):
        tokenizer_cfg.pop("repository", None)
        tokenizer_cfg.pop("revision", None)
        tokenizer_cfg.pop("subdir", None)
        tokenizer_cfg["tokenizer_type"] = str(processor_path)
        return True
    if target == "create_qwen2_tokenizer_with_download" and tokenizer_cfg.get("config_variant") == "hf":
        tokenizer_cfg["pretrained_model_name"] = str(processor_path)
        return True
    return False


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

        # A multiview sample is batched alone, as in the reference implementation. Its batch dict carries
        # per-sample view metadata (num_views_per_vision_item and a camera-major video whose
        # frame count is views x frames); merging two of them produces a batch that is silently
        # wrong about how many views each item has rather than failing.
        if sa.multiview is not None:
            if cur:
                yield cur
                cur = []
                running_tokens = 0
                running_seqs = 0
            yield [item]
            continue

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


def _available_cpu_count() -> int:
    """Return the CPUs this rank and its FFmpeg children may schedule on."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


_MIN_MULTIVIEW_ENCODER_THREADS = 2
_MAX_MULTIVIEW_ENCODER_THREADS = 4


def _multiview_save_thread_budget(num_jobs: int, *, cpu_count: int | None = None) -> tuple[int, int]:
    """Return bounded writer concurrency and FFmpeg threads within this rank's CPU budget.

    The thread count is part of the output: x264 encodes the same frames to different bytes under a
    different ``-threads`` value, so a writer whose files must match the reference implementation's
    byte for byte takes its count from this budget, as the reference does.
    """
    affinity_cpu_count = _available_cpu_count() if cpu_count is None else max(1, cpu_count)
    minimum_encoder_threads = min(_MIN_MULTIVIEW_ENCODER_THREADS, affinity_cpu_count)
    workers = max(1, min(num_jobs, affinity_cpu_count // minimum_encoder_threads))
    threads = min(_MAX_MULTIVIEW_ENCODER_THREADS, max(minimum_encoder_threads, affinity_cpu_count // workers))
    return workers, threads


def _loads_training_experiment(setup_args: SetupArgs) -> bool:
    """Whether the model is built from a training experiment (a DCP checkpoint with a module config).

    Only an exported checkpoint records its caption layout (``per_view_captions`` in its
    ``config.json``). Nothing on this route does, so ``OmniInference.per_view_captions`` stays False
    there whatever the checkpoint was trained with.
    """
    return setup_args.checkpoint_type == CheckpointType.DCP and setup_args.config_file_type == ConfigFileType.MODULE


def _compute_num_tokens_for_sample(sample_args: OmniSampleArgs, model: OmniMoTModel) -> int:
    """Estimate the number of tokens for a single inference sample.

    Follows the counting logic in
    ``JointDataLoader._compute_num_tokens_per_sample`` (vision + text + EOS).
    """
    w, h = sample_args.vision_size
    T = sample_args.num_frames

    vision_tokenizer = model.tokenizer_vision_gen
    if vision_tokenizer is None:
        spatial_cf = cast(int, model.config.tokenizer.spatial_compression_factor)
        temporal_cf = cast(int, model.config.tokenizer.temporal_compression_factor)
    else:
        spatial_cf = vision_tokenizer.spatial_compression_factor
        temporal_cf = vision_tokenizer.temporal_compression_factor
    patch_spatial: int = model.config.diffusion_expert_config.patch_spatial

    vae_spatial_downsample = spatial_cf * patch_spatial
    vae_temporal_downsample = temporal_cf

    latent_h = h // vae_spatial_downsample
    latent_w = w // vae_spatial_downsample
    latent_t = 1 + (T - 1) // vae_temporal_downsample
    num_vision_tokens = latent_h * latent_w * latent_t

    if sample_args.multiview is not None:
        num_vision_items = 2 if sample_args.multiview.has_control else 1
        num_vision_tokens *= len(sample_args.multiview.views) * num_vision_items

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
    truncate_duration: bool = False,
) -> str:
    """Append duration/fps and resolution metadata to a prompt."""
    prompt = prompt.strip()
    if duration_template is not None and (num_frames > 1 or force_duration_template):
        duration = num_frames / fps
        if truncate_duration:
            duration = int(duration)
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


def _format_sample_prompt(
    raw_prompt: str,
    sample_args: OmniSampleArgs,
    *,
    h: int,
    w: int,
    duration_template: str | None,
    truncate_duration: bool = False,
) -> str:
    """Decorate one caption with the duration and resolution metadata training appended to it.

    Extracted so the per-view caption path can apply exactly this, once per camera, instead of
    growing a second copy that agrees today and drifts later. The decoration is what the model was
    trained to read; two versions of it would diverge invisibly.
    """
    prompt_obj = _parse_json_object_prompt(raw_prompt)
    if prompt_obj is not None:
        return _format_json_prompt_with_template(
            prompt_obj,
            fps=sample_args.fps,
            num_frames=sample_args.num_frames,
            aspect_ratio=sample_args.aspect_ratio,
            h=h,
            w=w,
            include_temporal_metadata=sample_args.num_frames > 1,
        )
    if not sample_args.native_prompt_upsampling:
        return _format_prompt_with_template(
            raw_prompt,
            fps=sample_args.fps,
            num_frames=sample_args.num_frames,
            duration_template=duration_template,
            resolution_template=sample_args.resolution_template,
            h=h,
            w=w,
            truncate_duration=truncate_duration,
        )
    # If native prompt upsampling is enabled, the duration and resolution metadata are added into
    # the upsampled JSON prompt directly.
    #
    # Returned unstripped, matching the reference implementation. This branch used to .strip() here, which only
    # differs for a non-JSON prompt carrying leading or trailing whitespace under
    # native_prompt_upsampling=True -- latent at the defaults, and a token-level difference
    # between the two runtimes wherever it is not.
    return raw_prompt


def _get_per_view_prompt_sample_data(
    sample_args: OmniSampleArgs,
    model: OmniMoTModel,
    view_prompts: Sequence[str],
    *,
    h: int,
    w: int,
    duration_template: str | None,
) -> dict:
    """The caption entry for a checkpoint that tokenizes one caption per camera.

    Every caption is decorated on its own, the way training's augmentors decorate each entry of
    the per-view caption list, and the sample's captions are nested one level deeper so the model
    reads a list of captions per sample instead of one prompt per sample.

    By default no negative caption is emitted.
    With no ``neg_<caption_key>`` in the batch the model falls to ``[""] * len(cond_tokens)`` --
    one empty caption per (sample, camera), which is the unconditional branch training's caption
    dropout produced. Emitting the sample's real negative instead would send it through
    ``_expand_negative_captions`` to be fanned out across the cameras, conditioning the
    unconditional pass on text training never paired with these weights. An explicit
    per_view_negative_prompt opts into a shared negative caption expanded per camera by the model.
    """
    if not view_prompts:
        raise ValueError("Per-view captions were requested for a sample that has no camera captions.")
    per_view_negative_prompt = getattr(sample_args, "per_view_negative_prompt", None)
    if sample_args.negative_prompt and per_view_negative_prompt is None:
        log.warning(
            "Ignoring negative_prompt: this checkpoint tokenizes one caption per camera, and its "
            "unconditional branch uses an empty caption per camera, as training's dropout did."
        )
    captions = [
        _format_sample_prompt(
            view_prompt,
            sample_args,
            h=h,
            w=w,
            duration_template=duration_template,
            # Multiview training uses whole seconds for both text and JSON captions. The JSON branch
            # already truncates (``int(num_frames / fps)`` in _format_json_prompt_with_template);
            # without this the text branch appends a fractional duration, so a 297-frame clip reads
            # "9.9 seconds" where training wrote "9 seconds". The video stays plausible and every
            # camera is conditioned on text the weights never saw.
            truncate_duration=True,
        )
        for view_prompt in view_prompts
    ]
    out: dict[str, Any] = {model.input_caption_key: [list(captions) for _ in range(sample_args.num_outputs)]}
    if per_view_negative_prompt:
        negative_caption = _format_sample_prompt(
            per_view_negative_prompt,
            sample_args,
            h=h,
            w=w,
            duration_template=duration_template,
            truncate_duration=True,
        )
        # A sample-level negative string is expanded to each camera by _expand_negative_captions.
        out["neg_" + model.input_caption_key] = [negative_caption] * sample_args.num_outputs
    return out


def _get_prompt_sample_data(
    sample_args: OmniSampleArgs,
    model: OmniMoTModel,
    *,
    h: int,
    w: int,
    device: Any,
    view_prompts: list[str] | None = None,
) -> dict:
    duration_template = sample_args.duration_template
    inverse_duration_template = sample_args.inverse_duration_template

    if view_prompts is not None:
        return _get_per_view_prompt_sample_data(
            sample_args,
            model,
            view_prompts,
            h=h,
            w=w,
            duration_template=duration_template,
        )

    prompt = _format_sample_prompt(
        sample_args.prompt,
        sample_args,
        h=h,
        w=w,
        duration_template=duration_template,
    )
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


def _decode_reasoner_video(vision_path: str, video_fps: float | None) -> dict[str, Any]:
    """Decode a local video file into the frame-list payload the Qwen3-VL processor expects.

    Returns ``{"frames": [PIL.Image, ...], "fps": float}``. Uses the same TorchCodec
    decode the rest of the inference path relies on (no ``decord`` dependency), then
    uniformly samples frames toward ``video_fps`` (default 2.0) via Qwen's
    ``smart_nframes``. The repo ``Qwen3VLProcessor`` runs with ``do_sample_frames=False``,
    so it consumes this pre-sampled frame list as-is and handles its own per-frame resize."""
    frames, src_fps = decode_video_thwc_uint8(Path(vision_path))  # [T,H,W,C] uint8
    total_frames = int(frames.shape[0])
    if total_frames == 0:
        raise ValueError(f"Decoded zero frames from reasoner video: {vision_path}")
    src_fps = src_fps or 1.0
    target_fps = video_fps if video_fps is not None else 2.0
    nframes = smart_nframes({"fps": target_fps}, total_frames=total_frames, video_fps=src_fps)
    idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
    pil_frames = [Image.fromarray(frames[i].numpy()) for i in idx]
    sample_fps = nframes / total_frames * src_fps
    return {"frames": pil_frames, "fps": sample_fps}


def _get_reasoner_sample_data(sample_args: OmniSampleArgs, model: OmniMoTModel) -> dict[str, Any]:
    """Sample batch for reasoner text generation: prompt + optional conditioning image or video."""
    image: Image.Image | None = None
    video: dict[str, Any] | None = None
    if sample_args.vision_path is not None:
        if Path(sample_args.vision_path).suffix.lower() in VIDEO_EXTENSIONS:
            video = _decode_reasoner_video(str(sample_args.vision_path), sample_args.video_fps)
        else:
            image = Image.open(sample_args.vision_path).convert("RGB")
    # Both keys are emitted for every sample (``None`` when absent) so the batch
    # builder can positionally align them and the three-way homogeneity check in
    # ``_generate_reasoner_batch`` reliably detects an image/video/text mix.
    return {
        model.input_caption_key: [sample_args.prompt],
        "reasoner_images": [image],
        "reasoner_videos": [video],
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

    # A causal multiview student loads its synchronized controls and caption timeline itself, chunk
    # by chunk (inference/multiview_causal.py). That covers its joint camera+LiDAR samples too, so
    # this precedes the joint branch below, whose batch belongs to the bidirectional joint sampler.
    if sample_args.multiview is not None and is_causal_multiview_config(getattr(model, "config", None)):
        return {}

    # Before the multiview branch: a joint sample has multiview views and would be captured by it,
    # but it builds its batch in generate_joint_cam_lidar_sample instead -- and would be refused
    # here anyway, since get_multiview_sample_data accepts a single transfer hint and joint
    # carries two (wsm + lidar). Empty dict for the same reason plain transfer returns one: the
    # generation path owns batch assembly.
    if sample_args.is_joint_cam_lidar_transfer:
        return {}

    if sample_args.multiview is not None:
        from cosmos_framework.inference.multiview import get_multiview_sample_data

        return get_multiview_sample_data(sample_args, model, device=device)

    if sample_args.transfer_hints:
        return {}

    if sample_args.camera_trajectory is not None:
        # Special pathway for interactive AR generation with a camera trajectory:
        # the action tensor is derived from the trajectory rather than read from an
        # action file, so this precedes the action-mode branch.
        from cosmos_framework.inference.camera import get_camera_sample_data

        if sample_args.resolution is None:
            raise ValueError("'resolution' is required for action-conditioned inference.")
        if sample_args.num_frames < 2:
            raise ValueError("'num_frames' must be >= 2 for action-conditioned inference.")

        action_chunk_size = sample_args.num_frames - 1
        if sample_args.camera_num_frames is not None:
            # Consume a fixed number of camera poses regardless of the frame count
            # the tokenizer aligned to.
            action_chunk_size = sample_args.camera_num_frames - 1

        return get_camera_sample_data(
            input_video_key=model.input_video_key,
            batch_size=sample_args.num_outputs,
            prompt=sample_args.prompt,
            vision_path=sample_args.vision_path,
            vision_size=sample_args.vision_size,
            camera_trajectory=sample_args.camera_trajectory,
            domain_name=sample_args.domain_name,
            resolution=sample_args.resolution,
            content_crop_box=sample_args.content_crop_box,
            action_chunk_size=action_chunk_size,
            max_action_dim=model.config.max_action_dim,
            camera_translation_scale=sample_args.camera_translation_scale,
            camera_action_normalization=sample_args.camera_action_normalization,
            camera_pose_convention=sample_args.camera_pose_convention,
            fps=sample_args.fps,
            device=device,
            duration_template=sample_args.duration_template,
            resolution_template=sample_args.resolution_template,
            output_dir=sample_args.output_dir,
        )

    if sample_args.model_mode.is_action:
        from cosmos_framework.inference.action import get_action_sample_data

        assert sample_args.vision_path is not None
        if getattr(model, "_bimanual_action_contract", None) is not None:
            if sample_args.num_frames != sample_args.action_chunk_size + 1:
                raise ValueError("Bimanual num_frames must equal action_chunk_size + 1 after tokenizer alignment")
        return get_action_sample_data(
            model_config=model,
            batch_size=sample_args.num_outputs,
            prompt=sample_args.prompt,
            vision_path=sample_args.vision_path,
            model_mode=sample_args.model_mode,
            action_path=sample_args.action_path,
            domain_name=sample_args.domain_name,
            view_point=sample_args.view_point,
            raw_action_dim=sample_args.raw_action_dim,
            format_prompt_as_json=sample_args.format_prompt_as_json,
            resolution=str(sample_args.image_size),
            content_crop_box=sample_args.content_crop_box,
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

    All mutations (key renames, tensor → list unbind, or list-item replacement)
    are applied to a fresh copy so the caller's input dict is never modified. This keeps
    the "no aliasing" responsibility localized at the single place where the
    mutation happens, instead of forcing every producer of a data dict (e.g.
    seed-expansion fan-out in ``create_batches_from_dataset``, dummy padding
    batches in ``create_batches``) to defensively copy before handing the
    dict to ``generate_batch``.

    The top-level dict and list containers are copied; tensors and other values
    inside those lists remain shared. Copying each list is necessary because
    downstream preprocessing replaces video-list items with normalized tensors.

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
    data_batch = {key: list(value) if isinstance(value, list) else value for key, value in data_batch.items()}

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
class _SamplePreparationFailure:
    index: int
    message: str
    stack_trace: str
    recoverable: bool


@dataclass
class OmniInference(Inference):
    # pyrefly: ignore[bad-override]
    model: OmniMoTModel
    vae_decode_stream: torch.cuda.Stream | None = None
    # Exported caption metadata selects per-camera text on the caption-chunk path.
    # See inference/common/multiview_captions.py for the serialized field names.
    per_view_captions: bool = False
    per_view_caption_task: str | None = None

    @property
    def model_config(self) -> "OmniMoTModelConfig":
        return self.model.config

    @classmethod
    def _get_parallelism_config(cls, setup_args: ParallelismArgs) -> ParallelismConfig:
        # The VAE work-sharing group is one inference replica, the cp x cfgp ranks that generate one
        # sample together, so its ranks can split the per-sample media loads, VAE encodes and VAE
        # decodes they would otherwise all repeat (see model/generator/mot/replica_partition.py).
        # The separate-VAE-GPU mode rebinds encode/decode on the replica's first rank only, so its
        # ranks would not run the same VAE: no sharing there.
        # This applies to every mode run at cp x cfgp > 1, as in the reference. A shared encode runs
        # each camera view or LiDAR item on one rank and broadcasts its latent, which reproduces the
        # unshared encode's bytes only when the tokenizer's output does not depend on random draws:
        # true of Wan2.2 (the default, and every shipped config) and of the LiDAR VAE (posterior
        # mean by default), not of the SD and Flux VAEs, which sample their posterior.
        replica_size = setup_args.cp_size * setup_args.cfgp_size
        return ParallelismConfig(
            enable_inference_mode=True,
            data_parallel_shard_degree=setup_args.dp_shard_size,
            context_parallel_shard_degree=setup_args.cp_size,
            cfg_parallel_shard_degree=setup_args.cfgp_size,
            vae_load_balance_group_size=1 if setup_args.use_separate_pipeline_vision_decode_gpu else replica_size,
        )

    @classmethod
    def _get_compile_config(cls, setup_args: ParallelismArgs) -> CompileConfig:
        return CompileConfig(
            # Translate the flat ``OmniSetupOverrides.use_torch_compile`` public
            # surface into R1's nested ``CompileConfig.enabled`` knob.
            enabled=setup_args.use_torch_compile,
            use_cuda_graphs=setup_args.use_cuda_graphs
            and setup_args.dp_shard_size * setup_args.cp_size * setup_args.cfgp_size == 1,
            cuda_graph_scope=setup_args.cuda_graph_scope,
            compiled_region=setup_args.compiled_region,
            compile_dynamic=setup_args.compile_dynamic,
            ar_post_saturation_mode=setup_args.ar_post_saturation_mode,
        )

    @classmethod
    def _get_quantization_config(cls, setup_args: SetupArgs) -> QuantizationConfig:
        return QuantizationConfig(
            method=setup_args.quantization_method,
            fp8_granularity=setup_args.quantization_fp8_granularity,
            include_regex=list(setup_args.quantization_include_regex),
            exclude_regex=list(setup_args.quantization_exclude_regex),
            mixed_precision_first_steps=setup_args.mixed_precision_first_steps,
            mixed_precision_last_steps=setup_args.mixed_precision_last_steps,
            mixed_precision_reasoner_policy=setup_args.mixed_precision_reasoner_policy,
            mixed_precision_w8a16_cache=setup_args.mixed_precision_w8a16_cache,
        )

    @override
    @classmethod
    def _create(cls, setup_args: SetupArgs, **kwargs: Any) -> Self:
        assert isinstance(setup_args, OmniSetupArgs)
        assert setup_args.output_dir is not None

        sampler_override = setup_args.sampler
        parallelism_config = cls._get_parallelism_config(setup_args)
        attention_io_layout = setup_args.attention_io_layout
        compile_config = cls._get_compile_config(setup_args)
        quantization_config = cls._get_quantization_config(setup_args)
        # Only an exported checkpoint records these; the module-config path has no equivalent.
        exported_per_view_captions = False
        exported_per_view_caption_task: str | None = None
        exported_metadata: Any = None
        if _loads_training_experiment(setup_args):
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
                quantization_config=attrs.asdict(quantization_config),
                load_ema_to_reg=setup_args.use_ema_weights,
                experiment_opts=[
                    *setup_args.experiment_overrides,
                    f"model.config.rectified_flow_inference_config.scheduler_type={sampler_override}",
                    f"model.config.attention_io_layout={attention_io_layout}",
                ],
                use_cache_checkpoint=setup_args.checkpoint_cache_dir is not None,
                cache_checkpoint_rootdir=str(setup_args.checkpoint_cache_dir or ""),
            )
            model = cast("OmniMoTModel", model)
            Cosmos3OmniModel.after_load_model(model)
            # Thread a local checkpoint dir to the reasoner LM (consumed by
            # Edge's lazy ``_ensure_vision_tower``) so a checkpoint-bundled
            # ``vision_encoder/`` is preferred over a hub download. s3 DCP
            # loads have no local dir to offer — skip them.
            if "://" not in str(setup_args.checkpoint_path) and Path(setup_args.checkpoint_path).is_dir():
                language_model = getattr(getattr(model, "net", None), "language_model", None)
                if language_model is not None:
                    language_model._local_checkpoint_dir = str(setup_args.checkpoint_path)
            save_config(config, setup_args.output_dir)
        else:
            checkpoint_path = _download_on_rank0(setup_args.download_checkpoint)
            if setup_args.config_file_type == ConfigFileType.MODULE:
                config = None
            else:
                model_dict = setup_args.load_model_config_dict()
                tokenizer_cfg = model_dict["config"]["vlm_config"]["tokenizer"]
                has_bundled_processor = _checkpoint_has_processor_files(checkpoint_path)
                if setup_args.vlm_processor_from_checkpoint:
                    # Source the VLM processor from the loaded checkpoint's own
                    # bundled files instead of the repository hardcoded in the
                    # model config. Drops the redundant base-model download.
                    processor_path = checkpoint_path
                elif has_bundled_processor and _bundled_processor_serves_node(tokenizer_cfg, checkpoint_path):
                    # Local-first: exported checkpoints bundle processor/tokenizer
                    # files at their root (export_model); prefer them over the
                    # repository hardcoded in the model config so exported dirs
                    # stay offline-capable. Gated on the bundle actually being
                    # dispatchable for this tokenizer node.
                    log.info(f"VLM processor: using checkpoint-bundled processor files at {checkpoint_path}")
                    processor_path = checkpoint_path
                else:
                    if has_bundled_processor:
                        log.info(
                            f"VLM processor: checkpoint-bundled files at {checkpoint_path} cannot serve this "
                            "tokenizer config; falling back to the configured source"
                        )
                    if repository := tokenizer_cfg.get("repository"):
                        revision = tokenizer_cfg.get("revision")
                        if revision is None:
                            raise ValueError("VLM processor 'revision' is required when 'repository' is set")
                        processor_path = _download_on_rank0(
                            CheckpointDirHf(
                                repository=repository,
                                revision=revision,
                                subdirectory=tokenizer_cfg.get("subdir", ""),
                                include=_PROCESSOR_HF_INCLUDE,
                            ).download
                        )
                    else:
                        processor_path = None
                if processor_path is not None and not _point_tokenizer_node_at_dir(tokenizer_cfg, processor_path):
                    log.info(
                        "VLM processor: unrecognized 'vlm_config.tokenizer' node shape; leaving the configured "
                        "tokenizer source untouched"
                    )
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
            if setup_args.kv_cache_inference_size is not None or setup_args.attention_sink_size is not None:
                # Local HF checkpoints using a module config do not consume Hydra
                # overrides. Apply public cache settings before model construction.
                if config is None:
                    config = Cosmos3OmniConfig.from_pretrained(checkpoint_path)
                if setup_args.kv_cache_inference_size is not None:
                    config.model["config"]["kv_cache_inference_size"] = setup_args.kv_cache_inference_size or None
                if setup_args.attention_sink_size is not None:
                    config.model["config"]["attention_sink_size"] = setup_args.attention_sink_size
            # The requested attention layout overrides the checkpoint's. A config that does not record
            # one already runs the model default (sequence_sharded), so it is only written when needed.
            if config is None:
                probe = Cosmos3OmniConfig.from_pretrained(checkpoint_path)
                recorded_layout = probe.model.get("config", {}).get("attention_io_layout")
                if attention_io_layout != _DEFAULT_ATTENTION_IO_LAYOUT or (
                    recorded_layout is not None and recorded_layout != attention_io_layout
                ):
                    config = probe
            if config is not None:
                model_config_dict = config.model.setdefault("config", {})
                if "attention_io_layout" in model_config_dict or attention_io_layout != _DEFAULT_ATTENTION_IO_LAYOUT:
                    model_config_dict["attention_io_layout"] = attention_io_layout
            hf_model = Cosmos3OmniModel.from_pretrained_dcp(
                checkpoint_path,
                config=config,
                parallelism_config=parallelism_config,
                compile_config=compile_config,
                quantization_config=quantization_config,
            )
            # The training dataloaders that carry the caption layout do not travel with an
            # exported checkpoint, so the exporter records the answer as top-level keys in
            # config.json. Read that file, not the loaded wrapper: on this path `config` above is
            # `Cosmos3OmniConfig(model=model_dict)`, built from the file's `model` section alone,
            # so every other top-level key is gone by the time the model exists. Reading the
            # wrapper instead reported "not per-view" for a checkpoint that says otherwise, and
            # the refusal below silently never fired (measured on the camera+LiDAR production
            # export, 2026-09-16).
            #
            # The wrapper is still the fallback for the branch that passes `config=None` and lets
            # `from_pretrained_dcp` parse config.json itself. getattr, not attribute access: it is
            # stubbed in tests as a bare object carrying only `.model`, and a wrapper with no
            # config has nothing to declare, which the readers treat as "not per-view".
            exported_metadata = exported_checkpoint_metadata(checkpoint_path)
            if exported_metadata is None:
                exported_metadata = getattr(hf_model, "config", None)
            exported_per_view_captions = exports_per_view_captions(exported_metadata)
            exported_per_view_caption_task = exported_caption_task(exported_metadata)
            model = hf_model.model
            if model.config.rectified_flow_inference_config.scheduler_type != sampler_override:
                model.config.rectified_flow_inference_config.scheduler_type = sampler_override
                model.set_up_scheduler_and_sampler()
                log.debug(f"Sampler overridden to: {sampler_override}")

        _configure_vae_compile(model, setup_args)
        if hasattr(model, "maybe_install_ar_post_saturation_mode"):
            model.maybe_install_ar_post_saturation_mode()
        vae_decode_stream: torch.cuda.Stream | None = None
        vision_tokenizer = model.tokenizer_vision_gen
        if setup_args.use_separate_pipeline_vision_decode_gpu and vision_tokenizer is None:
            log.info("Separate vision decode GPU setup skipped because the generation vision tokenizer is not loaded")
        elif setup_args.use_separate_pipeline_vision_decode_gpu:
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
                vae = vision_tokenizer.model
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

        # Runtime-only policy: preserve immutable published checkpoint files and the shared
        # argument-builder signature. CLI, Ray and Python callers all use pipe.model_config.
        checkpoint_hf = setup_args.checkpoint_hf
        setattr(
            model.config,
            "_auto_camera_fps",
            auto_camera_fps(
                exported_metadata, repository=checkpoint_hf.repository if checkpoint_hf is not None else None
            ),
        )
        pipe = cls(setup_args=setup_args, model=model, vae_decode_stream=vae_decode_stream, **kwargs)
        pipe.per_view_captions = exported_per_view_captions
        pipe.per_view_caption_task = exported_per_view_caption_task
        if pipe.per_view_captions:
            log.info(
                "This checkpoint packs one caption per camera "
                f"(task={pipe.per_view_caption_task or 'unrecorded'}). Multiview generation reads "
                "each camera's own caption; this needs multiview.load_caption_from_data=True with a "
                "caption_path per camera, and is refused without it."
            )
        # Install here (not only in scripts/inference.py) so all callers —
        # cosmos-benchmarks EvalModel, Ray serve, CLI — get the default-on cache.
        cls._maybe_install_diffusion_cache(pipe, setup_args)
        return pipe

    @staticmethod
    def _maybe_install_diffusion_cache(pipe: "OmniInference", setup_args: SetupArgs) -> None:
        """Install SeaCache on ``pipe.model`` when ``setup_args.diffusion_cache`` is set.

        ``num_steps`` is adopted per ``generate_samples_from_batch`` call via
        ``begin_generation``, so install-time ``sample_args_list`` can be empty.
        """
        if not setup_args.diffusion_cache:
            return
        # The setup contract turns the cache off for an exported causal multiview checkpoint; this also
        # covers checkpoints whose config.json it does not read (DCP with a training config).
        if is_causal_multiview_config(getattr(pipe.model, "config", None)):
            log.info("Diffusion cache is disabled for causal multiview replay; its KV cache is model-controlled.")
            return
        from cosmos_framework.model.generator.mot.diffusion_cache import install_diffusion_cache

        config_overrides: dict[str, Any] = {}
        if setup_args.diffusion_cache_thresh is not None:
            config_overrides["diffusion_cache_thresh"] = setup_args.diffusion_cache_thresh
        if setup_args.diffusion_cache_residual_order is not None:
            config_overrides["residual_order"] = setup_args.diffusion_cache_residual_order
        if setup_args.diffusion_cache_max_consecutive_cached is not None:
            config_overrides["max_consecutive_cached"] = setup_args.diffusion_cache_max_consecutive_cached
        install_diffusion_cache(
            pipe=pipe,
            enabled=True,
            sample_args_list=[],
            config_overrides=config_overrides or None,
        )

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
    def create_batches(self, sample_args_list: Sequence[SampleArgs]) -> Generator[PreparedBatch]:
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

        if self.setup_args.keep_going:
            yield from self._create_batches_keep_going(
                dataset, sampler_indices, batch_position_lists, global_max_batches
            )
            return

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

    def _create_batches_keep_going(
        self,
        dataset: SampleDataset,
        sampler_indices: list[int],
        batch_position_lists: list[list[int]],
        global_max_batches: int,
    ) -> Generator[PreparedBatch]:
        """Isolate input failures without desynchronizing inference replicas.

        All ranks exchange preparation outcomes once per scheduled batch. A
        sample rejected by any rank is excluded on every rank of its replica.
        Empty replicas use a valid sample as padding; globally empty rounds do
        not generate. Unexpected runtime/CUDA errors remain fatal on all ranks.
        """
        for batch_index in range(global_max_batches):
            positions = batch_position_lists[batch_index] if batch_index < len(batch_position_lists) else []
            prepared: dict[int, tuple[SampleArgs, dict[str, Any]]] = {}
            failures: list[_SamplePreparationFailure] = []
            for pos in positions:
                sample_index = sampler_indices[pos]
                try:
                    prepared[sample_index] = dataset[sample_index]
                except Exception as error:
                    failures.append(
                        _SamplePreparationFailure(
                            sample_index,
                            f"{type(error).__name__}: {error}",
                            traceback.format_exc(),
                            isinstance(error, (ValueError, OSError, TypeError, KeyError)),
                        )
                    )

            outcome = (list(prepared), failures)
            outcomes = [outcome]
            if torch.distributed.is_initialized():
                gathered: list[Any] = [None] * get_world_size()
                torch.distributed.all_gather_object(gathered, outcome)
                outcomes = gathered
            all_failures = [failure for _, errors in outcomes for failure in errors]
            fatal = next((failure for failure in all_failures if not failure.recoverable), None)
            if fatal is not None:
                raise RuntimeError(f"Sample preparation failed: {fatal.message}\n{fatal.stack_trace}")
            rejected = {failure.index: failure for failure in all_failures}

            valid_indices = sorted({index for indices, _ in outcomes for index in indices} - rejected.keys())
            local_samples = [sample for index, sample in prepared.items() if index not in rejected]
            error_outputs: list[SampleOutputs] = []
            # Every replica still makes one generate_batch call in an active
            # round. Never use dataset[0] as padding: it may be the bad sample.
            with sync_distributed_errors():
                for pos in positions:
                    sample_index = sampler_indices[pos]
                    if sample_index not in rejected:
                        continue
                    sample_args = dataset._sample_args_list[sample_index]
                    if self.should_process_sample(sample_args):
                        failure = rejected[sample_index]
                        message = f"Error preparing sample '{sample_args.name}': {failure.message}"
                        log.error(message)
                        output = SampleOutputs(
                            args=sample_args.model_dump(mode="json"),
                            status="error",
                            message=message,
                            stack_trace=failure.stack_trace,
                        )
                        assert sample_args.output_dir is not None
                        sample_args.output_dir.mkdir(parents=True, exist_ok=True)
                        (sample_args.output_dir / "sample_outputs.json").write_text(output.model_dump_json())
                        error_outputs.append(output)
                if local_samples:
                    chunk_args = [args for args, _ in local_samples]
                    chunk_data = _merge_data_batches([data for _, data in local_samples])
                    if self.setup_args.debug:
                        for args, data in local_samples:
                            if self.should_process_sample(args):
                                assert args.output_dir is not None
                                self.save_data(data, output_dir=args.output_dir, output_name="sample_data")
                elif valid_indices:
                    padding_args, chunk_data = dataset[valid_indices[0]]
                    chunk_args = [
                        padding_args.model_copy(
                            update={"output_dir": None, "name": "padding", "num_steps": 1, "guidance": 1.0}
                        )
                    ]
            yield from error_outputs
            if valid_indices:
                yield chunk_args, chunk_data

    def open_bimanual_session(
        self,
        *,
        vision_path: Path,
        prompt: str,
        domain_name: str = "agibotworld",
        fps: int = 30,
        seed: int = 0,
        resolution: str = "480",
    ) -> "BimanualSession":
        """Start an eager, single-process session without a precomputed action file.

        See ``docs/bimanual_sessions.md`` for decoded feedback, live controls,
        actual retained-history inspection and reset/close behavior.
        """
        from cosmos_framework.inference.action import build_action_batch
        from cosmos_framework.inference.bimanual_session import BimanualSession, require_idle_bimanual_model
        from cosmos_framework.inference.vision import read_media_frames

        require_idle_bimanual_model(self.model)
        if self.vae_decode_stream is not None:
            raise ValueError("Bimanual sessions do not support a separate VAE decode device")
        contract = getattr(self.model, "_bimanual_action_contract", None)
        if contract is None or domain_name not in contract.domain_ids:
            raise ValueError("Select a schema-5 Bimanual checkpoint and one of its supported action domains")
        frames, _ = read_media_frames(Path(vision_path), max_frames=1)
        rows = (
            self.model.tokenizer_vision_gen.temporal_compression_factor
            * self.model.config.teacher_forcing_frames_per_chunk
        )
        # Seed-batch placeholders are never played back: every transition is
        # supplied by session.step(). Reuse the native image/action preprocessing.
        data = build_action_batch(
            video=frames,
            action=torch.zeros(rows, contract.model_action_dim),
            raw_action_dim=contract.action_dim,
            prompt=prompt,
            view_point="ego_view",
            domain_name=domain_name,
            model_mode=ModelMode.FORWARD_DYNAMICS,
            action_chunk_size=rows,
            fps=fps,
            resolution=resolution,
            input_video_key=self.model.input_video_key,
            action_domain_ids=torch.tensor(contract.domain_ids[domain_name], dtype=torch.long),
            format_prompt_as_json=False,
            device=self.model.tensor_kwargs["device"],
        )
        return BimanualSession(
            self.model,
            _finalize_data_batch(data, 1, self.model),
            domain_name=domain_name,
            fps=fps,
            seed=seed,
        )

    @torch.no_grad()
    @override
    def generate_batch(
        self,
        sample_args_list: Sequence[SampleArgs],
        data_batch: dict[str, Any],
        *,
        save_outputs: bool = True,
    ) -> list[SampleOutputs]:
        from cosmos_framework.inference.bimanual_session import require_idle_bimanual_model

        require_idle_bimanual_model(self.model)
        assert all(isinstance(sa, OmniSampleArgs) for sa in sample_args_list)
        for sample in sample_args_list:
            # Also protect callers that construct or modify already-built OmniSampleArgs.
            check_auto_fps_options(cast(OmniSampleArgs, sample), self.model.config)

        # A multiview sample driven by a multiview caption file uses the chunkwise path, as in
        # the reference implementation. Without this dispatch load_caption_from_data was accepted, validated
        # and downloaded, then silently ignored: generation fell through to the one-shot
        # builder with the raw prompt, full num_frames, and the unsliced control head.
        multiview_flags = [cast(OmniSampleArgs, sa).multiview is not None for sa in sample_args_list]
        if any(multiview_flags):
            assert all(multiview_flags), "Cannot mix multiview and non-multiview samples in a batch"
            assert len(sample_args_list) == 1, "Batching is not supported for multiview inference"
            mv_args = cast(OmniSampleArgs, sample_args_list[0])
            assert mv_args.multiview is not None
            # A causal multiview student (e.g. Cosmos3-Nano-Sim-Auto) is sampled chunk by chunk with a
            # rolling KV cache, RGB and joint camera+LiDAR samples alike. Placed before every branch
            # below: each would accept its samples and run the one-shot bidirectional sampler on
            # them, producing a plausible but wrong video with no error.
            if is_causal_multiview_config(self.model.config):
                return self._generate_causal_multiview_batch(mv_args, save_outputs=save_outputs)
            # Joint camera+LiDAR denoises both streams in one pass, so it takes its own path
            # rather than either multiview branch below. Placed ahead of them because it handles both
            # caption modes itself and raises its own message when per-view text is required.
            if mv_args.is_joint_cam_lidar_transfer:
                return self._generate_joint_cam_lidar_batch(mv_args, save_outputs=save_outputs)
            if self.per_view_captions and not mv_args.multiview.load_caption_from_data:
                # The weights read one caption per camera, and per-camera text only exists on the
                # caption-chunk path -- the one-shot builder below has a single prompt and nowhere
                # to get eleven. Serving anyway costs nothing visible: the video is plausible and
                # only the text the model was conditioned on differs from training. So refuse
                # instead of producing output that cannot be told apart from a correct run.
                #
                # Narrowed from refusing every per-view checkpoint: the caption-chunk path can now
                # pack per-camera text, so it is no longer turned away.
                raise NotImplementedError(
                    "This checkpoint was trained with separate_view_text_tokenization: each camera "
                    "attends to its own caption"
                    + (f" under the '{self.per_view_caption_task}' task" if self.per_view_caption_task else "")
                    + ". Per-camera text is only available on the caption-chunk path; set "
                    "multiview.load_caption_from_data=True with a caption_path per camera."
                )
            if mv_args.multiview.load_caption_from_data:
                return self._generate_multiview_caption_chunks_batch(mv_args, save_outputs=save_outputs)

        transfer_flags = [
            bool(sa.transfer_hints) and cast(OmniSampleArgs, sa).multiview is None for sa in sample_args_list
        ]
        if any(transfer_flags):
            assert all(transfer_flags), "Cannot mix transfer and non-transfer samples in a batch"
            assert len(sample_args_list) == 1, "Batching is not supported for transfer inference"
            return self._generate_transfer_batch(sample_args_list[0], save_outputs=save_outputs)

        reasoner_flags = [cast(OmniSampleArgs, sa).model_mode.is_reasoner for sa in sample_args_list]
        if any(reasoner_flags):
            assert all(reasoner_flags), "Cannot mix reasoner and non-reasoner samples in a batch"
            return self._generate_reasoner_batch(sample_args_list, data_batch, save_outputs=save_outputs)

        # Process inputs
        try:
            with self._get_timer(f"{self.__class__.__name__}.prepare_inputs"), sync_distributed_errors():
                for sample_args in sample_args_list:
                    if self.should_process_sample(sample_args) and save_outputs:
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
            # Apply the same error policy to warmup and measured passes:
            # fail fast unless keep_going is enabled.
            return [
                self._handle_sample_exception(args, e) for args in sample_args_list if self.should_process_sample(args)
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
                _steps_t = torch.tensor([local_num_steps], device=self.model.tensor_kwargs["device"], dtype=torch.int32)
                torch.distributed.all_reduce(
                    _steps_t, op=torch.distributed.ReduceOp.MAX, group=parallel_dims.dp_shard_mesh.get_group()
                )
                align_num_steps = int(_steps_t.item())

            # Autoregressive generation decodes each yielded latent chunk inside a
            # single cached-decoder scope. The causal VAE carries decoder state across
            # chunks, so decoding the assembled clip in one call yields different
            # pixels than the native runtime, which decodes chunk by chunk.
            if bool(getattr(omni_sample_args_list[0], "autoregressive", False)):
                assert len(omni_sample_args_list) == 1, "Autoregressive generation requires exactly one sample"
                ar_sample_args = omni_sample_args_list[0]
                # Action-conditioned interactive generation drives the AR loop in
                # forward_dynamics regardless of the declared sample modality.
                if ar_sample_args.action_condition:
                    ar_mode = "forward_dynamics"
                else:
                    ar_mode = ar_sample_args.model_mode.value
                sampler_mode = "distilled" if is_distilled else "rf"
                decoded_vision_frames: list[torch.Tensor] = []
                frustum_geometry: dict[str, Any] | None = None
                if ar_sample_args.kv_cache_frustum_path is not None:
                    from cosmos_framework.inference.bimanual import load_bimanual_frustum_geometry

                    frustum_geometry = load_bimanual_frustum_geometry(
                        ar_sample_args.kv_cache_frustum_path,
                        ar_sample_args.action_path,
                    )
                # With a separate decode GPU, submit each clean chunk through the
                # callback before KV refresh so decode overlaps refresh.
                use_separate_gpu_decode = self.vae_decode_stream is not None and should_decode_outputs

                def decode_and_append_vision_chunk(vision_latent: torch.Tensor) -> None:
                    decoded_vision_frames.append(decode_vision(vision_latent))

                with (
                    self.model.tokenizer_vision_gen.use_cached_decoder(),
                    self._get_timer(f"{self.model.__class__.__name__}.iter_samples_from_batch_autoregressive_decode"),
                ):
                    for ar_output in self.model.iter_samples_from_batch_autoregressive(
                        data_batch,
                        mode=ar_mode,
                        guidance=guidance,
                        seed=seed[0],
                        num_steps=ar_sample_args.num_steps,
                        shift=_getattr(sample_args_list, "shift"),
                        normalize_cfg=_getattr(sample_args_list, "normalize_cfg"),
                        sampler_mode=sampler_mode,
                        on_clean_vision_chunk=(decode_and_append_vision_chunk if use_separate_gpu_decode else None),
                        has_negative_prompt=neg_key in data_batch,
                        frustum_geometry=frustum_geometry,
                    ):
                        if not use_separate_gpu_decode:
                            decode_and_append_vision_chunk(ar_output["vision"])

                if self.vae_decode_stream is not None:
                    torch.cuda.current_stream(device=decoded_vision_frames[0].device).wait_stream(
                        self.vae_decode_stream
                    )
                outputs = {"vision": [torch.cat(decoded_vision_frames, dim=2).cpu()]}

            if outputs is None:
                # Multiview transfer honours control_guidance, as it does in the reference implementation. Without
                # this the value is accepted by arg validation and then discarded: the extra
                # no-control forward pass and the velocity mix never happen, so a run with
                # control_guidance != 1.0 silently produces a different video. At the pinned
                # control_guidance=1.0 used for parity this branch is inert, which is why the
                # bit-exact runs passed without it.
                generation_kwargs: dict[str, Any] = {}
                mv_sample_args = sample_args_list[0]
                if mv_sample_args.multiview is not None and mv_sample_args.transfer_hints:
                    from cosmos_framework.inference.transfer import build_control_cfg_postprocess

                    generation_kwargs["velocity_postprocess_builder"] = build_control_cfg_postprocess(
                        control_guidance=mv_sample_args.control_guidance,
                        control_guidance_interval=mv_sample_args.control_guidance_interval,
                    )

                with self._get_timer(f"{self.model.__class__.__name__}.generate_samples_from_batch"):
                    outputs = self.model.generate_samples_from_batch(
                        data_batch,
                        sampler=sampler,
                        **generation_kwargs,
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
                        use_batched_cfg=getattr(self.setup_args, "use_batched_cfg", False),
                        upsample_task=upsample_task,
                        upsample_max_new_tokens=_getattr(omni_sample_args_list, "prompt_upsampler_max_tokens"),
                        upsample_temperature=_getattr(omni_sample_args_list, "prompt_upsampler_temperature"),
                        upsample_top_k=_getattr(omni_sample_args_list, "prompt_upsampler_top_k"),
                        upsample_top_p=_getattr(omni_sample_args_list, "prompt_upsampler_top_p"),
                        upsample_repetition_penalty=_getattr(
                            omni_sample_args_list, "prompt_upsampler_repetition_penalty"
                        ),
                        upsample_presence_penalty=_getattr(omni_sample_args_list, "prompt_upsampler_presence_penalty"),
                        upsample_seed=_getattr(omni_sample_args_list, "prompt_upsampler_seed"),
                    )

                with self._get_timer(f"{self.model.__class__.__name__}.decode"):
                    output_vision = outputs.pop("vision")
                    if "enable_per_camera_vae_encoding" in data_batch:
                        from cosmos_framework.utils.generator.multiview import decode_multiview_latent_per_view

                        _mv_n_views = int(data_batch["sample_n_views"].reshape(-1)[0].item())
                        _mv_frames_per_view = int(data_batch["num_video_frames_per_view"].reshape(-1)[0].item())

                        def _decode_for_multiview(vision_latent: torch.Tensor) -> torch.Tensor:
                            decoded = decode_vision(vision_latent)
                            if self.vae_decode_stream is not None:
                                torch.cuda.current_stream(device=decoded.device).wait_stream(self.vae_decode_stream)
                            return decoded

                        decoded_vision = [
                            decode_multiview_latent_per_view(
                                _decode_for_multiview, vision, _mv_n_views, _mv_frames_per_view
                            )
                            for vision in output_vision
                        ]
                    else:
                        decoded_vision = [decode_vision(vision) for vision in output_vision]
                    outputs["vision"] = [cast(torch.Tensor, vision) for vision in decoded_vision]
                    if self.vae_decode_stream is not None:
                        # If we are using a separate GPU for VAE decoding, wait for results to be ready
                        torch.cuda.current_stream(device=outputs["vision"][0].device).wait_stream(
                            self.vae_decode_stream
                        )
        for k, v in outputs.items():
            if len(v) != len(sample_args_list):
                raise ValueError(f"Output key '{k}' has length {len(v)} but expected {len(sample_args_list)}")

        if "sound" in outputs:
            with self._get_timer(f"{self.model.__class__.__name__}.decode_sound"):
                outputs["sound"] = [self.model.decode_sound(sound) for sound in outputs.pop("sound")]

        if not save_outputs:
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

                        # The prompt guardrail runs before the multiview branch, as it does in
                        # the reference implementation. Leaving it below the `continue` meant every multiview sample
                        # skipped the only text guardrail call: with --guardrails enabled, a prompt
                        # the reference implementation rejects was generated and written to disk here instead.
                        self._run_text_guardrail(
                            str(sample_args.output_dir), data_batch[self.model.input_caption_key][sample_idx]
                        )

                        # Multiview: per-view decode and save — bypasses the single-view path below
                        # to produce named per-camera files. The video guardrail stays on the
                        # single-view path, matching the reference implementation.
                        if sample_args.multiview is not None:
                            sample_outputs.append(
                                self._save_multiview_sample(sample_args, data_batch, output, vision_cthw)
                            )
                            continue

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

    def _save_multiview_view_videos(
        self,
        sample_args: OmniSampleArgs,
        view_videos: list[torch.Tensor],
        *,
        prefix: str,
        fps: int,
        encoder_threads: int | None = None,
        max_workers: int = 1,
    ) -> tuple[list[Path], dict[str, list[str]]]:
        """Save one file per camera view and return (file list, files-by-camera dict).

        Filenames follow the parity spec: ``{prefix}_view{NN}_{safe_camera_key}.mp4``,
        e.g. ``vision_view00_camera_front_wide_120fov.mp4``. ``encoder_threads`` pins FFmpeg's
        thread count for multi-frame videos (see :func:`_multiview_save_thread_budget`); ``None``
        leaves it to the encoder. ``max_workers`` encodes up to that many views at once, each with
        the same ``encoder_threads``; the files come back in camera order either way.
        """
        from cosmos_framework.utils.generator.multiview import safe_multiview_camera_name

        assert sample_args.multiview is not None
        assert sample_args.output_dir is not None
        output_dir = sample_args.output_dir
        if len(view_videos) != len(sample_args.multiview.views):
            raise ValueError(
                f"Expected {len(sample_args.multiview.views)} view videos, "
                f"got {len(view_videos)} for prefix={prefix!r}."
            )
        if max_workers < 1:
            raise ValueError(f"max_workers must be positive, got {max_workers}.")

        quality = sample_args.video_save_quality
        views = list(enumerate(zip(view_videos, sample_args.multiview.views, strict=True)))

        def save_view(indexed_view: tuple[int, tuple[torch.Tensor, MultiviewCameraArgs]]) -> Path:
            view_index, (view, camera) = indexed_view  # view: [C,F,H,W]
            safe_name = safe_multiview_camera_name(camera.camera_key, view_index)
            vision_file = output_dir / f"{prefix}_{safe_name}{sample_args.vision_extension}"
            ffmpeg_params = (
                None if encoder_threads is None or view.shape[1] <= 1 else ["-threads", str(encoder_threads)]
            )
            save_img_or_video(
                view, str(vision_file.with_suffix("")), fps=fps, quality=quality, ffmpeg_params=ffmpeg_params
            )
            assert vision_file.is_file(), vision_file
            return vision_file

        if max_workers == 1:
            files = [save_view(indexed_view) for indexed_view in views]
        else:
            # x264 runs in an FFmpeg child process per view, and the frame conversion and pipe writes
            # release the GIL, so the encodes overlap.
            with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=f"multiview-save-{prefix}") as executor:
                files = list(executor.map(save_view, views))  # camera order, whatever order they finish in
        files_by_camera = {
            camera.camera_key: [str(vision_file)]
            for vision_file, camera in zip(files, sample_args.multiview.views, strict=True)
        }
        return files, files_by_camera

    def _save_multiview_sample(
        self,
        sample_args: OmniSampleArgs,
        data_batch: dict[str, Any],
        output: dict[str, Any],
        vision_cthw: torch.Tensor,
    ) -> SampleOutputs:  # vision_cthw: [C, V*F, H, W] — decoded camera-major pixel tensor
        """Split a camera-major output tensor into per-view clips and write named files.

        Per-view filenames produced:
          ``vision_view{NN}_{safe_camera_key}{ext}``
          ``control_view{NN}_{safe_camera_key}{ext}``  (transfer mode only)
        """
        from cosmos_framework.utils.generator.multiview import split_multiview_video_by_view

        assert sample_args.multiview is not None
        assert sample_args.output_dir is not None

        sample_n_views = int(data_batch["sample_n_views"].reshape(-1)[0].item())
        num_video_frames_per_view = int(data_batch["num_video_frames_per_view"].reshape(-1)[0].item())

        view_videos = split_multiview_video_by_view(
            vision_cthw,
            sample_n_views=sample_n_views,
            num_video_frames_per_view=num_video_frames_per_view,
        )  # list[[C, F, H, W]]

        guardrailed_view_videos = [
            self._run_video_guardrail(f"{sample_args.output_dir}/{camera.camera_key}", view)
            for view, camera in zip(view_videos, sample_args.multiview.views, strict=True)
        ]

        files: list[Path] = []
        if self.setup_args.debug:
            output["vision"] = torch.cat(guardrailed_view_videos, dim=1)  # [C, V*F, H, W]
            files.extend(self.save_data(output, output_dir=sample_args.output_dir, output_name="output"))

        vision_files, vision_files_by_camera = self._save_multiview_view_videos(
            sample_args,
            guardrailed_view_videos,
            prefix="vision",
            fps=sample_args.video_output_fps,
        )
        files.extend(vision_files)

        content: dict[str, Any] = {
            "multiview_cameras": sample_args.multiview.camera_keys,
            "vision_files_by_camera": vision_files_by_camera,
        }

        if sample_args.transfer_hints:
            # The first video item in the batch is the control video (see get_multiview_sample_data).
            control_video = data_batch[self.model.input_video_key][0]  # [1, C, V*F, H, W]
            # Cast to float32 on the host to avoid a large on-device copy.
            control_cthw = (control_video[0].cpu().float() / 255.0).detach()  # [C, V*F, H, W]
            control_views = split_multiview_video_by_view(
                control_cthw,
                sample_n_views=sample_n_views,
                num_video_frames_per_view=num_video_frames_per_view,
            )  # list[[C, F, H, W]]
            control_files, control_files_by_camera = self._save_multiview_view_videos(
                sample_args,
                control_views,
                prefix="control",
                fps=sample_args.video_output_fps,
            )
            files.extend(control_files)
            content["control_files_by_camera"] = control_files_by_camera

        sample_output = SampleOutputs(
            args=sample_args.model_dump(mode="json"),
            outputs=[SampleOutput(content=content, files=files)],
        )
        sample_outputs_file = sample_args.output_dir / "sample_outputs.json"
        sample_outputs_file.write_text(sample_output.model_dump_json())
        log.success(f"Saved multiview sample outputs to '{sample_outputs_file}'", rank0_only=False)
        return sample_output

    def _generate_causal_multiview_batch(
        self,
        sample_args: OmniSampleArgs,
        *,
        save_outputs: bool = True,
    ) -> list[SampleOutputs]:
        """Adapt causal multiview sampling to the shared batching, timing, and output lifecycle.

        Generation and saving live in ``inference/multiview_causal.py``; this wrapper owns the timer,
        the prompt guardrail, and error handling, as ``generate_causal_multiview_batch`` does in the
        reference implementation.
        """
        from cosmos_framework.inference.multiview_causal import (
            generate_causal_multiview_sample,
            save_causal_multiview_outputs,
        )

        # Model sampling includes collectives: do not catch rank-local sampler failures
        # and continue into the next sample, which could strand other ranks.
        if self.setup_args.benchmark:
            torch.cuda.synchronize()
        # Same key as the reference, so benchmark.json compares directly. The sampler times its input
        # preparation (inference.prepare_inputs, mostly media decode) and its rollout plus decode
        # (inference.generate_decode) inside this span.
        with self._get_timer("inference.encode_generate_decode"):
            result = generate_causal_multiview_sample(
                self, sample_args, save_latents=sample_args.save_latents and save_outputs
            )
            if self.setup_args.benchmark:
                torch.cuda.synchronize()
        if not save_outputs:
            return []
        try:
            with sync_distributed_errors(), torch_host_threads(host_threads_per_local_rank()):
                if self.should_process_sample(sample_args):
                    # These are the actual formatted captions consumed by the sampler,
                    # including every camera and each consumed rolling/caption segment.
                    for chunk in result.chunks:
                        prompts = chunk["prompt"]
                        for prompt in [prompts] if isinstance(prompts, str) else prompts:
                            self._run_text_guardrail(str(sample_args.output_dir), prompt)
            with sync_distributed_errors(), torch_host_threads(host_threads_per_local_rank()):
                if self.should_process_sample(sample_args):
                    with self._get_timer("inference.save_outputs"):
                        return [save_causal_multiview_outputs(self, sample_args, result)]
        except Exception as e:
            return [self._handle_sample_exception(sample_args, e)] if self.should_process_sample(sample_args) else []
        return []

    @torch.no_grad()
    @torch.no_grad()
    def _generate_multiview_caption_chunks_batch(
        self,
        sample_args: OmniSampleArgs,
        *,
        save_outputs: bool = True,
    ) -> list[SampleOutputs]:
        """Chunkwise multiview generation driven by a multiview caption file.

        Generation lives in ``inference/multiview.py``; this wrapper owns arg persistence, the prompt
        guardrail, saving, and error handling, matching ``_generate_transfer_batch``.
        """
        from cosmos_framework.inference.multiview import generate_multiview_caption_chunks

        try:
            with sync_distributed_errors():
                if self.should_process_sample(sample_args) and save_outputs:
                    assert sample_args.output_dir is not None
                    sample_args.output_dir.mkdir(parents=True, exist_ok=True)
                    sample_args_file = sample_args.output_dir / "sample_args.json"
                    sample_args_file.write_text(sample_args.model_dump_json())
                    log.info(f"Saved sample args to '{sample_args_file}'", rank0_only=False)
        except Exception as e:
            return [self._handle_sample_exception(sample_args, e)] if self.should_process_sample(sample_args) else []

        debug_save = None
        if self.setup_args.debug and self.should_process_sample(sample_args) and save_outputs:
            assert sample_args.output_dir is not None
            output_dir = sample_args.output_dir

            def debug_save(data_batch: dict[str, Any], name: str) -> None:
                self.save_data(data_batch, output_dir=output_dir, output_name=name)

        chunks_output = generate_multiview_caption_chunks(
            sample_args,
            self.model,
            debug_save=debug_save,
            per_view_captions=self.per_view_captions,
        )

        if not save_outputs:
            return []

        sample_outputs: list[SampleOutputs] = []
        try:
            with sync_distributed_errors():
                if self.should_process_sample(sample_args):
                    assert sample_args.output_dir is not None
                    final_data_batch: dict[str, Any] = {
                        self.model.input_caption_key: [chunks_output.joined_prompt],
                        "sample_n_views": torch.tensor([chunks_output.sample_n_views]),
                        "num_video_frames_per_view": torch.tensor([chunks_output.total_frames_per_view]),
                    }
                    if chunks_output.control_video is not None:
                        final_data_batch[self.model.input_video_key] = [chunks_output.control_video]

                    self._run_text_guardrail(
                        str(sample_args.output_dir),
                        final_data_batch[self.model.input_caption_key][0],
                    )
                    sample_outputs.append(
                        self._save_multiview_sample(
                            sample_args,
                            final_data_batch,
                            {"caption_chunks": chunks_output.chunk_content},
                            chunks_output.vision_cthw,
                        )
                    )
        except Exception as e:
            return [self._handle_sample_exception(sample_args, e)] if self.should_process_sample(sample_args) else []

        return sample_outputs

    def _generate_joint_cam_lidar_batch(
        self,
        sample_args: OmniSampleArgs,
        *,
        save_outputs: bool = True,
    ) -> list[SampleOutputs]:
        """Generate camera views and a LiDAR rangemap together, then write both streams.

        The two run on different clocks, so the cameras are written as per-view mp4s at the
        camera fps and the LiDAR as a safetensors range view carrying its own frame ids. There
        is no single file that holds both.
        """
        from cosmos_framework.inference.lidar_transfer import (
            generate_joint_cam_lidar_sample,
            lidar_range_projection,
            save_lidar_range_view,
        )

        try:
            with sync_distributed_errors():
                if self.should_process_sample(sample_args) and save_outputs:
                    assert sample_args.output_dir is not None
                    sample_args.output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            if self.should_process_sample(sample_args):
                return [self._handle_sample_exception(sample_args, e)]
            return []

        joint_output = generate_joint_cam_lidar_sample(
            sample_args=sample_args,
            model=self.model,
            per_view_captions=self.per_view_captions,
        )
        # Per-view captions replace the prompt and take num_frames from the caption's own span, so
        # record what generation resolved rather than what the caller asked for. sample_args.json is
        # written below, after this call, for that reason: before it the frame count is still the
        # request and disagrees with the media written beside it.
        sample_args = joint_output.resolved_sample_args

        if not save_outputs:
            return []

        sample_outputs: list[SampleOutputs] = []
        try:
            with sync_distributed_errors():
                if self.should_process_sample(sample_args):
                    assert sample_args.output_dir is not None
                    sample_args_file = sample_args.output_dir / "sample_args.json"
                    sample_args_file.write_text(sample_args.model_dump_json())
                    log.info(f"Saved sample args to '{sample_args_file}'", rank0_only=False)
                    files: list[Path] = []

                    vision_cthw = ((1.0 + joint_output.output_video.squeeze(0)) / 2).clamp(0, 1)
                    view_videos = list(
                        torch.chunk(vision_cthw, joint_output.sample_n_views, dim=1)
                    )  # camera-major: [C, V*F, H, W] -> V x [C,F,H,W]
                    view_files, files_by_camera = self._save_multiview_view_videos(
                        sample_args,
                        view_videos,
                        prefix="vision",
                        fps=int(round(joint_output.fps)),
                    )
                    files.extend(view_files)

                    lidar_file = save_lidar_range_view(
                        joint_output.output_lidar,
                        sample_args.output_dir / "lidar.safetensors",
                        range_projection=lidar_range_projection(self.model),
                        frame_ids=joint_output.lidar_frame_ids,
                    )
                    files.append(lidar_file)

                    content: dict[str, Any] = {
                        "files_by_camera": files_by_camera,
                        "lidar": str(lidar_file),
                        "lidar_fps": joint_output.lidar_fps,
                        "fps": joint_output.fps,
                    }
                    sample_output = SampleOutputs(
                        args=sample_args.model_dump(mode="json"),
                        outputs=[SampleOutput(content=content, files=files)],
                    )
                    sample_outputs_file = sample_args.output_dir / "sample_outputs.json"
                    sample_outputs_file.write_text(sample_output.model_dump_json())
                    log.success(f"Saved sample outputs to '{sample_outputs_file}'", rank0_only=False)
                    sample_outputs.append(sample_output)
        except Exception as e:
            if self.should_process_sample(sample_args):
                return [self._handle_sample_exception(sample_args, e)]
            return []
        return sample_outputs

    def _generate_transfer_batch(
        self,
        sample_args: OmniSampleArgs,
        *,
        save_outputs: bool = True,
    ) -> list[SampleOutputs]:
        """Handle transfer inference using the autoregressive generate_transfer_sample path."""
        from cosmos_framework.inference.transfer import (
            DirectTransferInputs,
            _load_direct_transfer_inputs,
            generate_transfer_sample,
        )
        from cosmos_framework.inference.transfer_streaming import generate_transfer_batch

        prepared_direct_inputs: DirectTransferInputs | None = None

        try:
            with sync_distributed_errors():
                if self.should_process_sample(sample_args) and save_outputs:
                    log.debug(f"{sample_args.__class__.__name__}({sample_args})")
                    assert sample_args.output_dir is not None
                    sample_args.output_dir.mkdir(parents=True, exist_ok=True)
                    sample_args_file = sample_args.output_dir / "sample_args.json"
                    sample_args_file.write_text(sample_args.model_dump_json())
                    log.info(f"Saved sample args to '{sample_args_file}'", rank0_only=False)
                if getattr(sample_args, "transfer_resize_mode", None) == "direct_resize":
                    if self.setup_args.condition_stream_encode:
                        raise ValueError("Direct Transfer resizing requires condition_stream_encode=false")
                    prepared_direct_inputs = _load_direct_transfer_inputs(sample_args)
        except Exception as e:
            if self.should_process_sample(sample_args):
                return [self._handle_sample_exception(sample_args, e)]
            return []

        if self.setup_args.condition_stream_encode:
            transfer_outputs = generate_transfer_batch(
                sample_args_list=[sample_args], model=self.model, materialize_outputs=save_outputs
            )
            if not save_outputs:
                return []
            transfer_output = transfer_outputs[0]
        elif prepared_direct_inputs is not None:
            transfer_output = generate_transfer_sample(
                sample_args=sample_args, model=self.model, prepared_direct_inputs=prepared_direct_inputs
            )
        else:
            transfer_output = generate_transfer_sample(sample_args=sample_args, model=self.model)

        if not save_outputs:
            return []

        sample_outputs: list[SampleOutputs] = []
        try:
            with sync_distributed_errors():
                if self.should_process_sample(sample_args):
                    assert sample_args.output_dir is not None
                    content: dict[str, Any] = {}
                    files: list[Path] = []

                    if self.setup_args.debug:
                        raw_output = {
                            "vision": _unbatch_transfer_debug_tensor(
                                transfer_output.output_video, name="vision"
                            )  # [C,T,H,W]
                        }
                        raw_output.update(
                            {
                                f"control_{hint_key}": _unbatch_transfer_debug_tensor(
                                    control_video, name=f"control_{hint_key}"
                                )  # [C,T,H,W]
                                for hint_key, control_video in transfer_output.control_videos.items()
                            }
                        )
                        files.extend(
                            self.save_data(raw_output, output_dir=sample_args.output_dir, output_name="output")
                        )

                    vision_cthw = ((1.0 + transfer_output.output_video.squeeze(0)) / 2).clamp(0, 1)

                    if vision_cthw.shape[1] == 1:
                        quality = sample_args.image_save_quality
                    else:
                        quality = sample_args.video_save_quality
                    output_fps = transfer_output.fps
                    model_control = next(iter(transfer_output.control_videos.values()))  # [1,C,T,Hm,Wm]
                    files.extend(
                        _save_transfer_rgb(
                            vision_cthw,
                            original_hw=transfer_output.original_hw,
                            model_hw=(model_control.shape[-2], model_control.shape[-1]),
                            fps=output_fps,
                            sample_args=sample_args,
                            content=content,
                        )
                    )

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
        save_outputs: bool = True,
    ) -> list[SampleOutputs]:
        """Reasoner AR text generation. Each prompt writes ``reasoner_text.txt`` and
        ``SampleOutput.content["reasoner_text"]``. Mixing image-conditioned and
        text-only samples in one batch is rejected."""
        sample_args_list = cast(list[OmniSampleArgs], sample_args_list)

        prompts: list[str] = data_batch[self.model.input_caption_key]
        raw_images: list[Image.Image | None] = data_batch["reasoner_images"]
        raw_videos: list[dict[str, Any] | None] | None = data_batch.get("reasoner_videos")

        n_img = sum(img is not None for img in raw_images)
        n_vid = sum(v is not None for v in (raw_videos or []))
        if n_img and n_vid:
            raise ValueError("Reasoner batch mixes image- and video-conditioned samples. Split into separate batches.")
        if 0 < n_img < len(raw_images):
            raise ValueError(
                "Reasoner batch mixes image-conditioned and text-only samples "
                f"({n_img}/{len(raw_images)} have an image vision_path). Split into separate batches."
            )
        if raw_videos is not None and 0 < n_vid < len(raw_videos):
            raise ValueError(
                "Reasoner batch mixes video-conditioned and text-only samples "
                f"({n_vid}/{len(raw_videos)} have a video vision_path). Split into separate batches."
            )
        images: list[Image.Image] | None = cast(list[Image.Image], raw_images) if n_img == len(raw_images) else None
        videos: list[dict[str, Any]] | None = (
            cast(list[dict[str, Any]], raw_videos) if raw_videos is not None and n_vid == len(raw_videos) else None
        )

        try:
            with sync_distributed_errors():
                for sa, prompt in zip(sample_args_list, prompts):
                    if self.should_process_sample(sa) and save_outputs:
                        log.debug(f"{sa.__class__.__name__}({sa})")
                        assert sa.output_dir is not None
                        sa.output_dir.mkdir(parents=True, exist_ok=True)
                        (sa.output_dir / "sample_args.json").write_text(sa.model_dump_json())
                        self._run_text_guardrail(str(sa.output_dir), prompt)
        except Exception as e:
            return [self._handle_sample_exception(sa, e) for sa in sample_args_list if self.should_process_sample(sa)]

        # Collective call: every rank must enter so FSDP unshard/reshard and the
        # cross-rank early-exit reduction stay in lockstep. Not wrapped in try/except.
        with self._get_timer(f"{self.model.__class__.__name__}.generate_reasoner_text"):
            texts = self.model.generate_reasoner_text(
                prompts,
                max_new_tokens=sample_args_list[0].max_new_tokens,
                images=images,
                videos=videos,
                do_sample=sample_args_list[0].do_sample,
                temperature=sample_args_list[0].temperature,
                top_k=sample_args_list[0].top_k,
                top_p=sample_args_list[0].top_p,
                repetition_penalty=sample_args_list[0].repetition_penalty,
                presence_penalty=sample_args_list[0].presence_penalty,
                seed=sample_args_list[0].seed,
            )

        if not save_outputs:
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
