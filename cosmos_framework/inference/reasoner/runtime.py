# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Framework-native model loading and multimodal generation."""

from __future__ import annotations

import copy
import json
import logging
import os
import types
from collections import OrderedDict
from pathlib import Path
from typing import Any

from cosmos_framework.checkpoint.reasoner import distributed_identity, ensure_evaluation_checkpoint

logger = logging.getLogger(__name__)


def _read_model_type(model_path: str) -> str | None:
    try:
        return json.loads((Path(model_path) / "config.json").read_text()).get("model_type")
    except (OSError, ValueError):
        return None


def _torch_dtype(name: str):
    import torch

    if name in ("auto", None):
        return "auto"
    aliases = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}
    resolved = aliases.get(str(name).lower(), str(name).lower())
    if not hasattr(torch, resolved):
        raise ValueError(f"Unsupported torch dtype: {name}")
    return getattr(torch, resolved)


class _RankLocalVideoProcessorCache:
    """Transparent LRU for deterministic Qwen video-processor outputs."""

    def __init__(self, delegate: Any, runtime: "CosmosFrameworkRuntime") -> None:
        self._delegate = delegate
        self._runtime = runtime

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    @staticmethod
    def _copy_output(value: Any) -> Any:
        copier = getattr(value, "copy", None)
        return copier() if callable(copier) else copy.copy(value)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        runtime = self._runtime
        active_key = runtime._video_feature_cache_active_key
        if active_key is None or runtime._video_processor_cache_capacity <= 0:
            return self._delegate(*args, **kwargs)

        cache = runtime._video_processor_cache
        if active_key in cache:
            runtime._video_processor_cache_hits += 1
            cached = cache.pop(active_key)
            cache[active_key] = cached
            if runtime._video_processor_cache_hits == 1:
                print(
                    "COSMOS_VIDEO_PROCESSOR_CACHE_HIT_ATTESTATION "
                    f"pid={os.getpid()} rank={getattr(runtime, 'rank', -1)} "
                    f"capacity={runtime._video_processor_cache_capacity} "
                    "cache_boundary=video_processor",
                    flush=True,
                )
            return self._copy_output(cached)

        runtime._video_processor_cache_misses += 1
        processed = self._delegate(*args, **kwargs)
        cache[active_key] = processed
        while len(cache) > runtime._video_processor_cache_capacity:
            cache.popitem(last=False)
        return self._copy_output(processed)


class CosmosFrameworkRuntime:
    """Load an exported Cosmos Framework model and generate responses.

    Native VLM exports (including ``cosmos3_edge``) use Transformers plus the
    framework-registered model/processor.  General ``cosmos3_omni`` exports use
    the official Framework inference model's reasoner-only generation method.
    """

    def __init__(
        self,
        model_path: str,
        *,
        config_file: str | None = None,
        export_dir: str | None = None,
        vit_checkpoint_path: str | None = None,
        enable_lora: bool = False,
        base_model_path: str | None = None,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        attn_implementation: str | None = None,
    ) -> None:
        import torch

        self.rank, self.world_size, self.local_rank = distributed_identity()
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.device = f"cuda:{self.local_rank}"
        else:
            self.device = "cpu"
        self.model_path = ensure_evaluation_checkpoint(
            model_path,
            enable_lora=enable_lora,
            base_model_path=base_model_path,
            config_file=config_file,
            export_dir=export_dir,
            vit_checkpoint_path=vit_checkpoint_path,
        )
        self.model_type = _read_model_type(self.model_path)
        self.dtype = dtype
        self.device_map = device_map
        self.attn_implementation = str(attn_implementation).strip() if attn_implementation else None
        self.model: Any = None
        self.processor: Any = None
        self.backend: str = ""
        self._video_feature_cache: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
        self._video_feature_cache_capacity = 0
        self._video_feature_cache_active_key: tuple[Any, ...] | None = None
        self._video_feature_cache_wrapped = False
        self._video_feature_cache_hits = 0
        self._video_feature_cache_misses = 0
        self._video_processor_cache: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
        self._video_processor_cache_capacity = 0
        self._video_processor_cache_wrapped = False
        self._video_processor_cache_hits = 0
        self._video_processor_cache_misses = 0
        self._video_input_cache: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
        self._video_input_cache_hits = 0
        self._video_input_cache_misses = 0
        self._video_prefix_cache: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()
        self._video_prefix_cache_capacity = 0
        self._video_prefix_cache_hits = 0
        self._video_prefix_cache_misses = 0
        self._load()

    def _load(self) -> None:
        logger.info(
            "Loading Cosmos Framework model from %s on %s (model_type=%s)",
            self.model_path,
            self.device,
            self.model_type,
        )
        if self.model_type == "cosmos3_omni":
            from cosmos_framework.inference.model import Cosmos3OmniModel

            wrapper = Cosmos3OmniModel.from_pretrained_dcp(Path(self.model_path))
            wrapper.eval()
            self.model = wrapper.model
            self.processor = getattr(self.model, "vlm_processor", None)
            self.backend = "omni"
            return

        from transformers import AutoModelForImageTextToText, AutoProcessor

        if self.model_type == "cosmos3_edge":
            # Registers the native Edge config/model with Transformers Auto classes.
            import cosmos_framework.model.generator.reasoner.cosmos3_edge  # noqa: F401
            from cosmos_framework.data.generator.processors.cosmos3_edge_processing import (
                build_cosmos3_edge_processor,
            )

            self.processor = build_cosmos3_edge_processor(self.model_path)
        else:
            self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)

        device_map: str | dict[str, str]
        if self.device == "cpu":
            device_map = "cpu"
        elif self.device_map == "auto":
            # One process owns one GPU; an explicit map prevents Accelerate from
            # spreading a replica across all visible devices.
            device_map = {"": self.device}
        else:
            device_map = self.device_map
        model_load_kwargs: dict[str, Any] = {
            "torch_dtype": _torch_dtype(self.dtype),
            "device_map": device_map,
            "trust_remote_code": True,
        }
        if self.attn_implementation:
            model_load_kwargs["attn_implementation"] = self.attn_implementation
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            **model_load_kwargs,
        )
        if self.model_type == "qwen3_vl":
            from cosmos_framework.model.generator.qwen3_vl_compat import apply_qwen3_vl_patch_embed_compat

            apply_qwen3_vl_patch_embed_compat(self.model, model_type=self.model_type, mode="auto")
        self.model.eval()
        if self.attn_implementation:
            actual_attn = getattr(getattr(self.model, "config", None), "_attn_implementation", None)
            if actual_attn != self.attn_implementation:
                raise RuntimeError(
                    "Requested attention implementation was not selected: "
                    f"requested={self.attn_implementation!r} actual={actual_attn!r}"
                )
            print(
                "COSMOS_ATTENTION_IMPLEMENTATION_ATTESTATION "
                f"requested={self.attn_implementation} actual={actual_attn} "
                f"pid={os.getpid()} rank={self.rank}",
                flush=True,
            )
        if getattr(self.processor, "tokenizer", None) is not None:
            self.processor.tokenizer.padding_side = "left"
        self.backend = "transformers"

    def _install_video_feature_cache(self, capacity: int) -> None:
        """Install an opt-in rank-local cache around the deterministic VLM encoder."""
        capacity = int(capacity)
        if capacity <= 0:
            self._video_feature_cache_capacity = 0
            self._video_feature_cache.clear()
            return

        self._video_feature_cache_capacity = capacity
        if getattr(self, "_video_feature_cache_wrapped", False):
            return

        feature_owner = getattr(self.model, "model", None)
        if feature_owner is None or not hasattr(feature_owner, "get_video_features"):
            if hasattr(self.model, "get_video_features"):
                feature_owner = self.model
            else:
                raise RuntimeError("vision.video_feature_cache_size requires a model with get_video_features")

        original_get_video_features = feature_owner.get_video_features
        runtime = self

        def cached_get_video_features(
            _feature_owner: Any,
            pixel_values_videos: Any,
            video_grid_thw: Any = None,
        ) -> Any:
            active_key = runtime._video_feature_cache_active_key
            if active_key is None or runtime._video_feature_cache_capacity <= 0:
                return original_get_video_features(pixel_values_videos, video_grid_thw)

            if video_grid_thw is None:
                grid_key: tuple[int, ...] = ()
            else:
                grid_key = tuple(int(value) for value in video_grid_thw.detach().reshape(-1).cpu().tolist())
            cache_key = (
                active_key,
                tuple(int(value) for value in pixel_values_videos.shape),
                str(pixel_values_videos.dtype),
                str(pixel_values_videos.device),
                grid_key,
            )
            cache = runtime._video_feature_cache
            if cache_key in cache:
                runtime._video_feature_cache_hits += 1
                cached = cache.pop(cache_key)
                cache[cache_key] = cached
                if runtime._video_feature_cache_hits == 1:
                    print(
                        "COSMOS_VIDEO_FEATURE_CACHE_HIT_ATTESTATION "
                        f"pid={os.getpid()} rank={getattr(runtime, 'rank', -1)} "
                        f"capacity={runtime._video_feature_cache_capacity} "
                        "cache_boundary=get_video_features",
                        flush=True,
                    )
                    logger.info(
                        "COSMOS_VIDEO_FEATURE_CACHE_HIT_ATTESTATION pid=%s rank=%s "
                        "capacity=%s cache_boundary=get_video_features",
                        os.getpid(),
                        getattr(runtime, "rank", -1),
                        runtime._video_feature_cache_capacity,
                    )
                return cached

            runtime._video_feature_cache_misses += 1
            encoded = original_get_video_features(pixel_values_videos, video_grid_thw)
            cache[cache_key] = encoded
            while len(cache) > runtime._video_feature_cache_capacity:
                cache.popitem(last=False)
            return encoded

        feature_owner.get_video_features = types.MethodType(cached_get_video_features, feature_owner)
        self._video_feature_cache_wrapped = True
        print(
            "COSMOS_VIDEO_FEATURE_CACHE_ENABLED_ATTESTATION "
            f"pid={os.getpid()} rank={getattr(self, 'rank', -1)} "
            f"capacity={capacity} cache_boundary=get_video_features",
            flush=True,
        )
        logger.info(
            "Rank-local video feature cache enabled: capacity=%s rank=%s",
            capacity,
            getattr(self, "rank", -1),
        )

    def _install_video_processor_cache(self, capacity: int) -> None:
        """Install an opt-in cache before deterministic video normalization."""
        capacity = int(capacity)
        if capacity <= 0:
            self._video_processor_cache_capacity = 0
            self._video_processor_cache.clear()
            return
        self._video_processor_cache_capacity = capacity
        if getattr(self, "_video_processor_cache_wrapped", False):
            return
        delegate = getattr(self.processor, "video_processor", None)
        if delegate is None:
            raise RuntimeError("vision.video_processor_cache_size requires processor.video_processor")
        self.processor.video_processor = _RankLocalVideoProcessorCache(delegate, self)
        self._video_processor_cache_wrapped = True
        print(
            "COSMOS_VIDEO_PROCESSOR_CACHE_ENABLED_ATTESTATION "
            f"pid={os.getpid()} rank={getattr(self, 'rank', -1)} "
            f"capacity={capacity} cache_boundary=video_processor",
            flush=True,
        )

    def video_cache_stats(self) -> dict[str, int]:
        return {
            "feature_hits": int(getattr(self, "_video_feature_cache_hits", 0)),
            "feature_misses": int(getattr(self, "_video_feature_cache_misses", 0)),
            "feature_entries": len(getattr(self, "_video_feature_cache", {})),
            "processor_hits": int(getattr(self, "_video_processor_cache_hits", 0)),
            "processor_misses": int(getattr(self, "_video_processor_cache_misses", 0)),
            "processor_entries": len(getattr(self, "_video_processor_cache", {})),
            "video_input_hits": int(getattr(self, "_video_input_cache_hits", 0)),
            "video_input_misses": int(getattr(self, "_video_input_cache_misses", 0)),
            "video_input_entries": len(getattr(self, "_video_input_cache", {})),
            "prefix_hits": int(getattr(self, "_video_prefix_cache_hits", 0)),
            "prefix_misses": int(getattr(self, "_video_prefix_cache_misses", 0)),
            "prefix_entries": len(getattr(self, "_video_prefix_cache", {})),
        }

    @staticmethod
    def _fork_dynamic_cache(cache: Any) -> Any:
        """Fork a DynamicCache without copying immutable prefix tensors."""
        layers = getattr(cache, "layers", None)
        if layers is None:
            raise TypeError(
                "vision.video_prefix_cache_size requires a Transformers cache with independently forkable layers"
            )
        forked = copy.copy(cache)
        forked.layers = [copy.copy(layer) for layer in layers]
        return forked

    def _video_prefix_boundary(self, input_ids: Any) -> int:
        """Return the boundary immediately after the final video segment."""
        import torch

        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("vision.video_prefix_cache_size preserves singleton generation only")
        video_token_id = int(self.model.config.video_token_id)
        vision_end_token_id = int(self.model.config.vision_end_token_id)
        video_positions = torch.nonzero(input_ids[0] == video_token_id, as_tuple=False).flatten()
        end_positions = torch.nonzero(input_ids[0] == vision_end_token_id, as_tuple=False).flatten()
        if video_positions.numel() == 0 or end_positions.numel() == 0:
            raise ValueError("vision.video_prefix_cache_size requires tokenized video content")
        prefix_length = int(end_positions[-1].item()) + 1
        if int(video_positions[-1].item()) >= prefix_length:
            raise RuntimeError("video token found after the selected prefix boundary")
        if prefix_length >= int(input_ids.shape[1]):
            raise ValueError("video prefix leaves no task-specific generation suffix")
        return prefix_length

    def _generate_with_video_prefix_cache(
        self,
        inputs: Any,
        *,
        active_key: tuple[Any, ...],
        capacity: int,
        generation_kwargs: dict[str, Any],
    ) -> Any:
        """Generate from a reusable video-prefix KV cache with isolated branches."""
        import torch

        capacity = int(capacity)
        if capacity <= 0:
            return self.model.generate(**inputs, **generation_kwargs)
        if int(inputs["input_ids"].shape[0]) != 1:
            raise ValueError(
                "vision.video_prefix_cache_size preserves singleton generation only; evaluation.batch_size must be 1"
            )

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        prefix_length = self._video_prefix_boundary(input_ids)
        prefix_ids = input_ids[:, :prefix_length]
        cache_key = (
            active_key,
            tuple(int(token) for token in prefix_ids[0].detach().cpu().tolist()),
        )

        model_core = getattr(self.model, "model", None)
        if model_core is None or not hasattr(model_core, "get_rope_index"):
            raise RuntimeError("vision.video_prefix_cache_size requires Qwen3-VL mRoPE support")
        full_position_ids, rope_deltas = model_core.get_rope_index(
            input_ids,
            inputs.get("image_grid_thw"),
            inputs.get("video_grid_thw"),
            attention_mask=attention_mask,
        )

        cache = self._video_prefix_cache
        entry = cache.pop(cache_key, None)
        if entry is None:
            self._video_prefix_cache_misses += 1
            prefix_kwargs: dict[str, Any] = {
                "input_ids": prefix_ids,
                "attention_mask": attention_mask[:, :prefix_length],
                "position_ids": full_position_ids[..., :prefix_length],
                "cache_position": torch.arange(prefix_length, device=input_ids.device, dtype=torch.long),
                "use_cache": True,
                "return_dict": True,
                "logits_to_keep": 1,
            }
            for name in (
                "pixel_values",
                "pixel_values_videos",
                "image_grid_thw",
                "video_grid_thw",
            ):
                if inputs.get(name) is not None:
                    prefix_kwargs[name] = inputs[name]
            prefix_outputs = self.model(**prefix_kwargs)
            prefix_cache = prefix_outputs.past_key_values
            if prefix_cache is None or int(prefix_cache.get_seq_length()) != prefix_length:
                raise RuntimeError("video-prefix prefill did not produce the expected KV-cache length")
            entry = {
                "past_key_values": prefix_cache,
                "prefix_length": prefix_length,
            }
            if self._video_prefix_cache_misses == 1:
                print(
                    "COSMOS_VIDEO_PREFIX_CACHE_ENABLED_ATTESTATION "
                    f"pid={os.getpid()} rank={getattr(self, 'rank', -1)} "
                    f"capacity={capacity} prefix_tokens={prefix_length} "
                    "cache_boundary=final_vision_end",
                    flush=True,
                )
        else:
            self._video_prefix_cache_hits += 1
            if int(entry["prefix_length"]) != prefix_length:
                raise RuntimeError("cached video-prefix boundary changed for one key")
            if self._video_prefix_cache_hits == 1:
                print(
                    "COSMOS_VIDEO_PREFIX_CACHE_HIT_ATTESTATION "
                    f"pid={os.getpid()} rank={getattr(self, 'rank', -1)} "
                    f"capacity={capacity} prefix_tokens={prefix_length} "
                    "cache_boundary=final_vision_end",
                    flush=True,
                )

        cache[cache_key] = entry
        while len(cache) > capacity:
            cache.popitem(last=False)
        self._video_prefix_cache_capacity = capacity

        reusable_cache = entry["past_key_values"]
        reusable_length = int(reusable_cache.get_seq_length())
        branch_cache = self._fork_dynamic_cache(reusable_cache)
        model_core.rope_deltas = rope_deltas
        generated = self.model.generate(
            **inputs,
            past_key_values=branch_cache,
            **generation_kwargs,
        )
        if int(reusable_cache.get_seq_length()) != reusable_length:
            raise RuntimeError("generation mutated the reusable video-prefix cache")
        return generated

    @staticmethod
    def _video_feature_key(task: dict[str, Any], vision_config: dict[str, Any]) -> tuple[Any, ...]:
        media_paths = tuple(str(path) for path in task.get("media_paths", []))
        if task.get("media_mode", "image") != "video" or len(media_paths) != 1:
            raise ValueError("vision.video_feature_cache_size requires exactly one video per task")
        return (
            media_paths,
            int(vision_config.get("num_frames", vision_config.get("nframes", 0))),
            vision_config.get("fps"),
            vision_config.get("min_pixels"),
            vision_config.get("max_pixels"),
            vision_config.get("total_pixels"),
        )

    @staticmethod
    def _task_conversation(task: dict[str, Any], vision_config: dict[str, Any]) -> list[dict[str, Any]]:
        conversation = copy.deepcopy(task.get("prompt", []))
        if not conversation:
            conversation = [{"role": "user", "content": task.get("question", "")}]

        # Multimodal ProcessorMixin scans every turn for visual content and
        # therefore expects the structured content-list form even for a
        # text-only system message.
        for turn in conversation:
            content = turn.get("content", "")
            if not isinstance(content, list):
                turn["content"] = [{"type": "text", "text": str(content)}]

        user = next((turn for turn in conversation if turn.get("role") == "user"), None)
        if user is None:
            user = {
                "role": "user",
                "content": [{"type": "text", "text": str(task.get("question", ""))}],
            }
            conversation.append(user)
        text = user.get("content", "")
        if isinstance(text, list):
            text_content = text
        else:
            text_content = [{"type": "text", "text": str(text)}]

        media_content: list[dict[str, Any]] = []
        media_mode = task.get("media_mode", "image")
        media_paths = task.get("media_paths", [])
        decoded_media = task.get("_framework_decoded_media")
        if decoded_media is not None and len(decoded_media) != len(media_paths):
            raise ValueError("Framework decoded-media count must match media_paths")
        for media_index, media_path in enumerate(media_paths):
            media_value = media_path
            if decoded_media is not None:
                media_value = decoded_media[media_index]["frames"]
            item = {"type": media_mode, media_mode: media_value}
            if decoded_media is not None:
                item["fps"] = decoded_media[media_index]["fps"]
            for key in ("fps", "min_pixels", "max_pixels", "total_pixels"):
                if vision_config.get(key) is not None:
                    item[key] = vision_config[key]
            configured_frames = vision_config.get("nframes", vision_config.get("num_frames"))
            if configured_frames is not None:
                # qwen-vl-utils consumes ``nframes`` on each video content item.
                # Preserve Cosmos's public ``num_frames`` spelling alongside that
                # runtime alias so the submitted/evaluated semantics remain
                # directly attestable instead of silently changing at the
                # framework boundary.
                configured_frames = int(configured_frames)
                item["num_frames"] = configured_frames
                item["nframes"] = configured_frames
            media_content.append(item)
        user["content"] = media_content + text_content
        return conversation

    @staticmethod
    def _decode_video_exact(video_path: str, num_frames: int | None, fps: float | None) -> dict[str, Any]:
        import torch
        from PIL import Image

        from cosmos_framework.inference.vision import decode_video_thwc_uint8

        frames, src_fps = decode_video_thwc_uint8(Path(video_path))
        total = int(frames.shape[0])
        if total == 0:
            raise ValueError(f"Decoded zero frames from {video_path}")
        src_fps = float(src_fps or 1.0)
        if num_frames is None:
            target_fps = float(fps or 2.0)
            num_frames = max(2, min(total, int(round(total * target_fps / src_fps))))
        num_frames = max(1, min(total, int(num_frames)))
        indices = torch.linspace(0, total - 1, num_frames).round().long().tolist()
        pil_frames = [Image.fromarray(frames[index].numpy()) for index in indices]
        return {"frames": pil_frames, "fps": num_frames / total * src_fps}

    def _generate_omni(
        self,
        tasks: list[dict[str, Any]],
        generation_config: dict[str, Any],
        vision_config: dict[str, Any],
    ) -> tuple[list[str], list[float]]:
        from PIL import Image

        outputs: list[str] = []
        max_new_tokens = int(generation_config.get("max_tokens", generation_config.get("max_new_tokens", 32)))
        for task in tasks:
            prompt = str(task.get("question", ""))
            system_prompt = ""
            for turn in task.get("prompt", []):
                if turn.get("role") == "system" and isinstance(turn.get("content"), str):
                    system_prompt = turn["content"].strip()

            def prompt_builder(text: str) -> list[dict[str, Any]]:
                messages: list[dict[str, Any]] = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": text})
                return messages

            media_paths = task.get("media_paths", [])
            if len(media_paths) > 1:
                raise ValueError("Cosmos Framework reasoner inference currently supports one media file per sample")
            images = videos = None
            if media_paths:
                if task.get("media_mode") == "video":
                    videos = [
                        self._decode_video_exact(
                            media_paths[0],
                            vision_config.get("num_frames"),
                            vision_config.get("fps"),
                        )
                    ]
                else:
                    images = [Image.open(media_paths[0]).convert("RGB")]
            result = self.model.generate_reasoner_text(
                [prompt],
                max_new_tokens=max_new_tokens,
                images=images,
                videos=videos,
                prompt_builder=prompt_builder,
                do_sample=float(generation_config.get("temperature", 0.0)) > 0,
                temperature=float(generation_config.get("temperature", 0.0)) or 1.0,
                top_k=generation_config.get("top_k"),
                top_p=generation_config.get("top_p"),
                repetition_penalty=float(generation_config.get("repetition_penalty", 1.0)),
                presence_penalty=float(generation_config.get("presence_penalty", 0.0)),
                seed=int(generation_config.get("seed", 1)),
            )
            outputs.append(result[0])
        return outputs, [0.0] * len(outputs)

    def _generate_transformers(
        self,
        tasks: list[dict[str, Any]],
        generation_config: dict[str, Any],
        vision_config: dict[str, Any],
    ) -> tuple[list[str], list[float]]:
        import torch

        conversations = [self._task_conversation(task, vision_config) for task in tasks]
        max_new_tokens = int(generation_config.get("max_tokens", generation_config.get("max_new_tokens", 32)))
        if not conversations:
            return [], []

        feature_cache_capacity = int(vision_config.get("video_feature_cache_size", 0))
        processor_cache_capacity = int(vision_config.get("video_processor_cache_size", 0))
        prefix_cache_capacity = int(vision_config.get("video_prefix_cache_size", 0))
        if feature_cache_capacity > 0 or processor_cache_capacity > 0 or prefix_cache_capacity > 0:
            if len(tasks) != 1:
                raise ValueError(
                    "vision video feature/processor caches preserve singleton generation only; "
                    "evaluation.batch_size must be 1"
                )
            if feature_cache_capacity > 0:
                self._install_video_feature_cache(feature_cache_capacity)
            if processor_cache_capacity > 0:
                self._install_video_processor_cache(processor_cache_capacity)
            self._video_feature_cache_active_key = self._video_feature_key(tasks[0], vision_config)

        # The evaluator already groups tasks according to evaluation.batch_size.
        # Materialize and generate that group as one real model batch instead of
        # silently looping over singleton forwards. Qwen's processor accepts a
        # list of conversations and preserves the text-to-video placeholder
        # mapping while padding text on the configured left side.
        try:
            inputs = self._prepare_transformers_batch(conversations).to(self.device)
            with torch.inference_mode():
                generation_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": float(generation_config.get("temperature", 0.0)) > 0,
                    "temperature": float(generation_config.get("temperature", 0.0)) or None,
                    "top_k": generation_config.get("top_k"),
                    "top_p": generation_config.get("top_p"),
                    "repetition_penalty": float(generation_config.get("repetition_penalty", 1.0)),
                    "return_dict_in_generate": True,
                    "output_scores": True,
                }
                generated = self._generate_with_video_prefix_cache(
                    inputs,
                    active_key=self._video_feature_cache_active_key,
                    capacity=prefix_cache_capacity,
                    generation_kwargs=generation_kwargs,
                )
        finally:
            self._video_feature_cache_active_key = None
        input_length = inputs["input_ids"].shape[-1]
        new_ids = generated.sequences[:, input_length:]
        outputs = self.processor.batch_decode(new_ids, skip_special_tokens=True)

        eos_token_id = getattr(getattr(self.processor, "tokenizer", None), "eos_token_id", None)
        if eos_token_id is None:
            eos_token_ids: set[int] = set()
        elif isinstance(eos_token_id, (list, tuple, set)):
            eos_token_ids = {int(value) for value in eos_token_id}
        else:
            eos_token_ids = {int(eos_token_id)}
        losses: list[float] = []
        for sample_index in range(new_ids.shape[0]):
            token_nll: list[float] = []
            for step, logits in enumerate(generated.scores or []):
                if step >= new_ids.shape[1]:
                    break
                token_id = new_ids[sample_index, step]
                token_nll.append(-torch.log_softmax(logits[sample_index].float(), dim=-1)[token_id].item())
                # Batched generation pads samples that finish before their
                # peers. Include EOS exactly as singleton generation does, then
                # exclude synthetic trailing pad steps from the NLL.
                if int(token_id) in eos_token_ids:
                    break
            losses.append(sum(token_nll) / len(token_nll) if token_nll else 0.0)
        return outputs, losses

    def _process_vision_info_cached(self, conversations: list[list[dict[str, Any]]]) -> tuple[Any, Any, dict[str, Any]]:
        """Reuse exact qwen-vl-utils inputs for adjacent questions on one video.

        This rank-local LRU stores only the deterministic, in-memory video
        tensor produced during evaluation. It does not prewarm or write a disk
        cache, and it remains keyed by the same complete visual-input identity
        as the processor and feature caches.
        """
        from qwen_vl_utils import process_vision_info

        active_key = self._video_feature_cache_active_key
        capacity = int(getattr(self, "_video_processor_cache_capacity", 0))
        if active_key is None or capacity <= 0:
            return process_vision_info(
                conversations,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )

        cache = self._video_input_cache
        if active_key in cache:
            self._video_input_cache_hits += 1
            cached = cache.pop(active_key)
            cache[active_key] = cached
            if self._video_input_cache_hits == 1:
                print(
                    "COSMOS_VIDEO_INPUT_CACHE_HIT_ATTESTATION "
                    f"pid={os.getpid()} rank={getattr(self, 'rank', -1)} "
                    f"capacity={capacity} cache_boundary=process_vision_info",
                    flush=True,
                )
            return cached

        self._video_input_cache_misses += 1
        prepared = process_vision_info(
            conversations,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        cache[active_key] = prepared
        while len(cache) > capacity:
            cache.popitem(last=False)
        return prepared

    def _prepare_transformers_batch(self, conversations: list[list[dict[str, Any]]]) -> Any:
        """Materialize a true padded multimodal batch through Qwen's GPU reader."""
        texts = [
            self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
            for conversation in conversations
        ]
        image_inputs, video_inputs, video_kwargs = self._process_vision_info_cached(conversations)
        processor_kwargs: dict[str, Any] = {
            "text": texts,
            "images": image_inputs,
            "padding": True,
            "return_tensors": "pt",
        }
        if video_inputs:
            videos, video_metadata = zip(*video_inputs)
            processor_kwargs.update(
                {
                    "videos": list(videos),
                    "video_metadata": list(video_metadata),
                    "do_resize": False,
                }
            )
        return self.processor(**processor_kwargs, **video_kwargs)

    def _prepare_transformers_inputs(self, conversation: list[dict[str, Any]]) -> Any:
        """Materialize visual inputs through the registered Qwen GPU reader."""
        from qwen_vl_utils import process_vision_info

        text = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            conversation,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        processor_kwargs: dict[str, Any] = {
            "text": [text],
            "images": image_inputs,
            "return_tensors": "pt",
        }
        if video_inputs:
            videos, video_metadata = zip(*video_inputs)
            processor_kwargs.update(
                {
                    "videos": list(videos),
                    "video_metadata": list(video_metadata),
                    "do_resize": False,
                }
            )
        return self.processor(**processor_kwargs, **video_kwargs)

    def generate_tasks(
        self,
        tasks: list[dict[str, Any]],
        *,
        generation_config: dict[str, Any] | None = None,
        vision_config: dict[str, Any] | None = None,
    ) -> tuple[list[str], list[float]]:
        generation_config = generation_config or {}
        vision_config = vision_config or {}
        if self.backend == "omni":
            return self._generate_omni(tasks, generation_config, vision_config)
        return self._generate_transformers(tasks, generation_config, vision_config)
