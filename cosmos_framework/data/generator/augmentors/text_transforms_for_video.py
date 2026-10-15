# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
import math
import os
import random
from typing import Optional

from cosmos_framework.data.imaginaire.webdataset.augmentors.v3_text_transforms import pad_and_resize
from cosmos_framework.data.imaginaire.webdataset.augmentors.augmentor import Augmentor
from cosmos_framework.utils import log


class TextTransformForVideo(Augmentor):
    def __init__(self, input_keys: dict, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

        # our caption is saved in json with format: {"<key>": "xxx", "<caption_windows_key1>": [{"start_frame": x, "end_frame": x, "<caption_type>": xxx}, ...], "<caption_windows_key2>": [{"start_frame":...]}
        # our t5 embedding is saved in pickle with format: [{"<embedding_caption_type1>": array1, "<embedding_caption_type2>": array2}, ...]
        self.captions_key: str = args[
            "captions_key"
        ]  # s3 folder that saves the captions; this get mapped to the key in data_dict to fetch the caption field
        self.embeddings_key: Optional[str] = args[
            "embeddings_key"
        ]  # s3 folder that saves the embeddings; this get mapped to the key in data_dict to fetch the embedding field
        self.caption_windows_key: str = args[
            "caption_windows_key"
        ]  # key to get the caption windows from the caption field
        self.caption_type: str = args["caption_type"]  # key of caption type to fetch the caption from caption windows

        self._load_embeddings = self.embeddings_key is not None

        if not self._load_embeddings:
            # In this case, we don't load the embeddings
            log.info("No embeddings key provided, we will not load embeddings")
            self.embedding_caption_type = None
            self.t5_tokens_num = None
            self.is_mask_all_ones = None
            self.embedding_style_mapping = None
        else:
            self.embedding_caption_type: str = args[
                "embedding_caption_type"
            ]  # key to get the embedding of a particular caption type from the embedding field
            self.t5_tokens_num = args["t5_tokens"]["num"]  # number of tokens we cap after padding
            self.is_mask_all_ones = args["is_mask_all_ones"]  # if true, set mask for t5 to all ones

            self.embedding_style_mapping = {
                "long": self.embedding_caption_type,
                "short": f"{self.embedding_caption_type}_short",
                "medium": f"{self.embedding_caption_type}_medium",
                "user": f"{self.embedding_caption_type}_user",
            }

        self.caption_probs: dict[str, float] = args[
            "caption_probs"
        ]  # probabilities for user/short/medium/long captions
        self.caption_style_mapping = {
            "long": self.caption_type,
            "short": f"{self.caption_type}_short",
            "medium": f"{self.caption_type}_medium",
            "user": f"{self.caption_type}_user",
        }
        assert self.caption_probs.keys() == self.caption_style_mapping.keys(), (
            "The keys for caption_probs, caption_style_mapping, and embedding_style_mapping should match"
        )

        if self._load_embeddings:
            assert self.caption_style_mapping.keys() == self.embedding_style_mapping.keys(), (
                "The keys for caption_style_mapping and embedding_style_mapping should match"
            )

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs text transformation.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict with captions and t5 embeddings added
        """

        try:
            windows = data_dict[self.captions_key][self.caption_windows_key]
            n_windows = len(windows)
            chunk_index = data_dict["chunk_index"]

            if chunk_index == n_windows:
                # This will only happen when the number of captions does not match number of chunks due to re-transcoding the videos.
                log.warning(
                    f"TextTransform dataloader error: Found {data_dict['n_orig_video_frames']} in video but captioning is done with videos of {windows[-1]['end_frame']} frames. This mismatch is due to video re-transcoding.",
                    rank0_only=False,
                )
                chunk_index -= 1

            selected_caption_window = windows[chunk_index]
        except Exception as e:
            log.warning(
                f"TextTransform dataloader error -- url: {data_dict['__url__']}, key: {data_dict['__key__']}, chunk_index: {data_dict['chunk_index']}\n error {e}",
                rank0_only=False,
            )
            return None

        sampled_caption_style = None
        try:
            available_caption_styles = []
            for k in selected_caption_window.keys():
                caption_style = k.replace(self.caption_type, "").replace("_", "")
                if caption_style == "":  # it is long caption by default
                    available_caption_styles.append("long")
                elif caption_style in self.caption_style_mapping:
                    available_caption_styles.append(caption_style)
                else:
                    assert caption_style in ["startframe", "endframe"], f"Unsupported caption_type {caption_style}"

            probabilities_for_available_caption_styles = {
                k: v for k, v in self.caption_probs.items() if k in available_caption_styles
            }
            sampled_caption_style = random.choices(
                list(probabilities_for_available_caption_styles),
                weights=probabilities_for_available_caption_styles.values(),
            )[0]
            data_dict["ai_caption"] = selected_caption_window[self.caption_style_mapping[sampled_caption_style]]
        except Exception as e:
            log.warning(
                f"TextTransform dataloader error -- url: {data_dict['__url__']}, key: {data_dict['__key__']}, selected_caption_window: {selected_caption_window}\n error {e}",
                rank0_only=False,
            )
            return None
        if data_dict["ai_caption"] == "":
            log.warning(
                f"TextTransform dataloader error -- empty caption! url: {data_dict['__url__']}, key: {data_dict['__key__']}, selected_caption_window: {selected_caption_window}",
                rank0_only=False,
            )
            return None

        assert data_dict["ai_caption"] is not None and sampled_caption_style is not None
        data_dict["sampled_caption_style"] = sampled_caption_style

        del data_dict[self.captions_key]  # delete the field as we have extracted ai_caption from it

        if self._load_embeddings:
            ai_caption_embedding_data = data_dict[self.embeddings_key]
            try:
                if self.embedding_caption_type == "vila_caption":
                    t5_embedding = ai_caption_embedding_data[chunk_index]
                else:
                    t5_embedding = ai_caption_embedding_data[chunk_index][
                        self.embedding_style_mapping[sampled_caption_style]
                    ]
            except Exception as e:
                log.warning(
                    f"TextTransform dataloader error -- url: {data_dict['__url__']}, key: {data_dict['__key__']}, chunk_index: {data_dict['chunk_index']}, n embeddings: {len(ai_caption_embedding_data)}, n captions: {n_windows} \n error {e}",
                    rank0_only=False,
                )
                return None
            out_t5, out_t5_mask = pad_and_resize(
                t5_embedding,
                self.t5_tokens_num,
                is_mask_all_ones=self.is_mask_all_ones,
            )
            data_dict["t5_text_embeddings"] = out_t5
            data_dict["t5_text_mask"] = out_t5_mask
            del data_dict[self.embeddings_key]  # delete the field as we have extracted t5 embedding from it

        return data_dict


class TextTransformForVideoWithFullFrames(Augmentor):
    """
    Pair use with VideoParsingWithFullFrames to get the full frames of the video.
    The caption is assumed to be for the entire video frames, rather than TextTransformForVideo
    which assumes captions are for a specific chunk of frames.

    Audio captions are handled separately by AudioCaptionAppender, which appends
    audio descriptions to the video caption after this augmentor runs.
    """

    def __init__(self, input_keys: dict, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)
        assert len(input_keys) == 3, "TextTransformForVideoWithFullFrames augmentor only supports three input keys"
        self.meta_key = input_keys[0]
        self.video_key = input_keys[1]
        self.sequence_plan_key = input_keys[2]
        self.args = args
        self.keep_metas = args.get("keep_metas", False) if args else False
        self.caption_prefix = args.get("caption_prefix", None) if args else None

    def _apply_caption_prefix(self, data_dict: dict) -> None:
        """Prepend caption_prefix to ai_caption if configured."""
        if not self.caption_prefix or not isinstance(data_dict.get("ai_caption"), str):
            return
        original = data_dict["ai_caption"]
        data_dict["ai_caption"] = self.caption_prefix + " " + original.lstrip()
        log.debug(
            f"[caption_prefix] before: {original[:120]!r}... | after: {data_dict['ai_caption'][:120]!r}...",
            rank0_only=False,
        )

    @staticmethod
    def _resolve_multi_chunk_caption(raw_caption: str) -> str:
        """Resolve a caption that may be in multi-chunk JSON format.

        Multi-chunk captions are JSON strings encoding a dict of chunks, e.g.:
            {"chunk_0_300": {"caption": "...", "start_frame": 0, "end_frame": 300}, ...}
        When detected, a chunk is randomly selected and its "caption" text returned.
        Plain string captions are returned unchanged.
        """
        if not isinstance(raw_caption, str):
            return raw_caption
        try:
            parsed = json.loads(raw_caption)
        except (json.JSONDecodeError, TypeError):
            return raw_caption
        if not isinstance(parsed, dict) or len(parsed) == 0:
            return raw_caption
        chunk = random.choice(list(parsed.values()))
        if isinstance(chunk, dict) and "caption" in chunk:
            return chunk["caption"]
        return raw_caption

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs text transformation.

        Samples a video caption from metadata based on caption_config ratios.
        Supports both plain-string captions and multi-chunk JSON captions
        (randomly selects one chunk when multiple chunks are present).
        Audio captions are handled separately by AudioCaptionAppender.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict with captions and t5 embeddings added
        """
        caption_config = self.args["caption_config"]
        meta_dict = data_dict[self.meta_key]

        for caption_type in caption_config:
            assert caption_type in meta_dict, (
                f"Caption type {caption_type} not found in meta_dict (keys = {meta_dict.keys()})"
            )

        # First check if we are doing image to world or video to world
        if self.sequence_plan_key in data_dict:
            sequence_plan = data_dict[self.sequence_plan_key]
            conditioning_frame_indexes_vision = sequence_plan.condition_frame_indexes_vision
            if len(conditioning_frame_indexes_vision) > 0:
                sampled_caption = self._resolve_multi_chunk_caption(meta_dict["caption_temporal"])
                data_dict["ai_caption"] = sampled_caption
                data_dict["sampled_caption_style"] = "caption_temporal"

                self._apply_caption_prefix(data_dict)
                if not self.keep_metas:
                    del data_dict[self.meta_key]
                return data_dict

        # Text-to-world: sample from short, medium, long captions
        caption_keys = list(caption_config.keys())
        caption_ratios = [caption_config[k]["ratio"] for k in caption_keys]
        sampled_caption_type = random.choices(caption_keys, weights=caption_ratios, k=1)[0]
        data_dict["ai_caption"] = self._resolve_multi_chunk_caption(meta_dict[sampled_caption_type])
        data_dict["sampled_caption_style"] = sampled_caption_type

        self._apply_caption_prefix(data_dict)

        # Clean up - delete the caption fields that were sampled from
        for caption_type in caption_config.keys():
            if caption_type in meta_dict:
                del meta_dict[caption_type]

        # Delete metas unless keep_metas=True (set when AudioCaptionAppender runs downstream)
        if not self.keep_metas:
            del data_dict[self.meta_key]

        return data_dict


class CaptionSchemaError(ValueError):
    """A decoded transfer caption cannot provide the configured representations."""


class TextTransformForVideoTransferFullFrames(Augmentor):
    """Read structured captions for the full-frame transfer pipeline.

    Two-level lookup:

    1. A caption-source key is sampled (with weights) from ``caption_config``.
       This key identifies the WebDataset folder / metadata field whose value
       is a dict of annotations (e.g.
       ``"structured_captions_qwen3-vl-8b-lora-v1.5-merged"``). The sampled
       value is looked up first in ``data_dict`` (top-level) and then in
       ``meta_dict``.
    2. Inside that caption dict the field ``caption_structured`` is hardcoded
       as the JSON-encoded chunked annotation, of the form
       ``{"chunk_0_300": {"caption": "<json-encoded structured payload>",
       "start_frame": ..., "end_frame": ...}}``.

    The full-frame pipeline always decodes from the start of the video, so the
    first chunk is always selected and its inner JSON-encoded structured payload
    is parsed back into a dict before being serialized as ``ai_caption``.
    A source can opt into ``dense_caption_probability`` to emit that same
    payload's ``temporal_caption`` as plain text for a fraction of samples.
    """

    CAPTION_FIELD: str = "caption_structured"

    def __init__(self, input_keys: dict, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)
        assert len(input_keys) >= 1, "TextTransformForVideoTransferFullFrames requires a metadata input key"
        self.meta_key = input_keys[0]
        self.args = args or {}
        self.keep_metas = self.args.get("keep_metas", False)
        self.caption_options = self._normalize_caption_config(self.args["caption_config"])
        for caption_key, _, option in self.caption_options:
            try:
                dense_probability = float(option.get("dense_caption_probability", 0.0))
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid dense_caption_probability for {caption_key!r}.") from error
            if not math.isfinite(dense_probability) or not 0.0 <= dense_probability <= 1.0:
                raise ValueError(
                    f"dense_caption_probability for {caption_key!r} must be finite and in [0, 1], "
                    f"got {dense_probability}."
                )
        # This fixes transfer datasets that mix caption chunks with different
        # lengths. Each caption source needs its own stride so the sampled video
        # stays within the token budget while matching the selected caption.
        self.min_stride_key = self.args.get("min_stride_key", "_full_frames_min_stride")
        self._caption_stats_pid: int = os.getpid()
        self._caption_attempted: int = 0
        self._caption_yielded: int = 0
        self._caption_rejected: dict[str, int] = {}

    def _record_caption_attempt(self) -> None:
        # Forked DataLoader workers must not inherit counts from the parent.
        pid = os.getpid()
        if pid != self._caption_stats_pid:
            self._caption_stats_pid = pid
            self._caption_attempted = 0
            self._caption_yielded = 0
            self._caption_rejected = {}
        self._caption_attempted += 1

    def _reject_caption(
        self,
        data_dict: dict,
        reason: str,
        detail: str,
        caption_key: str | None = None,
        chunk_key: str | None = None,
    ) -> None:
        count = self._caption_rejected.get(reason, 0) + 1
        self._caption_rejected[reason] = count
        # Report zero-yield schema outages immediately, then bound log volume
        # independently for each reason so decode failures cannot hide them.
        if count <= 3 or count % 100 == 0:
            log.warning(
                f"{type(self).__name__}: {reason}: {detail} "
                f"caption_source={caption_key!r}, chunk={chunk_key!r}, "
                f"url: {data_dict.get('__url__')}, key: {data_dict.get('__key__')}; "
                f"worker-local transform counts (pid={self._caption_stats_pid}): "
                f"attempted={self._caption_attempted}, yielded={self._caption_yielded}, "
                f"rejected={self._caption_rejected}. Logging first 3 and every 100 failures per reason.",
                rank0_only=False,
            )

    @staticmethod
    def _normalize_caption_config(caption_config: dict | list) -> list[tuple[str, float, dict]]:
        if isinstance(caption_config, dict):
            options: list[tuple[str, float, dict]] = []
            for caption_key, config in caption_config.items():
                if isinstance(config, dict):
                    ratio = config.get("ratio", 1.0)
                    # Keep more than the sampling ratio because source-specific
                    # settings, like min_stride, are part of how caption/video
                    # alignment is preserved.
                    options.append((caption_key, float(ratio), dict(config)))
                else:
                    options.append((caption_key, float(config), {}))
            return options

        options = []
        for item in caption_config:
            if isinstance(item, str):
                options.append((item, 1.0, {}))
            elif isinstance(item, dict):
                caption_key = item.get("key") or item.get("caption_key") or item.get("caption_type") or item.get("name")
                if caption_key is None:
                    raise ValueError(f"Caption config entry is missing a caption key: {item}")
                options.append((caption_key, float(item.get("ratio", 1.0)), dict(item)))
            else:
                caption_key, ratio = item
                options.append((caption_key, float(ratio), {}))
        return options

    def _lookup_caption_dict(self, data_dict: dict, meta_dict: dict | None, caption_key: str) -> dict | None:
        candidate = data_dict.get(caption_key)
        if candidate is None and isinstance(meta_dict, dict):
            candidate = meta_dict.get(caption_key)
        if isinstance(candidate, dict):
            return candidate
        return None

    @staticmethod
    def _format_caption(structured: object, option: dict) -> str:
        """Choose structured JSON or dense prose from one already-selected chunk."""
        dense_probability = float(option.get("dense_caption_probability", 0.0))
        if dense_probability == 0.0:
            return json.dumps(structured)
        if not isinstance(structured, dict):
            raise CaptionSchemaError("Dense caption mixing requires a structured caption dictionary.")
        dense_caption = structured.get("temporal_caption")
        if not isinstance(dense_caption, str) or not dense_caption.strip():
            raise CaptionSchemaError("Dense caption mixing requires a nonempty temporal_caption string.")
        # Validate both representations before choosing one so malformed dense
        # captions cannot change the requested mixture through selective rejection.
        structured_caption = json.dumps(structured)
        return dense_caption if random.random() < dense_probability else structured_caption

    def __call__(self, data_dict: dict) -> dict | None:
        self._record_caption_attempt()
        meta_dict = data_dict.get(self.meta_key)

        available_options: list[tuple[str, float, dict]] = []
        for key, ratio, option in self.caption_options:
            if ratio <= 0:
                continue
            if self._lookup_caption_dict(data_dict, meta_dict, key) is not None:
                available_options.append((key, ratio, option))

        if not available_options:
            self._reject_caption(
                data_dict,
                "missing_source",
                f"None of the configured caption keys {[key for key, _, _ in self.caption_options]} "
                "hold a caption dict in metadata/sample keys.",
            )
            return None

        sampled_caption_key = random.choices(
            [key for key, _, _ in available_options],
            weights=[ratio for _, ratio, _ in available_options],
            k=1,
        )[0]
        sampled_caption_option = next(option for key, _, option in available_options if key == sampled_caption_key)
        caption_dict = self._lookup_caption_dict(data_dict, meta_dict, sampled_caption_key)
        if caption_dict is None or self.CAPTION_FIELD not in caption_dict:
            self._reject_caption(
                data_dict,
                "missing_field",
                f"Caption dict is missing the hardcoded {self.CAPTION_FIELD} field.",
                sampled_caption_key,
            )
            return None

        sampled_chunk_key: str | None = None
        try:
            chunks = json.loads(caption_dict[self.CAPTION_FIELD])
            sampled_chunk_key, first_chunk = next(iter(chunks.items()))
            structured = json.loads(first_chunk["caption"])
            caption_text = self._format_caption(structured, sampled_caption_option)
        except CaptionSchemaError as error:
            self._reject_caption(data_dict, "schema_error", str(error), sampled_caption_key, sampled_chunk_key)
            return None
        except Exception as e:
            self._reject_caption(
                data_dict,
                "decode_error",
                f"Failed to decode {self.CAPTION_FIELD}: {e}",
                sampled_caption_key,
                sampled_chunk_key,
            )
            return None

        data_dict["ai_caption"] = caption_text
        data_dict["sampled_caption_style"] = sampled_caption_key
        if "min_stride" in sampled_caption_option:
            # Without this override, 200-frame and 400-frame caption sources
            # would share one stride and could either waste context or overflow
            # the intended token length.
            data_dict[self.min_stride_key] = int(sampled_caption_option["min_stride"])

        if not self.keep_metas:
            data_dict.pop(self.meta_key, None)
        for caption_key, _, _ in self.caption_options:
            if caption_key in data_dict:
                del data_dict[caption_key]
        self._caption_yielded += 1
        return data_dict


class TextTransformForVideoTransferChunkedFrames(TextTransformForVideoTransferFullFrames):
    """Read structured captions and sample one chunk for transfer training.

    This keeps the full-frame caption-source sampling behavior, including
    per-source options such as ``min_stride``, but emits the sampled chunk's
    frame range so the downstream parser can decode the matching RGB/control
    frames.
    """

    def __init__(self, input_keys: dict, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)
        # The parser still needs metadata for fps/resolution after this transform.
        self.keep_metas = self.args.get("keep_metas", True)
        self.min_num_frames = int(self.args.get("min_num_frames", 5))
        self.target_num_frames: int | None = self.args.get("target_num_frames")

    def _supports_target_length(self, start_frame: int, end_frame: int, option: dict, meta_dict: dict) -> bool:
        """Keep complete caption windows that can be resampled without duplicating frames."""
        if self.target_num_frames is None:
            return True
        if self.target_num_frames < 1 or end_frame - start_frame < self.target_num_frames:
            return False
        try:
            source_num_frames = int(meta_dict["nb_frames"])
            source_fps = float(meta_dict["framerate"])
        except (KeyError, TypeError, ValueError):
            return False
        if start_frame < 0 or end_frame > source_num_frames:
            return False
        if self.target_num_frames == 1:
            # A single generated frame cannot represent an entire multi-frame
            # caption window, even when VAE alignment maps it to one latent.
            if end_frame - start_frame != 1:
                return False
            stride = 1.0
        else:
            stride = (end_frame - start_frame - 1) / (self.target_num_frames - 1)
        min_stride = int(option.get("min_stride", self.args.get("min_stride", 1)))
        max_stride = max(int(self.args.get("max_stride", 3)), min_stride)
        min_fps = float(self.args.get("min_fps", 0.0))
        max_fps = float(self.args.get("max_fps", float("inf")))
        return min_stride <= stride <= max_stride and min_fps <= source_fps / stride <= max_fps

    def __call__(self, data_dict: dict) -> dict | None:
        self._record_caption_attempt()
        meta_dict = data_dict.get(self.meta_key)

        available_options: list[tuple[str, float, dict]] = []
        for key, ratio, option in self.caption_options:
            if ratio <= 0:
                continue
            if self._lookup_caption_dict(data_dict, meta_dict, key) is not None:
                available_options.append((key, ratio, option))

        if not available_options:
            self._reject_caption(
                data_dict,
                "missing_source",
                f"None of the configured caption keys {[key for key, _, _ in self.caption_options]} "
                "hold a caption dict in metadata/sample keys.",
            )
            return None

        sampled_caption_key = random.choices(
            [key for key, _, _ in available_options],
            weights=[ratio for _, ratio, _ in available_options],
            k=1,
        )[0]
        sampled_caption_option = next(option for key, _, option in available_options if key == sampled_caption_key)
        caption_dict = self._lookup_caption_dict(data_dict, meta_dict, sampled_caption_key)
        if caption_dict is None or self.CAPTION_FIELD not in caption_dict:
            self._reject_caption(
                data_dict,
                "missing_field",
                f"Caption dict is missing the hardcoded {self.CAPTION_FIELD} field.",
                sampled_caption_key,
            )
            return None

        sampled_chunk_key: str | None = None
        try:
            chunks = json.loads(caption_dict[self.CAPTION_FIELD])
            if not isinstance(chunks, dict) or len(chunks) == 0:
                self._reject_caption(data_dict, "empty_chunks", "Expected a nonempty chunk dict.", sampled_caption_key)
                return None

            eligible_chunk_keys: list[str] = []
            for chunk_key, chunk in chunks.items():
                try:
                    start_frame = int(chunk["start_frame"])
                    end_frame = int(chunk["end_frame"])
                except (KeyError, TypeError, ValueError):
                    continue
                if end_frame - start_frame >= self.min_num_frames and self._supports_target_length(
                    start_frame, end_frame, sampled_caption_option, meta_dict
                ):
                    eligible_chunk_keys.append(chunk_key)

            if not eligible_chunk_keys:
                self._reject_caption(
                    data_dict,
                    "no_eligible_chunks",
                    f"No chunks meet min_num_frames={self.min_num_frames} "
                    f"and target_num_frames={self.target_num_frames} constraints.",
                    sampled_caption_key,
                )
                return None

            sampled_chunk_key = random.choice(eligible_chunk_keys)
            sampled_chunk = chunks[sampled_chunk_key]
            chunk_start_frame = int(sampled_chunk["start_frame"])
            chunk_end_frame = int(sampled_chunk["end_frame"])
            structured = json.loads(sampled_chunk["caption"])
            if self.target_num_frames is not None and not isinstance(structured, dict):
                raise CaptionSchemaError("Exact target lengths require a structured caption dictionary.")
            if self.target_num_frames is not None:
                # Remove source metadata before the downstream augmentor appends
                # the sampled clip's duration/FPS. Keep the teacher's established
                # serialized-caption format and preserve all narrative fields.
                structured.pop("duration", None)
                structured.pop("fps", None)
            caption_text = self._format_caption(structured, sampled_caption_option)
        except CaptionSchemaError as error:
            self._reject_caption(data_dict, "schema_error", str(error), sampled_caption_key, sampled_chunk_key)
            return None
        except Exception as e:
            self._reject_caption(
                data_dict,
                "decode_error",
                f"Failed to decode {self.CAPTION_FIELD}: {e}",
                sampled_caption_key,
                sampled_chunk_key,
            )
            return None

        data_dict["chunk_start_frame"] = chunk_start_frame
        data_dict["chunk_end_frame"] = chunk_end_frame
        data_dict["ai_caption"] = caption_text
        data_dict["sampled_caption_style"] = sampled_caption_key
        data_dict["sampled_chunk_key"] = sampled_chunk_key
        if "min_stride" in sampled_caption_option:
            data_dict[self.min_stride_key] = int(sampled_caption_option["min_stride"])

        if not self.keep_metas:
            data_dict.pop(self.meta_key, None)
        for caption_key, _, _ in self.caption_options:
            if caption_key in data_dict:
                del data_dict[caption_key]
        self._caption_yielded += 1
        return data_dict


class TextTransformForVideoJsonCaption(Augmentor):
    """Transform a chunk-keyed caption and expose its matching frame range.

    The outer caption is a JSON object keyed by chunk. Inner captions may be
    structured dictionaries, JSON-encoded dictionaries, or plain prose. Plain
    prose is normalized to ``{"description": prose}`` so all downstream text
    transforms receive a dictionary while ``VideoParsingChunkedFrames`` receives
    the selected chunk's frame boundaries.

    By default, a non-empty metadata audio caption is injected into the caption
    dictionary. With ``natural_language_caption=True``, the selected chunk's
    description is emitted as text instead so a downstream AudioCaptionAppender
    can add the audio caption before a final video metadata section.
    """

    def __init__(self, input_keys: dict, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)
        assert len(input_keys) >= 2, (
            "TextTransformForVideoJsonCaption augmentor requires at least two input keys: [meta_key, video_key]"
        )
        self.meta_key: str = input_keys[0]
        self.video_key: str = input_keys[1]
        self.args: dict = args or {}
        self.keep_metas: bool = self.args.get("keep_metas", False)
        self.caption_key: str = self.args.get("caption_key", "caption")
        self.natural_language_caption: bool = self.args.get("natural_language_caption", False)

    @staticmethod
    def _json_dict_or_none(raw_caption: object) -> dict | None:
        if isinstance(raw_caption, dict):
            return raw_caption
        if not isinstance(raw_caption, str) or len(raw_caption) == 0:
            return None
        try:
            parsed_caption = json.loads(raw_caption)
        except (json.JSONDecodeError, TypeError):
            return None
        return parsed_caption if isinstance(parsed_caption, dict) else None

    @staticmethod
    def _caption_payload_to_dict(raw_caption: object) -> dict:
        """Normalize a structured JSON or plain-text chunk caption."""
        if isinstance(raw_caption, dict):
            return dict(raw_caption)
        if not isinstance(raw_caption, str) or not raw_caption.strip():
            raise ValueError("chunk caption must be a non-empty string or dictionary")

        try:
            parsed_caption = json.loads(raw_caption)
        except (json.JSONDecodeError, TypeError):
            return {"description": raw_caption}

        if isinstance(parsed_caption, dict):
            return parsed_caption
        if isinstance(parsed_caption, str) and parsed_caption.strip():
            return {"description": parsed_caption}
        raise ValueError("chunk caption JSON must decode to a non-empty string or dictionary")

    @staticmethod
    def _frame_count_or_large_bound(meta_dict: dict) -> int:
        nb_frames = meta_dict.get("nb_frames")
        if isinstance(nb_frames, int) and nb_frames > 0:
            return nb_frames
        length = meta_dict.get("length")
        framerate = meta_dict.get("framerate")
        if isinstance(length, (float, int)) and isinstance(framerate, (float, int)) and length > 0 and framerate > 0:
            return max(1, int(round(length * framerate)))
        # VideoParsingChunkedFrames clamps to the decoder length, so a large end frame is safe.
        return 10**9

    def _find_audio_caption(self, meta_dict: dict) -> str | None:
        audio_caption = meta_dict.get("caption_audio")
        if isinstance(audio_caption, str) and len(audio_caption) > 0:
            return audio_caption
        for value in meta_dict.values():
            if isinstance(value, dict):
                caption_sound = value.get("caption_sound")
                if isinstance(caption_sound, str) and len(caption_sound) > 0:
                    return caption_sound
        return None

    def _parse_legacy_full_video_caption(self, meta_dict: dict) -> dict[str, dict] | None:
        """Build one full-video chunk from older caption schemas that do not have ``caption``."""
        caption_json = self._json_dict_or_none(meta_dict.get("caption_structured"))
        if caption_json is None:
            for caption_key in (
                "caption_rewrite_dense",
                "caption_dense",
                "caption_descriptive",
                "caption_base",
                "caption_temporal",
                "caption_short",
            ):
                caption_text = meta_dict.get(caption_key)
                if isinstance(caption_text, str) and len(caption_text) > 0:
                    caption_json = {"description": caption_text}
                    break
        if caption_json is None:
            return None

        end_frame = self._frame_count_or_large_bound(meta_dict)
        return {
            f"chunk_0_{end_frame}": {
                "start_frame": 0,
                "end_frame": end_frame,
                "caption_json": caption_json,
            }
        }

    def _parse_audio_caption_chunks(self, meta_dict: dict) -> dict[str, dict] | None:
        """Build chunk metadata from nested audio-caption metas when visual captions are absent."""
        chunks: dict[str, dict] = {}
        for key, value in meta_dict.items():
            if not isinstance(key, str) or not isinstance(value, dict):
                continue
            caption_sound = value.get("caption_sound")
            if not isinstance(caption_sound, str) or len(caption_sound) == 0:
                continue

            try:
                start_frame, end_frame = [int(part) for part in key.split("_", maxsplit=1)]
            except ValueError:
                start_frame = value.get("start_frame")
                end_frame = value.get("end_frame")
                if not isinstance(start_frame, int) or not isinstance(end_frame, int):
                    continue

            chunks[key] = {
                "start_frame": start_frame,
                "end_frame": end_frame,
                "caption_json": {"audio_description": caption_sound},
            }

        return chunks or None

    def __call__(self, data_dict: dict) -> dict | None:
        r"""Performs text transformation.

        Parses the per-chunk caption JSON, randomly samples one chunk, and writes
        the chunk's frame range into ``data_dict`` so a downstream
        ``VideoParsingChunkedFrames`` can decode only that frame range. Output is
        either the existing structured caption or, when configured, the selected
        description as natural-language text.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict with captions and t5 embeddings added
        """
        caption_config = self.args["caption_config"]
        json_field_dropout_rate = caption_config["json_field_dropout_rate"]

        try:
            meta_dict = data_dict[self.meta_key]
            raw_caption = meta_dict.get(self.caption_key)
            if raw_caption is not None:
                caption = self._json_dict_or_none(raw_caption)
                if caption is None:
                    raise ValueError(f"{self.caption_key} is not a JSON object")
            else:
                # Some sound midtrain shards use older full-video visual captions
                # (caption_base/caption_structured/...) instead of the chunked
                # ``caption`` field. Prefer those visual captions when present;
                # otherwise fall back to nested audio-only chunks.
                caption = self._parse_legacy_full_video_caption(meta_dict)
                if caption is None:
                    caption = self._parse_audio_caption_chunks(meta_dict)
                if caption is None:
                    raise KeyError(self.caption_key)

            # Contents of caption
            # caption = {
            #    "chunk_0_300": {
            #        "caption": "...",
            #        "start_frame": 0,
            #        "end_frame": 300,
            #    },
            #    "chunk_300_435": {
            #        "caption": "...",
            #        "start_frame": 300,
            #        "end_frame": 435,
            #    },
            # }
            chunk_keys = list(caption.keys())
            if len(chunk_keys) == 0:
                log.warning(
                    f"TextTransformForVideoJsonCaption: empty caption dict. url: {data_dict.get('__url__')}, key: {data_dict.get('__key__')}",
                    rank0_only=False,
                )
                return None

            sampled_key = random.choice(chunk_keys)
            sampled_chunk = caption[sampled_key]

            data_dict["chunk_index"] = chunk_keys.index(sampled_key)
            data_dict["chunk_start_frame"] = int(sampled_chunk["start_frame"])
            data_dict["chunk_end_frame"] = int(sampled_chunk["end_frame"])

            caption_payload = sampled_chunk.get("caption_json", sampled_chunk.get("caption"))
            caption_json = self._caption_payload_to_dict(caption_payload)
        except Exception as e:
            log.warning(
                f"TextTransformForVideoJsonCaption dataloader error -- url: {data_dict.get('__url__')}, key: {data_dict.get('__key__')}\n error {e}",
                rank0_only=False,
            )
            return None

        # Randomly dropout json keys during training
        if json_field_dropout_rate > 0:
            for key in list(caption_json.keys()):
                if random.random() < json_field_dropout_rate:
                    caption_json.pop(key)

        if self.natural_language_caption:
            caption_text = caption_json.get("description")
            if not isinstance(caption_text, str) or not caption_text.strip():
                log.warning(
                    f"TextTransformForVideoJsonCaption: natural-language chunk is missing a description. "
                    f"url: {data_dict.get('__url__')}, key: {data_dict.get('__key__')}",
                    rank0_only=False,
                )
                return None
            data_dict["ai_caption"] = caption_text
        else:
            # Inject audio caption from metas as a new field when available. Added after the field
            # dropout above so it is preserved whenever upstream metadata provides it.
            audio_caption = self._find_audio_caption(meta_dict)
            if isinstance(audio_caption, str) and len(audio_caption) > 0:
                caption_json["audio_description"] = audio_caption
            data_dict["ai_caption"] = caption_json

        # Delete metas unless keep_metas=True (set when downstream augmentors still need them,
        # e.g. VideoParsingChunkedFrames needs framerate/width/height/nb_frames).
        if not self.keep_metas:
            del data_dict[self.meta_key]

        return data_dict
