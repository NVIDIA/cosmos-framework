# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import time
from typing import Any

import torch
from torch.utils.data._utils.collate import default_collate

from cosmos_framework.data.generator.packing_iterable_dataset import FP8_PAD_MULTIPLE
from cosmos_framework.data.generator.processors.audio_utils import collate_audio_processor_outputs

_AUDIO_PROCESSOR_KEYS = ("audio_features", "audio_feature_lengths", "audio_token_lengths")


def custom_collate(batch: list[dict[str, Any]], pad_to_multiple_of: int = FP8_PAD_MULTIPLE) -> dict[str, Any]:
    if pad_to_multiple_of < FP8_PAD_MULTIPLE or pad_to_multiple_of % FP8_PAD_MULTIPLE:
        raise ValueError("pad_to_multiple_of must be a positive multiple of FP8_PAD_MULTIPLE")
    # batch is a list of dicts
    # Packing-time telemetry: wall-time spent ASSEMBLING the batch here in the dataloader worker
    # (padding+stacking on the padded path, concat+cu_seqlens on the true-packing path). Both layouts
    # share the same dynamic-batch SELECTION upstream, so this collate is the one packing-specific CPU
    # cost that differs between them. Emitted below as a sync-free CPU float ``collate_ms`` (same
    # contract as content_tokens: plain Python float, passes through misc.to(...) untouched, stripped
    # before the HF forward via HFModel._COLLATE_NON_MODEL_KEYS) so VLMTokensPerSec can attribute how
    # much of ``timer/dataloader_train`` (the main-process fetch wait) is packing vs upstream I/O --
    # WITHOUT adding any device sync to the timed step.
    t_collate_start = time.perf_counter()
    keys_processed = []
    special = {}
    regular = {}
    batch_size = len(batch)
    if "collated" in batch[0] and batch[0]["collated"]:
        # Skip collated batch (already assembled upstream -> no packing work happens here, so we do
        # NOT stamp collate_ms; the callback simply skips the metric for this step).
        return batch[0]

    # First assert all keys are 1D
    for key in ["input_ids", "token_mask", "attention_mask", "labels"]:
        assert all([item[key].ndim == 1 for item in batch]), f"Key {key} is not 1D"

    # Forward-safety: padded multi-sample batching is correct because every row is RIGHT-padded
    # (pads appended below), mask-aware backends consume attention_mask, the cosmos causal path
    # cannot attend from real tokens into a later right-pad tail, and label pads are ignore_index.
    # The one cross-sample invariant the collate cannot enforce structurally is a shared pad /
    # ignore id -- mixed ids across a packed batch would silently corrupt masking and the loss.
    # Cheap (a set over <=mbs items); guards the mbs>1 path that this MR turns on in production.
    if batch_size > 1:
        assert len({int(item["pad_token_id"]) for item in batch}) == 1, "Mixed pad_token_id in a packed batch"
        assert len({int(item["ignore_index"]) for item in batch}) == 1, "Mixed ignore_index in a packed batch"

    if any("mm_token_type_ids" in item for item in batch):
        if not all(
            "mm_token_type_ids" in item and item["mm_token_type_ids"].shape == item["input_ids"].shape for item in batch
        ):
            raise ValueError("mm_token_type_ids must align with input_ids for every sample before collation")
        if bool(batch[0].get("true_packing", False)):
            raise ValueError("Qwen3.5 modality IDs require padded batches; recurrent true packing is not supported")

    # True sequence packing: when the packer marks the batch, build
    # ONE B=1 concatenated row + cu_seqlens for block-diagonal varlen attention instead of padding and
    # stacking. Read (do not pop) the seed marker here; the assembler pops it from every item.
    if bool(batch[0].get("true_packing", False)):
        if pad_to_multiple_of != FP8_PAD_MULTIPLE:
            raise ValueError("Sequence bucketing applies only to padded batches")
        result = _collate_true_packing(batch)
        result["collate_ms"] = (time.perf_counter() - t_collate_start) * 1000.0
        return result

    # max length over input_ids
    max_seq_length = max([item["input_ids"].shape[0] for item in batch])
    # Content (non-pad) token count = sum of real input_ids lengths, captured HERE
    # before the padding loop below mutates input_ids. Emitted as a plain Python int so
    # it passes through misc.to(..., device="cuda") untouched and the VLMTokensPerSec
    # callback can read packing efficiency WITHOUT a per-step device sync. Dropped
    # before the HF forward, which allowlists its signature (HFModel._forward_keys).
    content_tokens = int(sum(int(item["input_ids"].shape[0]) for item in batch))
    # Extended packing telemetry, captured PRE-pad on the worker's CPU tensors (same
    # sync-free contract as content_tokens: plain Python ints, passed through misc.to(...)
    # untouched, dropped before the HF forward by HFModel._forward_keys, read by
    # VLMTokensPerSec with zero device work).
    #   supervised_tokens: U* = #positions that carry a loss (labels != ignore_index) ->
    #     supervision density rho_sup = U*/content_tokens. Computed before the labels padding
    #     loop below appends ignore_index, so pads are not counted.
    #   seq_max_len: the UNPADDED longest sample l_max (the padded row length is the /16-rounded
    #     value below) -> l_max distribution.
    #   sum_len_sq: sum_i L_i^2 -> "content" attention work; against the padded k*l_max^2 this
    #     is the attention-quadratic waste that true sequence packing would remove.
    supervised_tokens = 0
    for item in batch:
        supervised_mask = item["labels"][1:] != item["ignore_index"]  # [T-1]
        supervised_tokens += int(supervised_mask.sum())  # [] -> int
    seq_max_len = int(max_seq_length)
    sum_len_sq = int(sum(int(item["input_ids"].shape[0]) ** 2 for item in batch))
    # Packer-predicted per-step runtime (FLOP cost model), set by the dynamic batcher's
    # _best_fit_batch on the seed sample only. Pop it from EVERY item (not just the first match)
    # so it can NEVER reach the generic per-key collate loop below -- which builds
    # default_collate([item[key] for item in batch]) and would KeyError on the samples that don't
    # carry it should any future path tag more than one sample. We keep the first non-None value.
    # Surfaced as a batch-level key for the VLMTokensPerSec realized-vs-predicted metric; sync-free,
    # dropped before the HF forward by HFModel._forward_keys. Absent on the token-based
    # batching path -> stays None and the metric is simply skipped.
    predicted_runtime_ms = None
    for item in batch:
        v = item.pop("predicted_runtime_ms", None)
        if v is not None and predicted_runtime_ms is None:
            predicted_runtime_ms = float(v)
    # Packing diagnostics (sync-free CPU ints) set by the dynamic batcher on the seed sample only:
    #   singleton_cause: WHY a 1-sample step happened (SingletonCause value; 0 == not a singleton)
    #     so a high singleton_rate is attributable to the right lever (singleton counters).
    #   over_budget: 1 if this MULTI-sample batch's realized padded cost exceeds the correct budget
    #     -- the budget-gate symptom; 0 under the fixed gate. Both are popped from EVERY item (same
    #     KeyError-safety reason as predicted_runtime_ms) and stripped before the HF forward via
    #     HFModel._forward_keys.
    singleton_cause = None
    over_budget = None
    for item in batch:
        sc = item.pop("singleton_cause", None)
        ob = item.pop("over_budget", None)
        # Padded path: drop the true-packing marker (False here) from EVERY item so it cannot reach
        # the generic default_collate loop below (it is stamped only on the seed). The packed path is
        # taken above before this loop.
        item.pop("true_packing", None)
        if sc is not None and singleton_cause is None:
            singleton_cause = int(sc)
        if ob is not None and over_budget is None:
            over_budget = int(ob)
    # The default preserves the packer's FP8 alignment. Optional coarser buckets
    # bound shape-specialized kernel variants for padded singleton caption batches.
    max_seq_length = (max_seq_length + pad_to_multiple_of - 1) // pad_to_multiple_of * pad_to_multiple_of

    # pad for input_ids with pad_token_id
    for item in batch:
        item["input_ids"] = torch.cat(
            [
                item["input_ids"],
                torch.full((max_seq_length - item["input_ids"].shape[0],), item["pad_token_id"], dtype=torch.long),
            ]
        )  # [max_seq_length]
    regular["input_ids"] = torch.stack([item["input_ids"] for item in batch], dim=0)  # [B,max_seq_length]

    # pad for 'token_mask', 'attention_mask' with zeros
    for key in ["token_mask", "attention_mask"]:
        for item in batch:
            item[key] = torch.cat(
                [item[key], torch.full((max_seq_length - item[key].shape[0],), False, dtype=torch.bool)]
            )  # [max_seq_length]
        regular[key] = torch.stack([item[key] for item in batch], dim=0)  # [B,max_seq_length]

    if "mm_token_type_ids" in batch[0]:
        regular["mm_token_type_ids"] = torch.stack(
            [
                torch.cat(
                    [
                        item["mm_token_type_ids"],
                        item["mm_token_type_ids"].new_zeros(max_seq_length - item["mm_token_type_ids"].shape[0]),
                    ]
                )
                for item in batch
            ],
            dim=0,
        )

    # pad for 'labels' with ignore_index
    for item in batch:
        item["labels"] = torch.cat(
            [
                item["labels"],
                torch.full((max_seq_length - item["labels"].shape[0],), item["ignore_index"], dtype=torch.long),
            ]
        )  # [max_seq_length]
    regular["labels"] = torch.stack([item["labels"] for item in batch], dim=0)  # [B,max_seq_length]

    if any(any(key in item for key in _AUDIO_PROCESSOR_KEYS) for item in batch):
        regular.update(
            collate_audio_processor_outputs(
                audio_features=[item.get("audio_features") for item in batch],
                audio_feature_lengths=[item.get("audio_feature_lengths") for item in batch],
                audio_token_lengths=[item.get("audio_token_lengths") for item in batch],
                num_samples=batch_size,
            )
        )

    if any("raw_image" in item for item in batch):
        # Preserve per-sample, per-image boundaries because image sizes can differ.
        regular_raw_image: list[list[torch.Tensor]] = []
        for item in batch:
            raw_image = item.get("raw_image", [])
            if isinstance(raw_image, torch.Tensor):
                if raw_image.ndim == 3:
                    raw_image = raw_image[:, None]  # [3,1,H,W]
                raw_image = [
                    raw_image[:, image_idx : image_idx + 1] for image_idx in range(raw_image.shape[1])
                ]  # each: [3,1,H,W]
            regular_raw_image.append(raw_image)
        regular["raw_image"] = regular_raw_image

    if any("raw_video" in item for item in batch):
        # Preserve per-sample, per-video boundaries because video counts and sizes can differ.
        regular_raw_video: list[list[torch.Tensor]] = []
        for item in batch:
            raw_video = item.get("raw_video", [])
            if isinstance(raw_video, torch.Tensor):
                raw_video = [raw_video]
            regular_raw_video.append(raw_video)
        regular["raw_video"] = regular_raw_video

    all_keys = list(set([key for item in batch for key in item.keys()]))
    for key in all_keys:
        if key in regular:  # already collated
            continue
        if key in [
            "image_grid_thw",
            "video_grid_thw",
            "second_per_grid_ts",
            "pixel_values",
            "pixel_values_videos",
            "image_sizes",
        ]:
            special[key] = torch.cat([item[key] for item in batch if key in item], dim=0)  # [total_across_batch,...]
        elif key in ["second_per_grid_ts"]:
            # collect as-is (list instead of collating)
            list_of_tensor = [torch.tensor(item[key]) for item in batch if key in item]
            special[key] = torch.cat(list_of_tensor, dim=0)  # [total_across_batch]
        else:
            # let default_collate handle this key
            regular[key] = default_collate([item[key] for item in batch])

    # merge the dicts
    result = {
        **regular,
        **special,
        "collated": True,
        "content_tokens": content_tokens,
        "supervised_tokens": supervised_tokens,
        "seq_max_len": seq_max_len,
        "sum_len_sq": sum_len_sq,
        "attended_token_pairs": batch_size * max_seq_length**2,
        "logical_batch_size": batch_size,
        "logical_supervised_tokens": supervised_tokens,
    }
    # Only present on the FLOP-based batching path (None otherwise) -> emit conditionally so the
    # batch dict carries no None values into misc.to(...).
    if predicted_runtime_ms is not None:
        result["predicted_runtime_ms"] = predicted_runtime_ms
    # Packing diagnostics: present only when the VFM dynamic batcher built this batch. Plain ints
    # (like content_tokens) -> pass through misc.to(...) untouched; absent for non-packer collates.
    if singleton_cause is not None:
        result["singleton_cause"] = singleton_cause
    if over_budget is not None:
        result["over_budget"] = over_budget
    # Packing-time telemetry (see custom_collate header): sync-free CPU float, stamped last so it
    # captures the full padded-collate assembly cost. Its true-packing counterpart is stamped in
    # custom_collate around the _collate_true_packing call.
    result["collate_ms"] = (time.perf_counter() - t_collate_start) * 1000.0
    return result


def _collate_true_packing(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """True-packing assembler.

    Concatenates the ``k`` packed samples into ONE ``B=1`` row of ``S = sum(L_i)`` tokens (padded only
    once to ``FP8_PAD_MULTIPLE`` at the tail) and emits the varlen metadata that drives block-diagonal
    causal attention in the cosmos adapter:

      * ``packed_cu_seq_lens`` ``[0, L_1, L_1+L_2, ..., S(, S_pad)]`` (int32) -- one segment per
        sample, plus a trailing segment for the FP8 pad tail when ``S_pad > S`` so pad tokens never
        mix with a sample.
      * ``packed_max_length`` -- the longest segment (python int).
      * Boundary label masking: the first label of every non-first segment is set to ``ignore_index``
        so the global next-token shift in the CE loss carries NO cross-sample term. This loses
        no real supervision (a sample's first token is never an in-sample target).
      * Per-sample RoPE metadata: ``seq_lens`` so the owning packing dataset can reshape the real
        prefix into a ``[k, max_L]`` CPU batch and build per-sample-reset positions from the
        concatenated grids before the trainer's H2D copy. No per-sample image/video counts are
        needed -- ``get_rope_index`` consumes the concatenated grids in sample order via its own
        global counters.

    Telemetry (``content_tokens`` / ``supervised_tokens`` / ``seq_max_len`` / ``sum_len_sq``) is
    computed identically to the padded path so the A/B columns are directly comparable, with
    ``packed_tokens`` / ``pack_efficiency`` added for realized density.
    """
    # Marker + sync-free diagnostics are stamped on the seed only; pop from EVERY item so none reach
    # the generic key handling below (same KeyError-safety contract as the padded path).
    if any(any(key in item for key in _AUDIO_PROCESSOR_KEYS) for item in batch):
        raise NotImplementedError(
            "True packing is not implemented for audio processor inputs; use padded batching until "
            "audio position, collation, and loss parity are validated."
        )
    if any(not bool(item["attention_mask"].to(torch.bool).all()) for item in batch):
        raise ValueError("True packing requires each logical sample to be unpadded before concatenation")

    predicted_runtime_ms: float | None = None
    singleton_cause: int | None = None
    over_budget: int | None = None
    for item in batch:
        item.pop("true_packing", None)
        # attention_mask is unused on the true-packing forward: varlen attention is driven by
        # cu_seqlens, packed position_ids are built from seq_lens, and the pad tail is kept inert by
        # labels=ignore_index + token_mask=False. vlm_model pops it before the forward, so emitting a
        # per-step [1, S_pad] mask would only add a wasted H2D copy (true packing). Drop it here so
        # the variable-length per-sample masks also never reach the generic key handling below.
        item.pop("attention_mask", None)
        v = item.pop("predicted_runtime_ms", None)
        sc = item.pop("singleton_cause", None)
        ob = item.pop("over_budget", None)
        if v is not None and predicted_runtime_ms is None:
            predicted_runtime_ms = float(v)
        if sc is not None and singleton_cause is None:
            singleton_cause = int(sc)
        if ob is not None and over_budget is None:
            over_budget = int(ob)

    pad_token_id = int(batch[0]["pad_token_id"])
    ignore_index = int(batch[0]["ignore_index"])

    # Pre-pad, per-sample telemetry (identical definition to the padded path so the A/B is comparable).
    seq_lens = [int(item["input_ids"].shape[0]) for item in batch]
    content_tokens = int(sum(seq_lens))
    # Keep the pre-boundary number as a logical-data proxy. ``supervised_tokens`` below is
    # recomputed after boundary masking from the shifted labels the CE actually consumes.
    logical_supervised_tokens = int(sum(int((item["labels"][1:] != item["ignore_index"]).sum()) for item in batch))
    seq_max_len = int(max(seq_lens))
    sum_len_sq = int(sum(length**2 for length in seq_lens))

    # Concatenate (do NOT stack) the 1-D per-sample tensors into the single packed row.
    input_ids = torch.cat([item["input_ids"] for item in batch], dim=0)  # [S]
    labels = torch.cat([item["labels"] for item in batch], dim=0)  # [S]
    token_mask = torch.cat([item["token_mask"] for item in batch], dim=0)  # [S]
    seq_total = int(input_ids.shape[0])  # == content_tokens

    # Segment boundaries in the packed row. cu[i] is the start of sample i; cu[-1] == seq_total.
    cu: list[int] = [0]
    running: int = 0
    for length in seq_lens:
        running += length
        cu.append(running)
    # Boundary label masking: kill the cross-sample next-token pair at every non-first segment
    # start. cu[1:k] are the starts of samples 1..k-1 (sample 0's start is dropped by the global shift).
    for start in cu[1 : len(seq_lens)]:
        labels[start] = ignore_index

    # One FP8 pad tail for the whole row (not per sample). Its own varlen segment so it never mixes
    # with a real sample; labels=ignore_index and token_mask False keep it inert.
    seq_padded = (seq_total + FP8_PAD_MULTIPLE - 1) // FP8_PAD_MULTIPLE * FP8_PAD_MULTIPLE
    pad = seq_padded - seq_total
    if seq_padded > seq_total:
        input_ids = torch.cat([input_ids, torch.full((pad,), pad_token_id, dtype=torch.long)])
        labels = torch.cat([labels, torch.full((pad,), ignore_index, dtype=torch.long)])
        token_mask = torch.cat([token_mask, torch.full((pad,), False, dtype=torch.bool)])
        cu.append(seq_padded)

    packed_cu_seq_lens = torch.tensor(cu, dtype=torch.int32)  # [num_segments + 1]
    packed_max_length = int(max(cu[i + 1] - cu[i] for i in range(len(cu) - 1)))
    supervised_tokens = int((labels[1:] != ignore_index).sum())
    attended_token_pairs = sum_len_sq + pad**2

    regular: dict[str, Any] = {
        "input_ids": input_ids.unsqueeze(0),  # [1, S_pad]
        "labels": labels.unsqueeze(0),  # [1, S_pad]
        "token_mask": token_mask.unsqueeze(0),  # [1, S_pad]
    }

    # Visual + misc keys: concatenated across samples in sample order (identical to the padded path,
    # which already cats these). Order matches input_ids concatenation so masked_scatter / RoPE align.
    special: dict[str, Any] = {}
    # dict.fromkeys (not set) so key iteration order is deterministic across runs (reproducible result
    # assembly / snapshot tests); first-seen order is enough since values are concatenated per key.
    all_keys = list(dict.fromkeys(key for item in batch for key in item.keys()))
    for key in all_keys:
        if key in regular:
            continue
        if key in ["image_grid_thw", "video_grid_thw", "pixel_values", "pixel_values_videos", "image_sizes"]:
            special[key] = torch.cat([item[key] for item in batch if key in item], dim=0)
        elif key == "second_per_grid_ts":
            list_of_tensor = [torch.tensor(item[key]) for item in batch if key in item]
            special[key] = torch.cat(list_of_tensor, dim=0)
        elif key in ["pad_token_id", "ignore_index", "raw_image", "raw_video"]:
            # pad/ignore ids are uniform scalars (asserted); keep the seed's. raw_* are viz-only and
            # stripped before the forward -- skip the costly resize/cat the padded viz path does.
            special[key] = batch[0][key] if key in ["pad_token_id", "ignore_index"] else batch[0].get(key)
        else:
            special[key] = default_collate([item[key] for item in batch])
    # Drop any Nones we may have inserted for absent raw_* keys.
    special = {k: v for k, v in special.items() if v is not None}

    result: dict[str, Any] = {
        **regular,
        **special,
        "collated": True,
        "true_packing": True,
        "packed_cu_seq_lens": packed_cu_seq_lens,
        "packed_max_length": packed_max_length,
        "seq_lens": seq_lens,
        "content_tokens": content_tokens,
        "supervised_tokens": supervised_tokens,
        "logical_supervised_tokens": logical_supervised_tokens,
        "logical_batch_size": len(seq_lens),
        "seq_max_len": seq_max_len,
        "sum_len_sq": sum_len_sq,
        "attended_token_pairs": attended_token_pairs,
        "packed_tokens": int(seq_padded),
        "pack_efficiency": float(seq_total) / float(seq_padded),
    }
    if predicted_runtime_ms is not None:
        result["predicted_runtime_ms"] = predicted_runtime_ms
    if singleton_cause is not None:
        result["singleton_cause"] = singleton_cause
    if over_budget is not None:
        result["over_budget"] = over_budget
    return result
