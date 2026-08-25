# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Runtime SequencePack helpers used by attention and context parallel paths."""

import math
from dataclasses import dataclass
from typing import Any, List, Tuple

import torch

from cosmos_framework.utils import log

MAX_CAUSAL_LEN_IMAGE_BATCH = 0
MAX_FULL_LEN_IMAGE_BATCH = 0
MAX_CAUSAL_LEN_VIDEO_BATCH = 0
MAX_FULL_LEN_VIDEO_BATCH = 0


def get_padding_stats() -> dict[str, int]:
    """Return the current runtime sequence-packing padding stats."""
    return {
        "MAX_CAUSAL_LEN_IMAGE_BATCH": MAX_CAUSAL_LEN_IMAGE_BATCH,
        "MAX_FULL_LEN_IMAGE_BATCH": MAX_FULL_LEN_IMAGE_BATCH,
        "MAX_CAUSAL_LEN_VIDEO_BATCH": MAX_CAUSAL_LEN_VIDEO_BATCH,
        "MAX_FULL_LEN_VIDEO_BATCH": MAX_FULL_LEN_VIDEO_BATCH,
    }


SequencePack = dict[str, Any]


@dataclass(frozen=True)
class SequencePackMetadata:
    """Validated, device-specific metadata for one packed-sequence layout."""

    sample_lens: tuple[int, ...]
    split_lens: tuple[int, ...]
    attn_modes: tuple[str, ...]
    device: torch.device
    sample_offsets: torch.Tensor
    max_sample_len: int
    max_causal_len: int
    max_full_len: int
    causal_indices: torch.Tensor
    full_indices: torch.Tensor
    causal_seq_offsets: torch.Tensor
    full_only_seq_offsets: torch.Tensor
    causal_sample_ids: torch.Tensor  # [N_causal_tokens]
    full_only_sample_ids: torch.Tensor  # [N_full_tokens]
    num_causal_tokens: int
    num_full_tokens: int

    def matches_layout(
        self,
        sample_lens: list[int],
        split_lens: list[int],
        attn_modes: list[str],
        device: torch.device,
    ) -> bool:
        """Return whether this metadata describes the supplied layout."""
        return (
            self.sample_lens == tuple(sample_lens)
            and self.split_lens == tuple(split_lens)
            and self.attn_modes == tuple(attn_modes)
            and self.device == device
        )

    def as_sequence_pack_fields(self) -> dict[str, Any]:
        """Return the legacy SequencePack mapping backed by these tensors."""
        return {
            "sample_offsets": self.sample_offsets,
            "max_sample_len": self.max_sample_len,
            "max_causal_len": self.max_causal_len,
            "max_full_len": self.max_full_len,
            "_causal_indices": self.causal_indices,
            "_full_indices": self.full_indices,
            "_causal_seq_offsets": self.causal_seq_offsets,
            "_full_only_seq_offsets": self.full_only_seq_offsets,
            "_causal_sample_ids": self.causal_sample_ids,
            "_full_only_sample_ids": self.full_only_sample_ids,
            "_num_causal_tokens": self.num_causal_tokens,
            "_num_full_tokens": self.num_full_tokens,
            "split_lens": list(self.split_lens),
            "attn_modes": list(self.attn_modes),
        }


# ------------------------------------
# SequencePack: internal helpers
# ------------------------------------


def _find_non_causal_text_token_idx(
    attn_modes: List[str], split_lens: List[int], und_token_indexes: List[int]
) -> List[int]:
    """
    Find the indexes of the "und" tokens that are under the "full" mode.
    This are indices into the full_only_seq.
    """
    # Return indexes *into* full_only_seq, not into the original packed sequence.
    # The order within full_only_seq is the concatenation of each "full" split in order.
    out = []
    full_offset = 0
    packed_idx = 0
    und_token_set = set(und_token_indexes)
    for attn_mode, split_len in zip(attn_modes, split_lens):
        if attn_mode == "full":
            split_indices = range(packed_idx, packed_idx + split_len)
            # For this "full" split, find the und tokens within this split, mapped local to full_only_seq offset
            for local_idx, split_idx in enumerate(split_indices):
                if split_idx in und_token_set:
                    out.append(full_offset + local_idx)
            full_offset += split_len
        packed_idx += split_len
    return out


def _compute_mode_indices_and_offsets(
    split_lens: torch.Tensor | List[int], attn_modes: List[str], mode: str, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute indices from a joint tensor that are in the given mode.
    """
    indices = []
    offsets = [0]
    next_offset = 0
    start = 0

    if isinstance(split_lens, torch.Tensor):
        split_lens = split_lens.tolist()

    for split_len, attn_mode in zip(split_lens, attn_modes):
        if attn_mode == mode:
            indices.extend(range(start, start + split_len))
            next_offset += split_len
            offsets.append(next_offset)
        start += split_len

    return (
        torch.tensor(indices, dtype=torch.int32, device=device),
        torch.tensor(offsets, dtype=torch.int32, device=device),
    )  # [N_mode_tokens], [N_mode_splits+1]


def _get_padded_size(n: int, cp_world_size: int = 1, pad_for_cuda_graphs: bool = False, alignment: int = 1) -> int:
    """Return the length a stream of ``n`` tokens has to be padded to.

    ``alignment`` is the caller's own requirement on the padded length (e.g. the
    FlexAttention block size for the GEN stream); CUDA-graph bucketing and CP
    divisibility are folded into it so the result satisfies all three at once.
    """
    if pad_for_cuda_graphs:
        # Reduce recompilations / CUDA graph re-captures by bucketing lengths.
        # <= 2K: 128,  <= 4K: 256,  <= 8K: 512,  <= 16K: 1024,  > 16K: 2048
        if n <= 2048:
            bucket = 128
        elif n <= 4096:
            bucket = 256
        elif n <= 8192:
            bucket = 512
        elif n <= 16384:
            bucket = 1024
        else:
            bucket = 2048
        alignment = math.lcm(alignment, bucket)

    # ensure it's divisible by cp_world_size
    if cp_world_size > 1:
        alignment = math.lcm(alignment, cp_world_size)

    if alignment > 1:
        n = ((n + alignment - 1) // alignment) * alignment

    return n


# The only place padding is materialised: _get_padded_size (plus _grow_cuda_graph_bounds
# under CUDA graphs) decides the target length, this zero-fills a stream up to it.
def _pad_to_size(size: int, x: torch.Tensor, pad_value: int | float = 0) -> torch.Tensor:
    assert x.shape[0] <= size
    padded = x.new_full((size, *x.shape[1:]), pad_value)  # [size,...]
    padded[: x.shape[0]] = x
    return padded


def _pad_stream_and_sample_ids(
    sequence: torch.Tensor,  # [N,...]
    sample_ids: torch.Tensor,  # [N]
    padded_len: int,
    padding_sample_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad one token stream and its per-token sample IDs to the same length."""
    assert sequence.shape[0] == sample_ids.shape[0]
    sequence = _pad_to_size(padded_len, sequence)  # [N_padded,...]
    sample_ids = _pad_to_size(padded_len, sample_ids, pad_value=padding_sample_id)  # [N_padded]
    return sequence, sample_ids


def _append_pad_segment(offsets: torch.Tensor, padded_len: int) -> torch.Tensor:
    """Return ``offsets`` with ``padded_len`` appended, adding a final segment for the padding.

    ``offsets`` ends at the real token count, so the appended entry describes exactly the rows
    that :func:`_pad_to_size` zero-filled.
    """
    return torch.cat((offsets, offsets.new_full((1,), padded_len)))


def _pad(
    causal_seq: torch.Tensor, full_only_seq: torch.Tensor, padded_causal_len: int, padded_full_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    causal_seq = _pad_to_size(padded_causal_len, causal_seq)
    full_only_seq = _pad_to_size(padded_full_len, full_only_seq)
    return causal_seq, full_only_seq


def _ensure_core_metadata(pack: SequencePack) -> None:
    required = [
        "sample_offsets",
        "max_sample_len",
        "max_causal_len",
        "max_full_len",
        "_causal_indices",
        "_full_indices",
        "_causal_seq_offsets",
        "_full_only_seq_offsets",
        "is_sharded",
    ]
    for key in required:
        if key not in pack:
            raise KeyError(f"Missing required pack field: {key}")


def _build_sequence_pack_metadata(
    sample_lens: list[int],
    split_lens: list[int],
    attn_modes: list[str],
    device: torch.device,
) -> SequencePackMetadata:
    """Build device tensors and scalar metadata for one sequence layout."""
    _max_sample_len = max(sample_lens)
    _max_causal_len = max((split_lens[i] for i in range(len(split_lens)) if attn_modes[i] == "causal"), default=0)
    _max_full_len = max((split_lens[i] for i in range(len(split_lens)) if attn_modes[i] == "full"), default=0)

    sample_lens_cu = torch.tensor([0] + sample_lens, device=device, dtype=torch.int32)  # [N_samples+1]
    _sample_offsets = torch.cumsum(sample_lens_cu, dim=0, dtype=torch.int32)  # [N_samples+1]
    sample_lens_tensor = torch.tensor(sample_lens, device=device, dtype=torch.int64)  # [N_samples]
    sample_ids = torch.repeat_interleave(
        torch.arange(len(sample_lens), device=device, dtype=torch.int64),
        sample_lens_tensor,
        output_size=sum(sample_lens),
    )  # [N_tokens]

    _causal_indices, _causal_seq_offsets = _compute_mode_indices_and_offsets(split_lens, attn_modes, "causal", device)
    _full_indices, _full_only_seq_offsets = _compute_mode_indices_and_offsets(split_lens, attn_modes, "full", device)
    _causal_sample_ids = sample_ids[_causal_indices]  # [N_causal_tokens]
    _full_only_sample_ids = sample_ids[_full_indices]  # [N_full_tokens]

    return SequencePackMetadata(
        sample_lens=tuple(sample_lens),
        split_lens=tuple(split_lens),
        attn_modes=tuple(attn_modes),
        device=device,
        sample_offsets=_sample_offsets,
        max_sample_len=_max_sample_len,
        max_causal_len=_max_causal_len,
        max_full_len=_max_full_len,
        causal_indices=_causal_indices,
        full_indices=_full_indices,
        causal_seq_offsets=_causal_seq_offsets,
        full_only_seq_offsets=_full_only_seq_offsets,
        causal_sample_ids=_causal_sample_ids,
        full_only_sample_ids=_full_only_sample_ids,
        num_causal_tokens=len(_causal_indices),
        num_full_tokens=len(_full_indices),
    )


def prepare_sequence_pack_metadata(
    sample_lens: list[int],
    split_lens: list[int],
    attn_modes: list[str],
    packed_und_token_indexes: torch.Tensor,
    device: torch.device,
) -> SequencePackMetadata:
    """Validate and prepare reusable metadata for one packed-sequence layout."""
    non_causal_text_idxs = _find_non_causal_text_token_idx(
        attn_modes,
        split_lens,
        packed_und_token_indexes.tolist(),
    )
    assert len(non_causal_text_idxs) == 0, "non_causal_text_idxs should be empty"
    return _build_sequence_pack_metadata(sample_lens, split_lens, attn_modes, device)


# ------------------------------------
# SequencePack constructors
# ------------------------------------


def _grow_cuda_graph_bounds(need_causal: int, need_full: int, is_image_batch: bool) -> tuple[int, int]:
    """Raise the per-batch-kind high-water marks to cover ``need_*`` and return them.

    CUDA graphs are captured per shape, so every step has to pad to the largest length
    seen so far rather than to its own length: the marks only ever grow.
    """
    global MAX_CAUSAL_LEN_IMAGE_BATCH, MAX_FULL_LEN_IMAGE_BATCH, MAX_CAUSAL_LEN_VIDEO_BATCH, MAX_FULL_LEN_VIDEO_BATCH
    if is_image_batch:
        if need_causal > MAX_CAUSAL_LEN_IMAGE_BATCH:
            MAX_CAUSAL_LEN_IMAGE_BATCH = need_causal
            log.info(f"Growing MAX_CAUSAL_LEN_IMAGE_BATCH to {MAX_CAUSAL_LEN_IMAGE_BATCH}", rank0_only=False)
        if need_full > MAX_FULL_LEN_IMAGE_BATCH:
            MAX_FULL_LEN_IMAGE_BATCH = need_full
            log.info(f"Growing MAX_FULL_LEN_IMAGE_BATCH to {MAX_FULL_LEN_IMAGE_BATCH}", rank0_only=False)
        return MAX_CAUSAL_LEN_IMAGE_BATCH, MAX_FULL_LEN_IMAGE_BATCH

    if need_causal > MAX_CAUSAL_LEN_VIDEO_BATCH:
        MAX_CAUSAL_LEN_VIDEO_BATCH = need_causal
        log.info(f"Growing MAX_CAUSAL_LEN_VIDEO_BATCH to {MAX_CAUSAL_LEN_VIDEO_BATCH}", rank0_only=False)
    if need_full > MAX_FULL_LEN_VIDEO_BATCH:
        MAX_FULL_LEN_VIDEO_BATCH = need_full
        log.info(f"Growing MAX_FULL_LEN_VIDEO_BATCH to {MAX_FULL_LEN_VIDEO_BATCH}", rank0_only=False)
    return MAX_CAUSAL_LEN_VIDEO_BATCH, MAX_FULL_LEN_VIDEO_BATCH


def sequence_pack_from_packed_sequence(
    packed_sequence: torch.Tensor,
    attn_modes: List[str],
    split_lens: List[int],
    sample_lens: List[int],
    packed_und_token_indexes: torch.Tensor,
    packed_gen_token_indexes: torch.Tensor,
    is_image_batch: bool = False,
    cp_world_size: int = 1,
    pad_for_cuda_graphs: bool = False,
    full_seq_alignment: int = 1,
    causal_seq_alignment: int = 1,
    prepared_metadata: SequencePackMetadata | None = None,
) -> SequencePack:
    """
    Create a sequence pack from a packed sequence and metadata.
    NOTE: Some arguments seem redundant because they in principle support more flexible sequence setups.
          This constructor checks that the required invariants for SequencePack are satisfied.
    NOTE: This constructor checks that there are no "und" tokens under "full" mode, and no "gen" tokens under "causal" mode,
          since this is a requirement for SequencePack.
    Args:
        packed_sequence (torch.Tensor): Tensor containing all tokens in the batch of sequences.
        attn_modes (List[str]): List of attention modes. Must be alternating ["causal", "full", ... "causal", "full"]
        split_lens (List[int]): Length of each subsequence. len(split_lens) == len(attn_modes)
        sample_lens (List[int]): Length of each sequence. len(sample_lens) == number of samples.
        packed_und_token_indexes (torch.Tensor): The indexes of the understanding tokens in the packed sequence.
        packed_gen_token_indexes (torch.Tensor): The indexes of the generating tokens in the packed sequence.
        full_seq_alignment (int): Pad the full (GEN) stream to a multiple of this. FlexAttention
            requires a block-aligned GEN length; satisfying it here means the attention path never
            has to re-pad q/k/v and metadata per layer.
        causal_seq_alignment (int): Pad the causal (UND) stream to a multiple of this. The fused
            FlexAttention path keys GEN queries against ``[UND | GEN]``, so the UND stream needs the
            same block alignment as the GEN one for the boundary between them to fall on a block
            boundary.
    """
    del packed_gen_token_indexes

    if prepared_metadata is None:
        prepared_metadata = prepare_sequence_pack_metadata(
            sample_lens=sample_lens,
            split_lens=split_lens,
            attn_modes=attn_modes,
            packed_und_token_indexes=packed_und_token_indexes,
            device=packed_sequence.device,
        )
    elif not prepared_metadata.matches_layout(sample_lens, split_lens, attn_modes, packed_sequence.device):
        raise ValueError("Prepared sequence-pack metadata does not match the current packed-sequence layout")

    assert sum(sample_lens) == packed_sequence.shape[0], (
        "sum(sample_lens) must be equal to the length of the packed sequence"
    )

    meta = prepared_metadata.as_sequence_pack_fields()
    causal_seq = packed_sequence[meta["_causal_indices"]]  # [N_causal_tokens,D]
    full_only_seq = packed_sequence[meta["_full_indices"]]  # [N_full_tokens,D]
    causal_sample_ids = meta["_causal_sample_ids"]  # [N_causal_tokens]
    full_only_sample_ids = meta["_full_only_sample_ids"]  # [N_full_tokens]

    # The pad segment below pairs the two streams segment for segment, so it only applies when
    # they have the same, non-zero segment count, i.e. every sample contributes both a causal and
    # a full split. The AR no-text packs carry full splits only (see
    # test_init_sequence_pack_no_causal_splits), and keep the plain lengths and offsets.
    pad_segment_supported = (
        meta["_causal_seq_offsets"].shape[0] == meta["_full_only_seq_offsets"].shape[0]
        and meta["_causal_seq_offsets"].shape[0] > 1
    )

    # Attention sees the padding as a trailing segment of its own (see the offsets below), and
    # that segment has to be non-empty on both streams: the gen->und pass pairs GEN queries with
    # und keys, and an empty key range would turn those rows into an empty softmax. So once
    # either stream is padded, give both at least one padded row, re-rounded so the alignment
    # and CUDA-graph bucketing still hold.
    len_causal = int(causal_seq.shape[0])
    len_full = int(full_only_seq.shape[0])
    if pad_segment_supported:
        need_causal = len_causal + 1
        need_full = len_full + 1
    else:
        need_causal = len_causal
        need_full = len_full

    need_causal = _get_padded_size(need_causal, cp_world_size, pad_for_cuda_graphs, causal_seq_alignment)
    need_full = _get_padded_size(need_full, cp_world_size, pad_for_cuda_graphs, full_seq_alignment)
    if pad_for_cuda_graphs:
        need_causal, need_full = _grow_cuda_graph_bounds(need_causal, need_full, is_image_batch)

    if need_causal != len_causal or need_full != len_full:
        padding_sample_id = meta["sample_offsets"].shape[0] - 1
        causal_seq, causal_sample_ids = _pad_stream_and_sample_ids(
            causal_seq,
            causal_sample_ids,
            need_causal,
            padding_sample_id,
        )
        full_only_seq, full_only_sample_ids = _pad_stream_and_sample_ids(
            full_only_seq,
            full_only_sample_ids,
            need_full,
            padding_sample_id,
        )

    pack: SequencePack = {
        **meta,
        "max_num_tokens": sum(sample_lens),
        "causal_seq": causal_seq,
        "full_only_seq": full_only_seq,
        "_causal_sample_ids": causal_sample_ids,
        "_full_only_sample_ids": full_only_sample_ids,
        "is_sharded": False,
    }

    # Trailing padding rows belong to no sample, and varlen attention leaves rows outside its
    # cumulative ranges unwritten in both directions: the forward output rows keep whatever was
    # in the buffer, and the backward skips the matching dq/dk/dv rows, which then reach the
    # projection weight gradients with no zero factor to cancel them. So describe the padding as
    # one extra trailing segment per stream -- the offsets already end at the real token count,
    # so appending the padded length is enough. Padding then attends only to padding, every real
    # query keeps its exact range, and both streams gain the same one extra segment, which keeps
    # the query and key segment counts equal for the gen->und pass. No real sample grows, so each
    # maximum is whichever is longer, the longest real sample or the padding itself.
    pad_causal = int(causal_seq.shape[0]) - meta["_num_causal_tokens"]
    pad_full = int(full_only_seq.shape[0]) - meta["_num_full_tokens"]
    if pad_segment_supported and (pad_causal > 0 or pad_full > 0):
        assert pad_causal > 0 and pad_full > 0, (
            "Padding must land on both streams so the pad segment is non-empty on every side, "
            f"got pad_causal={pad_causal}, pad_full={pad_full}."
        )
        pack["_causal_seq_offsets_pad_segment"] = _append_pad_segment(
            meta["_causal_seq_offsets"], int(causal_seq.shape[0])
        )
        pack["max_causal_len_pad_segment"] = max(meta["max_causal_len"], pad_causal)
        pack["_full_only_seq_offsets_pad_segment"] = _append_pad_segment(
            meta["_full_only_seq_offsets"], int(full_only_seq.shape[0])
        )
        pack["max_full_len_pad_segment"] = max(meta["max_full_len"], pad_full)
        # The two-way dense full pass keys GEN queries against the interleaved stream rather than
        # against either tower, so covering its padded queries needs a pad segment on
        # ``sample_offsets`` too -- those are the offsets that describe that stream. Same shape as
        # the other two: the offsets already end at the real token count, so appending the padded
        # length is the whole of it. That padded length is both towers' padded lengths, which is
        # what :func:`get_all_seq_padded` materialises.
        #
        # Asserted rather than guarded, and asserted here rather than folded into
        # ``pad_segment_supported``, because the three consumers have to agree. Emitting the tower
        # segments without this one would leave the causal pass covered and the dense full pass
        # back on plain offsets -- the exact asymmetry the segment exists to remove, reappearing
        # silently on a layout nobody tested. Skipping all three instead would be no better now
        # that an uncovered row is known to survive into the gradients on some backends. So a
        # layout that breaks the pairing has to stop here and be looked at.
        assert meta["sample_offsets"].shape[0] == meta["_full_only_seq_offsets"].shape[0], (
            "The pad segments pair a GEN query segment with the interleaved keys of its own sample, which "
            f"needs one full split per sample: got {meta['sample_offsets'].shape[0] - 1} samples and "
            f"{meta['_full_only_seq_offsets'].shape[0] - 1} full splits. The packer emits them one to one "
            "(see SequencePlan.finish_sample); a layout that does not has to say how its GEN queries and "
            "interleaved keys line up before it can carry a pad segment."
        )
        pack["_sample_offsets_pad_segment"] = _append_pad_segment(
            meta["sample_offsets"], int(causal_seq.shape[0]) + int(full_only_seq.shape[0])
        )
        pack["max_sample_len_pad_segment"] = max(meta["max_sample_len"], pad_causal + pad_full)

    return pack


def zeros_like(orig: SequencePack, shape: Tuple[int, ...] | torch.Size | None = None) -> SequencePack:
    """
    Create a new sequence pack with the same metadata as the original, but with all tokens set to zero.
    Args:
        orig (SequencePack): The original sequence pack to copy metadata from.
        shape (Tuple[int, ...] | torch.Size | None): The shape of the new sequence pack. If None, the shape will be the same as the original.
    """
    _ensure_core_metadata(orig)
    if shape is None:
        shape_causal = orig["causal_seq"].shape
        shape_full = orig["full_only_seq"].shape
    else:
        assert len(shape) >= 1 and shape[0] == -1
        shape_causal = (orig["causal_seq"].shape[0],) + tuple(shape)[1:]
        shape_full = (orig["full_only_seq"].shape[0],) + tuple(shape)[1:]
    causal_seq = torch.zeros(
        shape_causal, device=orig["causal_seq"].device, dtype=orig["causal_seq"].dtype
    )  # [N_causal_tokens,D]
    full_only_seq = torch.zeros(
        shape_full, device=orig["full_only_seq"].device, dtype=orig["full_only_seq"].dtype
    )  # [N_full_tokens,D]
    return from_mode_splits(causal_seq, full_only_seq, orig)


def from_all_seq(packed_sequence: torch.Tensor, metadata_source: SequencePack) -> SequencePack:
    """
    Create a new sequence pack from all tokens and another sequence pack with the same metadata.
    Args:
        packed_sequence (torch.Tensor): Tensor containing all tokens in the batch of sequences.
        metadata_source (SequencePack): The metadata source to copy from.
    """
    _ensure_core_metadata(metadata_source)
    if metadata_source["is_sharded"]:
        # Use sharded sequences as is when is_sharded is True (used in Context Parallel)
        causal_seq = packed_sequence[: len(metadata_source["causal_seq"])]  # [N_causal_tokens,D]
        full_only_seq = packed_sequence[len(metadata_source["causal_seq"]) :]  # [N_full_tokens,D]
    else:
        causal_seq = packed_sequence[metadata_source["_causal_indices"]]  # [N_causal_tokens,D]
        full_only_seq = packed_sequence[metadata_source["_full_indices"]]  # [N_full_tokens,D]
        causal_seq, full_only_seq = _pad(
            causal_seq,
            full_only_seq,
            padded_causal_len=metadata_source["causal_seq"].shape[0],
            padded_full_len=metadata_source["full_only_seq"].shape[0],
        )

    return from_mode_splits(causal_seq, full_only_seq, metadata_source)


def from_mode_splits(
    causal_seq: torch.Tensor,
    full_only_seq: torch.Tensor,
    orig: SequencePack,
    is_sharded: bool | None = None,
) -> SequencePack:
    """
    Create a new sequence pack from two mode splits.
    Args:
        causal_seq (torch.Tensor): The causal sequence.
        full_only_seq (torch.Tensor): The full-only sequence.
        orig (SequencePack): The metadata source to copy from.
        is_sharded (bool | None): If True, create a local pack for context parallel.
                                  If None, inherits from orig.
    """
    _ensure_core_metadata(orig)
    if is_sharded is None:
        is_sharded = orig.get("is_sharded", False)

    out = dict(orig)
    out["causal_seq"] = causal_seq
    out["full_only_seq"] = full_only_seq
    out["is_sharded"] = is_sharded
    return out


def from_und_gen_splits(und_seq: torch.Tensor, gen_seq: torch.Tensor, orig: SequencePack) -> SequencePack:
    """
    Create a new sequence pack from two und/gen splits.
    Args:
        und_seq (torch.Tensor): The understanding sequence.
        gen_seq (torch.Tensor): The generating sequence.
        orig (SequencePack): The metadata source to copy from.
    """
    # The supported SequencePack layout maps und/gen directly to causal/full.
    return from_mode_splits(und_seq, gen_seq, orig)


# ------------------------------------
# Getters and setters for SequencePack
# ------------------------------------
def get_und_seq(pack: SequencePack) -> torch.Tensor:
    """
    Get all understanding tokens in a sequence pack in a single tensor.

    Args:
        pack (SequencePack): The sequence pack to get the understanding sequence from.
    Returns:
        torch.Tensor: All understanding tokens concatenated over all sequences in the batch.
    """
    return pack["causal_seq"]


def set_und_seq(pack: SequencePack, value: torch.Tensor) -> None:
    """
    Override the understanding tokens in a sequence pack.
    The order of tokens passed in must correspond to the order of tokens returned by get_und_seq.

    Args:
        pack (SequencePack): The sequence pack to set the understanding sequence in.
        value (torch.Tensor): The understanding sequence to set.
    """
    pack["causal_seq"] = value


def get_gen_seq(pack: SequencePack) -> torch.Tensor:
    """
    Get all generating tokens in a sequence pack in a single tensor.
    Args:
        pack (SequencePack): The sequence pack to get the generating sequence from.
    Returns:
        torch.Tensor: All generating tokens concatenated over all sequences in the batch.
    """
    return pack["full_only_seq"]


def set_gen_seq(pack: SequencePack, value: torch.Tensor) -> None:
    """
    Override the generating tokens in a sequence pack.
    The order of tokens passed in must correspond to the order of tokens returned by get_gen_seq.
    Args:
        pack (SequencePack): The sequence pack to set the generating sequence in.
        value (torch.Tensor): The generating sequence to set.
    """
    pack["full_only_seq"] = value


def get_all_seq(pack: SequencePack) -> torch.Tensor:
    """
    Get all tokens in a sequence pack in a single tensor.
    Args:
        pack (SequencePack): The sequence pack to get the all sequence from.
    Returns:
        torch.Tensor: All tokens concatenated over all sequences in the batch.
    """
    _ensure_core_metadata(pack)
    if pack["is_sharded"]:
        assert False, "get_all_seq is not supported in context parallel sharded mode"
    out = pack["causal_seq"].new_zeros(
        int(pack["_causal_indices"].shape[0] + pack["_full_indices"].shape[0]), *pack["causal_seq"].shape[1:]
    )  # [seq_len,D]
    if pack["causal_seq"].shape[0] > 0:
        out[pack["_causal_indices"]] = pack["causal_seq"][: pack["_causal_indices"].shape[0]]
    if pack["full_only_seq"].shape[0] > 0:
        out[pack["_full_indices"]] = pack["full_only_seq"][: pack["_full_indices"].shape[0]]
    return out


def get_all_seq_padded(pack: SequencePack) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Like :func:`get_all_seq`, but returns the interleaved stream and its matching cumulative
    offsets/length, preferring the pack's pad-segment variant when it has one.

    See :func:`has_pad_segment` for why the pad-segment variant exists and should be preferred.
    Unlike :func:`get_causal_seq_padded`/:func:`get_full_only_seq_padded`, the offsets/length here
    are the pack's own ``sample_offsets``/``max_sample_len`` rather than something
    ``get_all_seq`` itself returns -- they're the cumulative ranges the two-way dense full pass
    keys its GEN queries against, and the padded stream below is only correct alongside the
    matching padded ``sample_offsets``, which is why the two travel together here.

    ``get_all_seq`` returns real tokens only, which is what its other callers want: the context
    parallel gather and the OSS pipeline both take it as the model's hidden state. The two-way
    dense full pass is the exception -- it uses this stream as the keys for a *padded* query
    stream, so its keys have to reach as far as the queries do or the padded queries fall outside
    every cumulative range and the kernel leaves their rows, and the matching gradient rows,
    exactly as it found them.

    The padded length is both towers' padded lengths, so the trailing rows are the two towers'
    padding taken together, and ``_sample_offsets_pad_segment`` covers them as one extra segment.
    They stay zero: the real tokens scatter into their packed positions as before, and what is
    left is padding attending to padding, which needs no content, only a range.

    Nothing scatters into the tail, so no gradient flows back from it into either tower -- the
    index assignments below gather only the real positions.

    Args:
        pack (SequencePack): The sequence pack to get the all-sequence from.
    Returns:
        Tuple[torch.Tensor, torch.Tensor, int]: The all-tokens stream (interleaved, padded to
        ``_sample_offsets_pad_segment``'s length when the pack carries a pad segment, else
        ``get_all_seq``'s real-tokens-only variant), and the matching
        ``sample_offsets``/``max_sample_len``.
    """
    if not has_pad_segment(pack):
        return get_all_seq(pack), pack["sample_offsets"], pack["max_sample_len"]

    _ensure_core_metadata(pack)
    if pack["is_sharded"]:
        assert False, "get_all_seq_padded is not supported in context parallel sharded mode"
    out = pack["causal_seq"].new_zeros(
        int(pack["causal_seq"].shape[0] + pack["full_only_seq"].shape[0]), *pack["causal_seq"].shape[1:]
    )  # [padded_causal+padded_full,D]
    if pack["causal_seq"].shape[0] > 0:
        out[pack["_causal_indices"]] = pack["causal_seq"][: pack["_causal_indices"].shape[0]]
    if pack["full_only_seq"].shape[0] > 0:
        out[pack["_full_indices"]] = pack["full_only_seq"][: pack["_full_indices"].shape[0]]
    return out, pack["_sample_offsets_pad_segment"], pack["max_sample_len_pad_segment"]


def set_all_seq(pack: SequencePack, value: torch.Tensor) -> None:
    """
    Override the all tokens in a sequence pack.
    The order of tokens passed in must correspond to the order of tokens returned by get_all_seq.
    Args:
        pack (SequencePack): The sequence pack to set the all sequence in.
        value (torch.Tensor): The all sequence to set.
    """
    _ensure_core_metadata(pack)
    pack["causal_seq"][: pack["_causal_indices"].shape[0]] = value[pack["_causal_indices"]]
    pack["full_only_seq"][: pack["_full_indices"].shape[0]] = value[pack["_full_indices"]]


def get_causal_seq(pack: SequencePack) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get the causal sequence and its offsets in a sequence pack.
    Args:
        pack (SequencePack): The sequence pack to get the causal sequence from.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The concatenated causal sub-sequences and the starting offset for each sub-sequence.
    """
    _ensure_core_metadata(pack)
    return pack["causal_seq"], pack["_causal_seq_offsets"]


def get_full_only_seq(pack: SequencePack) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get the full-only sequence and its offsets in a sequence pack.
    Args:
        pack (SequencePack): The sequence pack to get the full-only sequence from.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The concatenated full-only sub-sequences and the starting offset for each sub-sequence.
    """
    _ensure_core_metadata(pack)
    return pack["full_only_seq"], pack["_full_only_seq_offsets"]


def has_pad_segment(pack: SequencePack) -> bool:
    """
    Whether the pack carries a trailing pad segment (padded causal/full-only offsets and lengths).

    Trailing padding rows belong to no sample, and varlen attention leaves rows outside its
    cumulative ranges unwritten in both directions: the forward output rows keep whatever was in
    the buffer, and the backward skips the matching dq/dk/dv rows, which then reach the
    projection weight gradients with no zero factor to cancel them. The pack describes its
    padding as one extra trailing segment, so a caller that holds one should always prefer it
    over the plain offsets, which make padding attend only to padding while each real query
    keeps its exact range. ``sequence_pack_from_packed_sequence`` emits the pad-segment fields as
    a set or not at all, so checking for any one of them (here, the causal offsets) answers for
    all of them.

    Args:
        pack (SequencePack): The sequence pack to check.
    Returns:
        bool: True if the pack carries a pad segment.
    """
    return "_causal_seq_offsets_pad_segment" in pack


def get_causal_seq_padded(pack: SequencePack) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Like :func:`get_causal_seq`, but the offsets/length prefer the pack's pad-segment variant.

    See :func:`has_pad_segment` for why the pad-segment offsets exist and should be preferred.

    Args:
        pack (SequencePack): The sequence pack to get the causal sequence from.
    Returns:
        Tuple[torch.Tensor, torch.Tensor, int]: The causal sub-sequences (same tensor
        ``get_causal_seq`` returns) and the offsets/max length (the pad-segment variant when the
        pack carries one, else the plain ``get_causal_seq`` offsets and ``max_causal_len``).
    """
    seq, offsets = get_causal_seq(pack)
    if has_pad_segment(pack):
        # The offsets index the whole padded stream, so they only fit a pack that holds it whole.
        assert not pack["is_sharded"], (
            "Pad-segment offsets describe the unsharded stream, so a context parallel local shard "
            "needs offsets rebased onto the shard."
        )
        offsets = pack["_causal_seq_offsets_pad_segment"]
        max_len = pack["max_causal_len_pad_segment"]
    else:
        max_len = pack["max_causal_len"]

    return seq, offsets, max_len


def get_full_only_seq_padded(pack: SequencePack) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Like :func:`get_full_only_seq`, but the offsets/length prefer the pack's pad-segment variant.

    Mirrors :func:`get_causal_seq_padded` for the full/GEN side; see :func:`has_pad_segment` for
    why the pad-segment offsets exist.

    Args:
        pack (SequencePack): The sequence pack to get the full-only sequence from.
    Returns:
        Tuple[torch.Tensor, torch.Tensor, int]: The full-only sub-sequences (same tensor
        ``get_full_only_seq`` returns) and the offsets/max length (the pad-segment variant when
        the pack carries one, else the plain ``get_full_only_seq`` offsets and ``max_full_len``).
    """
    seq, offsets = get_full_only_seq(pack)
    if has_pad_segment(pack):
        assert not pack["is_sharded"], (
            "Pad-segment offsets describe the unsharded stream, so a context parallel local shard "
            "needs offsets rebased onto the shard."
        )
        offsets = pack["_full_only_seq_offsets_pad_segment"]
        max_len = pack["max_full_len_pad_segment"]
    else:
        max_len = pack["max_full_len"]

    return seq, offsets, max_len


def num_local_real_tokens(num_real_tokens: int, rank: int, shard_len: int) -> int:
    """
    Count how many of a stream's real (non-padding) tokens land on ``rank``'s contiguous shard.

    Paired with :func:`get_num_real_tokens`: sharding uses this to record the local counts that the
    getter later reads back.

    Args:
        num_real_tokens (int): Real token count for the whole stream.
        rank (int): Context parallel rank owning the shard.
        shard_len (int): Length of each rank's shard of the stream.
    Returns:
        int: Real token count within this rank's shard.
    """
    return max(0, min(shard_len, num_real_tokens - rank * shard_len))


def get_num_real_tokens(pack: SequencePack) -> Tuple[int, int]:
    """
    Get the number of real (non-padding) und and gen tokens in the sequences this pack holds.

    ``_num_causal_tokens`` / ``_num_full_tokens`` count the whole batch, so a context-parallel local
    pack needs its own counts: it holds one contiguous shard of each stream, and since padding sits
    at the end of a stream, the last shard has fewer real tokens than its length while the earlier
    shards have none of the padding at all. Reading the batch-wide counts against a local shard
    over-counts silently, because slicing past the end of a tensor clamps instead of raising.

    Args:
        pack (SequencePack): The sequence pack to get the token counts from.
    Returns:
        Tuple[int, int]: Real und token count and real gen token count.
    """
    _ensure_core_metadata(pack)
    if pack["is_sharded"]:
        assert "_num_causal_tokens_local" in pack and "_num_full_tokens_local" in pack, (
            "A context parallel local pack must carry _num_causal_tokens_local and "
            "_num_full_tokens_local; build it with get_context_parallel_sharded_sequence."
        )
        return pack["_num_causal_tokens_local"], pack["_num_full_tokens_local"]
    return pack["_num_causal_tokens"], pack["_num_full_tokens"]


def get_device_and_dtype(pack: SequencePack) -> Tuple[torch.device, torch.dtype]:
    """
    Get the device and dtype of a sequence pack.
    Args:
        pack (SequencePack): The sequence pack to get the device and dtype from.
    Returns:
        Tuple[torch.device, torch.dtype]: The device and dtype of the sequence pack.
    """
    return pack["causal_seq"].device, pack["causal_seq"].dtype


def get_und_position_ids(position_ids: torch.Tensor, meta: dict[str, Any]) -> torch.Tensor:
    """
    Get the understanding position ids in a sequence pack.
    Args:
        position_ids (torch.Tensor): The position ids. Shape (seq_len,) for 1D RoPE
            or (3, seq_len) for 3D mRoPE.
        meta (dict[str, Any]): The metadata.
    Returns:
        torch.Tensor: The understanding position ids.
    """
    assert not meta["is_sharded"], "get_und_position_ids is not supported in context parallel sharded mode"
    if position_ids.dim() == 2:
        # 3D mRoPE: position_ids is (3, seq_len)
        return position_ids[:, meta["_causal_indices"]]  # [3,N_causal_tokens]
    return position_ids[meta["_causal_indices"]]  # [N_causal_tokens]


def get_gen_position_ids(position_ids: torch.Tensor, meta: dict[str, Any]) -> torch.Tensor:
    """
    Get the generating position ids in a sequence pack.
    Args:
        position_ids (torch.Tensor): The position ids. Shape (seq_len,) for 1D RoPE
            or (3, seq_len) for 3D mRoPE.
        meta (dict[str, Any]): The metadata.
    Returns:
        torch.Tensor: The generating position ids.
    """
    assert not meta["is_sharded"], "get_gen_position_ids is not supported in context parallel sharded mode"
    if position_ids.dim() == 2:
        # 3D mRoPE: position_ids is (3, seq_len)
        return position_ids[:, meta["_full_indices"]]  # [3,N_full_tokens]
    return position_ids[meta["_full_indices"]]  # [N_full_tokens]
