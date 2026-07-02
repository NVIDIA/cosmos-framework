# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Runtime SequencePack helpers used by attention and context parallel paths."""

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

    for i, (split_len, attn_mode) in enumerate(zip(split_lens, attn_modes)):
        if attn_mode == mode:
            indices.extend(range(start, start + split_len))
            next_offset += split_len
            offsets.append(next_offset)
        start += split_len
    return torch.tensor(indices, dtype=torch.int32, device=device), torch.tensor(  # [N_mode_tokens], [N_mode_splits+1]
        offsets, dtype=torch.int32, device=device
    )


# Pad causal_seq and full_only_seq to have length 2048 if not already at that size
def _pad_to_N(N, x: torch.Tensor) -> torch.Tensor:
    assert x.shape[0] <= N
    padded = x.new_zeros((N, *x.shape[1:]))
    padded[: x.shape[0]] = x
    return padded


def _round_up_to_N(n: int, cp_world_size: int = 1, pad_for_cuda_graphs: bool = False) -> int:
    if pad_for_cuda_graphs:
        # Reduce recompilations / CUDA graph re-captures by bucketing lengths.
        # <= 2K: 128,  <= 4K: 256,  <= 8K: 512,  <= 16K: 1024,  > 16K: 2048
        if n <= 2048:
            alignment = 128
        elif n <= 4096:
            alignment = 256
        elif n <= 8192:
            alignment = 512
        elif n <= 16384:
            alignment = 1024
        else:
            alignment = 2048
        n = ((n + alignment - 1) // alignment) * alignment

    # ensure it's divisible by cp_world_size
    if cp_world_size > 1:
        remainder = n % cp_world_size
        if remainder != 0:
            n += cp_world_size - remainder

    return n


def _pad(
    causal_seq: torch.Tensor, full_only_seq: torch.Tensor, max_causal_len: int, max_full_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    causal_seq = _pad_to_N(max_causal_len, causal_seq)
    full_only_seq = _pad_to_N(max_full_len, full_only_seq)
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


def init_sequence_pack(
    sample_lens: List[int],
    split_lens: List[int],
    attn_modes: List[str],
    device: torch.device,
) -> dict[str, Any]:
    _max_sample_len = max(sample_lens)
    _max_causal_len = max((split_lens[i] for i in range(len(split_lens)) if attn_modes[i] == "causal"), default=0)
    _max_full_len = max((split_lens[i] for i in range(len(split_lens)) if attn_modes[i] == "full"), default=0)

    sample_lens_cu = torch.tensor([0] + sample_lens, device=device, dtype=torch.int32)  # [N_samples+1]
    _sample_offsets = torch.cumsum(sample_lens_cu, dim=0, dtype=torch.int32)  # [N_samples+1]

    _causal_indices, _causal_seq_offsets = _compute_mode_indices_and_offsets(split_lens, attn_modes, "causal", device)
    _full_indices, _full_only_seq_offsets = _compute_mode_indices_and_offsets(split_lens, attn_modes, "full", device)

    return dict(
        sample_offsets=_sample_offsets,
        max_sample_len=_max_sample_len,
        max_causal_len=_max_causal_len,
        max_full_len=_max_full_len,
        _causal_indices=_causal_indices,
        _full_indices=_full_indices,
        _causal_seq_offsets=_causal_seq_offsets,
        _full_only_seq_offsets=_full_only_seq_offsets,
        _num_causal_tokens=len(_causal_indices),
        _num_full_tokens=len(_full_indices),
        split_lens=split_lens,
        attn_modes=attn_modes,
    )


# ------------------------------------
# SequencePack constructors
# ------------------------------------


def _round_up_for_cuda_graphs_or_cp(
    causal_seq: torch.Tensor,
    full_only_seq: torch.Tensor,
    need_causal: int,
    need_full: int,
    is_image_batch: bool,
    pad_for_cuda_graphs: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad causal/full sequences to the required lengths, growing global bounds for CUDA graphs."""
    if pad_for_cuda_graphs:
        global \
            MAX_CAUSAL_LEN_IMAGE_BATCH, \
            MAX_FULL_LEN_IMAGE_BATCH, \
            MAX_CAUSAL_LEN_VIDEO_BATCH, \
            MAX_FULL_LEN_VIDEO_BATCH
        if is_image_batch:
            if need_causal > MAX_CAUSAL_LEN_IMAGE_BATCH:
                MAX_CAUSAL_LEN_IMAGE_BATCH = need_causal
                log.info(f"Growing MAX_CAUSAL_LEN_IMAGE_BATCH to {MAX_CAUSAL_LEN_IMAGE_BATCH}", rank0_only=False)
            if need_full > MAX_FULL_LEN_IMAGE_BATCH:
                MAX_FULL_LEN_IMAGE_BATCH = need_full
                log.info(f"Growing MAX_FULL_LEN_IMAGE_BATCH to {MAX_FULL_LEN_IMAGE_BATCH}", rank0_only=False)
            causal_seq, full_only_seq = _pad(
                causal_seq,
                full_only_seq,
                max_causal_len=MAX_CAUSAL_LEN_IMAGE_BATCH,
                max_full_len=MAX_FULL_LEN_IMAGE_BATCH,
            )
        else:
            if need_causal > MAX_CAUSAL_LEN_VIDEO_BATCH:
                MAX_CAUSAL_LEN_VIDEO_BATCH = need_causal
                log.info(f"Growing MAX_CAUSAL_LEN_VIDEO_BATCH to {MAX_CAUSAL_LEN_VIDEO_BATCH}", rank0_only=False)
            if need_full > MAX_FULL_LEN_VIDEO_BATCH:
                MAX_FULL_LEN_VIDEO_BATCH = need_full
                log.info(f"Growing MAX_FULL_LEN_VIDEO_BATCH to {MAX_FULL_LEN_VIDEO_BATCH}", rank0_only=False)
            causal_seq, full_only_seq = _pad(
                causal_seq,
                full_only_seq,
                max_causal_len=MAX_CAUSAL_LEN_VIDEO_BATCH,
                max_full_len=MAX_FULL_LEN_VIDEO_BATCH,
            )
    elif need_causal != int(causal_seq.shape[0]) or need_full != int(full_only_seq.shape[0]):
        causal_seq, full_only_seq = _pad(causal_seq, full_only_seq, need_causal, need_full)
    return causal_seq, full_only_seq


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
    """
    del packed_gen_token_indexes

    non_causal_text_idxs = _find_non_causal_text_token_idx(attn_modes, split_lens, packed_und_token_indexes.tolist())
    assert len(non_causal_text_idxs) == 0, "non_causal_text_idxs should be empty"

    assert sum(sample_lens) == packed_sequence.shape[0], (
        "sum(sample_lens) must be equal to the length of the packed sequence"
    )

    meta = init_sequence_pack(sample_lens, split_lens, attn_modes, packed_sequence.device)
    causal_seq = packed_sequence[meta["_causal_indices"]]  # [N_causal_tokens,D]
    full_only_seq = packed_sequence[meta["_full_indices"]]  # [N_full_tokens,D]

    need_causal = _round_up_to_N(int(causal_seq.shape[0]), cp_world_size, pad_for_cuda_graphs)
    need_full = _round_up_to_N(int(full_only_seq.shape[0]), cp_world_size, pad_for_cuda_graphs)

    causal_seq, full_only_seq = _round_up_for_cuda_graphs_or_cp(
        causal_seq,
        full_only_seq,
        need_causal,
        need_full,
        is_image_batch,
        pad_for_cuda_graphs,
    )

    pack: SequencePack = {
        **meta,
        "max_num_tokens": sum(sample_lens),
        "causal_seq": causal_seq,
        "full_only_seq": full_only_seq,
        "is_sharded": False,
    }
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
            max_causal_len=metadata_source["causal_seq"].shape[0],
            max_full_len=metadata_source["full_only_seq"].shape[0],
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
    if "all_seq" in pack:
        return pack["all_seq"]
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
