# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU plans for bounded varlen workspaces without splitting independent sequences."""


def plan_varlen_attention_chunks(
    lengths: list[int], max_padded_tokens: int
) -> tuple[tuple[int, int, int, int, int], ...]:
    """Bound sequence count times maximum length per attention call.

    Entries are (first_sequence, end_sequence, first_token, end_token, max_length).
    Preserve sequence order and boundaries, including any isolated padding tail.
    A single sequence longer than the budget stays intact.
    """
    if max_padded_tokens <= 0 or not lengths or any(length <= 0 for length in lengths):
        raise ValueError("Sequence lengths and the padded-token budget must be positive")
    chunks: list[tuple[int, int, int, int, int]] = []
    first_sequence = first_token = end_token = max_length = 0
    for sequence, length in enumerate(lengths):
        proposed_max = max(max_length, length)
        if sequence > first_sequence and (sequence - first_sequence + 1) * proposed_max > max_padded_tokens:
            chunks.append((first_sequence, sequence, first_token, end_token, max_length))
            first_sequence, first_token, max_length = sequence, end_token, 0
        end_token += length
        max_length = max(max_length, length)
    chunks.append((first_sequence, len(lengths), first_token, end_token, max_length))
    return tuple(chunks)
