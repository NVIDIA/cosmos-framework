# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Triton kernels for maskless replay: row layouts, FP32 accumulation and softmax merging.

Both offsets and dynamic lengths must be int64: annotating a length as Python ``int`` narrows
it to int32, making the bounds mask false for every element of a tensor larger than 2**31.
"""

import triton
import triton.language as tl


@triton.jit
def _global_rows(
    rows: tl.tensor, indices: tl.tensor, valid: tl.tensor, row_start: tl.int64, INDEXED: tl.constexpr
) -> tl.tensor:  # [BLOCK], [N], [BLOCK] -> [BLOCK]
    if INDEXED:
        return tl.load(indices + rows, mask=valid, other=0).to(tl.int64)  # [BLOCK]
    return (rows + row_start).to(tl.int64)  # [BLOCK]


@triton.jit
def accumulate(
    source_k: tl.tensor,  # [K_span,K_ROW]
    source_v: tl.tensor,  # [K_span,V_ROW]
    target_k: tl.tensor,  # [KV,K_ROW]
    target_v: tl.tensor,  # [KV,V_ROW]
    starts: tl.tensor,  # [B]
    lengths: tl.tensor,  # [B]
    target_row: tl.int64,
    K_ROW: tl.constexpr,
    V_ROW: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
) -> None:
    # One program adds BLOCK_ROWS rows of one segment into the FP32 accumulators.
    block = tl.program_id(0)  # []
    segment = tl.program_id(1)  # []
    start = tl.load(starts + segment)  # []
    length = tl.load(lengths + segment)  # []
    if block * BLOCK_ROWS < length:
        rows = block * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)  # [BLOCK_ROWS]
        source_rows = (start + rows).to(tl.int64)[:, None]  # [BLOCK_ROWS,1]
        target_rows = (target_row + start + rows).to(tl.int64)[:, None]  # [BLOCK_ROWS,1]
        valid = (rows < length)[:, None]  # [BLOCK_ROWS,1]
        k_columns = tl.arange(0, BLOCK_K)[None, :]  # [1,BLOCK_K|BLOCK_V]
        k_mask = valid & (k_columns < K_ROW)  # [BLOCK_ROWS,BLOCK_K]
        k_target = target_rows * K_ROW + k_columns  # [BLOCK_ROWS,BLOCK_K]
        k_value = tl.load(source_k + source_rows * K_ROW + k_columns, mask=k_mask).to(
            tl.float32
        )  # [BLOCK_ROWS,BLOCK_K]
        tl.store(target_k + k_target, tl.load(target_k + k_target, mask=k_mask) + k_value, mask=k_mask)
        v_columns = tl.arange(0, BLOCK_V)[None, :]  # [1,BLOCK_K|BLOCK_V]
        v_mask = valid & (v_columns < V_ROW)  # [BLOCK_ROWS,BLOCK_V]
        v_target = target_rows * V_ROW + v_columns  # [BLOCK_ROWS,BLOCK_V]
        v_value = tl.load(source_v + source_rows * V_ROW + v_columns, mask=v_mask).to(
            tl.float32
        )  # [BLOCK_ROWS,BLOCK_V]
        tl.store(target_v + v_target, tl.load(target_v + v_target, mask=v_mask) + v_value, mask=v_mask)


@triton.jit
def query_layout(
    global_tensor: tl.tensor,  # [Q,H,D]
    part: tl.tensor,  # [N*G,H_KV,D]
    indices: tl.tensor,  # [N], unused for a contiguous part
    length: tl.int64,
    row_start: tl.int64,
    KV_HEADS: tl.constexpr,
    FOLD: tl.constexpr,
    DIM: tl.constexpr,
    INDEXED: tl.constexpr,
    ADD: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)  # [BLOCK], folded part layout
    valid = offsets < length  # [BLOCK]
    rows = offsets // (FOLD * KV_HEADS * DIM)  # [BLOCK]
    rows = _global_rows(rows, indices, valid, row_start, INDEXED)  # [BLOCK]
    head = offsets // DIM % KV_HEADS  # [BLOCK]
    group = offsets // (DIM * KV_HEADS) % FOLD  # [BLOCK]
    global_offsets = ((rows * KV_HEADS + head) * FOLD + group) * DIM + offsets % DIM  # [BLOCK]
    if ADD:
        value = tl.load(part + offsets, mask=valid, other=0).to(tl.float32)  # [BLOCK]
        current = tl.load(global_tensor + global_offsets, mask=valid, other=0)  # [BLOCK]
        # Query rows are unique within a part, including gathered passes; no atomics are needed.
        tl.store(global_tensor + global_offsets, current + value, mask=valid)
    else:
        value = tl.load(global_tensor + global_offsets, mask=valid, other=0)  # [BLOCK]
        tl.store(part + offsets, value, mask=valid)


@triton.jit
def merge_lse_update(
    source: tl.tensor,  # [H_KV,N_part*G]
    top: tl.tensor,  # [Q,H]
    total: tl.tensor,  # [Q,H]
    indices: tl.tensor,  # [N_part], unused for a contiguous part
    length: tl.int64,
    row_start: tl.int64,
    HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    INDEXED: tl.constexpr,
    SUM: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)  # [BLOCK]
    valid = offsets < length  # [BLOCK]
    rows = offsets // HEADS  # [BLOCK]
    rows = _global_rows(rows, indices, valid, row_start, INDEXED)  # [BLOCK]
    destinations = rows * HEADS + offsets % HEADS  # [BLOCK]
    fold = HEADS // KV_HEADS
    source_offsets = (
        offsets % HEADS // fold * (length // KV_HEADS) + offsets // HEADS * fold + offsets % fold
    )  # [BLOCK]
    lse = tl.load(source + source_offsets, mask=valid, other=-float("inf"))  # [BLOCK]
    maximum = tl.load(top + destinations, mask=valid, other=-float("inf"))  # [BLOCK]
    if SUM:
        safe_top = tl.where(tl.abs(maximum) < float("inf"), maximum, 0.0)  # [BLOCK]
        current = tl.load(total + destinations, mask=valid, other=0.0)  # [BLOCK]
        tl.store(total + destinations, current + tl.exp(lse - safe_top), mask=valid)
    else:
        tl.store(top + destinations, tl.maximum(maximum, lse), mask=valid)


@triton.jit
def merge_lse_finish(
    top: tl.tensor,  # [Q,H]
    total: tl.tensor,  # [Q,H]
    result: tl.tensor,  # [Q,H]
    length: tl.int64,
    BLOCK: tl.constexpr,
) -> None:
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)  # [BLOCK]
    valid = offsets < length  # [BLOCK]
    maximum = tl.load(top + offsets, mask=valid, other=0.0)  # [BLOCK]
    maximum = tl.where(tl.abs(maximum) < float("inf"), maximum, 0.0)  # [BLOCK]
    summed = tl.load(total + offsets, mask=valid, other=0.0)  # [BLOCK]
    tl.store(result + offsets, maximum + tl.log(summed), mask=valid)


@triton.jit
def merge_output_update(
    source: tl.tensor,  # [N_part*G,H_KV,Dv]
    source_lse: tl.tensor,  # [H_KV,N_part*G]
    merged_lse: tl.tensor,  # [Q,H]
    result: tl.tensor,  # [Q,H,Dv] FP32
    indices: tl.tensor,  # [N_part], unused for a contiguous part
    length: tl.int64,
    row_start: tl.int64,
    HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    DIM: tl.constexpr,
    INDEXED: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)  # [BLOCK]
    valid = offsets < length  # [BLOCK]
    rows = offsets // (HEADS * DIM)  # [BLOCK]
    rows = _global_rows(rows, indices, valid, row_start, INDEXED)  # [BLOCK]
    destinations = rows * (HEADS * DIM) + offsets % (HEADS * DIM)  # [BLOCK]
    fold = HEADS // KV_HEADS
    local_rows = offsets // (HEADS * DIM)  # [BLOCK]
    head = offsets // DIM % HEADS // fold  # [BLOCK]
    group = offsets // DIM % fold  # [BLOCK]
    lse_offsets = head * (length // (KV_HEADS * DIM)) + local_rows * fold + group  # [BLOCK]
    local_lse = tl.load(source_lse + lse_offsets, mask=valid, other=-float("inf"))  # [BLOCK]
    global_lse = tl.load(merged_lse + destinations // DIM, mask=valid, other=0.0)  # [BLOCK]
    safe_lse = tl.where(tl.abs(global_lse) < float("inf"), global_lse, 0.0)  # [BLOCK]
    weight = tl.exp(local_lse - safe_lse)  # [BLOCK]
    source_offsets = ((local_rows * fold + group) * KV_HEADS + head) * DIM + offsets % DIM  # [BLOCK]
    value = tl.load(source + source_offsets, mask=valid, other=0.0).to(tl.float32)  # [BLOCK]
    current = tl.load(result + destinations, mask=valid, other=0.0)  # [BLOCK]
    # A part's query rows are unique; sequential part launches need no atomics.
    tl.store(result + destinations, current + weight * value, mask=valid)
