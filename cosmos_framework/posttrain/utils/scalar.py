# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Reading an arbitrary value as a scalar float, safely.

Anything that consumes numbers it did not produce needs this: metric dicts returned by a role body,
values crossing a WorkerGroup edge, anything read from a config or a checkpoint. The value may be a
Python number, a 0-d torch tensor, a multi-element tensor, a bool, a string, or None — and only some
of those are a number.

WHY THIS IS SHARED. The question was answered in two places, differently, and neither module could
import the other (`utils/report.py` is control plane, `workers/replica.py` is driver-side next to
Ray), so each kept its own answer and they drifted:

- `workers/replica.py::_as_scalar` was careful — rejected bools, multi-element tensors and 0-d bool
  tensors, duck-typed `.item()`.
- `utils/report.py::_is_scalar` was a bare `float(v)` try/except. That is not a laxer version of the
  same rule, it is a bug: `float("1.5")` succeeds, so a STRING metric was logged to W&B as the
  number 1.5 — a value that was never measured, charted as if it had been.

Consolidated here so there is one rule. The callers differ in exactly one respect, `allow_bool`,
and the difference is real rather than a disagreement:

- REDUCING a value (cross-replica mean): a bool must NOT be averaged. `mean(True, False) = 0.5` is
  not a status, and `Tensor.new_tensor` coerces it straight back to `True` — a flag reported true
  when half the replicas said false.
- REPORTING a value (W&B): a bool SHOULD be logged, as 0/1. A flag over time is a useful chart and
  nothing downstream averages it.

TORCH-FREE AT IMPORT, deliberately. `workers/replica.py` runs driver-side alongside Ray and must not
pull torch into that process, so tensors are recognized by duck-typing (`.item()` / `.numel()` /
`.dtype`) rather than by `isinstance`. This also means the module works unchanged for any
tensor-like that follows the same protocol.
"""

from __future__ import annotations

from typing import Any


def as_scalar(x: Any, *, allow_bool: bool = False) -> float | None:
    """`x` as a float when it is genuinely a scalar number, else None.

    Returning None rather than raising is the point: callers are iterating over values they do not
    control (a metric dict, a gathered batch), and one unreducible entry must not abort the rest.
    Every caller decides what to do with None — skip the key, keep an upstream value, log nothing.

    What is REJECTED, and why each case is a real one that was hit:

    - **Strings, even parseable ones.** `float("1.5")` succeeds, so a bare try/except turns a string
      metric into a number nobody measured. A string is never a measurement.
    - **Bools**, unless `allow_bool`. A bool is a status, not a quantity; see the module docstring.
    - **0-d bool tensors**, likewise. `isinstance(x, bool)` does NOT catch these — a tensor is not a
      bool — which is exactly how they slipped through before.
    - **Multi-element tensors.** A value that is genuinely a vector has no single meaningful scalar,
      and silently taking element 0 or the mean would invent one.
    - **Anything whose `.item()` raises.** Reporting and reduction must never kill the caller, so an
      exotic tensor-like degrades to None rather than propagating.

    What is ACCEPTED: Python `int`/`float`, and any tensor-like exposing `.item()` with exactly one
    element (including integer tensors — promoting the result to float is the caller's business, and
    `workers/replica.py` does so to avoid truncating a fractional mean into an int dtype).

    Args:
        x: the value to interpret.
        allow_bool: treat bools and 0-d bool tensors as 1.0/0.0 rather than rejecting them. Leave
            False when the result will be AVERAGED; pass True when it will only be recorded.

    Returns:
        The value as a float, or None when `x` is not a scalar number under the rules above.
    """
    if isinstance(x, bool):
        return float(x) if allow_bool else None
    if isinstance(x, (int, float)):
        return float(x)
    # Duck-typed so this module stays torch-free (it is imported driver-side alongside Ray).
    if hasattr(x, "item"):
        try:
            if getattr(x, "numel", None) is not None and x.numel() != 1:
                return None
            # A bool TENSOR is the same case as a Python bool; the isinstance check above does not
            # catch it, because a tensor is not a bool.
            dtype = getattr(x, "dtype", None)
            if dtype is not None and getattr(dtype, "is_floating_point", None) is False and "bool" in str(dtype):
                return float(x.item()) if allow_bool else None
            return float(x.item())
        except Exception:  # noqa: BLE001 — an exotic .item() must not break the caller's loop
            return None
    return None


def as_report_scalar(x: Any) -> int | float | bool | None:
    """Detach a scalar metric without erasing its integer or boolean schema type.

    Reduction callers intentionally promote values to float via :func:`as_scalar`. Reporting and
    checkpoint journals have a different contract: ``*_count`` values finalized as integers must
    remain integers, and status flags must remain booleans. Tensor-like inputs are admitted only
    when ``item()`` returns one of those three primitive scalar types.
    """
    if type(x) is bool:
        return x
    if type(x) is int:
        return x
    if type(x) is float:
        return x
    if not hasattr(x, "item"):
        return None
    try:
        if getattr(x, "numel", None) is not None and x.numel() != 1:
            return None
        item = x.item()
    except Exception:  # noqa: BLE001 — an exotic .item() must not break reporting
        return None
    if type(item) in (bool, int, float):
        return item
    return None
