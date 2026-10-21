# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Comparison instruments shared by the two GR00T parity gates (`tests/gr00t_parity_cli.py`,
`tests/gr00t_train_parity_cli.py`): a difference report that measures without judging, the tolerance rule
(`evaluate`: a row absent from `TOLERANCES` must be bitwise equal), and capture of the random draws each
side actually made."""

from __future__ import annotations

import contextlib
import dataclasses
from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np
import torch


@dataclasses.dataclass
class Diff:
    """One tensor comparison, reported whether or not it passes."""

    name: str
    shape: tuple[int, ...]
    max_abs: float
    max_rel: float
    exact: bool

    def as_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "max_abs": self.max_abs,
            "max_rel": self.max_rel,
            "bitwise_equal": self.exact,
        }


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        # float32 so bf16 pairs are compared in a type that can represent their difference.
        return (
            value.detach().to(torch.float32).cpu().numpy()
            if value.is_floating_point()
            else value.detach().cpu().numpy()
        )
    return np.asarray(value)


def compare(name: str, a: Any, b: Any) -> Diff:
    """Report the difference between two tensors without deciding whether it is acceptable."""
    x, y = _to_numpy(a), _to_numpy(b)
    if x.shape != y.shape:
        raise ValueError(f"{name}: shape mismatch {x.shape} vs {y.shape} -- not a tolerance question")
    exact = bool(np.array_equal(x, y))
    xf, yf = x.astype(np.float64), y.astype(np.float64)
    abs_diff = np.abs(xf - yf)
    scale = np.maximum(np.abs(xf), np.abs(yf))
    # Relative difference only where there is something to be relative to; 0 vs 0 is not an error.
    rel = np.where(scale > 0, abs_diff / np.maximum(scale, 1e-12), 0.0)
    return Diff(name, tuple(x.shape), float(abs_diff.max(initial=0.0)), float(rel.max(initial=0.0)), exact)


def flatten_tensors(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten a nested mapping / `BatchFeature` into `dot.separated` key -> tensor."""
    if isinstance(obj, (torch.Tensor, np.ndarray)):
        return {prefix: obj} if prefix else {}
    items: Any = None
    if isinstance(obj, Mapping):
        items = obj.items()
    elif isinstance(getattr(obj, "data", None), Mapping):
        items = obj.data.items()
    if items is None:
        return {}
    out: dict[str, Any] = {}
    for key, value in items:
        out.update(flatten_tensors(value, f"{prefix}.{key}" if prefix else str(key)))
    return out


def render_diff_table(diffs: list[dict[str, Any]]) -> list[str]:
    """Markdown table rows for a list of `Diff.as_json()` records."""
    lines = ["| tensor | shape | max abs | max rel | bitwise equal |", "|---|---|---|---|---|"]
    for d in diffs:
        lines.append(
            f"| `{d['name']}` | {tuple(d['shape'])} | {d['max_abs']:.3e} | {d['max_rel']:.3e} | {d['bitwise_equal']} |"
        )
    return lines


def evaluate(
    diffs: list[dict[str, Any]],
    tolerances: Mapping[str, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split reported diffs into `(over-tolerance, not-bitwise)` failures. A name absent from `tolerances`
    is required to be bitwise equal; tolerances are a per-row exception list, never a global slack."""
    failed = [d for d in diffs if d["name"] in tolerances and d["max_abs"] > tolerances[d["name"]]]
    exact_failed = [d for d in diffs if d["name"] not in tolerances and not d["bitwise_equal"]]
    return failed, exact_failed


@contextlib.contextmanager
def _patched(owner: Any, name: str, replacement: Any) -> Iterator[None]:
    """Swap `owner.name` for the block and restore the original binding -- deleting rather than
    re-assigning when the attribute was inherited (`Beta.sample`), so the class `__dict__` is unchanged."""
    own = name in vars(owner)
    original = getattr(owner, name)
    setattr(owner, name, replacement)
    try:
        yield
    finally:
        if own:
            setattr(owner, name, original)
        else:
            delattr(owner, name)


@dataclasses.dataclass
class DrawLog:
    """Every stochastic draw one forward made, plus the dropout call count.

    The GR00T action head's random sources, in order (`gr00t_n1d7.py:220-278`): `vl_self_attention`
    dropout -> `torch.rand(B)` state dropout -> `torch.randn(actions.shape)` noise -> a CPU-generator
    `Beta(1.5, 1.0).sample([B])` time -> DiT dropout. Dropout masks are not capturable as values, so
    they are pinned by count and their equality is proven by the loss being bitwise equal.
    """

    randn: list[torch.Tensor] = dataclasses.field(default_factory=list)
    rand: list[torch.Tensor] = dataclasses.field(default_factory=list)
    beta: list[torch.Tensor] = dataclasses.field(default_factory=list)
    dropout_calls: int = 0


@contextlib.contextmanager
def capture_draws() -> Iterator[DrawLog]:
    """Record `torch.randn` / `torch.rand` / `Beta.sample` draws and count `F.dropout` calls inside the
    block, then restore the patched symbols. Isaac-GR00T calls all four through their module or class."""
    import torch.nn.functional as functional
    from torch.distributions import Beta

    log = DrawLog()
    real_randn, real_rand, real_sample, real_dropout = torch.randn, torch.rand, Beta.sample, functional.dropout

    def _recording_randn(*args: Any, **kwargs: Any) -> torch.Tensor:
        out = real_randn(*args, **kwargs)
        log.randn.append(out.detach().clone())
        return out

    def _recording_rand(*args: Any, **kwargs: Any) -> torch.Tensor:
        out = real_rand(*args, **kwargs)
        log.rand.append(out.detach().clone())
        return out

    def _recording_sample(self: Any, sample_shape: Any = torch.Size()) -> torch.Tensor:
        out = real_sample(self, sample_shape)
        log.beta.append(out.detach().clone())
        return out

    def _counting_dropout(*args: Any, **kwargs: Any) -> torch.Tensor:
        log.dropout_calls += 1
        return real_dropout(*args, **kwargs)

    with contextlib.ExitStack() as stack:
        stack.enter_context(_patched(torch, "randn", _recording_randn))
        stack.enter_context(_patched(torch, "rand", _recording_rand))
        stack.enter_context(_patched(Beta, "sample", _recording_sample))
        stack.enter_context(_patched(functional, "dropout", _counting_dropout))
        yield log


@contextlib.contextmanager
def capture_initial_noise(store: list[torch.Tensor]) -> Iterator[list[torch.Tensor]]:
    """Stage 1's form: append every `[B, H, D]` `torch.randn` draw made inside the block to `store`."""
    real_randn = torch.randn

    def _recording_randn(*args: Any, **kwargs: Any) -> torch.Tensor:
        out = real_randn(*args, **kwargs)
        if out.ndim == 3:
            store.append(out.detach().clone())
        return out

    with _patched(torch, "randn", _recording_randn):
        yield store
