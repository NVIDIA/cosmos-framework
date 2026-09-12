# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Teacher-forced replay of the diffusion sampler, for local (single-step) error measurement.

The end-to-end image is a chaotic function of the network numerics (a perturbation flips the
sampled mode), so PSNR of the final image cannot rank quantization variants. This tool measures
the *local* error instead: every sampler step evaluates the network on the reference run's input.

Environment:
  COSMOS_TF_MODE=dump   record, per sampler step, (timestep, noise_x, cond_v, uncond_v, v_pred) of this
                        run into ``COSMOS_TF_DIR/sample<k>.pt`` (k = k-th sample generated in the process).
  COSMOS_TF_MODE=replay before each step, replace the sampler's noise_x with the recorded one from
                        ``COSMOS_TF_REF_DIR/sample<k>.pt`` (teacher forcing) and record this run's outputs
                        into ``COSMOS_TF_DIR/sample<k>.pt`` in the same format.
Unset: everything is a no-op. Compare dumps with ``~/gb300/tools/tf_local_error.py``.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable
from pathlib import Path

import torch

_SAMPLE_COUNTER = 0


def _cpu(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    return [t.detach().float().cpu() for t in tensors]


class _Session:
    def __init__(self, model, mode: str, out_dir: Path, ref_dir: Path | None, sample_idx: int) -> None:
        self.model = model
        self.mode = mode
        self.out_path = out_dir / f"sample{sample_idx}.pt"
        self.ref = None
        if mode == "replay":
            ref_path = (ref_dir or out_dir) / f"sample{sample_idx}.pt"
            self.ref = torch.load(ref_path, map_location="cpu")
        self.step = 0
        self.records: list[dict] = []
        self.last_cond: list[torch.Tensor] | None = None
        self.last_uncond: list[torch.Tensor] | None = None
        self._orig_cfg = None

    # -- capture cond/uncond velocities via the model's CFG helper
    def install(self) -> None:
        orig = self.model._run_classifier_free_guidance
        self._orig_cfg = orig
        session = self

        def cfg_wrapper(*args, **kwargs):
            cond_v, uncond_v = orig(*args, **kwargs)
            session.last_cond, session.last_uncond = _cpu(cond_v), _cpu(uncond_v)
            return cond_v, uncond_v

        self.model._run_classifier_free_guidance = cfg_wrapper

    def restore(self) -> None:
        if self._orig_cfg is not None:
            # remove the instance attribute so the class method is visible again
            try:
                del self.model._run_classifier_free_guidance
            except AttributeError:
                self.model._run_classifier_free_guidance = self._orig_cfg

    def wrap(self, velocity_fn: Callable) -> Callable:
        session = self

        def wrapped(noise_x: list[torch.Tensor], timestep: torch.Tensor):
            i = session.step
            _set_sim_step(i)
            t_val = float(timestep.flatten()[0].item())
            if session.mode == "replay":
                steps = session.ref["steps"]
                if i >= len(steps):
                    raise RuntimeError(f"teacher forcing: reference has {len(steps)} steps, sampler asked for step {i}")
                rec = steps[i]
                if abs(rec["timestep"] - t_val) > 1e-3 * max(1.0, abs(t_val)):
                    raise RuntimeError(
                        f"teacher forcing: timestep mismatch at step {i}: ref {rec['timestep']} vs {t_val}"
                    )
                noise_x = [r.to(device=x.device, dtype=x.dtype) for r, x in zip(rec["noise_x"], noise_x, strict=True)]
            session.last_cond = session.last_uncond = None
            out = velocity_fn(noise_x, timestep)
            session.records.append(
                {
                    "step": i,
                    "timestep": t_val,
                    "noise_x": _cpu(noise_x),
                    "cond_v": session.last_cond,
                    "uncond_v": session.last_uncond,
                    "v_pred": _cpu(out),
                }
            )
            session.step += 1
            return out

        return wrapped

    def save(self) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"mode": self.mode, "steps": self.records}, self.out_path)


@contextlib.contextmanager
def teacher_forcing_session(model):
    """Yield an object with ``.wrap(velocity_fn)``; a no-op passthrough unless COSMOS_TF_MODE is set."""
    global _SAMPLE_COUNTER
    mode = os.environ.get("COSMOS_TF_MODE", "").strip().lower()
    if mode not in ("dump", "replay"):
        yield _Passthrough()
        return
    out_dir = Path(os.environ["COSMOS_TF_DIR"])
    ref_dir = Path(os.environ["COSMOS_TF_REF_DIR"]) if os.environ.get("COSMOS_TF_REF_DIR") else None
    if mode == "replay" and ref_dir is None:
        raise RuntimeError("COSMOS_TF_MODE=replay needs COSMOS_TF_REF_DIR")
    session = _Session(model, mode, out_dir, ref_dir, _SAMPLE_COUNTER)
    _SAMPLE_COUNTER += 1
    session.install()
    try:
        yield session
    finally:
        session.restore()
        if session.records:
            session.save()


def _set_sim_step(step: int) -> None:
    # keeps the Q/DQ attention simulator's step counter in sync (step / layer gating, error log)
    from cosmos_framework.utils.generator.qdq_sim_edges import set_sampler_step

    set_sampler_step(step)


class _Passthrough:
    """No dump/replay; still counts sampler steps for the Q/DQ simulator."""

    def wrap(self, velocity_fn: Callable) -> Callable:
        counter = {"i": 0}

        def wrapped(noise_x, timestep):
            _set_sim_step(counter["i"])
            counter["i"] += 1
            return velocity_fn(noise_x, timestep)

        return wrapped


__all__ = ["teacher_forcing_session"]
