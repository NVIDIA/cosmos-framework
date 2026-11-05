# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Six-layer training parity: our GR00T trainer vs the REAL official `Gr00tTrainer.train()`.

  T0 construction  parameters, freezing, dtypes, tf32, optimizer groups
  T1 data          our collated batch vs the reference's, after a codec round trip
  T2 backbone      backbone_features, cloned before the head overwrites them in place
  T3 loss          per-step loss, plus the random draws each side actually made
  T4 gradient      clipped action_head gradients and the reported grad-norm
  T5 K steps       parameters and AdamW moments after every step

A row absent from `TOLERANCES` must be bitwise equal. `--self-consistency` runs the reference against itself
(expected: every row 0) and is what makes T4/T5 readable; `--perturb` / `--share-action-head` are negative
controls that must FAIL. Single process, ONE visible GPU: at `n_gpu > 1` HF silently wraps `nn.DataParallel`
and takes `loss.mean()`. Results and the determinism lever: `operator GR00T training documentation`.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
import sys
from typing import Any

import torch

from cosmos_framework.posttrain.tests.helpers.parity_diff import (
    Diff,
    capture_draws,
    compare,
    evaluate,
    flatten_tensors,
    render_diff_table,
)
from cosmos_framework.posttrain.utils.seed import derive_u64_seeds

# Empty by measurement (2026-09-05, GB200, B=2 K=4, seeds 1234 and 4321): every row is bitwise equal, so
# every row is required to stay exact. A row turning non-exact is a finding, never a number to absorb.
TOLERANCES: dict[str, float] = {}

# Isaac's own lever for a deterministic DiT attention backward. Without it the official trainer cannot
# reproduce its own gradients, so no comparison against it is readable; `--self-consistency` measures this.
_DETERMINISTIC_SDPA = "math"

# `vl_self_attention` (4 layers) + DiT (32 layers), 4 `F.dropout` calls each. Dropout masks cannot be captured,
# so the count pins the module structure and train/eval state; mask equality is proven by the loss being bitwise.
EXPECTED_DROPOUT_CALLS_PER_FORWARD = 144

_SEED_DOMAIN = b"posttrain.gr00t.train_parity.v1\0"

# The collated tensors T1 compares; a difference in any of them means the two sides fed different inputs.
_T1_KEYS = (
    "input_ids",
    "attention_mask",
    "pixel_values",
    "image_grid_thw",
    "state",
    "embodiment_id",
    "action",
    "action_mask",
)


class _StepSeeder:
    """Reseed the global RNG in a pre-forward hook on the action head: the last point common to both sides
    before the first draw (only `set_frozen_modules_to_eval_mode` and `vlln` run in between, neither stochastic)."""

    def __init__(self, action_head: Any, root_seed: int) -> None:
        self.domain = _SEED_DOMAIN + int(root_seed).to_bytes(8, "big")
        self.calls = 0
        self._handle = action_head.register_forward_pre_hook(self._hook)

    def _hook(self, _module: Any, _args: Any) -> None:
        torch.manual_seed(derive_u64_seeds([self.calls], domain=self.domain)[0])
        self.calls += 1

    def remove(self) -> None:
        self._handle.remove()


class _BackboneCapture:
    """Record the backbone's raw features. `clone()` because `process_backbone_output` overwrites
    `backbone_output["backbone_features"]` in place (`gr00t_n1d7.py:174-179`)."""

    def __init__(self, backbone: Any) -> None:
        self.features: list[torch.Tensor] = []
        self._handle = backbone.register_forward_hook(self._hook)

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> None:
        self.features.append(output["backbone_features"].detach().clone())

    def remove(self) -> None:
        self._handle.remove()


class _CapturingCollator:
    """Wrap a collator, keeping a detached copy of every batch and the features it came from."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.batches: list[dict] = []
        self.features: list[list[dict]] = []

    def __call__(self, features: list[dict]) -> Any:
        collated = self.inner(features)
        self.features.append(features)
        self.batches.append(
            {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in collated["inputs"].items()}
        )
        return collated


class _MemoryIterableDataset(torch.utils.data.IterableDataset):
    """The K x B pre-processed samples in a fixed order; iterable so no sampler shuffles them."""

    def __init__(self, samples: list[dict]) -> None:
        self.samples = samples

    def __iter__(self):
        return iter(self.samples)


def _snapshot(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().float().cpu().clone() for key, value in state.items()}


def _adam_moments(optimizer: Any, named: dict[int, str]) -> dict[str, torch.Tensor]:
    """AdamW moments keyed by parameter NAME, so T5 compares moments rather than group orderings."""
    out: dict[str, torch.Tensor] = {}
    for group in optimizer.param_groups:
        for param in group["params"]:
            name, state = named[id(param)], optimizer.state[param]
            out[f"{name}.exp_avg"] = state["exp_avg"].detach().float().cpu().clone()
            out[f"{name}.exp_avg_sq"] = state["exp_avg_sq"].detach().float().cpu().clone()
            step = state["step"]
            out[f"{name}.step"] = (
                step.detach().float().cpu().clone() if torch.is_tensor(step) else torch.tensor(float(step))
            )
    return out


def _grads(net: Any) -> dict[str, torch.Tensor]:
    return {
        name: param.grad.detach().float().cpu().clone()
        for name, param in net.named_parameters()
        if param.grad is not None
    }


def _strip_prefix(state: dict[str, Any], prefix: str = "action_head.") -> dict[str, Any]:
    """Our `net` names carry `action_head.`; the reference's action-head state dict does not."""
    return {key.removeprefix(prefix): value for key, value in state.items()}


# ---------------------------------------------------------------------------
# The two sides
# ---------------------------------------------------------------------------


class _ReferenceRun:
    """The official `Gr00tTrainer.train()` -- the REAL trainer, not a loop over HF pieces -- instrumented at
    the six seams. `TrainingArguments` takes the official `TrainingConfig` values; the deviations (constant LR,
    `max_steps=K`, batch B, no workers, logging every step, no save/eval) isolate the update rule."""

    def __init__(
        self,
        *,
        model: Any,
        samples: list[dict],
        collator: Any,
        steps: int,
        batch: int,
        seed: int,
        output_dir: str,
    ) -> None:
        from gr00t.configs.training.training_config import TrainingConfig
        from gr00t.experiment.trainer import Gr00tTrainer
        from transformers import TrainerCallback, TrainingArguments

        official = TrainingConfig()
        self.official_config = official
        self.model = model
        self.collator = _CapturingCollator(collator)

        args = TrainingArguments(
            output_dir=output_dir,
            max_steps=steps,
            per_device_train_batch_size=batch,
            gradient_accumulation_steps=1,
            learning_rate=official.learning_rate,
            weight_decay=official.weight_decay,
            max_grad_norm=official.max_grad_norm,
            bf16=official.bf16,
            fp16=official.fp16,
            tf32=official.tf32,
            optim="adamw_torch",  # launch_finetune forces this over TrainingConfig's fused default
            seed=42,  # gr00t's DataConfig().seed
            remove_unused_columns=False,  # per-sample dicts would otherwise be filtered to nothing
            lr_scheduler_type="constant",
            warmup_ratio=0.0,
            warmup_steps=0,
            logging_steps=1,  # grad_norm is only recorded per step at 1
            dataloader_num_workers=0,
            save_strategy="no",
            eval_strategy="no",
            report_to="none",
        )

        outer = self

        class _Probe(TrainerCallback):
            def on_pre_optimizer_step(self, args, state, control, **kwargs):
                # After clipping, before the step -- the only place the clipped gradient exists.
                outer.grads.append(_grads(outer.model.action_head))
                outer.losses.append(float(outer.trainer.loss.detach()))

            def on_step_end(self, args, state, control, **kwargs):
                outer.params.append(_snapshot(outer.model.action_head.state_dict()))
                outer.moments.append(_adam_moments(outer.trainer.optimizer, outer.param_names))

            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs and "grad_norm" in logs:
                    outer.grad_norms.append(float(logs["grad_norm"]))

        self.grads: list[dict[str, torch.Tensor]] = []
        self.losses: list[float] = []
        self.params: list[dict[str, torch.Tensor]] = []
        self.moments: list[dict[str, torch.Tensor]] = []
        self.grad_norms: list[float] = []
        self.param_names = {id(p): n for n, p in model.action_head.named_parameters()}

        self.trainer = Gr00tTrainer(
            model=model,
            args=args,
            train_dataset=_MemoryIterableDataset(samples),
            data_collator=self.collator,
            callbacks=[_Probe()],
        )
        if self.trainer.args.n_gpu != 1:
            raise RuntimeError(
                f"the reference trainer sees n_gpu={self.trainer.args.n_gpu}; at >1 HF silently wraps "
                "nn.DataParallel and takes loss.mean(). Export CUDA_VISIBLE_DEVICES with exactly one device."
            )
        self.backbone = _BackboneCapture(model.backbone)
        self.seeder = _StepSeeder(model.action_head, seed)

    def run(self) -> None:
        with capture_draws() as log:
            self.trainer.train()
        self.draws = log


class _PosttrainRun:
    """`Gr00tN1d7TrainPolicy` + flow-BC objective + `LLMTrainer`, instrumented identically."""

    def __init__(self, *, policy: Any, config: Any, seed: int) -> None:
        from cosmos_framework.posttrain.config.build import build_optimizer
        from cosmos_framework.posttrain.configs.gr00t_n1d7_droid import make_objective, make_optimizer_config
        from cosmos_framework.posttrain.roles.trainer.llm import LLMTrainer

        self.policy = policy
        self.config = config
        self.optimizer_config = make_optimizer_config()
        self.optimizer = build_optimizer(policy, self.optimizer_config)
        self.trainer = LLMTrainer(
            policy,
            make_objective(config, policy.gr00t.collator),
            self.optimizer,
            grad_accum=1,
            n_mini=1,
            max_grad_norm=self.optimizer_config.grad_clip_norm,
        )
        self.param_names = {id(p): n for n, p in policy.net.named_parameters()}
        self.grads: list[dict[str, torch.Tensor]] = []
        self.losses: list[float] = []
        self.params: list[dict[str, torch.Tensor]] = []
        self.moments: list[dict[str, torch.Tensor]] = []
        self.grad_norms: list[float] = []
        self.batches: list[dict] = []
        self.backbone = _BackboneCapture(policy.backbone)
        self.seeder = _StepSeeder(policy.gr00t.action_head, seed)

    def run(self, step_batches: list[list]) -> None:
        from cosmos_framework.posttrain.model.gr00t.packer import collate_transitions

        with capture_draws() as log:
            for trajectories in step_batches:
                # Capture what we feed, so T1 compares the batch the model actually saw.
                inputs = collate_transitions(
                    trajectories,
                    self.policy.gr00t.collator,
                    action_dim=self.config.action_dim,
                    training_prefix=self.config.training_prefix,
                )
                self.batches.append({k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in inputs.items()})
                metrics = self.trainer.train(trajectories)
                # `zero_grad` happens at the START of the next attempt, so the clipped gradient is still live
                # here -- the same instant the reference's on_pre_optimizer_step sees.
                self.grads.append(_grads(self.policy.net))
                self.losses.append(float(metrics["train/loss_avg"]))
                self.grad_norms.append(float(metrics["train/grad_norm"]))
                self.params.append(_snapshot(self.policy.net.state_dict()))
                self.moments.append(_adam_moments(self.optimizer, self.param_names))
        self.draws = log


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------


def layer_t0(
    reference: _ReferenceRun,
    ours: _PosttrainRun,
    config: Any,
    initial_action_head: dict[str, dict[str, torch.Tensor]],
) -> tuple[list[Diff], list[str]]:
    """Construction: two independently loaded, identically shaped models. `initial_action_head` was snapshotted
    BEFORE either side ran; structural properties read the live objects."""
    diffs: list[Diff] = []
    notes: list[str] = []

    ref_initial = initial_action_head["reference"]
    our_initial = _strip_prefix(initial_action_head["ours"])
    ref_params = dict(reference.model.action_head.named_parameters())
    our_params = _strip_prefix(dict(ours.policy.net.named_parameters()))

    if set(ref_initial) != set(our_initial):
        notes.append(f"T0 action-head key sets differ: {sorted(set(ref_initial) ^ set(our_initial))[:5]}")
    else:
        diffs += [compare(f"t0.param.{key}", ref_initial[key], our_initial[key]) for key in sorted(ref_initial)]

    # Two models, not one compared with itself: sharing a module would make every layer bitwise for free.
    shared = {id(p) for p in ref_params.values()} & {id(p) for p in our_params.values()}
    if shared:
        notes.append(f"T0 CRITICAL: {len(shared)} action-head parameters are the SAME OBJECT on both sides")

    ref_grad = {k for k, v in ref_params.items() if v.requires_grad}
    our_grad = {k for k, v in our_params.items() if v.requires_grad}
    if ref_grad != our_grad:
        notes.append(f"T0 requires_grad sets differ ({len(ref_grad)} vs {len(our_grad)} trainable)")

    for label, model in (("reference", reference.model), ("ours", ours.policy.gr00t)):
        # The frozen COMPUTE submodules, not the wrapper flag: HF leaves the wrapper in train mode and
        # `Qwen3Backbone.forward` eval()s language_model/visual from inside.
        for part in ("language_model", "visual"):
            if getattr(model.backbone.model, part).training:
                notes.append(f"T0 {label}: backbone.{part} is in TRAIN mode; the frozen-module switch did not run")
        if any(p.requires_grad for p in model.backbone.parameters()):
            notes.append(f"T0 {label}: backbone has trainable parameters")
        non_fp32 = [n for n, p in model.named_parameters() if p.is_floating_point() and p.dtype is not torch.float32]
        if non_fp32:
            notes.append(f"T0 {label}: non-fp32 parameters present, e.g. {non_fp32[:3]}")

    if not (torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32):
        notes.append("T0: tf32 is not fully enabled; the official recipe sets both switches via tf32=True")

    ref_groups = [{reference.param_names[id(p)] for p in g["params"]} for g in reference.trainer.optimizer.param_groups]
    our_groups = [
        {ours.param_names[id(p)].removeprefix("action_head.") for p in g["params"]} for g in ours.optimizer.param_groups
    ]
    if ref_groups != our_groups:
        notes.append(f"T0 optimizer groups differ: {len(ref_groups)} vs {len(our_groups)} groups or their membership")

    official = reference.official_config
    for field, ours_value, ref_value in (
        ("lr", ours.optimizer_config.lr, official.learning_rate),
        ("weight_decay", ours.optimizer_config.weight_decay, official.weight_decay),
        ("grad_clip_norm", ours.optimizer_config.grad_clip_norm, official.max_grad_norm),
    ):
        if float(ours_value) != float(ref_value):
            notes.append(f"T0 recipe OptimizerConfig.{field}={ours_value} != official {ref_value}")
    if config.model_action_horizon != int(reference.model.config.action_horizon):
        notes.append("T0 recipe model_action_horizon disagrees with the checkpoint")
    return diffs, notes


def layer_t1(reference: _ReferenceRun, ours: _PosttrainRun) -> tuple[list[Diff], list[str]]:
    """Data: our assembled batch (codec round trip, batch assembly, derived mask) vs the reference's, per step.
    Both sides run the same collator class, so the collator itself is not under test here."""
    diffs: list[Diff] = []
    notes: list[str] = []
    if len(reference.collator.batches) != len(ours.batches):
        notes.append(f"T1 step counts differ: reference {len(reference.collator.batches)}, ours {len(ours.batches)}")
    for step, (ref_batch, our_batch) in enumerate(zip(reference.collator.batches, ours.batches)):
        ref_flat, our_flat = flatten_tensors(ref_batch), flatten_tensors(our_batch)
        for key in _T1_KEYS:
            if key not in ref_flat or key not in our_flat:
                notes.append(f"T1 step {step}: {key!r} missing (reference={key in ref_flat}, ours={key in our_flat})")
            else:
                diffs.append(compare(f"t1.step{step}.{key}", ref_flat[key], our_flat[key]))
    return diffs, notes


def layer_t2(reference: _ReferenceRun, ours: _PosttrainRun) -> tuple[list[Diff], list[str]]:
    """Backbone: raw `backbone_features` per forward."""
    notes: list[str] = []
    if len(reference.backbone.features) != len(ours.backbone.features):
        notes.append(f"T2 forward counts differ: {len(reference.backbone.features)} vs {len(ours.backbone.features)}")
    diffs = [
        compare(f"t2.step{step}.backbone_features", left, right)
        for step, (left, right) in enumerate(zip(reference.backbone.features, ours.backbone.features))
    ]
    return diffs, notes


def layer_t3(reference: _ReferenceRun, ours: _PosttrainRun, steps: int) -> tuple[list[Diff], list[str]]:
    """Loss, and the random draws that produced it."""
    diffs = [compare("t3.loss_sequence", torch.tensor(reference.losses), torch.tensor(ours.losses))]
    notes: list[str] = []
    for label in ("randn", "rand", "beta"):
        left, right = getattr(reference.draws, label), getattr(ours.draws, label)
        if len(left) != len(right):
            notes.append(f"T3 {label} draw COUNT differs: reference={len(left)}, ours={len(right)}")
        else:
            diffs += [compare(f"t3.draw.{label}.{index}", a, b) for index, (a, b) in enumerate(zip(left, right))]
    expected = EXPECTED_DROPOUT_CALLS_PER_FORWARD * steps
    for label, log in (("reference", reference.draws), ("ours", ours.draws)):
        if log.dropout_calls != expected:
            notes.append(f"T3 {label} made {log.dropout_calls} F.dropout calls, expected {expected}")
    return diffs, notes


def layer_t4(reference: _ReferenceRun, ours: _PosttrainRun) -> tuple[list[Diff], list[str]]:
    """Gradients: clipped `action_head.*` gradients, and the reported pre-clip norm."""
    diffs: list[Diff] = []
    notes: list[str] = []
    for step, (ref_grads, our_grads) in enumerate(zip(reference.grads, ours.grads)):
        our_stripped = _strip_prefix(our_grads)
        if set(ref_grads) != set(our_stripped):
            notes.append(f"T4 step {step}: gradient key sets differ ({len(ref_grads)} vs {len(our_stripped)})")
        else:
            diffs += [
                compare(f"t4.step{step}.grad.{key}", ref_grads[key], our_stripped[key]) for key in sorted(ref_grads)
            ]
    diffs.append(compare("t4.grad_norm_sequence", torch.tensor(reference.grad_norms), torch.tensor(ours.grad_norms)))
    return diffs, notes


def layer_t5(
    reference: _ReferenceRun,
    ours: _PosttrainRun,
    initial_backbone: dict[str, dict[str, torch.Tensor]],
) -> tuple[list[Diff], list[str]]:
    """K-step trajectory: parameters and AdamW moments after every step, and a frozen backbone on both sides."""
    diffs: list[Diff] = []
    notes: list[str] = []
    for step, (ref_params, our_params) in enumerate(zip(reference.params, ours.params)):
        our_stripped = _strip_prefix(our_params)
        if set(ref_params) != set(our_stripped):
            notes.append(f"T5 step {step}: parameter key sets differ")
        else:
            diffs += [
                compare(f"t5.step{step}.param.{key}", ref_params[key], our_stripped[key]) for key in sorted(ref_params)
            ]
    for step, (ref_moments, our_moments) in enumerate(zip(reference.moments, ours.moments)):
        our_stripped = _strip_prefix(our_moments)
        diffs += [
            compare(f"t5.step{step}.moment.{key}", ref_moments[key], our_stripped[key]) for key in sorted(ref_moments)
        ]
    for label, model in (("reference", reference.model), ("ours", ours.policy.gr00t)):
        after = model.backbone.state_dict()
        changed = [
            k
            for k, before in initial_backbone[label].items()
            if not torch.equal(before, after[k].detach().float().cpu())
        ]
        if changed:
            notes.append(f"T5 {label}: frozen backbone parameters CHANGED during training, e.g. {changed[:3]}")
    return diffs, notes


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


def _setup(*, checkpoint: str, batch: int, steps: int, seed: int, out_root: str, training_prefix: int | None) -> tuple:
    """Recipe config, the checkpoint's processor and K x B synthetic samples, shared by both sides.

    Below the full action horizon the recipe's mask is written into the samples so the reference minimizes the
    same objective we do; at the full horizon the derived mask equals the processor's own, and T1 validates that.
    """
    from cosmos_framework.posttrain.configs.gr00t_n1d7_droid import make_model_config
    from cosmos_framework.posttrain.model.gr00t.packer import action_mask_for
    from cosmos_framework.posttrain.model.gr00t.policy import load_processor
    from cosmos_framework.posttrain.tests.helpers.gr00t_synthetic import synthetic_samples

    overrides: dict[str, Any] = {"checkpoint_path": checkpoint}
    if training_prefix is not None:
        overrides["training_prefix"] = training_prefix
    config = make_model_config(**overrides)
    os.makedirs(os.path.join(out_root, "cfg"), exist_ok=True)
    processor = load_processor(checkpoint)
    samples = synthetic_samples(processor, config, steps * batch, seed=seed)
    if config.training_prefix != config.model_action_horizon:
        for sample in samples:
            sample["action_mask"] = action_mask_for(
                sample["action"], action_dim=config.action_dim, training_prefix=config.training_prefix
            )
    return config, processor, samples


def _step_batches(samples: list[dict], steps: int, batch: int) -> list[list]:
    """Slice the shared samples into per-step `Trajectory` lists, in the reference's consumption order."""
    from cosmos_framework.posttrain.model.gr00t.packer import transition_from_processed
    from cosmos_framework.posttrain.schema import Status, Trajectory

    return [
        [
            Trajectory(
                transitions=[transition_from_processed(copy.deepcopy(s))], id=f"step{step}-{i}", status=Status.COMPLETED
            )
            for i, s in enumerate(samples[step * batch : (step + 1) * batch])
        ]
        for step in range(steps)
    ]


def run_train_parity(
    *,
    checkpoint: str,
    batch: int,
    steps: int,
    seed: int,
    out_root: str,
    training_prefix: int | None = None,
    dropout: str = "on",
    device: str = "cuda:0",
    perturb: float = 0.0,
    share_action_head: bool = False,
) -> dict[str, Any]:
    """Run both trainers on identical batches and report all six layers. `perturb` and `share_action_head`
    are negative controls: a gate that has only ever passed is indistinguishable from one that cannot fail."""
    from cosmos_framework.posttrain.model.gr00t.policy import (
        Gr00tHeads,
        Gr00tN1d7TrainPolicy,
        build_official_gr00t_model,
    )

    config, processor, samples = _setup(
        checkpoint=checkpoint, batch=batch, steps=steps, seed=seed, out_root=out_root, training_prefix=training_prefix
    )
    cfg_dir = os.path.join(out_root, "cfg")
    report: dict[str, Any] = {
        "checkpoint": checkpoint,
        "batch": batch,
        "steps": steps,
        "seed": seed,
        "training_prefix": config.training_prefix,
        "dropout": dropout,
        "config_identity": config.identity_hash(),
        "diffs": [],
        "notes": [],
        "deviations": [],
    }
    if config.training_prefix != config.model_action_horizon:
        report["deviations"].append(
            f"training_prefix={config.training_prefix} < model_action_horizon={config.model_action_horizon}: the "
            "recipe's mask is fed to BOTH sides, so T1's action_mask row compares our derivation with itself"
        )

    reference_model = build_official_gr00t_model(checkpoint, save_cfg_dir=cfg_dir).to(device)
    policy = Gr00tN1d7TrainPolicy(config, device=device, save_cfg_dir=cfg_dir)

    if dropout == "off":
        # Attribution variant: if as-shipped mismatches but this aligns, the fault is in module train/eval
        # state or RNG consumption order rather than in the update rule.
        for model in (reference_model, policy.gr00t):
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = 0.0
            model.action_head.state_dropout_prob = 0.0
        report["deviations"].append("dropout=off: nn.Dropout.p and state_dropout_prob zeroed on BOTH sides")
    if share_action_head:
        # NEGATIVE CONTROL: our wire scope IS the reference's head; only T0's identity check can tell.
        policy.net = Gr00tHeads(reference_model.action_head)
        policy.gr00t.action_head = reference_model.action_head
        report["deviations"].append("NEGATIVE CONTROL share_action_head: T0 must report shared parameters")
    if perturb:
        # NEGATIVE CONTROL: move one weight; T0/T3/T4/T5 must all report it.
        with torch.no_grad():
            next(iter(policy.net.parameters())).add_(perturb)
        report["deviations"].append(f"NEGATIVE CONTROL perturb={perturb}: one of our parameters was moved; must FAIL")

    initial_backbone = {
        "reference": _snapshot(reference_model.backbone.state_dict()),
        "ours": _snapshot(policy.gr00t.backbone.state_dict()),
    }
    initial_action_head = {
        "reference": _snapshot(reference_model.action_head.state_dict()),
        "ours": _snapshot(policy.net.state_dict()),
    }

    reference = _ReferenceRun(
        model=reference_model,
        samples=copy.deepcopy(samples),  # a mutation by one side must not reach the other
        collator=processor.collator,
        steps=steps,
        batch=batch,
        seed=seed,
        output_dir=os.path.join(out_root, "reference_out"),
    )
    ours = _PosttrainRun(policy=policy, config=config, seed=seed)
    reference.run()
    ours.run(_step_batches(samples, steps, batch))
    for capture in (reference.backbone, ours.backbone, reference.seeder, ours.seeder):
        capture.remove()

    # Our slice of `samples` must be what the reference's collator actually saw, step by step.
    for step, features in enumerate(reference.collator.features[:steps]):
        expected = samples[step * batch : (step + 1) * batch]
        if len(features) != len(expected) or not all(
            torch.equal(f["action"], e["action"]) for f, e in zip(features, expected)
        ):
            report["notes"].append(
                f"SAMPLE ORDER: at step {step} the reference consumed different samples than our side"
            )
            break

    for layer, (diffs, notes) in (
        ("T0", layer_t0(reference, ours, config, initial_action_head)),
        ("T1", layer_t1(reference, ours)),
        ("T2", layer_t2(reference, ours)),
        ("T3", layer_t3(reference, ours, steps)),
        ("T4", layer_t4(reference, ours)),
        ("T5", layer_t5(reference, ours, initial_backbone)),
    ):
        report["diffs"].extend(d.as_json() for d in diffs)
        report["notes"].extend(f"[{layer}] {n}" for n in notes)
    report["losses"] = {"reference": reference.losses, "ours": ours.losses}
    report["grad_norms"] = {"reference": reference.grad_norms, "ours": ours.grad_norms}
    return report


def run_reference_self_consistency(
    *,
    checkpoint: str,
    batch: int,
    steps: int,
    seed: int,
    out_root: str,
    training_prefix: int | None = None,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Run the OFFICIAL trainer twice, identically, and compare it with itself. Every row is expected to be 0;
    a non-zero is a determinism defect to diagnose, never a tolerance to spend on T4/T5."""
    from cosmos_framework.posttrain.model.gr00t.policy import build_official_gr00t_model

    config, processor, samples = _setup(
        checkpoint=checkpoint, batch=batch, steps=steps, seed=seed, out_root=out_root, training_prefix=training_prefix
    )
    captured = []
    for attempt in range(2):  # sequential, releasing the first model before building the second
        model = build_official_gr00t_model(checkpoint, save_cfg_dir=os.path.join(out_root, "cfg")).to(device)
        run = _ReferenceRun(
            model=model,
            samples=copy.deepcopy(samples),
            collator=processor.collator,
            steps=steps,
            batch=batch,
            seed=seed,
            output_dir=os.path.join(out_root, f"self_consistency_{attempt}"),
        )
        run.run()
        run.backbone.remove()
        run.seeder.remove()
        captured.append(
            {"losses": list(run.losses), "grad_norms": list(run.grad_norms), "grads": run.grads, "params": run.params}
        )
        del run, model
        torch.cuda.empty_cache()

    first, second = captured
    diffs = [compare("reference_self_consistency_loss", torch.tensor(first["losses"]), torch.tensor(second["losses"]))]
    for kind in ("grads", "params"):
        for step, (left, right) in enumerate(zip(first[kind], second[kind])):
            diffs += [
                compare(f"reference_self_consistency_{kind}.step{step}.{key}", left[key], right[key])
                for key in sorted(left)
            ]
    non_exact = [d for d in diffs if not d.exact]
    notes = []
    if non_exact:
        worst = max(non_exact, key=lambda d: d.max_abs)
        notes.append(
            f"DETERMINISM IS NOT IN EFFECT: {len(non_exact)}/{len(diffs)} rows differ when the reference is run "
            f"twice identically (worst {worst.name} at {worst.max_abs:.3e}); diagnose the kernel, do not add a tolerance"
        )
    return {
        "checkpoint": checkpoint,
        "batch": batch,
        "steps": steps,
        "seed": seed,
        "training_prefix": config.training_prefix,
        "dropout": "on",
        "diffs": [d.as_json() for d in diffs],
        "notes": notes,
        "deviations": ["reference vs reference: no posttrain code participates in this comparison"],
        "losses": {"first": first["losses"], "second": second["losses"]},
        "grad_norms": {"first": first["grad_norms"], "second": second["grad_norms"]},
    }


def run_our_side(
    *,
    checkpoint: str,
    batch: int,
    steps: int,
    seed: int,
    out_root: str,
    training_prefix: int | None = None,
    device: str = "cuda:0",
    dp_shard: int = 1,
    precision_mode: str = "autocast",
) -> dict[str, Any]:
    """Run OUR trainer alone, optionally under FSDP2. There is no official multi-GPU reference (Isaac uses
    DDP/ZeRO), so the sharded run is compared with our own single-GPU golden. Every rank consumes the IDENTICAL
    micro-batch: `torch.randn` is shape-dependent, so a disjoint split would give the ranks different noise
    from the golden and measure the harness's seeding rather than FSDP.

    `precision_mode` picks the `Precision` preset: "autocast" is `OFFICIAL_PRECISION` (the gated recipe);
    "fsdp" is the all-bf16 RLinf-style preset, which has no reference and is only smoke-tested."""
    from cosmos_framework.posttrain.model.gr00t.policy import PRECISION_PRESETS, Gr00tN1d7TrainPolicy

    precision = PRECISION_PRESETS[precision_mode]

    config, _, samples = _setup(
        checkpoint=checkpoint, batch=batch, steps=steps, seed=seed, out_root=out_root, training_prefix=training_prefix
    )
    parallel_dims, rank = None, 0
    if dp_shard > 1:
        import torch.distributed as dist

        from cosmos_framework.posttrain.model.parallel_dims import ParallelDims

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        if dist.get_world_size() != dp_shard:
            raise ValueError(f"--dp-shard {dp_shard} but the torchrun world is {dist.get_world_size()}")
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        parallel_dims = ParallelDims(world_size=dp_shard, dp_shard=dp_shard, dp_replicate=1)
        parallel_dims.build_device_mesh("cuda")

    cfg_dir = os.path.join(out_root, "cfg", f"rank{rank}")  # one dir per rank: Isaac writes final_model_config.json
    os.makedirs(cfg_dir, exist_ok=True)
    policy = Gr00tN1d7TrainPolicy(
        config, device=device, parallel_dims=parallel_dims, save_cfg_dir=cfg_dir, precision=precision
    )
    ours = _PosttrainRun(policy=policy, config=config, seed=seed)
    ours.run(_step_batches(samples, steps, batch))
    ours.backbone.remove()
    ours.seeder.remove()

    if dp_shard > 1:
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

        full = get_model_state_dict(policy.net, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
        net_state = {k: v.detach().float().cpu().clone() for k, v in full.items()}
    else:
        net_state = _snapshot(policy.net.state_dict())
    return {
        "losses": ours.losses,
        "grad_norms": ours.grad_norms,
        "net": net_state,
        "rank": rank,
        "training_prefix": config.training_prefix,
        "config_identity": config.identity_hash(),
        "precision": dataclasses.asdict(precision),
        "dp_shard": dp_shard,
        "param_dtypes": sorted({str(p.dtype) for p in policy.net.parameters()}),
        "backbone_param_dtypes": sorted({str(p.dtype) for p in policy.backbone.parameters()}),
    }


def render(report: dict[str, Any]) -> str:
    lines = [
        "# GR00T-N1.7 training parity: posttrain LLMTrainer vs official Gr00tTrainer",
        "",
        f"checkpoint `{report['checkpoint']}` · B={report['batch']} · K={report['steps']} · seed={report['seed']} · "
        f"prefix={report['training_prefix']} · dropout={report['dropout']} · sdpa={os.environ.get('GR00T_DIT_SDPA_MODE', 'default')}",
        "",
        *(f"{label} losses: {value}" for label, value in report["losses"].items()),
        "",
        *render_diff_table(report["diffs"]),
    ]
    if report["notes"]:
        lines += ["", "## Notes", *(f"> {note}" for note in report["notes"])]
    if report["deviations"]:
        lines += ["", "## Deviations from the official run", *(f"- {d}" for d in report["deviations"])]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default=os.environ.get("GR00T_CKPT", "nvidia/GR00T-N1.7-DROID"))
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--training-prefix", type=int, default=None, help="override the recipe's prefix")
    parser.add_argument("--dropout", choices=("on", "off"), default="on")
    parser.add_argument(
        "--out-root", required=True, help="scratch root for the factory config and the reference output_dir"
    )
    parser.add_argument("--json", default="", help="write the raw report here")
    parser.add_argument("--assert", dest="do_assert", action="store_true")
    parser.add_argument(
        "--dump-golden", default="", help="run OUR side alone on one GPU and write golden.pt here (FSDP variant)"
    )
    parser.add_argument("--dp-shard", type=int, default=1, help="FSDP2 shard degree; run under torchrun with --golden")
    parser.add_argument("--golden", default="", help="directory holding golden.pt from --dump-golden")
    parser.add_argument(
        "--precision-mode",
        choices=("autocast", "fsdp"),
        default="autocast",
        help="OUR side's Precision preset: autocast = the official fp32+autocast recipe (the only gated one); "
        "fsdp = all-bf16 parameters + MixedPrecisionPolicy, smoke only (--dump-golden / --dp-shard)",
    )
    parser.add_argument(
        "--self-consistency", action="store_true", help="run the OFFICIAL trainer twice; every row must be 0"
    )
    parser.add_argument(
        "--perturb", type=float, default=0.0, help="NEGATIVE CONTROL: add this to one of our parameters"
    )
    parser.add_argument(
        "--share-action-head", action="store_true", help="NEGATIVE CONTROL: our net IS the reference's head"
    )
    args = parser.parse_args(argv)
    if args.precision_mode != "autocast" and not (args.dump_golden or args.dp_shard > 1):
        parser.error(
            "--precision-mode fsdp has no official reference; use it with --dump-golden or --dp-shard/--golden"
        )

    from cosmos_framework.posttrain.utils.determinism import enable_determinism

    enable_determinism()  # before any CUDA init: cuBLAS reads CUBLAS_WORKSPACE_CONFIG when it creates its handle
    os.environ.setdefault("GR00T_DIT_SDPA_MODE", _DETERMINISTIC_SDPA)

    common: dict[str, Any] = dict(
        checkpoint=args.checkpoint,
        batch=args.batch,
        steps=args.steps,
        seed=args.seed,
        out_root=args.out_root,
        training_prefix=args.training_prefix,
    )
    if args.dump_golden:
        result = run_our_side(**common, precision_mode=args.precision_mode)
        result.update(checkpoint=args.checkpoint, batch=args.batch, seed=args.seed)
        os.makedirs(args.dump_golden, exist_ok=True)
        torch.save(result, os.path.join(args.dump_golden, "golden.pt"))
        print(
            f"wrote {args.dump_golden}/golden.pt; precision={args.precision_mode}; "
            f"head param dtypes: {result['param_dtypes']}; backbone: {result['backbone_param_dtypes']}; "
            f"losses: {result['losses']}"
        )
        return 0

    if args.self_consistency:
        report = run_reference_self_consistency(**common)
    elif args.dp_shard > 1:
        result = run_our_side(**common, dp_shard=args.dp_shard, precision_mode=args.precision_mode)
        if result["rank"] != 0:
            return 0  # only rank 0 holds the gathered full state and reports
        golden = torch.load(os.path.join(args.golden, "golden.pt"), map_location="cpu", weights_only=False)
        # Refuse before diffing: `config_identity` does not cover the Precision preset, so a golden dumped under
        # the other `--precision-mode` would pass the identity check and then fail every row with empty notes.
        for key in ("param_dtypes", "backbone_param_dtypes"):
            if golden[key] != result[key]:
                raise ValueError(
                    f"golden.pt comes from another precision preset: {key} golden={golden[key]} ours={result[key]} "
                    f"(--precision-mode {args.precision_mode}); re-run --dump-golden with the same preset"
                )
        diffs = [
            compare("fsdp.loss_sequence", torch.tensor(golden["losses"]), torch.tensor(result["losses"])),
            compare("fsdp.grad_norm_sequence", torch.tensor(golden["grad_norms"]), torch.tensor(result["grad_norms"])),
            *(compare(f"fsdp.param.{key}", golden["net"][key], result["net"][key]) for key in sorted(golden["net"])),
        ]
        report = {
            "checkpoint": golden["checkpoint"],
            "batch": golden["batch"],
            "steps": len(result["losses"]),
            "seed": golden["seed"],
            "training_prefix": golden["training_prefix"],
            "dropout": "on",
            "precision": result["precision"],
            "dp_shard": args.dp_shard,
            "param_dtypes": result["param_dtypes"],
            "backbone_param_dtypes": result["backbone_param_dtypes"],
            "diffs": [d.as_json() for d in diffs],
            "notes": []
            if golden["config_identity"] == result["config_identity"]
            else ["golden has a different recipe identity"],
            "deviations": ["no official multi-GPU reference: our FSDP2 run vs our own single-GPU golden"],
            "losses": {"golden": golden["losses"], "ours": result["losses"]},
            "grad_norms": {"golden": golden["grad_norms"], "ours": result["grad_norms"]},
        }
    else:
        report = run_train_parity(
            **common, dropout=args.dropout, perturb=args.perturb, share_action_head=args.share_action_head
        )

    print(render(report))
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(report, handle, indent=2)
    if not args.do_assert:
        return 0
    failed, exact_failed = evaluate(report["diffs"], TOLERANCES)
    for d in failed + exact_failed:
        print(f"FAIL {d['name']}: max_abs={d['max_abs']:.3e}")
    for note in report["notes"]:
        print(f"NOTE {note}")
    ok = not failed and not exact_failed and not report["notes"]
    print("\nOVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
