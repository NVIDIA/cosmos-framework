# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The GR00T serving gate: three layers between the vLLM-Omni port and the Isaac-GR00T reference, plus the engine.

Shared by `tests/gr00t_parity_cli.py` (stage 1's regression: version 0, an engine of its own) and
`tests/gr00t_weight_sync_cli.py` (version k: the HOT engine after a weight sync, against a reference loaded from
the exported checkpoint). One implementation of the layers, so the two entries cannot drift:

  layer 1  processor tensors      exact equality (a difference here is a different input)
  layer 2  backbone_features      exact equality
  layer 3  actions                normalized (model space) and absolute (after decode_action), exact equality

Every cross-implementation row is required to be bitwise equal; `TOLERANCES` holds the ONE row that is not a
parity row (`serving_batch_invariance_features`, the serving side against itself). Layers 1 and 2 are internal to
the engine's worker and not observable through `infer_batch`, so they run against an in-process instantiation of
the same vLLM-Omni classes the worker builds; layer 3 also runs through a body's `infer_batch`, which separates an
adaptation bug from an engine effect and, for a hot body, is the end-to-end proof that the synced weights are what
the engine serves.

Noise: both action heads draw ``torch.randn(size=(B, H, D), ...)`` with no generator and no earlier random op, so
seeding the global RNG immediately before gives both sides the same noise. That is asserted by capturing the drawn
tensor on each side, not inferred from RNG ordering.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np
import torch

from cosmos_framework.posttrain.tests.helpers.gr00t_synthetic import build_observations
from cosmos_framework.posttrain.tests.helpers.parity_diff import (
    Diff,
    capture_initial_noise,
    compare,
    evaluate,
    flatten_tensors,
    render_diff_table,
)

# Calibrated 2026-09-03 on GB200 against `nvidia/GR00T-N1.7-DROID`, B in {1, 2} x seed in
# {1234, 4321}, after the image-transform correction. Every cross-implementation comparison is
# BITWISE EQUAL, so a name absent from this mapping is required to be bitwise equal. The single
# entry is not a parity row: `serving_batch_invariance_features` compares the serving side against
# ITSELF (B=2 vs two B=1 runs) and measured 0.750 on both seeds -- padding/masking and bf16
# accumulation order, which applies to the reference equally. Bounded at ~4x the observed value.
# A parity row becoming non-exact is a finding to investigate, not a number to absorb.
TOLERANCES: dict[str, float] = {"serving_batch_invariance_features": 3.0}

# Processor tensors that must match exactly: integer-valued or geometry-defining inputs.
_EXACT_PROCESSOR_KEYS = (
    "input_ids",
    "attention_mask",
    "pixel_values",
    "image_grid_thw",
    "state",
    "embodiment_id",
)


def batched_observation(obs_list: list[dict[str, Any]], config: Any) -> dict[str, Any]:
    """Env observations -> the nested ``{video, state, language}`` dict both policies consume.

    Built by the PRODUCTION adapter (``Gr00tPolicyBody._build_robot_obs``) so the reference and the
    serving side are handed a byte-identical input; the body is constructed without ``__init__`` (no
    engine) and ``_build_robot_obs`` reads only ``self.config``.
    """
    from cosmos_framework.posttrain.model.gr00t.serving.body import Gr00tPolicyBody

    body = object.__new__(Gr00tPolicyBody)
    body.config = config
    robot_obs, _prompt = body._build_robot_obs(obs_list)
    return robot_obs


def processor_identity(processor: Any) -> dict[str, Any]:
    """A comparable fingerprint of a loaded processor's tokenizer and image settings.

    vLLM-Omni's ``build_processor`` substitutes ``Qwen/Qwen3-VL-2B-Instruct`` for
    ``nvidia/Cosmos-Reason2-2B``, claiming the latter lacks a ``Qwen3VLProcessor``. It does not; the
    substitution is harmless only because the six processor artifacts are byte-identical between the
    two repos today. That equivalence is asserted here rather than assumed -- a fine-tuned Cosmos
    tokenizer would make it a real divergence, and layer 1 is the only place it would show.
    """
    # The Qwen3VLProcessor that tokenizes hangs off the GR00T wrapper as `.processor`; resolve rather
    # than assume a depth, so a re-wrap fails loudly instead of fingerprinting the wrong object.
    vlm_processor = processor
    for _ in range(4):
        if hasattr(vlm_processor, "tokenizer"):
            break
        inner = getattr(vlm_processor, "processor", None)
        if inner is None:
            raise AttributeError(
                f"cannot find the tokenizing processor inside {type(processor).__name__}; "
                "the wrapper layout changed and this fingerprint would compare the wrong objects"
            )
        vlm_processor = inner
    else:
        raise AttributeError(f"no .tokenizer within 4 levels of {type(processor).__name__}")

    tokenizer = vlm_processor.tokenizer
    vocab = tokenizer.get_vocab()
    vocab_digest = hashlib.sha256(json.dumps(vocab, sort_keys=True).encode()).hexdigest()[:16]
    image_processor = getattr(vlm_processor, "image_processor", None)
    image_cfg = image_processor.to_dict() if image_processor is not None else {}
    image_cfg = {k: v for k, v in sorted(image_cfg.items()) if not k.startswith("_")}
    return {
        # Provenance only, not compared: the two sides legitimately name different repos.
        "_model_name": getattr(processor, "model_name", None),
        "tokenizer_class": type(tokenizer).__name__,
        "vocab_size": len(vocab),
        "vocab_sha256_16": vocab_digest,
        "special_tokens": dict(sorted((tokenizer.special_tokens_map or {}).items())),
        "image_processor_class": type(image_processor).__name__ if image_processor else None,
        "image_processor_config": image_cfg,
    }


def _comparable(identity: dict[str, Any]) -> dict[str, Any]:
    """Drop the provenance-only fields (leading underscore) before comparing two fingerprints."""
    return {k: v for k, v in identity.items() if not k.startswith("_")}


class PolicyRun:
    """One implementation with the processor / backbone / noise seams captured.

    Both packages expose ``policy.collate_fn``, ``policy.model.backbone`` and ``policy.get_action``, so
    one wrapper instruments both sides identically by construction.
    """

    def __init__(self, policy: Any) -> None:
        self.policy = policy
        self.captured: dict[str, Any] = {}
        # Layer 1: the collated batch IS the model input, after tokenization, image preprocessing,
        # padding and stacking.
        inner_collate = policy.collate_fn

        def _capturing_collate(processed_inputs: Any) -> Any:
            collated = inner_collate(processed_inputs)
            self.captured["collated_inputs"] = collated
            return collated

        policy.collate_fn = _capturing_collate
        policy.model.backbone.register_forward_hook(
            lambda _m, _i, output: self.captured.__setitem__("backbone_output", output)
        )
        # Every Euler step's velocity: the decoder's output, last `action_horizon` rows (Isaac's
        # `get_action_with_features` slices it the same way before `actions += dt * pred_velocity`).
        head = policy.model.action_head
        horizon = head.config.action_horizon
        head.action_decoder.register_forward_hook(
            lambda _m, _i, output: self.captured.setdefault("velocities", []).append(
                output[:, -horizon:].detach().clone()
            )
        )
        # The normalized action, before `decode_action`: a normalized mismatch is the model, an
        # absolute-only mismatch is the statistics or the relative-to-absolute conversion.
        inner_get_action = policy.model.get_action

        def _capturing_get_action(*args: Any, **kwargs: Any) -> Any:
            prediction = inner_get_action(*args, **kwargs)
            self.captured["normalized_action"] = prediction["action_pred"]
            return prediction

        policy.model.get_action = _capturing_get_action
        # The decode call's own arguments (embodiment tag and the states the relative actions are
        # applied to), so a third forward can be decoded by the SAME processor on the same states.
        inner_decode = policy.processor.decode_action

        def _capturing_decode(normalized: Any, embodiment_tag: Any, states: Any) -> Any:
            self.captured["decode_args"] = (embodiment_tag, states)
            return inner_decode(normalized, embodiment_tag, states)

        policy.processor.decode_action = _capturing_decode

    def run(self, obs: dict[str, Any], seed: int) -> dict[str, Any]:
        self.captured.clear()
        noise: list[torch.Tensor] = []
        torch.manual_seed(seed)
        with capture_initial_noise(noise):
            action = self.policy.get_action(obs)
        self.captured["initial_noise"] = noise[0] if noise else None
        # `Gr00tPolicy.get_action` returns (actions, info); unwrap defensively.
        self.captured["absolute_action"] = action[0] if isinstance(action, tuple) else action
        # A copy: the hooks keep writing `self.captured` on later calls through this policy.
        return dict(self.captured)


#: The ``model_type`` both packages register their classes under.
_SHARED_MODEL_TYPE = "Gr00tN1d7"

#: Isaac-GR00T's attention-backend override. Its DiT forces the math SDPA kernel under ``math``; the vLLM-Omni port
#: has no reader for it and always runs PyTorch's default kernel selection.
DIT_SDPA_MODE_ENV = "GR00T_DIT_SDPA_MODE"


@contextlib.contextmanager
def stage1_conditions() -> Iterator[None]:
    """Run the two in-process implementations the way stage 1 measured them: no attention-backend override.

    The trainer needs ``GR00T_DIT_SDPA_MODE=math`` (Isaac's DiT is not deterministic across backward passes
    otherwise), so a process that drives training carries it. Isaac applies the override to its own attention, the
    port cannot follow it, and the two implementations then legitimately differ (measured 2026-09-09: with the
    override set, `backbone_features` off by 0.375 and every action row not bitwise; unset or ``default``, all
    rows bitwise, engine included). The gate compares the implementations, not the override, so it runs both
    sides with the variable removed; Isaac reads it at call time. The native diagnostic runs OUTSIDE this block,
    under whatever the process carries, because it is meant to be the trainer's own forward.
    """
    saved = os.environ.pop(DIT_SDPA_MODE_ENV, None)
    try:
        yield
    finally:
        if saved is not None:
            os.environ[DIT_SDPA_MODE_ENV] = saved


@contextlib.contextmanager
def _pinned_auto_classes(config_cls: type, model_cls: type, processor_cls: type) -> Iterator[None]:
    """Resolve the shared ``Gr00tN1d7`` auto-class key to one package's classes inside the block, then hand it back.

    Both packages register ``model_type="Gr00tN1d7"`` with ``AutoConfig``/``AutoModel`` and the string key with
    ``AutoProcessor`` at import, and the last import wins -- which import is last depends on whether an engine body
    (the port) or a reference (Isaac) was built first. Everything that goes through ``AutoModel.from_pretrained``
    (both ``Gr00tPolicy`` classes, and Isaac's factory behind `model/gr00t/policy.py::build_official_gr00t_model`) then
    gets the OTHER package's class: the factory fails on its own kwargs, a "reference" is the port compared against
    itself, or the port receives Isaac's processor (whose training mode demands actions). `processor_config.json`
    names the class ``Gr00tN1d7Processor``, a name both packages use, so the processor pin is by key too. The
    previous holders are handed back on exit.
    """
    from transformers import AutoConfig, AutoModel, AutoProcessor
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from transformers.models.auto.processing_auto import PROCESSOR_MAPPING

    previous_config = CONFIG_MAPPING._extra_content.get(_SHARED_MODEL_TYPE)
    previous_processor = PROCESSOR_MAPPING._extra_content.get(_SHARED_MODEL_TYPE)
    AutoConfig.register(_SHARED_MODEL_TYPE, config_cls, exist_ok=True)
    AutoModel.register(config_cls, model_cls, exist_ok=True)
    AutoProcessor.register(_SHARED_MODEL_TYPE, processor_cls, exist_ok=True)
    try:
        yield
    finally:
        if previous_config is not None:
            AutoConfig.register(_SHARED_MODEL_TYPE, previous_config, exist_ok=True)
        if previous_processor is not None:
            AutoProcessor.register(_SHARED_MODEL_TYPE, previous_processor, exist_ok=True)


@contextlib.contextmanager
def isaac_auto_classes() -> Iterator[None]:
    """Isaac-GR00T's classes for the shared key: the reference, and Isaac's model factory (exporter, trainer copy)."""
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7, Gr00tN1d7Config
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor

    with _pinned_auto_classes(Gr00tN1d7Config, Gr00tN1d7, Gr00tN1d7Processor):
        yield


@contextlib.contextmanager
def port_auto_classes() -> Iterator[None]:
    """vLLM-Omni's classes for the shared key: the in-process serving side, built exactly as the worker builds it."""
    from vllm_omni.diffusion.models.gr00t.modeling.gr00t_n1d7 import Gr00tN1d7, Gr00tN1d7Config
    from vllm_omni.diffusion.models.gr00t.modeling.processing_gr00t_n1d7 import Gr00tN1d7Processor

    with _pinned_auto_classes(Gr00tN1d7Config, Gr00tN1d7, Gr00tN1d7Processor):
        yield


def reference_run(checkpoint: str, embodiment_tag: str, device: str = "cuda:0") -> PolicyRun:
    """Isaac-GR00T: the reference implementation, built with Isaac's classes whatever was imported before."""
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    with isaac_auto_classes():
        return PolicyRun(Gr00tPolicy(embodiment_tag=embodiment_tag, model_path=checkpoint, device=device))


def serving_run(checkpoint: str, embodiment_tag: str, device: str = "cuda:0") -> PolicyRun:
    """vLLM-Omni's port, in-process -- the same classes the engine worker builds."""
    from vllm_omni.diffusion.models.gr00t.policy import Gr00tPolicy

    # Importing the RL pipeline module replaces the built-in registry entry, matching the worker.
    import cosmos_framework.posttrain.model.gr00t.serving.omni_pipeline  # noqa: F401
    from cosmos_framework.posttrain.model.gr00t.serving.image_transform import (
        install_reference_image_transform,
    )

    with port_auto_classes():
        policy = Gr00tPolicy(embodiment_tag=embodiment_tag, model_path=checkpoint, device=device)
    # The same install the pipeline does; otherwise `engine_vs_inprocess_serving` would report a
    # difference that exists only in this script.
    install_reference_image_transform(policy, checkpoint)
    return PolicyRun(policy)


def compare_processor_tensors(ref: dict[str, Any], srv: dict[str, Any]) -> tuple[list[Diff], list[str]]:
    """Layer 1: every collated model input, compared for EXACT equality."""
    ref_batch, srv_batch = ref.get("collated_inputs"), srv.get("collated_inputs")
    if ref_batch is None or srv_batch is None:
        return [], ["collated inputs were not captured; layer 1 did not run"]

    # Recurse: a top-level-only scan of `BatchFeature(data={"inputs": batch})` finds no tensors and
    # would look like a pass while the layer never ran.
    ref_flat, srv_flat = flatten_tensors(ref_batch), flatten_tensors(srv_batch)

    diffs: list[Diff] = []
    notes: list[str] = []
    shared = sorted(set(ref_flat) & set(srv_flat))
    only_ref, only_srv = sorted(set(ref_flat) - set(srv_flat)), sorted(set(srv_flat) - set(ref_flat))
    if only_ref or only_srv:
        notes.append(f"collated-input tensors differ in NAME: reference-only={only_ref}, serving-only={only_srv}")
    for key in shared:
        try:
            diffs.append(compare(f"processor.{key}", ref_flat[key], srv_flat[key]))
        except ValueError as exc:
            notes.append(str(exc))
    if not shared:
        notes.append(f"no comparable tensors found; reference tree={sorted(ref_flat)}, serving tree={sorted(srv_flat)}")
    missing = [k for k in _EXACT_PROCESSOR_KEYS if not any(f.endswith(k) for f in shared)]
    if missing:
        notes.append(f"expected processor tensors absent from the collated batch: {missing}")
    return diffs, notes


def feature_tensor(backbone_output: Any) -> tuple[torch.Tensor, str]:
    """Pull the backbone's feature tensor out of its container, and say WHICH key it came from.

    The two packages need not name it the same thing; silently resolving different keys would compare
    different quantities and report a model divergence.
    """
    for key in ("backbone_features", "last_hidden_state"):
        value = (
            backbone_output.get(key) if isinstance(backbone_output, Mapping) else getattr(backbone_output, key, None)
        )
        if value is not None:
            return value, key
    available = sorted(backbone_output) if isinstance(backbone_output, dict) else dir(backbone_output)
    raise KeyError(f"no backbone feature tensor in {type(backbone_output).__name__}; available: {available}")


def check_class_provenance(reference: Any, serving: Any) -> tuple[dict[str, str], list[str]]:
    """Verify the two sides really are two DIFFERENT implementations.

    Both packages register ``Gr00tN1d7`` into the same ``AutoConfig`` / ``AutoModel`` /
    ``AutoProcessor`` maps at import time, so whichever imports last wins and ``from_pretrained`` can
    hand back the other package's class -- a flawless pass that compared one implementation against
    itself.
    """
    modules = {
        "reference_model": type(reference.policy.model).__module__,
        "serving_model": type(serving.policy.model).__module__,
        "reference_processor": type(reference.policy.processor).__module__,
        "serving_processor": type(serving.policy.processor).__module__,
    }
    notes: list[str] = []
    for name, module in modules.items():
        expected = "gr00t" if name.startswith("reference") else "vllm_omni"
        if not module.startswith(expected):
            notes.append(
                f"CLASS PROVENANCE: {name} resolved to {module!r}, expected a {expected!r} module. "
                "Both packages register the same auto-class key, so this run may be comparing one "
                "implementation against itself -- the numbers below are not evidence of parity."
            )
    return modules, notes


def flat_action(action: Any) -> np.ndarray:
    """Concatenate a per-modality action dict in sorted key order, matching the serving body."""
    if isinstance(action, dict):
        return np.concatenate([np.asarray(action[k], dtype=np.float32) for k in sorted(action)], axis=-1)
    return np.asarray(action, dtype=np.float32)


@contextlib.contextmanager
def _fixed_initial_noise(noise: torch.Tensor) -> Iterator[None]:
    """Make the one ``[B, H, D]`` ``torch.randn`` draw inside the block return ``noise`` (in the requested dtype and
    device) so a third forward starts from exactly the serving run's noise instead of relying on RNG ordering."""
    real_randn = torch.randn

    def _fixed(*args: Any, **kwargs: Any) -> torch.Tensor:
        out = real_randn(*args, **kwargs)
        if out.ndim == 3:
            if tuple(out.shape) != tuple(noise.shape):
                raise RuntimeError(f"initial noise shape {tuple(noise.shape)} does not fit the draw {tuple(out.shape)}")
            return noise.to(dtype=out.dtype, device=out.device)
        return out

    torch.randn = _fixed
    try:
        yield
    finally:
        torch.randn = real_randn


def native_diagnostic(native: Any, serving_out: dict[str, Any], processor: Any) -> dict[str, Any]:
    """Diagnostic D: the trainer's OWN forward against the serving run, report-only.

    ``native`` is a `model/gr00t/policy.py::Gr00tN1d7TrainPolicy` holding the published weights unsharded (an eval copy
    loaded from the exported checkpoint -- `Gr00tN1d7.get_action` calls the head as a plain method, so FSDP2's
    pre-forward all-gather would not fire on the sharded trainer). It runs under its preset's compute context
    (fp32 master + autocast, or bf16 parameters) on the serving run's collated batch -- the collator's
    ``{"inputs": batch}``, called as both policies call it, ``get_action(**collated)``; Isaac's `prepare_input`
    then moves the batch to the model's device and casts floating inputs to the PARAMETER dtype, which is the
    trainer's own input handling -- with the serving run's initial noise. The rows are the train/serve numerical
    gap itself: backbone features, every Euler step's velocity, the normalized and the decoded absolute action.
    Nothing here decides PASS; the caller only requires the diagnostic to be complete.
    """
    model = native.gr00t
    model.eval()
    # Shallow copies: `prepare_input` pops keys from the batch it is handed.
    collated = {
        key: dict(value) if isinstance(value, Mapping) else value
        for key, value in dict(serving_out["collated_inputs"]).items()
    }
    noise = serving_out["initial_noise"]
    if noise is None:
        raise RuntimeError("the serving run captured no initial noise; the native diagnostic cannot share it")

    captured: dict[str, Any] = {}
    velocities: list[torch.Tensor] = []
    head = model.action_head
    horizon = head.config.action_horizon
    hooks = [
        model.backbone.register_forward_hook(lambda _m, _i, out: captured.__setitem__("backbone_output", out)),
        head.action_decoder.register_forward_hook(
            lambda _m, _i, out: velocities.append(out[:, -horizon:].detach().clone())
        ),
    ]
    try:
        with torch.no_grad(), native.compute_ctx, _fixed_initial_noise(noise):
            prediction = model.get_action(**collated)
    finally:
        for hook in hooks:
            hook.remove()

    rows: list[dict[str, Any]] = []
    native_feat, native_key = feature_tensor(captured["backbone_output"])
    srv_feat, srv_key = feature_tensor(serving_out["backbone_output"])
    rows.append(compare("native_vs_serving/backbone_features", native_feat, srv_feat).as_json())
    srv_velocities = serving_out.get("velocities") or []
    if len(srv_velocities) != len(velocities):
        raise RuntimeError(f"Euler step count differs: native {len(velocities)} vs serving {len(srv_velocities)}")
    for t, (native_v, srv_v) in enumerate(zip(velocities, srv_velocities, strict=True)):
        rows.append(compare(f"native_vs_serving/velocity_step_{t}", native_v, srv_v).as_json())
    normalized = prediction["action_pred"].float()
    rows.append(compare("native_vs_serving/normalized_action", normalized, serving_out["normalized_action"]).as_json())
    embodiment_tag, states = serving_out["decode_args"]
    absolute = processor.decode_action(normalized.cpu().numpy(), embodiment_tag, states)
    rows.append(
        compare(
            "native_vs_serving/absolute_action", flat_action(absolute), flat_action(serving_out["absolute_action"])
        ).as_json()
    )
    return {
        "rows": rows,
        "precision": dataclasses.asdict(native.precision),
        "backbone_feature_key": {"native": native_key, "serving": srv_key},
        "euler_steps": len(velocities),
    }


def run_gate(
    *,
    checkpoint: str,
    config: Any,
    batch: int,
    seed: int,
    body: Any | None = None,
    weight_version: int | None = None,
    device: str = "cuda:0",
    native_policy: Any | None = None,
) -> dict[str, Any]:
    """Run the three layers between a reference and the in-process port, plus layer 3 through ``body``.

    Args:
        checkpoint: what the reference AND the in-process serving side load. For the version-k gate this is the
            exported checkpoint, so both cold sides hold the trained head.
        config: the recipe (`make_model_config(...)`); its horizon and modality maps shape the observations.
        batch: observations per request.
        seed: the noise seed both sides are seeded with immediately before their forward.
        body: an already-constructed engine body (its lifetime is the caller's), or None to skip the engine rows.
        weight_version: passed to ``body.infer_batch`` so a hot body proves it serves THAT committed version.
        device: where the two in-process sides load.
        native_policy: an unsharded `Gr00tN1d7TrainPolicy` holding the same published weights, or None. When given,
            the report carries ``diagnostic`` (see `native_diagnostic`), which the gate rule does not score. It runs
            under the process's own ``GR00T_DIT_SDPA_MODE`` (the trainer's), while the two in-process implementations
            run under `stage1_conditions`; ``report["dit_sdpa_mode"]`` records what the diagnostic saw.

    Returns:
        The report: ``diffs`` (JSON rows), ``notes``, provenance and processor identity fields.
    """
    tag = config.resolve_embodiment_tag()
    obs_list = build_observations(config, batch)
    batched = batched_observation(obs_list, config)

    report: dict[str, Any] = {
        "checkpoint": checkpoint,
        "embodiment_tag": tag.value,
        "batch": batch,
        "seed": seed,
        "weight_version": weight_version,
        "config_identity": config.identity_hash(),
        "diffs": [],
        "notes": [],
    }

    with stage1_conditions():
        reference = reference_run(checkpoint, tag.name, device=device)
        serving = serving_run(checkpoint, tag.name, device=device)

    provenance, provenance_notes = check_class_provenance(reference, serving)
    report["class_provenance"] = provenance
    report["class_provenance_ok"] = not provenance_notes
    report["notes"].extend(provenance_notes)

    # --- processor provenance (see `processor_identity`) --------------------------------------
    ref_proc, srv_proc = processor_identity(reference.policy.processor), processor_identity(serving.policy.processor)
    report["processor_identity"] = {"reference": ref_proc, "serving": srv_proc}
    report["processor_identical"] = _comparable(ref_proc) == _comparable(srv_proc)
    report["processor_substituted"] = ref_proc.get("_model_name") != srv_proc.get("_model_name")
    if not report["processor_identical"]:
        report["notes"].append(
            "PROCESSORS DIFFER. vLLM-Omni substitutes Qwen/Qwen3-VL-2B-Instruct for "
            "nvidia/Cosmos-Reason2-2B; those artifacts were byte-identical when this was written, so "
            "a difference now means the substitution has become a real divergence. Fix it on our "
            "side (override the processor's model_name in the RL pipeline) rather than widening a "
            "tolerance."
        )

    with stage1_conditions():
        ref_out = reference.run(batched, seed)
        srv_out = serving.run(batched, seed)

    # --- layer 1: processor / collated model inputs ---------------------------------------------
    layer1, layer1_notes = compare_processor_tensors(ref_out, srv_out)
    report["diffs"].extend(d.as_json() for d in layer1)
    report["notes"].extend(layer1_notes)

    # --- noise ---------------------------------------------------------------------------------
    if ref_out["initial_noise"] is None or srv_out["initial_noise"] is None:
        report["notes"].append("initial noise was not captured on one side; layer-3 parity is not attributable")
    else:
        report["diffs"].append(compare("initial_noise", ref_out["initial_noise"], srv_out["initial_noise"]).as_json())

    # --- layer 2: backbone features ------------------------------------------------------------
    ref_feat, ref_key = feature_tensor(ref_out["backbone_output"])
    srv_feat, srv_key = feature_tensor(srv_out["backbone_output"])
    report["backbone_feature_key"] = {"reference": ref_key, "serving": srv_key}
    if ref_key != srv_key:
        report["notes"].append(
            f"backbone feature key differs (reference {ref_key!r}, serving {srv_key!r}); the layer-2 "
            "row compares DIFFERENT quantities and says nothing about parity"
        )
    report["diffs"].append(compare("backbone_features", ref_feat, srv_feat).as_json())

    # --- layer 3: normalized then absolute action -----------------------------------------------
    if "normalized_action" in ref_out and "normalized_action" in srv_out:
        report["diffs"].append(
            compare("normalized_action", ref_out["normalized_action"], srv_out["normalized_action"]).as_json()
        )
    ref_action, srv_action = flat_action(ref_out["absolute_action"]), flat_action(srv_out["absolute_action"])
    report["diffs"].append(compare("absolute_action", ref_action, srv_action).as_json())
    # Per modality as well: eef_9d, gripper and joint have different scales and consumers, and the
    # aggregate max hides which one disagrees.
    if isinstance(ref_out["absolute_action"], Mapping) and isinstance(srv_out["absolute_action"], Mapping):
        for key in sorted(set(ref_out["absolute_action"]) & set(srv_out["absolute_action"])):
            report["diffs"].append(
                compare(
                    f"absolute_action[{key}]",
                    np.asarray(ref_out["absolute_action"][key], dtype=np.float32),
                    np.asarray(srv_out["absolute_action"][key], dtype=np.float32),
                ).as_json()
            )

    # --- batch invariance ------------------------------------------------------------------------
    # Vectorized envs ride a B>1 `robot_obs` inside ONE request. Compared on backbone_features, not
    # actions: under one seed `randn((2, H, D))` and `randn((1, H, D))` are not prefixes of each other
    # on CUDA, so an action comparison would measure the harness. The backbone is noise-free.
    if batch > 1:
        with stage1_conditions():
            single_feats = [
                feature_tensor(serving.run(batched_observation([obs], config), seed)["backbone_output"])[0]
                for obs in obs_list
            ]
        report["diffs"].append(
            compare("serving_batch_invariance_features", torch.cat(single_feats, dim=0), srv_feat).as_json()
        )

    # --- layer 3 through the real serving path -------------------------------------------------
    if body is not None:
        engine_actions = body.infer_batch(obs_list, seed=seed, weight_version=weight_version)
        horizon = config.execution_horizon
        engine_action = np.stack([a["action"] for a in engine_actions], axis=0)

        def _truncate(value: np.ndarray) -> np.ndarray:
            return value[:, :horizon] if value.ndim == 3 else value[:horizon]

        # vs the reference: the end-to-end question.
        report["diffs"].append(compare("absolute_action_via_engine", _truncate(ref_action), engine_action).as_json())
        # vs the in-process serving side: the same vLLM-Omni code, so agreement here means the
        # engine contributes nothing and any gap above lies between the port and Isaac-GR00T.
        report["diffs"].append(compare("engine_vs_inprocess_serving", _truncate(srv_action), engine_action).as_json())
        report["pipeline"] = body.describe_pipeline()

    if native_policy is not None:
        report["dit_sdpa_mode"] = os.environ.get(DIT_SDPA_MODE_ENV)
        report["diagnostic"] = native_diagnostic(native_policy, srv_out, serving.policy.processor)

    return report


def evaluate_gate(report: dict[str, Any]) -> tuple[bool, list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply the gate's rule: every row bitwise except the tolerance list, identical processors, and two
    implementations (a reference that resolved to the port's class is not a reference).

    Returns ``(ok, over_tolerance_rows, not_bitwise_rows)``.
    """
    failed, exact_failed = evaluate(report["diffs"], TOLERANCES)
    ok = (
        not failed
        and not exact_failed
        and bool(report.get("processor_identical", False))
        and bool(report.get("class_provenance_ok", False))
    )
    return ok, failed, exact_failed


def render(report: dict[str, Any]) -> str:
    lines = [
        "# GR00T-N1.7 parity: vLLM-Omni serving vs Isaac-GR00T reference",
        "",
        f"checkpoint `{report['checkpoint']}` · tag `{report['embodiment_tag']}` · "
        f"B={report['batch']} · seed={report['seed']}"
        + (f" · weight_version={report['weight_version']}" if report.get("weight_version") is not None else ""),
        "",
        f"processor artifacts identical: **{report.get('processor_identical')}** "
        f"(reference `{report['processor_identity']['reference'].get('_model_name')}` vs "
        f"serving `{report['processor_identity']['serving'].get('_model_name')}`)",
        "",
    ]
    lines += render_diff_table(report["diffs"])
    diagnostic = report.get("diagnostic")
    if diagnostic is not None:
        precision = diagnostic["precision"]
        lines += [
            "",
            f"## Diagnostic D (report-only): native trainer forward ({precision['mode']}, master {precision['master']}, "
            f"{DIT_SDPA_MODE_ENV}={report.get('dit_sdpa_mode')!r}) vs serving",
            "",
        ]
        lines += render_diff_table(diagnostic["rows"])
    for note in report["notes"]:
        lines += ["", f"> {note}"]
    return "\n".join(lines)
