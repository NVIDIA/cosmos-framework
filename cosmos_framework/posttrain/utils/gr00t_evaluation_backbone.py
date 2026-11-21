# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""One-batch raw GR00T backbone reuse, before either native head transformation.

This state is evaluation-only and never enters a model state_dict or checkpoint.
Both calls still use the model's actual prepare_input and native action head.
"""

from __future__ import annotations

import weakref
from contextlib import contextmanager, nullcontext

MAX_RETAINED_BYTES = 16 * 1024**2
INPUT_KEYS = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
OUTPUT_KEYS = ("backbone_features", "backbone_attention_mask", "image_mask")


def reuse_identity(rl):
    return {
        "enabled": rl.reuse_gr00t_evaluation_backbone,
        "format": "gr00t-evaluation-raw-backbone-v1",
        "max_retained_bytes": MAX_RETAINED_BYTES,
    }


def validate_reuse(config, rl):
    if not rl.reuse_gr00t_evaluation_backbone:
        return
    prefix = "cosmos_framework.posttrain.model.gr00t.action_sft."
    expected = {
        "policy_factory": prefix + "build_policy",
        "data_factory": prefix + "build_dataset",
        "objective_factory": prefix + "build_objective",
        "evaluation_callback": prefix + "evaluate_actions",
    }
    adaptation = getattr(config.model, "backbone_adaptation", None)
    if (
        rl.micro_batch_size != 1
        or rl.reuse_evaluation_inputs
        or any(getattr(config, key, None) != value for key, value in expected.items())
        or adaptation is None
        or adaptation.mode not in {"nf4_lora", "bf16_lora"}
        or adaptation.profile != "gr00t_n1d7_reason2_2b_layer16_v1"
    ):
        raise ValueError("GR00T evaluation backbone reuse requires the pinned native NF4/BF16 LoRA one-row pipeline")


def _tensor_mapping(value, keys):
    import torch

    if set(value) != set(keys):
        raise ValueError("GR00T evaluation backbone tensor keys changed")
    if any(type(x) is not torch.Tensor or x.requires_grad or x.layout != torch.strided for x in value.values()):
        raise ValueError("GR00T evaluation backbone reuse requires detached dense tensors")
    return dict(value)


def _bytes(value):
    return sum(x.numel() * x.element_size() for x in value.values())


def _clone(value):
    import torch

    return {key: tensor.clone(memory_format=torch.contiguous_format) for key, tensor in value.items()}


def _prepared_inputs(prepared):
    if not all(key in prepared for key in INPUT_KEYS):
        raise ValueError("GR00T prepared backbone input keys are incomplete")
    selected = _tensor_mapping({key: prepared[key] for key in INPUT_KEYS}, INPUT_KEYS)
    ids = selected["input_ids"]
    if ids.ndim != 2 or ids.shape[0] != 1 or selected["attention_mask"].shape != ids.shape:
        raise ValueError("GR00T prepared backbone inputs require one padded row")
    return selected


def _equal(left, right):
    import torch

    return left.keys() == right.keys() and all(
        a.dtype == right[key].dtype
        and a.device == right[key].device
        and a.shape == right[key].shape
        and torch.equal(a, right[key])
        for key, a in left.items()
    )


class EvaluationBackboneReuse:
    """Own only raw clones between the loss and sampled prediction of one row."""

    def __init__(self, policy):
        self._model = weakref.ref(policy.gr00t)
        self.active = False
        self.inputs = self.raw = None
        self.calls = 0
        self.stats = {"misses": 0, "hits": 0, "retained_tensor_bytes_peak": 0}

    def require_evaluation(self):
        import torch

        model = self._model()
        if model is None:
            raise RuntimeError("GR00T evaluation backbone reuse owner has been released")
        if not self.active or torch.is_grad_enabled() or any(m.training for m in model.modules()):
            raise RuntimeError("GR00T raw backbone reuse is only valid inside a no-grad evaluation batch")

    @contextmanager
    def batch_scope(self, batch):
        if self.active:
            raise RuntimeError("nested GR00T evaluation backbone scopes are unsupported")
        if len(batch) != 1 or len(batch[0].transitions) != 1:
            raise ValueError("GR00T evaluation backbone reuse requires one single-transition row")
        self.active, self.calls = True, 0
        try:
            self.require_evaluation()
            yield
            if self.calls != 2 or self.raw is not None or self.inputs is not None:
                raise RuntimeError("GR00T evaluation must consume one loss and one prediction per batch")
        finally:
            self.inputs = self.raw = None
            self.active, self.calls = False, 0

    def begin_loss(self, prepared):
        self.require_evaluation()
        if self.calls or self.inputs is not None:
            raise RuntimeError("GR00T evaluation backbone loss must be the first call")
        # The pinned Qwen backbone consumes exactly these four keys. Native
        # prepare_input also carries action/state fields, which the sampler must
        # prepare independently without the demonstration's action/mask.
        selected = _prepared_inputs(prepared)
        if _bytes(selected) > MAX_RETAINED_BYTES:
            raise ValueError("GR00T evaluation backbone retained tensors exceed 16 MiB")
        self.inputs = _clone(selected)

    def finish_loss(self, prepared, raw):
        from transformers.feature_extraction_utils import BatchFeature

        self.require_evaluation()
        if self.inputs is None or self.calls:
            raise RuntimeError("GR00T evaluation backbone loss was not prepared")
        if not _equal(self.inputs, _prepared_inputs(prepared)):
            raise ValueError("native GR00T backbone mutated its prepared input")
        raw = _tensor_mapping(raw, OUTPUT_KEYS)
        import torch

        features = raw["backbone_features"]
        if (
            features.dtype != torch.bfloat16
            or features.ndim != 3
            or features.shape[:2] != self.inputs["input_ids"].shape
            or any(
                raw[key].dtype != torch.bool
                or raw[key].shape != features.shape[:2]
                or raw[key].device != features.device
                for key in ("backbone_attention_mask", "image_mask")
            )
        ):
            raise ValueError("GR00T raw backbone features/masks do not match the pinned BF16 one-row geometry")
        retained = _bytes(self.inputs) + _bytes(raw)
        if retained > MAX_RETAINED_BYTES:
            raise ValueError("GR00T evaluation backbone retained tensors exceed 16 MiB")
        # process_backbone_output replaces backbone_features in its mapping.
        # Clone both mapping AND tensors: no head owns the retained raw storage.
        self.raw = _clone(raw)
        self.calls = 1
        self.stats["misses"] += 1
        self.stats["retained_tensor_bytes_peak"] = max(self.stats["retained_tensor_bytes_peak"], retained)
        return BatchFeature(data=_clone(self.raw))

    def prediction(self, prepared):
        from transformers.feature_extraction_utils import BatchFeature

        self.require_evaluation()
        if self.calls != 1 or self.inputs is None or self.raw is None:
            raise RuntimeError("GR00T evaluation prediction requires the preceding loss")
        selected = _prepared_inputs(prepared)
        if not _equal(self.inputs, selected):
            raise ValueError("GR00T prediction prepared backbone inputs differ from the loss")
        result = BatchFeature(data=_clone(self.raw))
        self.inputs = self.raw = None
        self.calls = 2
        self.stats["hits"] += 1
        return result


def batch_scope(policy, batch):
    state = getattr(policy, "_evaluation_backbone_reuse", None)
    return state.batch_scope(batch) if state is not None else nullcontext()


def install_reuse(policy, config, rl):
    validate_reuse(config, rl)
    if not rl.reuse_gr00t_evaluation_backbone:
        return
    from cosmos_framework.posttrain.model.gr00t.action_sft import Gr00tActionPacker
    from cosmos_framework.posttrain.model.gr00t.policy import OFFICIAL_PRECISION, Gr00tN1d7TrainPolicy

    if (
        type(policy) is not Gr00tN1d7TrainPolicy
        or policy._ppo_enabled
        or policy.precision != OFFICIAL_PRECISION
        or policy.net is not policy.gr00t
        or not hasattr(policy, "family_state")
        or type(policy.packer) is not Gr00tActionPacker
    ):
        raise ValueError("GR00T evaluation backbone reuse requires the original native adapted policy and packer")
    if getattr(policy, "_evaluation_backbone_reuse", None) is not None:
        raise RuntimeError("GR00T evaluation backbone reuse was already configured")
    policy._evaluation_backbone_reuse = EvaluationBackboneReuse(policy)


def observations(policy):
    state = getattr(policy, "_evaluation_backbone_reuse", None)
    if state is None:
        return None
    if state.active or state.inputs is not None or state.raw is not None:
        raise RuntimeError("GR00T evaluation backbone state must clear before publication")
    return {"format": "gr00t-evaluation-raw-backbone-observations-v1", "scope": "current process only", **state.stats}
