# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Full-prefix replay qualification; fingerprints never serialize gradients or model weights."""

import hashlib
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from unittest.mock import patch

import numpy as np
import torch

from cosmos_framework.posttrain.algorithm.joint_ppo.objective import make_joint_objective
from cosmos_framework.posttrain.config.build import OptimizerParameterGroup, build_optimizer
from cosmos_framework.posttrain.model.pi05.records import TRACE_KEY
from cosmos_framework.posttrain.model.protocol import PPODecisionScores
from cosmos_framework.posttrain.tests.helpers.pi05_prepared import parity_recipe
from cosmos_framework.posttrain.utils.dist import run_rank_converged


def fingerprint(tensor, *, name=None):
    value = tensor.detach().cpu().contiguous()
    if not bool(torch.isfinite(value).all()):
        error = ValueError("qualification encountered a nonfinite tensor")
        if name is not None:
            error.add_note(name)
        raise error
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "stride": list(tensor.stride()),
        "sha256": hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy()).hexdigest(),
    }


def record_inputs(tensors, inputs):
    for index, image in enumerate(inputs.images):
        item = fingerprint(image)
        if index < 2:
            palette = torch.arange(256, device=image.device, dtype=torch.float32) / 255.0 * 2.0 - 1.0
            pixels = ((image + 1.0) * 127.5).round().to(torch.uint8)
            if not torch.equal(image, palette[pixels.long()]):
                raise ValueError("image preprocessing does not match its device's exhaustive uint8 palette")
            item.update(
                uint8_sha256=fingerprint(pixels)["sha256"],
                palette=palette.cpu().tolist(),
                used_values=pixels.unique().cpu().tolist(),
            )
        tensors[f"inputs.image.{index}"] = item
    for name in ("image_masks", "tokens", "token_mask", "state"):
        tensors[f"inputs.{name}"] = fingerprint(getattr(inputs, name))


def record_prefix(tensors, mask, cache, features):
    tensors["prefix.mask"] = fingerprint(mask, name="prefix.mask")
    tensors["prefix.features"] = fingerprint(features, name="prefix.features")
    if len(cache) != 18:
        raise ValueError("qualification requires all 18 prefix KV layers")
    for layer, (key, value) in enumerate(cache):
        tensors[f"prefix.{layer}.key"] = fingerprint(key, name=f"prefix.{layer}.key")
        tensors[f"prefix.{layer}.value"] = fingerprint(value, name=f"prefix.{layer}.value")


@contextmanager
def capture_local(policy, tensors):
    materialize, condition, velocity = policy.processor.materialize, policy.model.prepare_condition, policy.net.velocity
    score = policy.score_ppo_decisions

    def inputs(*args, **kwargs):
        result = materialize(*args, **kwargs)
        record_inputs(tensors, result)
        return result

    def prefix(*args, **kwargs):
        result = condition(*args, **kwargs)
        record_prefix(tensors, result.prefix_mask, result.prefix_cache, result.value_features)
        return result

    def predict(*args, **kwargs):
        result = velocity(*args, **kwargs)
        tensors["velocity"] = fingerprint(result)
        return result

    def scored(*args, **kwargs):
        result = score(*args, **kwargs)
        tensors["logprobs"], tensors["values"] = fingerprint(result.logprobs), fingerprint(result.values)
        return result

    with (
        patch.object(policy.processor, "materialize", inputs),
        patch.object(policy.model, "prepare_condition", prefix),
        patch.object(policy.net, "velocity", predict),
        patch.object(policy, "score_ppo_decisions", scored),
    ):
        yield


class ReferenceReplayPolicy:
    """Run RLinf's real default_forward from raw observations, rebuilding its full VLM prefix."""

    def __init__(self, model, samples, tensors):
        self.net, self.tensors = model, tensors
        self.ppo_padding = samples[0].transitions[0]

    def validate_ppo_record(self, step):
        # The common objective validates source/stats/targets; reference owns its raw-input validation.
        trace = step.policy_info[TRACE_KEY]
        row = step.policy_info["sample_index"]
        if type(row) is not int or not 0 <= row < len(trace.latent_before):
            raise ValueError("invalid original invocation row")

    def score_ppo_decisions(self, steps):
        from types import SimpleNamespace

        def prepare():
            for step in steps:
                self.validate_ppo_record(step)
            raw = [step.obs.representations["raw"] for step in steps]
            env_obs = {
                "main_images": np.stack([row["image"] for row in raw]),
                "wrist_images": np.stack([row["wrist_image"] for row in raw]),
                "states": np.stack([row["state"] for row in raw]),
                "task_descriptions": [row["prompt"] for row in raw],
                "extra_view_images": None,
            }
            traces = [step.policy_info[TRACE_KEY] for step in steps]
            rows = [step.policy_info["sample_index"] for step in steps]
            indices = torch.stack([trace.denoise_index[row] for trace, row in zip(traces, rows, strict=True)]).to(
                "cuda"
            )
            chains = torch.full((len(steps), 4, 10, 32), float("nan"), device="cuda")
            batch = torch.arange(len(steps), device="cuda")
            chains[batch, indices] = torch.stack(
                [trace.latent_before[row] for trace, row in zip(traces, rows, strict=True)]
            ).to("cuda")
            chains[batch, indices + 1] = torch.stack(
                [trace.latent_after[row] for trace, row in zip(traces, rows, strict=True)]
            ).to("cuda")
            return env_obs, indices, chains

        env_obs, indices, chains = run_rank_converged(prepare, "RLinf batch replay inputs")
        # Use the actual rollout producer, then replace only the selected
        # denoising transition. Its observation/token device contract is native.
        with torch.no_grad(), torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            _, sampled = self.net.predict_action_batch(env_obs, mode="train", compute_values=True)
        inputs = sampled["forward_inputs"]
        inputs.update(chains=chains, denoise_inds=indices[:, None])
        preprocess, prefix, velocity = (
            self.net._preprocess_observation,
            self.net._build_prefix_cache,
            self.net.get_velocity,
        )
        cache_info = {}

        def prepared(*args, **kwargs):
            result = preprocess(*args, **kwargs)
            images, masks, tokens, mask, state = result
            record_inputs(
                self.tensors,
                SimpleNamespace(
                    images=images, image_masks=torch.stack(masks, 1), tokens=tokens, token_mask=mask, state=state
                ),
            )
            return result

        def prefixed(*args, **kwargs):
            result = prefix(*args, **kwargs)
            cache_info["mask"], cache_info["cache"] = result[1:]
            return result

        def predicted(*args, **kwargs):
            result = velocity(*args, **kwargs)
            self.tensors["velocity"] = fingerprint(result[0])
            return result

        def value_inputs(_module, args):
            cache = cache_info["cache"]
            record_prefix(
                self.tensors, cache_info["mask"], tuple(zip(cache.key_cache, cache.value_cache, strict=True)), args[0]
            )

        with ExitStack() as stack:
            stack.enter_context(patch.object(self.net, "_preprocess_observation", prepared))
            stack.enter_context(patch.object(self.net, "_build_prefix_cache", prefixed))
            stack.enter_context(patch.object(self.net, "get_velocity", predicted))
            handle = self.net.value_head.register_forward_pre_hook(value_inputs)
            stack.callback(handle.remove)
            result = self.net.default_forward(inputs, compute_values=True)
        logprobs = result["logprobs"].float().sum((1, 2))
        values = result["values"].reshape(-1)
        self.tensors["logprobs"], self.tensors["values"] = fingerprint(logprobs), fingerprint(values)
        return PPODecisionScores(logprobs, values, (logprobs.sum() + values.sum()) * 0)


def qualify_update(policy, samples, *, reference=False):
    recipe = parity_recipe()
    policy.ppo_padding = samples[0].transitions[0]
    objective = make_joint_objective(
        policy_clip=recipe.policy_clip,
        value_clip=recipe.value_clip,
        dual_clip=recipe.dual_clip,
        huber_delta=recipe.huber_delta,
        identity=recipe.identity(),
        loss_reduction=recipe.loss_reduction,
        microbatch=16,
    )
    parameters = {name: value for name, value in policy.net.named_parameters() if value.requires_grad}
    actor = tuple(name for name in parameters if not name.startswith("value_head."))
    value = tuple(name for name in parameters if name.startswith("value_head."))
    optimizer = build_optimizer(
        policy,
        recipe.optimizer,
        parameter_groups=(
            OptimizerParameterGroup("actor", actor, recipe.actor_lr),
            OptimizerParameterGroup("value", value, recipe.value_lr),
        ),
    )
    loss, metrics = objective(policy, samples)
    loss.backward()
    missing = [name for name, value in parameters.items() if value.grad is None]
    unused = "paligemma_with_expert.gemma_expert.lm_head.weight"
    if missing != ([unused] if reference else []):
        raise ValueError(f"unexpected missing gradients: {missing}")

    def named(name):
        return name.replace("paligemma_with_expert.gemma_expert.model.", "expert.", 1)

    report = {
        "optimizer": asdict(recipe.optimizer),
        "actor_lr": recipe.actor_lr,
        "value_lr": recipe.value_lr,
        "loss": float(loss.detach()),
        "metrics": {key: float(value) for key, value in metrics.items()},
        "unused_reference_parameters": missing,
        "gradients": {
            named(name): fingerprint(value.grad) for name, value in parameters.items() if name not in missing
        },
    }
    if len(report["gradients"]) != 216:
        raise ValueError("qualification must cover exactly 216 trained parameter gradients")
    before_unused = fingerprint(parameters[unused]) if reference else None
    report["gradient_norm"] = float(
        torch.nn.utils.clip_grad_norm_(list(parameters.values()), recipe.optimizer.grad_clip_norm)
    )
    optimizer.step()
    report["parameters"] = {
        named(name): fingerprint(value) for name, value in parameters.items() if name not in missing
    }
    if reference:
        report["unused_head_unchanged"] = before_unused == fingerprint(parameters[unused])
    return report
