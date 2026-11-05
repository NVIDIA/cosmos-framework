# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Independent-process OpenPI/reference instruments; numerical kernels remain production code."""

from __future__ import annotations

import contextlib
import random
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from cosmos_framework.posttrain.utils.object_store import file_sha256 as sha256


def prepare_request(output: Path, *, batch: int, seed: int, observations: Path | None) -> None:
    rng = np.random.default_rng(seed)
    if observations is None:
        raw = {
            "image": rng.integers(0, 256, (batch, 256, 256, 3), dtype=np.uint8),
            "wrist_image": rng.integers(0, 256, (batch, 256, 256, 3), dtype=np.uint8),
            "state": rng.uniform(-0.1, 0.1, (batch, 8)).astype(np.float32),
            "prompt": np.asarray(["put the_black bowl on the plate\n"] * batch),
        }
        origin = "synthetic numerical probe; no simulator success claim"
    else:
        with np.load(observations, allow_pickle=False) as source:
            raw = {key: source[key].copy() for key in ("image", "wrist_image", "state", "prompt")}
        batch = len(raw["state"])
        origin = f"external observations sha256:{sha256(observations)}"
    if batch < 1:
        raise ValueError("parity observations must contain at least one row")
    for key in ("image", "wrist_image"):
        if raw[key].ndim != 4 or raw[key].shape[0] != batch or raw[key].shape[-1] != 3 or raw[key].dtype != np.uint8:
            raise ValueError(f"{key} must be uint8 [B,H,W,3]")
    if raw["state"].shape != (batch, 8) or not np.isfinite(raw["state"]).all():
        raise ValueError("state must be finite [B,8]")
    if raw["prompt"].shape != (batch,) or raw["prompt"].dtype.kind != "U":
        raise ValueError("prompt must be a Unicode [B] array")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        **raw,
        noise=rng.standard_normal((4, batch, 10, 32), dtype=np.float32),
        seed=np.asarray(seed, dtype=np.int64),
        selected_step=np.asarray(random.Random(seed).randint(0, 2), dtype=np.int64),
        observation_origin=np.asarray(origin),
    )


def read_request(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        result = {key: source[key].copy() for key in source.files}
    batch = len(result["state"])
    if result["noise"].shape != (4, batch, 10, 32) or result["noise"].dtype != np.float32:
        raise ValueError("parity noise must be float32 [4,B,10,32]")
    if int(result["selected_step"]) != random.Random(int(result["seed"])).randint(0, 2):
        raise ValueError("request selected_step differs from its production sampler seed")
    return result


def observations(request: dict[str, np.ndarray]) -> list[dict]:
    return [
        {
            key: str(request[key][row]) if key == "prompt" else request[key][row]
            for key in ("image", "wrist_image", "state", "prompt")
        }
        for row in range(len(request["state"]))
    ]


def snapshot(tensors: dict[str, torch.Tensor], name: str, value: torch.Tensor) -> None:
    tensors[name] = value.detach().cpu().clone()


def record_inputs(tensors: dict[str, torch.Tensor], images, image_masks, tokens, token_mask, state) -> None:
    for index, image in enumerate(images):
        snapshot(tensors, f"input.image.{index}", image)
    snapshot(tensors, "input.image_masks", image_masks)
    snapshot(tensors, "input.tokens", tokens)
    snapshot(tensors, "input.token_mask", token_mask)
    snapshot(tensors, "input.state", state)


def record_prefix(tensors: dict[str, torch.Tensor], mask, cache, pooled) -> None:
    snapshot(tensors, "prefix.mask", mask)
    snapshot(tensors, "prefix.pooled", pooled)
    if len(cache) != 18:
        raise ValueError(f"expected 18 prefix layers, got {len(cache)}")
    for index, (key, value) in enumerate(cache):
        snapshot(tensors, f"prefix.{index}.key", key)
        snapshot(tensors, f"prefix.{index}.value", value)


class NoiseTape:
    """Replace only a sampler's Gaussian draws, and reject changes in draw count/shape."""

    def __init__(self, noise: np.ndarray) -> None:
        self.noise = noise
        self.calls = 0

    def draw(self, shape, device) -> torch.Tensor:
        if self.calls >= len(self.noise) or tuple(shape) != self.noise.shape[1:]:
            raise ValueError(f"unexpected parity noise draw {self.calls} with shape {tuple(shape)}")
        value = torch.as_tensor(self.noise[self.calls], device=device)
        self.calls += 1
        return value

    def normal(self, mean, std, *, size, device, dtype, generator=None) -> torch.Tensor:
        if mean != 0.0 or std != 1.0 or dtype != torch.float32:
            raise ValueError("parity sampler changed its standard Gaussian contract")
        return self.draw(size, device)

    def verify(self) -> None:
        if self.calls != len(self.noise):
            raise ValueError(f"expected {len(self.noise)} noise draws, observed {self.calls}")


@contextlib.contextmanager
def python_seed(seed: int):
    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)


def metadata(args, request: dict[str, np.ndarray], model: torch.nn.Module) -> dict:
    import transformers
    from safetensors import safe_open

    checkpoint = args.checkpoint / "model.safetensors"
    with safe_open(checkpoint, framework="pt", device="cpu") as source:
        if not any(name.startswith("value_head.") for name in source.keys()):
            raise ValueError("parity requires a checkpoint containing the shared initialized value_head")
    return {
        "phase": args.phase,
        "request_sha256": sha256(args.request),
        "checkpoint_sha256": sha256(checkpoint),
        "tokenizer_sha256": sha256(args.tokenizer),
        "norm_stats_sha256": sha256(args.norm_stats),
        "observation_origin": str(request["observation_origin"]),
        "seed": int(request["seed"]),
        "selected_step": int(request["selected_step"]),
        "torch": str(torch.__version__),
        "transformers": str(transformers.__version__),
        "device": torch.cuda.get_device_name(),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "parameter_dtypes": {name: str(value.dtype) for name, value in model.named_parameters()},
    }


def record_gradients(tensors: dict[str, torch.Tensor], model: torch.nn.Module, logprobs, values) -> None:
    # A fixed scalar probes one selected transition's full expert/value backward.
    objective = -logprobs.mean() + values.square().mean() / 2
    snapshot(tensors, "score.objective", objective)
    model.zero_grad(set_to_none=True)
    objective.backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            snapshot(tensors, f"gradient.{name}", parameter.grad)
    if not any(name.startswith("gradient.action_out_proj.") for name in tensors):
        raise RuntimeError("selected trace did not produce action expert gradients")
    if not any(name.startswith("gradient.value_head.") for name in tensors):
        raise RuntimeError("selected trace did not produce value gradients")


def load_reference_model(args):
    import openpi.models.tokenizer as tokenizer_module
    from omegaconf import OmegaConf
    from rlinf.models.embodiment.openpi import get_model

    cfg = OmegaConf.create(
        {
            "model_path": str(args.checkpoint),
            "openpi_data": {"norm_stats_path": str(args.norm_stats)},
            "openpi": {
                "config_name": "pi05_libero",
                "action_chunk": 5,
                "action_env_dim": 7,
                "num_steps": 3,
                "noise_level": 0.5,
                "noise_method": "flow_sde",
                "num_images_in_input": 2,
                "train_expert_only": True,
                "add_value_head": True,
                "value_after_vlm": True,
                "value_vlm_mode": "mean_token",
                "joint_logprob": False,
            },
        }
    )
    original_download = tokenizer_module.download.maybe_download

    def download(url, *positional, **kwargs):
        if str(url) == "gs://big_vision/paligemma_tokenizer.model":
            return args.tokenizer
        return original_download(url, *positional, **kwargs)

    with patch.object(tokenizer_module.download, "maybe_download", download):
        model = get_model(cfg).to(device="cuda").eval()
    return model


def run_reference(args, request: dict[str, np.ndarray]) -> dict:
    from rlinf.models.embodiment.openpi import openpi_action_model

    model = load_reference_model(args)
    tensors: dict[str, torch.Tensor] = {}
    original_preprocess = model._preprocess_observation
    original_prefix = model._build_prefix_cache
    original_velocity = model.get_velocity
    captured_prefix = None
    pooled = None
    velocity_calls = 0

    def preprocess(*positional, **kwargs):
        result = original_preprocess(*positional, **kwargs)
        images, masks, tokens, token_mask, state = result
        record_inputs(tensors, images, torch.stack(masks, dim=1), tokens, token_mask, state)
        return result

    def prefix(*positional, **kwargs):
        nonlocal captured_prefix
        result = original_prefix(*positional, **kwargs)
        captured_prefix = result
        return result

    def value_input(module, positional):
        nonlocal pooled
        pooled = positional[0]

    def velocity(*positional, **kwargs):
        nonlocal velocity_calls
        result = original_velocity(*positional, **kwargs)
        snapshot(tensors, f"velocity.{velocity_calls}", result[0])
        velocity_calls += 1
        return result

    tape = NoiseTape(request["noise"])
    env_obs = {
        "main_images": request["image"],
        "wrist_images": request["wrist_image"],
        "states": request["state"],
        "task_descriptions": request["prompt"].tolist(),
        "extra_view_images": None,
    }
    handle = model.value_head.register_forward_pre_hook(value_input)
    try:
        with (
            patch.object(model, "_preprocess_observation", preprocess),
            patch.object(model, "_build_prefix_cache", prefix),
            patch.object(model, "get_velocity", velocity),
            patch.object(model, "sample_noise", tape.draw),
            python_seed(int(request["seed"])),
            torch.no_grad(),
        ):
            decoded, sample = model.predict_action_batch(env_obs, mode="train", compute_values=True)
        tape.verify()
        if velocity_calls != 3 or captured_prefix is None or pooled is None:
            raise RuntimeError("reference did not expose all three denoising steps and prefix/value features")
        _, mask, cache = captured_prefix
        record_prefix(tensors, mask, tuple(zip(cache.key_cache, cache.value_cache, strict=True)), pooled)
    finally:
        handle.remove()
    inputs = sample["forward_inputs"]
    index = inputs["denoise_inds"][:, 0]
    if not (index == int(request["selected_step"])).all():
        raise ValueError("reference selected a different stochastic transition")
    rows = torch.arange(len(index), device=index.device)
    snapshot(tensors, "trace.before", inputs["chains"][rows, index])
    snapshot(tensors, "trace.after", inputs["chains"][rows, index + 1])
    snapshot(tensors, "trace.index", index)
    for name in ("before", "after", "index"):
        snapshot(tensors, f"score.trace.{name}", tensors[f"trace.{name}"])
    snapshot(tensors, "sample.normalized_actions", sample["model_actions"])
    snapshot(tensors, "sample.decoded_actions", decoded)
    snapshot(tensors, "sample.logprobs", sample["prev_logprobs"].float().sum(dim=(1, 2)))
    snapshot(tensors, "sample.values", sample["prev_values"].reshape(-1))
    score = model.default_forward(inputs, compute_values=True)
    logprobs, values = score["logprobs"].float().sum(dim=(1, 2)), score["values"].reshape(-1)
    snapshot(tensors, "score.logprobs", logprobs)
    snapshot(tensors, "score.values", values)
    record_gradients(tensors, model, logprobs, values)
    info = metadata(args, request, model)
    info["reference_model_source"] = str(Path(openpi_action_model.__file__).resolve())
    info["reference_model_source_sha256"] = sha256(Path(openpi_action_model.__file__))
    info["scoring_trace_origin"] = "reference sampler"
    return {"metadata": info, "tensors": tensors}


def run_posttrain(args, request: dict[str, np.ndarray]) -> dict:
    from cosmos_framework.posttrain.model.pi05.config import Pi05ModelConfig
    from cosmos_framework.posttrain.model.pi05.policy import Pi05TrainPolicy
    from cosmos_framework.posttrain.model.pi05.ppo import Pi05PPOTrace

    config = Pi05ModelConfig(str(args.checkpoint), str(args.tokenizer), str(args.norm_stats))
    policy = Pi05TrainPolicy(config, device="cuda")
    tensors: dict[str, torch.Tensor] = {}
    original_prepare = policy.processor.materialize
    original_velocity = policy.net.velocity
    velocity_calls = 0

    def prepare(*positional, **kwargs):
        result = original_prepare(*positional, **kwargs)
        record_inputs(tensors, result.images, result.image_masks, result.tokens, result.token_mask, result.state)
        return result

    def velocity(*positional, **kwargs):
        nonlocal velocity_calls
        result = original_velocity(*positional, **kwargs)
        snapshot(tensors, f"velocity.{velocity_calls}", result)
        velocity_calls += 1
        return result

    with patch.object(policy.processor, "materialize", prepare):
        condition = policy.ppo_condition(observations(request))
    record_prefix(tensors, condition.prefix_mask, condition.prefix_cache, condition.value_features)
    tape = NoiseTape(request["noise"])
    with patch.object(policy.net, "velocity", velocity), patch.object(torch, "normal", tape.normal):
        sample = policy.ppo_sample(condition, seed=int(request["seed"]))
    tape.verify()
    if velocity_calls != 3:
        raise RuntimeError(f"posttrain exposed {velocity_calls} denoising steps, expected three")
    snapshot(tensors, "trace.before", sample.trace.latent_before)
    snapshot(tensors, "trace.after", sample.trace.latent_after)
    snapshot(tensors, "trace.index", sample.trace.denoise_index)
    snapshot(tensors, "sample.normalized_actions", sample.normalized_actions)
    snapshot(tensors, "sample.decoded_actions", policy.processor.decode(sample.normalized_actions))
    snapshot(tensors, "sample.logprobs", sample.scores.logprobs)
    snapshot(tensors, "sample.values", sample.scores.values)
    trace = sample.trace
    trace_origin = "posttrain sampler"
    if args.reference_trace is not None:
        reference = torch.load(args.reference_trace, map_location="cpu", weights_only=True, mmap=True)
        if reference["metadata"]["request_sha256"] != sha256(args.request):
            raise ValueError("reference trace belongs to a different immutable request")
        saved = reference["tensors"]
        trace = Pi05PPOTrace(
            saved["trace.before"].to("cuda"), saved["trace.after"].to("cuda"), saved["trace.index"].to("cuda")
        )
        trace_origin = f"reference record sha256:{sha256(args.reference_trace)}"
    snapshot(tensors, "score.trace.before", trace.latent_before)
    snapshot(tensors, "score.trace.after", trace.latent_after)
    snapshot(tensors, "score.trace.index", trace.denoise_index)
    score = policy.ppo_score(condition, trace)
    snapshot(tensors, "score.logprobs", score.logprobs)
    snapshot(tensors, "score.values", score.values)
    record_gradients(tensors, policy.model, score.logprobs, score.values)
    info = metadata(args, request, policy.model)
    info["scoring_trace_origin"] = trace_origin
    return {"metadata": info, "tensors": tensors}


def compare_records(reference: dict, posttrain: dict) -> dict:
    identities = (
        "request_sha256",
        "checkpoint_sha256",
        "tokenizer_sha256",
        "norm_stats_sha256",
        "selected_step",
        "parameter_dtypes",
        "matmul_precision",
        "cuda_matmul_allow_tf32",
        "cudnn_allow_tf32",
    )
    for key in identities:
        if reference["metadata"][key] != posttrain["metadata"][key]:
            raise ValueError(f"parity source mismatch: {key}")
    left, right = reference["tensors"], posttrain["tensors"]
    if left.keys() != right.keys():
        raise ValueError(
            f"parity tensor inventory differs: reference-only={sorted(left.keys() - right.keys())}; "
            f"posttrain-only={sorted(right.keys() - left.keys())}"
        )
    rows = []
    for name in sorted(left):
        x, y = left[name], right[name]
        if x.shape != y.shape:
            raise ValueError(f"parity shape mismatch at {name}: {x.shape} vs {y.shape}")
        finite = bool(torch.isfinite(x).all() and torch.isfinite(y).all())
        difference = (x.double() - y.double()).abs()
        scale = torch.maximum(x.double().abs(), y.double().abs()).clamp_min(1e-12)
        rows.append(
            {
                "name": name,
                "shape": list(x.shape),
                "reference_dtype": str(x.dtype),
                "posttrain_dtype": str(y.dtype),
                "finite": finite,
                "exact": finite and x.dtype == y.dtype and torch.equal(x, y),
                "max_abs": float(difference.max()) if difference.numel() else 0.0,
                "max_rel": float((difference / scale).max()) if difference.numel() else 0.0,
            }
        )
    return {
        "passed": all(row["exact"] for row in rows),
        "rows": rows,
        "reference": reference["metadata"],
        "posttrain": posttrain["metadata"],
    }
