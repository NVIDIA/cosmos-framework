# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Synchronous Ray-colocated Cosmos3 online DiffusionNFT entrypoint.

The production v1 topology is deliberately narrow: one Ray ``ActorGroup`` owns
source, rollout, reward, replay, and trainer bodies in every GPU process. Rollout
and trainer share one ``Cosmos3FlowPolicy`` instance, so an applied update is visible
to the next UniPC rollout without a second model or a weight-transfer edge.

Run the 8B recipe on one four-GPU node::

    PYTHONPATH=. python -m cosmos_framework.posttrain.entrypoints.nft -- \
      experiment=cosmos3_nft_8b_online_v1 \
      rl.reward.provider_revision=<immutable-service-revision>

Use ``--validate`` for a GPU-less manifest/config/environment preflight. The live reward
client reads ``REMOTE_REWARD_ENQUEUE_URL``, ``REMOTE_REWARD_FETCH_URL``, and
``REMOTE_REWARD_TOKEN`` from the environment.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable, Mapping
from typing import Any, Literal, cast

from cosmos_framework.posttrain.config.experiment.nft import NFT_T2I_PROFILES
from cosmos_framework.posttrain.config.rl import VIDEO_REWARD_REVISION_REQUIRED, ImageRewardConfig
from cosmos_framework.posttrain.utils.entry import cli, dump_resolved, render_manifest_validation, resolve
from cosmos_framework.posttrain.utils.entry import out_dir as _out_dir
from cosmos_framework.posttrain.utils.log import info0, logger

ROLE_SPECS = [
    ("data", "env"),
    ("rollout", "rollout"),
    ("rm", "reward"),
    ("replay", "buffer"),
    ("policy", "trainer"),
]
_REWARD_ENVIRONMENT = (
    "REMOTE_REWARD_ENQUEUE_URL",
    "REMOTE_REWARD_FETCH_URL",
    "REMOTE_REWARD_TOKEN",
)
_WFM_SEED_STRIDE = 10_000
_NFT_CONSOLE_KEYS = (
    "train/loss_avg",
    "train/policy_loss",
    "train/kl_loss",
    "rollout/reward_group_mean",
)
_NFT_ADVANTAGE_SUM_KEYS = frozenset(
    {
        "rollout/reward_group_informative",
        "rollout/reward_group_all_zero",
    }
)
_NFT_ADVANTAGE_MEAN_KEYS = frozenset(
    {
        "rollout/reward_group_mean",
        "rollout/reward_group_spread",
        "rollout/reward_nonzero_fraction",
        "rollout/advantage_group_std",
    }
)
_NFT_ADVANTAGE_MAX_KEYS = frozenset({"rollout/advantage_abs_max"})
_NFT_ADVANTAGE_IDENTICAL_KEYS = frozenset({"rollout/reward_group_size"})
_NFT_ADVANTAGE_KEYS = (
    _NFT_ADVANTAGE_SUM_KEYS | _NFT_ADVANTAGE_MEAN_KEYS | _NFT_ADVANTAGE_MAX_KEYS | _NFT_ADVANTAGE_IDENTICAL_KEYS
)


class RejectedNFTUpdate(RuntimeError):
    """The finite-update gate rejected an online NFT optimizer attempt."""

    def __init__(self, metrics: dict[str, Any]) -> None:
        super().__init__(str(metrics.get("optim/skip_reason", "nonfinite")))
        self.metrics = dict(metrics)


class _IdentifiedLoader:
    """Attach a resolved-config fingerprint to a structural streaming cursor."""

    def __init__(self, loader: Any, fingerprint: str) -> None:
        if not fingerprint:
            raise ValueError("online NFT source fingerprint must be non-empty")
        self.loader = loader
        self.fingerprint = fingerprint

    def __iter__(self) -> Any:
        return iter(self.loader)

    def set_start_iteration(self, iteration: int) -> None:
        setter = getattr(self.loader, "set_start_iteration", None)
        if not callable(setter):
            raise ValueError("online NFT dataloader must implement set_start_iteration()")
        setter(iteration)

    def cursor_fingerprint(self) -> str:
        return self.fingerprint

    def close(self) -> None:
        close = getattr(self.loader, "close", None)
        if callable(close):
            close()


def _source_fingerprint(dataloader_config: Any) -> str:
    """Hash the fully resolved dataloader declaration used for structural continuation."""
    from cosmos_framework.posttrain.utils.identity import canonical_config_value, canonical_json_sha256

    normalized = canonical_config_value(dataloader_config)
    digest = canonical_json_sha256(normalized, context="online NFT dataloader configuration")
    return f"config-sha256:{digest}"


def _all_execution_values(value: Any, *, operation: str) -> list[Any]:
    """Resolve every local cell and every torchrun rank into one ordered inventory."""
    from cosmos_framework.posttrain.utils.entry import resolve_all
    from cosmos_framework.posttrain.utils.rank_state import gather_rank_states

    local_values = resolve_all(value)
    gathered = gather_rank_states(local_values, context=f"{operation} preparation")
    return [item for rank in sorted(gathered) for item in gathered[rank]]


def _identical_metric_frame(value: Any, *, operation: str) -> dict[str, Any]:
    """Require one primitive metric frame from every participating execution cell."""
    from cosmos_framework.posttrain.utils.dist import run_rank_converged
    from cosmos_framework.posttrain.utils.entry import resolve_all
    from cosmos_framework.posttrain.utils.rank_state import gather_rank_states
    from cosmos_framework.posttrain.utils.report import reportable_metrics

    def prepare_local_frames() -> list[dict[str, Any]]:
        frames: list[dict[str, Any]] = []
        for raw in resolve_all(value):
            if not isinstance(raw, dict):
                raise TypeError(f"{operation} must return metric dictionaries, got {type(raw).__name__}")
            frames.append(dict(reportable_metrics(raw)))
        return frames

    local_frames = run_rank_converged(prepare_local_frames, f"{operation} metric scalarization")
    gathered = gather_rank_states(local_frames, context=f"{operation} metric-frame gathering")
    frames = [frame for rank in sorted(gathered) for frame in gathered[rank]]
    if not frames:
        raise RuntimeError(f"{operation} returned no metric frames")
    first = frames[0]
    if any(frame != first for frame in frames[1:]):
        raise RuntimeError(f"{operation} returned inconsistent metric frames: {frames!r}")
    return first


def _aggregate_advantage_window(rows: list[dict[str, Any]]) -> dict[str, int | float]:
    """Reduce the declared complete-group WFM telemetry across accumulation rows."""
    if not rows:
        raise ValueError("cannot aggregate an empty NFT advantage window")
    for index, row in enumerate(rows):
        if set(row) != _NFT_ADVANTAGE_KEYS:
            raise ValueError(
                "NFT advantage telemetry must expose the exact declared schema; "
                f"row {index} has {sorted(row)}, expected {sorted(_NFT_ADVANTAGE_KEYS)}"
            )

    result: dict[str, int | float] = {}
    for key in _NFT_ADVANTAGE_SUM_KEYS:
        values = [row[key] for row in rows]
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
            raise ValueError(f"NFT advantage count {key!r} must contain non-negative integers, got {values!r}")
        result[key] = sum(values)
    for key in _NFT_ADVANTAGE_MEAN_KEYS:
        values = [row[key] for row in rows]
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            raise TypeError(f"NFT advantage metric {key!r} must be numeric, got {values!r}")
        normalized = [float(value) for value in values]
        if any(not math.isfinite(value) for value in normalized):
            raise ValueError(f"NFT advantage metric {key!r} must be finite, got {normalized!r}")
        result[key] = sum(normalized) / len(normalized)
    for key in _NFT_ADVANTAGE_MAX_KEYS:
        values = [row[key] for row in rows]
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            raise TypeError(f"NFT advantage metric {key!r} must be numeric, got {values!r}")
        normalized = [float(value) for value in values]
        if any(not math.isfinite(value) for value in normalized):
            raise ValueError(f"NFT advantage metric {key!r} must be finite, got {normalized!r}")
        result[key] = max(normalized)
    for key in _NFT_ADVANTAGE_IDENTICAL_KEYS:
        values = [row[key] for row in rows]
        first = values[0]
        if any(type(value) is not type(first) or value != first for value in values[1:]):
            raise ValueError(f"NFT advantage metric {key!r} must be identical, got {values!r}")
        if isinstance(first, bool) or not isinstance(first, int) or first < 1:
            raise ValueError(f"NFT advantage metric {key!r} must be a positive integer, got {first!r}")
        result[key] = first
    return result


def _provider_reports(value: Any, *, operation: str) -> list[Any]:
    """Resolve and validate serializable provider-only reports from every execution cell."""
    from cosmos_framework.posttrain.roles.reward import ProviderMetricReport

    reports = _all_execution_values(value, operation=operation)
    if not reports:
        raise RuntimeError(f"{operation} returned no provider reports")
    if any(not isinstance(report, ProviderMetricReport) for report in reports):
        raise TypeError(f"{operation} must return ProviderMetricReport from every cell, got {reports!r}")
    provenance = reports[0].provenance
    if any(report.provenance != provenance for report in reports[1:]):
        raise RuntimeError(f"{operation} returned inconsistent provider provenance: {reports!r}")
    return reports


def declare_wgs(
    ranks: int,
    *,
    shapes: dict[str, tuple[int, int]] | None = None,
    policy_parallelism: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Declare the five role bodies with explicit execution shapes and NFT dispatch."""
    from cosmos_framework.posttrain.roles import roles as role_factory

    def _shape(name: str) -> tuple[int, int]:
        return shapes[name] if shapes is not None else (1, ranks)

    def _kwargs(name: str, kind: str) -> dict[str, Any]:
        replicas, role_ranks = _shape(name)
        kwargs: dict[str, Any] = {
            "replicas": replicas,
            "ranks": role_ranks,
            "dispatch_profile": "nft",
        }
        if kind == "reward":
            # Native WFM reward providers decode rollout-owned latent tensors.
            kwargs["needs_bulk"] = True
        if kind == "trainer":
            kwargs["model_parallelism"] = policy_parallelism
        return kwargs

    return {name: getattr(role_factory, kind)(name, **_kwargs(name, kind)) for name, kind in ROLE_SPECS}


def _nested_value(root: Any, *path: str) -> Any:
    value = root
    for name in path:
        value = value.get(name) if isinstance(value, dict) else getattr(value, name, None)
        if value is None:
            break
    return value


def _matches_number(value: Any, expected: float) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return float(value) == expected
    except (TypeError, ValueError):
        return False


def _is_sha256(value: Any) -> bool:
    """Return whether ``value`` is one explicit lowercase hexadecimal SHA-256 digest."""
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _nft_role_shapes(config: Any, group_size: int) -> dict[str, tuple[int, int]]:
    """Resolve the uniform one-member-per-rank WFM shape at the config boundary."""
    raw_shapes = _nested_value(config, "grid", "shapes")
    if raw_shapes:
        try:
            policy_shape = tuple(int(axis) for axis in raw_shapes["policy"])
            rollout_shape = tuple(int(axis) for axis in raw_shapes["rollout"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"grid.shapes must declare integer policy and rollout [R,K] pairs: {error}") from error
        if len(policy_shape) != 2 or len(rollout_shape) != 2:
            raise ValueError(
                f"grid.shapes policy and rollout entries must be [R,K], got {policy_shape} and {rollout_shape}"
            )
    else:
        replicas = _nested_value(config, "grid", "num_replicas")
        ranks = _nested_value(config, "grid", "ranks")
        if isinstance(replicas, bool) or not isinstance(replicas, int) or replicas < 1:
            raise ValueError(f"grid.num_replicas must be a positive integer, got {replicas!r}")
        if isinstance(ranks, bool) or not isinstance(ranks, int) or ranks < 1:
            raise ValueError(f"grid.ranks must be a positive integer, got {ranks!r}")
        policy_shape = (1, ranks)
        rollout_shape = (replicas, ranks)
    expected = (1, group_size)
    if policy_shape != expected or rollout_shape != expected:
        raise ValueError(
            "online NFT currently requires one execution replica and one WFM member per model rank: "
            f"policy={policy_shape}, rollout={rollout_shape}, expected={expected}"
        )
    return {name: expected for name, _kind in ROLE_SPECS}


def validate_nft_runtime_config(
    config: Any,
    rl: Any,
    *,
    environ: Mapping[str, str] | None = None,
) -> list[str]:
    """Return GPU-less configuration/preflight errors without reading secrets."""
    errors: list[str] = []
    if str(_nested_value(config, "grid", "backend")) != "ray":
        errors.append("production online NFT requires grid.backend='ray'")
    if str(_nested_value(config, "grid", "grouping")) != "single":
        errors.append("grid.grouping must be 'single'")
    if str(_nested_value(config, "grid", "gpu_placement")) != "colocated":
        errors.append("grid.gpu_placement must be 'colocated'")
    if _nested_value(config, "grid", "gpus_per_role") not in ({}, None):
        errors.append("colocated online NFT does not accept grid.gpus_per_role overrides")
    try:
        role_shapes = _nft_role_shapes(config, int(rl.group_size))
    except (TypeError, ValueError) as error:
        errors.append(f"online NFT role topology is invalid: {error}")
        role_shapes = None
    modality = str(rl.modality)
    profile_name = str(rl.profile)
    image_profile = NFT_T2I_PROFILES.get(profile_name) if modality == "image" else None
    if modality == "video" and profile_name:
        errors.append("video NFT uses the native unprofiled contract; rl.profile must be empty")
    elif modality == "image" and image_profile is None:
        supported = ", ".join(sorted(NFT_T2I_PROFILES))
        errors.append(
            f"image NFT requires an explicit supported rl.profile; got {profile_name!r}; supported profiles: {supported}"
        )
    if _nested_value(config, "dataloader_train") is None:
        errors.append("dataloader_train must be configured")
    elif modality == "video":
        max_sequence_length = _nested_value(config, "dataloader_train", "max_sequence_length")
        max_samples_per_batch = _nested_value(config, "dataloader_train", "max_samples_per_batch")
        if max_sequence_length is not None:
            errors.append("online NFT dataloader_train.max_sequence_length must be null")
        if max_samples_per_batch != 1:
            errors.append("online NFT dataloader_train.max_samples_per_batch must be 1")
        if list(rl.init_keys_to_skip_loading):
            errors.append("video NFT does not support rl.init_keys_to_skip_loading")
        if str(rl.rollout_seed_clock) != "source_row":
            errors.append("video NFT requires rl.rollout_seed_clock='source_row'")
        if isinstance(rl.reward, ImageRewardConfig):
            errors.append("video NFT requires the decoded-video reward configuration")
    if not str(rl.init_from).strip():
        errors.append("rl.init_from must name the base checkpoint")
    provider_revision = str(rl.reward.provider_revision).strip()
    if not provider_revision or provider_revision == VIDEO_REWARD_REVISION_REQUIRED:
        errors.append(
            "rl.reward.provider_revision must identify the immutable deployed scorer contract; "
            f"override {VIDEO_REWARD_REVISION_REQUIRED!r} before launch"
        )
    if not bool(rl.old.build_old) or int(rl.old.update_every) < 1:
        errors.append("online NFT requires a moving old policy with rl.old.build_old=true and update_every>=1")
    if int(rl.ref.update_every) != 0:
        errors.append("online NFT keeps the KL reference frozen; rl.ref.update_every must be 0")
    scheduler_type = _nested_value(config, "model", "config", "rectified_flow_inference_config", "scheduler_type")
    if str(scheduler_type).lower() != "unipc":
        errors.append("model.config.rectified_flow_inference_config.scheduler_type must be 'unipc'")
    base_shift = _nested_value(config, "model", "config", "rectified_flow_inference_config", "shift")
    if not _matches_number(base_shift, 1.0):
        errors.append("native UniPC base shift must be 1.0; rl.sampler_shift applies the runtime shift once")
    training_shift = _nested_value(config, "model", "config", "rectified_flow_training_config", "shift")
    expected_training_shift = (
        1.0 if modality == "video" else image_profile.training_shift if image_profile is not None else None
    )
    if expected_training_shift is not None and (
        isinstance(training_shift, bool)
        or not isinstance(training_shift, int)
        or training_shift != int(expected_training_shift)
    ):
        errors.append(
            "model.config.rectified_flow_training_config.shift must be the integer "
            f"{int(expected_training_shift)} for modality={modality!r}; float scalars are not accepted by OmniMoT"
        )
    compile_enabled = _nested_value(config, "model", "config", "compile", "enabled")
    expected_compile_enabled = (
        False if modality == "video" else image_profile.compile_enabled if image_profile is not None else None
    )
    if expected_compile_enabled is not None and compile_enabled is not expected_compile_enabled:
        errors.append("model.config.compile.enabled must be false for the selected NFT recipe")
    latent_side = _nested_value(
        config,
        "model",
        "config",
        "diffusion_expert_config",
        "max_vae_latent_side_after_patchify",
    )
    expected_latent_side = (
        45
        if modality == "video"
        else image_profile.max_vae_latent_side_after_patchify
        if image_profile is not None
        else None
    )
    if expected_latent_side is not None and latent_side != expected_latent_side:
        errors.append(
            "model.config.diffusion_expert_config.max_vae_latent_side_after_patchify must be "
            f"{expected_latent_side} for modality={modality!r}"
        )
    ema_enabled = _nested_value(config, "model", "config", "ema", "enabled")
    ema_rate = _nested_value(config, "model", "config", "ema", "rate")
    ema_iteration_shift = _nested_value(config, "model", "config", "ema", "iteration_shift")
    if modality == "video" and (ema_enabled is not True or ema_rate != 0.1 or ema_iteration_shift != 0):
        errors.append("video NFT must enable power EMA with rate=0.1 and iteration_shift=0")
    if image_profile is not None and ema_enabled is not image_profile.ema_enabled:
        errors.append("the matched T2I HPSv3 recipe disables model EMA")
    if modality == "image":
        if not isinstance(rl.reward, ImageRewardConfig):
            errors.append("image NFT requires ImageRewardConfig")
    if image_profile is not None:
        from cosmos_framework.posttrain.data.datasets import NFTPromptImageDataLoader

        if (
            str(rl.reward.provider_name) != image_profile.provider_name
            or str(rl.reward.score_key) != image_profile.score_key
        ):
            errors.append("image NFT currently requires the HPSv3 image reward contract")
        source_coordinates = {
            "prompt_key": _nested_value(config, "dataloader_train", "prompt_key"),
            "expected_rows": _nested_value(config, "dataloader_train", "expected_rows"),
            "height": _nested_value(config, "dataloader_train", "height"),
            "width": _nested_value(config, "dataloader_train", "width"),
            "fps": _nested_value(config, "dataloader_train", "fps"),
            "shuffle": _nested_value(config, "dataloader_train", "shuffle"),
            "shuffle_seed": _nested_value(config, "dataloader_train", "shuffle_seed"),
            "num_replicas": _nested_value(config, "dataloader_train", "num_replicas"),
            "source_rank": _nested_value(config, "dataloader_train", "source_rank"),
            "drop_last": _nested_value(config, "dataloader_train", "drop_last"),
        }
        prompt_sha = _nested_value(config, "dataloader_train", "expected_sha256")
        source_target = _nested_value(config, "dataloader_train", "_target_")
        if (
            source_target is not NFTPromptImageDataLoader
            or not _is_sha256(prompt_sha)
            or source_coordinates
            != {
                "prompt_key": image_profile.prompt_key,
                "expected_rows": image_profile.expected_rows,
                "height": image_profile.height,
                "width": image_profile.width,
                "fps": image_profile.fps,
                "shuffle": image_profile.shuffle,
                "shuffle_seed": image_profile.shuffle_seed,
                "num_replicas": image_profile.num_replicas,
                "source_rank": image_profile.source_rank,
                "drop_last": image_profile.drop_last,
            }
        ):
            errors.append(
                "the matched T2I HPSv3 source requires NFTPromptImageDataLoader, an explicit lowercase SHA-256, "
                "10,000 rows, 720x720 at 24 fps, and rank-zero DistributedSampler(seed=0, replicas=8) ordering"
            )
        if (
            int(rl.group_size) != image_profile.group_size
            or int(rl.grad_accum) != image_profile.grad_accum
            or int(rl.sample_steps) != image_profile.sample_steps
            or not _matches_number(rl.sampler_shift, image_profile.sampler_shift)
            or str(_nested_value(config, "model", "config", "resolution")) != image_profile.resolution
            or _nested_value(config, "model", "config", "state_t") != image_profile.state_t
        ):
            errors.append(
                "the matched T2I HPSv3 execution profile requires group_size=8, grad_accum=4, "
                "20 rollout steps, shift 5, resolution 720, and state_t=24"
            )
        model_config = _nested_value(config, "model", "config")
        if (
            _nested_value(model_config, "sound_gen") is not image_profile.sound_gen
            or _nested_value(model_config, "action_gen") is not image_profile.action_gen
            or _nested_value(model_config, "lora_enabled") is not image_profile.lora_enabled
            or _nested_value(model_config, "lora_rank") != image_profile.lora_rank
            or _nested_value(model_config, "lora_alpha") != image_profile.lora_alpha
            or str(_nested_value(model_config, "lora_target_modules")) != image_profile.lora_target_modules
        ):
            errors.append(
                "the matched T2I HPSv3 model profile disables sound/action and requires rank-32, "
                "alpha-64 LoRA on generation-MoE q/k/v/o projections"
            )
        if (
            not _matches_number(rl.beta, image_profile.beta)
            or not _matches_number(rl.kl_beta, image_profile.kl_beta)
            or not _matches_number(rl.policy_loss_scale, image_profile.policy_loss_scale)
            or bool(rl.ref.build_ref) is not image_profile.build_ref
            or int(rl.old.update_every) != image_profile.old_update_every
            or not _matches_number(rl.advantage_low, image_profile.advantage_low)
            or not _matches_number(rl.advantage_high, image_profile.advantage_high)
            or not _matches_number(rl.advantage_eps, image_profile.advantage_eps)
            or str(rl.r_map) != image_profile.r_map
            or not _matches_number(rl.r_tau, image_profile.r_tau)
            or bool(rl.uniform_timestep) is not image_profile.uniform_timestep
        ):
            errors.append(
                "the matched T2I HPSv3 objective/continuation profile requires beta=1, kl_beta=0, "
                "policy loss scale 5, sigmoid tau=0.3, bounds [-5,5], eps=1e-4, uniform RF time, "
                "no frozen reference, and moving-old refresh every 10 applied updates"
            )
        optimizer = rl.optimizer
        if (
            str(optimizer.name) != image_profile.optimizer_name
            or not _matches_number(optimizer.lr, image_profile.optimizer_lr)
            or list(optimizer.betas) != list(image_profile.optimizer_betas)
            or not _matches_number(optimizer.eps, image_profile.optimizer_eps)
            or not _matches_number(optimizer.weight_decay, image_profile.optimizer_weight_decay)
            or list(optimizer.train_keys) != list(image_profile.optimizer_train_keys)
            or not _matches_number(optimizer.grad_clip_norm, image_profile.optimizer_grad_clip_norm)
            or bool(optimizer.force_finite_grads) is not image_profile.optimizer_force_finite_grads
        ):
            errors.append(
                "the matched T2I HPSv3 optimizer requires AdamW lr=3e-4, betas=(0.9,0.95), "
                "eps=1e-6, zero weight decay, finite gradients, grad clip 1, and LoRA-only ownership"
            )
        scheduler = rl.scheduler
        if (
            int(scheduler.warm_up_steps) != image_profile.scheduler_warm_up_steps
            or int(scheduler.cycle_length) != image_profile.scheduler_cycle_length
            or not _matches_number(scheduler.f_start, image_profile.scheduler_f_start)
            or not _matches_number(scheduler.f_max, image_profile.scheduler_f_max)
            or not _matches_number(scheduler.f_min, image_profile.scheduler_f_min)
        ):
            errors.append("the matched T2I HPSv3 scheduler requires warmup 50, cycle 500000, and factors 1e-6/1/0.5")
        if (
            not _matches_number(rl.reward.reward_clip_min, image_profile.reward_clip_min)
            or not _matches_number(rl.reward.reward_clip_max, image_profile.reward_clip_max)
            or not _matches_number(rl.reward.scale, image_profile.reward_scale)
        ):
            errors.append("the matched T2I HPSv3 reward transform requires fp32 clamp [-50,50] and scale 1")
        if list(rl.init_keys_to_skip_loading) != list(image_profile.source_only_keys):
            errors.append("the matched T2I HPSv3 warm start requires the exact five source-only sound keys")
        if not _matches_number(rl.guidance, image_profile.guidance):
            errors.append("the matched native T2I recipe requires rl.guidance=1 for conditional-only sampling")
        if str(rl.rollout_seed_clock) != image_profile.rollout_seed_clock:
            errors.append("the matched T2I recipe requires rl.rollout_seed_clock='optimizer_step'")
        use_system_prompt = _nested_value(config, "model", "config", "vlm_config", "use_system_prompt")
        if use_system_prompt is not image_profile.use_system_prompt:
            errors.append("the matched T2I HPSv3 recipe requires vlm_config.use_system_prompt=false")
        tokenizer_type = _nested_value(
            config,
            "model",
            "config",
            "vlm_config",
            "tokenizer",
            "tokenizer_type",
        )
        if str(tokenizer_type) != image_profile.tokenizer_type:
            errors.append("the matched T2I HPSv3 recipe requires the Qwen/Qwen3-VL-32B-Instruct tokenizer")
        json_file = _nested_value(
            config,
            "model",
            "config",
            "vlm_config",
            "model_instance",
            "config",
            "base_config",
            "json_file",
        )
        if str(json_file) != image_profile.vlm_json_file:
            errors.append(f"the matched T2I HPSv3 recipe requires VLM JSON {image_profile.vlm_json_file!r}")
        qk_norm_for_text = _nested_value(
            config,
            "model",
            "config",
            "vlm_config",
            "model_instance",
            "config",
            "qk_norm_for_text",
        )
        if qk_norm_for_text is not image_profile.qk_norm_for_text:
            errors.append("the matched T2I HPSv3 recipe requires effective text QK normalization")
        if (
            str(_nested_value(config, "model", "config", "joint_attn_implementation"))
            != image_profile.joint_attn_implementation
        ):
            errors.append("the matched T2I HPSv3 recipe requires two-way OmniMoT joint attention")
        backbone_path = _nested_value(
            config,
            "model",
            "config",
            "vlm_config",
            "pretrained_weights",
            "backbone_path",
        )
        if str(backbone_path) != image_profile.backbone_path:
            errors.append("the matched T2I HPSv3 recipe requires the private Cosmos3 32B reasoner backbone")
    try:
        from cosmos_framework.posttrain.roles.env import WFMSeedContract

        WFMSeedContract(
            base_seed=int(rl.seed),
            clock=cast(Literal["optimizer_step", "source_row"], str(rl.rollout_seed_clock)),
            group_size=int(rl.group_size),
        )
    except (TypeError, ValueError) as error:
        errors.append(f"invalid online NFT rollout seed contract: {error}")
    declared_world = int(rl.group_size)
    if declared_world > _WFM_SEED_STRIDE:
        errors.append(
            f"online NFT group_size={declared_world} exceeds the seed stride {_WFM_SEED_STRIDE}; "
            "adjacent source rows would overlap"
        )
    try:
        from cosmos_framework.posttrain.config.build import resolve_policy_parallelism

        policy_world = role_shapes["policy"][1] if role_shapes is not None else declared_world
        replicate, shard = resolve_policy_parallelism(config, policy_world)
    except (AttributeError, TypeError, ValueError) as error:
        errors.append(f"online NFT policy parallelism is incompatible with policy ranks={policy_world}: {error}")
    else:
        declared_world = replicate * shard
    environment = os.environ if environ is None else environ
    missing = [name for name in _REWARD_ENVIRONMENT if not str(environment.get(name, "")).strip()]
    if missing:
        errors.append(f"missing reward environment variable(s): {', '.join(missing)}")
    return errors


def validate_nft_dataloader(
    config: Any,
    *,
    modality: Literal["image", "video"] = "video",
    grad_accum: int = 1,
    instantiate_fn: Callable[[Any], Any] | None = None,
) -> None:
    """Instantiate the configured model-free source and prepare exactly one row.

    Static config checks cannot detect empty shard selection, caption-schema
    mismatch, or token packing that produces multiple source rows. ``--validate``
    runs this smoke before allocating the model.
    """
    if config.dataloader_train is None:
        raise ValueError("dataloader_train must be configured")
    if instantiate_fn is None:
        from cosmos_framework.utils.lazy_config import instantiate

        instantiate_fn = instantiate
    from cosmos_framework.posttrain.roles.env import WFMDataEnv

    dataloader = instantiate_fn(config.dataloader_train)
    try:
        source = WFMDataEnv(dataloader, grad_accum=grad_accum, modality=modality)
        profile = "T2I" if modality == "image" else "T2V"
        try:
            samples = source.get_samples(1)
        except StopIteration as error:
            raise RuntimeError(f"online NFT dataloader preflight produced no {profile} sample") from error
        if len(samples) != 1:
            raise RuntimeError(f"online NFT dataloader preflight expected one {profile} sample, got {len(samples)}")
        if modality == "image":
            from cosmos_framework.posttrain.schema import WFM_CONDITIONING_KEY, ObsBundle, WFMConditioningInput
            from cosmos_framework.posttrain.utils.wfm import repeat_conditioning_batch

            observation = samples[0]
            if not isinstance(observation, ObsBundle):
                raise TypeError(f"online NFT T2I preflight expected ObsBundle, got {type(observation).__name__}")
            conditioning = observation.representations.get(WFM_CONDITIONING_KEY)
            if not isinstance(conditioning, WFMConditioningInput):
                raise TypeError("online NFT T2I preflight expected a typed WFM conditioning input")
            repeat_conditioning_batch(conditioning, 1)
    finally:
        close = getattr(dataloader, "close", None)
        if callable(close):
            close()


def restore_wfm_source(source: Any, trainer: Any, data_state: dict[str, Any]) -> None:
    """Restore one fail-closed trainer/source optimizer boundary on every rank."""
    from cosmos_framework.posttrain.utils.dist import raise_if_any_rank_failed

    restore_error: Exception | None = None
    try:
        optimizer_step = data_state.get("optimizer_step")
        committed_rows = data_state.get("committed_rows")
        expected_rows = trainer.iteration() * source.grad_accum
        if optimizer_step != trainer.iteration() or committed_rows != expected_rows:
            raise RuntimeError(
                "online NFT trainer/source checkpoint boundary differs: "
                f"trainer_step={trainer.iteration()}, source_step={optimizer_step!r}, "
                f"trainer_rows={expected_rows}, source_rows={committed_rows!r}"
            )
        source.load_state_dict(data_state)
    except Exception as error:  # noqa: BLE001 — converge rank-local checkpoint corruption
        restore_error = error
    raise_if_any_rank_failed(restore_error, "online NFT source restore")


def build_shared_bodies(
    config: Any,
    rl: Any,
    *,
    resume_checkpoint: str | None = None,
    reward_client: Any | None = None,
) -> dict[str, Any]:
    """Build all roles around one shared Cosmos3 model instance."""
    from cosmos_framework.utils.lazy_config import instantiate
    from cosmos_framework.posttrain.algorithm.annotators.advantage import distributed_group_relative
    from cosmos_framework.posttrain.algorithm.nft import NFTObjectiveConfig, make_objective
    from cosmos_framework.posttrain.algorithm.reward import RewardCombiner
    from cosmos_framework.posttrain.config.build import build_backbone, build_optimizer, build_wfm_lr_scheduler
    from cosmos_framework.posttrain.control import ExperienceBuffer
    from cosmos_framework.posttrain.control.checkpointing import CheckpointLifecycle
    from cosmos_framework.posttrain.model.cosmos3 import Cosmos3FlowPolicy, validate_cosmos3_cold_lora_inventory
    from cosmos_framework.posttrain.model.unipc import build_unipc_sampler
    from cosmos_framework.posttrain.roles.env import WFMDataEnv, WFMSeedContract
    from cosmos_framework.posttrain.roles.reward import RewardWorker
    from cosmos_framework.posttrain.roles.reward.image import HTTPImageRewardClient, build_image_reward_provider
    from cosmos_framework.posttrain.roles.reward.video import HTTPVideoRewardClient, build_video_reward_provider
    from cosmos_framework.posttrain.roles.rollout import WFMOnlineRollout
    from cosmos_framework.posttrain.roles.trainer import CopyInventoryPolicy, DiffusionTrainer
    from cosmos_framework.posttrain.utils.dist import raise_if_any_rank_failed, world_size_or_1
    from cosmos_framework.posttrain.utils.rank_state import state_for_current_rank

    if config.dataloader_train is None:
        raise ValueError("online NFT needs config.dataloader_train")
    checkpoint_lifecycle = CheckpointLifecycle.from_config(
        config,
        rl,
        resume_checkpoint=resume_checkpoint,
    )
    process_world = world_size_or_1()
    if str(_nested_value(config, "grid", "backend")) == "ray" and process_world != int(rl.group_size):
        raise ValueError(
            "Ray-colocated online NFT requires one WFM member per actor rank before model construction: "
            f"live world={process_world}, group_size={rl.group_size}"
        )
    local_group_size = int(rl.group_size) if process_world == 1 else 1
    model = build_backbone(config, process_world)
    policy_model = Cosmos3FlowPolicy(
        model,
        build_ref=bool(rl.ref.build_ref),
        build_old=bool(rl.old.build_old),
    )
    uses_frozen_reference = bool(rl.ref.build_ref) and int(rl.ref.update_every) <= 0
    optimizer = build_optimizer(model, rl.optimizer)
    scheduler = build_wfm_lr_scheduler(optimizer, rl.scheduler)
    objective_config = NFTObjectiveConfig(
        beta=float(rl.beta),
        kl_beta=float(rl.kl_beta),
        policy_loss_scale=float(rl.policy_loss_scale),
        advantage_high=float(rl.advantage_high),
        advantage_low=float(rl.advantage_low),
        r_map=cast(Literal["linear", "sigmoid"], str(rl.r_map)),
        r_tau=float(rl.r_tau),
        uniform_timestep=bool(rl.uniform_timestep),
    )
    trainer = DiffusionTrainer(
        policy_model,
        make_objective(objective_config),
        optimizer,
        scheduler=scheduler,
        update_ema_after_iter=policy_model.net_ema is not None,
        grad_accum=int(rl.grad_accum),
        update_old_every_iter=int(rl.old.update_every),
        update_ref_every_iter=int(rl.ref.update_every),
        grad_clip_norm=float(rl.optimizer.grad_clip_norm),
        force_finite_grads=bool(rl.optimizer.force_finite_grads),
        checkpoint_manager=checkpoint_lifecycle.manager,
        defer_frozen_reference_identity=bool(rl.init_from) and uses_frozen_reference,
    )
    modality = cast(Literal["image", "video"], str(rl.modality))
    if rl.init_from:
        inventory_policy = (
            CopyInventoryPolicy(
                source_only_keys=tuple(rl.init_keys_to_skip_loading),
                target_only_validator=validate_cosmos3_cold_lora_inventory,
            )
            if modality == "image"
            else None
        )
        trainer.load(
            checkpoint_lifecycle.localize_init_from(rl.init_from),
            copy_mode=True,
            copy_inventory_policy=inventory_policy,
        )
        info0(f"initialized policy+old+reference from base {rl.init_from}")

    seed_contract = WFMSeedContract(
        base_seed=int(rl.seed),
        clock=cast(Literal["optimizer_step", "source_row"], str(rl.rollout_seed_clock)),
        group_size=int(rl.group_size),
    )
    source = WFMDataEnv(
        _IdentifiedLoader(
            instantiate(config.dataloader_train),
            _source_fingerprint(config.dataloader_train),
        ),
        grad_accum=int(rl.grad_accum),
        modality=modality,
        seed_contract=seed_contract,
    )
    rollout = WFMOnlineRollout(
        policy_model,
        sampler=build_unipc_sampler(policy_model),
        num_steps=int(rl.sample_steps),
        shift=float(rl.sampler_shift),
        guidance=float(rl.guidance),
        group_size=int(rl.group_size),
    )
    if modality == "image":
        client = reward_client or HTTPImageRewardClient(rl.reward)
        provider = build_image_reward_provider(
            rl.reward,
            client,
            policy_model.video_reward_decoder(),
            run_id=str(rl.run_ts or config.job.name),
        )
    else:
        client = reward_client or HTTPVideoRewardClient(rl.reward)
        provider = build_video_reward_provider(
            rl.reward,
            client,
            policy_model.video_reward_decoder(),
            run_id=str(rl.run_ts or config.job.name),
        )
    reward = RewardWorker(
        [provider],
        RewardCombiner(),
        annotators=(
            distributed_group_relative(
                eps=float(rl.advantage_eps),
                group_size=int(rl.group_size),
                correction=1,
            ),
        ),
        # Live v1 uses one trajectory on every model rank. The world-size-one
        # model-free fixture instead owns the complete local group.
        training_group_size=local_group_size,
        failure_convergence=raise_if_any_rank_failed,
    )
    replay = ExperienceBuffer(staleness=0, exact_batch_size=local_group_size)
    if rollout.policy is not trainer.model:
        raise RuntimeError("online NFT rollout and trainer must share one policy object")
    selected_resume = checkpoint_lifecycle.resume_checkpoint
    if selected_resume:
        # Construct every component before exact-load restores RNG. No setup work may
        # consume randomness after the durable trainer clock is installed.
        loaded = checkpoint_lifecycle.load_trainer_and_component(trainer, "data", selected_resume)
        if loaded is None:
            raise RuntimeError("online NFT resume checkpoint unexpectedly resolved to none")
        iteration, data_component = loaded
        data_state = state_for_current_rank(data_component, on_world_size_mismatch="raise")
        if data_state is None:
            raise RuntimeError(
                "online NFT training resume requires the checkpoint data component; "
                "use rl.init_from for model-only initialization"
            )
        restore_wfm_source(source, trainer, data_state)
        info0(f"resumed online NFT trainer/source from {selected_resume} (iter {iteration})")
    from cosmos_framework.posttrain.control.loop.weight_sync import weight_sync

    # A fresh run publishes version zero; an exact resume publishes the restored
    # optimizer step. Shared-model sync is byte-free but still proves object identity.
    weight_sync(rollout, trainer)
    return {
        "data": source,
        "rollout": rollout,
        "rm": reward,
        "replay": replay,
        "policy": trainer,
    }


def make_setup(config_path: str, opts: list[str], resume_checkpoint: str | None) -> Any:
    """Return the one-colocation setup closure run independently in each Ray actor."""

    def setup() -> dict[str, Any]:
        from cosmos_framework.utils.config import load_config
        from cosmos_framework.utils.lazy_config import instantiate
        from cosmos_framework.utils.misc import set_random_seed

        config = load_config(config_path, ["--", *opts])
        rl = instantiate(config.rl)
        if bool(rl.deterministic):
            from cosmos_framework.posttrain.utils.determinism import enable_determinism

            enable_determinism()
        # Match the pinned Cosmos-RL WFM fresh-run convention before constructing
        # model, dataloader, or objective state. Process-group rank is available here.
        set_random_seed(int(rl.seed), by_rank=True)
        torch_cudnn = getattr(getattr(config, "trainer", None), "cudnn", None)
        if torch_cudnn is not None and not bool(rl.deterministic):
            import torch

            torch.backends.cudnn.deterministic = bool(getattr(torch_cudnn, "deterministic", False))
            torch.backends.cudnn.benchmark = bool(getattr(torch_cudnn, "benchmark", False))
        return build_shared_bodies(config, rl, resume_checkpoint=resume_checkpoint)

    return setup


def online_nft_step(units: dict[str, Any], rl: Any, *, step: int) -> dict[str, Any]:
    """Run one atomic source→rollout→reward→accumulated optimizer transaction."""
    from cosmos_framework.posttrain.control.loop.offload import offload_trainer, onload_trainer
    from cosmos_framework.posttrain.control.loop.weight_sync import weight_sync
    from cosmos_framework.posttrain.roles.env import WFMSeedContract, wfm_seed_set
    from cosmos_framework.posttrain.schema import finalize_metric_report, merge_metric_reports
    from cosmos_framework.posttrain.utils.dist import raise_if_any_rank_failed
    from cosmos_framework.posttrain.utils.entry import resolve_all
    from cosmos_framework.posttrain.utils.report import merge_training_metric_window

    data = units["data"]
    rollout = units["rollout"]
    reward = units["rm"]
    replay = units["replay"]
    policy = units["policy"]
    grad_accum = int(rl.grad_accum)
    window_metrics: list[dict[str, Any]] = []
    advantage_rows: list[dict[str, Any]] = []
    provider_reports: list[Any] = []
    stage_seconds = {
        "timer/source_acquire_s": 0.0,
        "timer/rollout_s": 0.0,
        "timer/reward_s": 0.0,
        "timer/replay_s": 0.0,
        "timer/train_s": 0.0,
        "timer/source_stage_s": 0.0,
        "timer/weight_sync_s": 0.0,
        "timer/source_commit_s": 0.0,
    }
    try:
        weight_version = int(resolve(policy.iteration()))
        if weight_version != step:
            raise RuntimeError(
                f"online NFT loop step={step} differs from authoritative policy weight_version={weight_version}"
            )
        seed_contract = WFMSeedContract(
            base_seed=int(rl.seed),
            clock=cast(Literal["optimizer_step", "source_row"], str(rl.rollout_seed_clock)),
            group_size=int(rl.group_size),
        )

        for accumulation_row in range(grad_accum):
            source_row = weight_version * grad_accum + accumulation_row
            # The contract selects one global base. Rollout expands it across stable
            # rank/member lanes so the caller remains placement-agnostic.
            rollout_seed = wfm_seed_set(
                seed_contract,
                source_row=source_row,
                optimizer_step=weight_version,
            ).start

            stage_started = time.monotonic()
            conditioning = data.get_samples(1)
            stage_seconds["timer/source_acquire_s"] += time.monotonic() - stage_started

            generated: Any | None = None
            rollout_error: Exception | None = None
            stage_started = time.monotonic()
            # Sampling runs on the shared policy net; the trainer parks its update-only state.
            offload_trainer(policy)
            try:
                generated = rollout.generate(
                    conditioning,
                    seed=rollout_seed,
                    weight_version=weight_version,
                )
            except Exception as error:  # noqa: BLE001 — converge before reward collectives
                rollout_error = error
            finally:
                stage_seconds["timer/rollout_s"] += time.monotonic() - stage_started
            raise_if_any_rank_failed(rollout_error, "online NFT rollout")
            if generated is None:
                raise RuntimeError("online NFT rollout converged without generated trajectories")

            stage_started = time.monotonic()
            scored = reward.score_with_provider_report(generated)
            # Settle every rank-local score call and surface failures before consuming
            # the reports published by those same ordered actor calls.
            resolve_all(scored)
            provider_handle = reward.provider_report()
            annotation_handle = reward.annotation_metrics()
            try:
                row_provider_reports = _provider_reports(
                    provider_handle,
                    operation="online NFT provider report",
                )
                annotation_metrics = _identical_metric_frame(
                    annotation_handle,
                    operation="online NFT advantage telemetry",
                )
            finally:
                stage_seconds["timer/reward_s"] += time.monotonic() - stage_started
            provider_reports.extend(row_provider_reports)
            advantage_rows.append(annotation_metrics)

            batch: Any | None = None
            replay_error: Exception | None = None
            stage_started = time.monotonic()
            try:
                resolve_all(replay.put_samples(scored))
                batch = replay.get_exact_samples()
            except Exception as error:  # noqa: BLE001 — converge before trainer collectives
                replay_error = error
            finally:
                stage_seconds["timer/replay_s"] += time.monotonic() - stage_started
            raise_if_any_rank_failed(replay_error, "online NFT replay assembly")
            if batch is None:
                raise RuntimeError("online NFT replay assembly converged without a batch")

            stage_started = time.monotonic()
            onload_trainer(policy)
            try:
                metrics = _identical_metric_frame(
                    policy.train(batch),
                    operation="online NFT trainer",
                )
            finally:
                stage_seconds["timer/train_s"] += time.monotonic() - stage_started
            expected_outcome = "applied" if accumulation_row + 1 == grad_accum else "accumulating"
            if metrics.get("optim/update_outcome") != expected_outcome:
                failure_metrics = dict(metrics)
                if provider_reports:
                    failure_metrics.update(
                        finalize_metric_report(merge_metric_reports([report.metrics for report in provider_reports]))
                    )
                raise RejectedNFTUpdate(failure_metrics)

            stage_error: Exception | None = None
            stage_started = time.monotonic()
            try:
                resolve_all(data.stage())
            except Exception as error:  # noqa: BLE001 — converge before the next row/telemetry collectives
                stage_error = error
            finally:
                stage_seconds["timer/source_stage_s"] += time.monotonic() - stage_started
            raise_if_any_rank_failed(stage_error, "online NFT source-window stage")
            window_metrics.append(metrics)

        aggregated: dict[str, Any] | None = None
        telemetry_error: Exception | None = None
        try:
            aggregated = merge_training_metric_window(window_metrics)
            aggregated.update(_aggregate_advantage_window(advantage_rows))
            if not provider_reports:
                raise RuntimeError("online NFT completed without provider telemetry")
            provenance = provider_reports[0].provenance
            if any(report.provenance != provenance for report in provider_reports[1:]):
                raise RuntimeError(
                    f"online NFT provider provenance changed within one optimizer window: {provider_reports!r}"
                )
            aggregated.update(
                finalize_metric_report(merge_metric_reports([report.metrics for report in provider_reports]))
            )
            from cosmos_framework.posttrain.algorithm.nft import optimality_weight_std

            aggregated["train/optimality_std"] = optimality_weight_std(
                aggregated["train/optimality_mean"],
                aggregated["train/optimality_second_moment"],
            )
            aggregated["rollout/source_window_rows"] = grad_accum
            aggregated["rollout/reward_trajectory_count"] = int(aggregated["rollout/reward_group_size"]) * grad_accum
        except Exception as error:  # noqa: BLE001 — converge schema drift before weight/source publication
            telemetry_error = error
        raise_if_any_rank_failed(telemetry_error, "online NFT telemetry finalization")
        if aggregated is None:
            raise RuntimeError("online NFT telemetry converged without aggregated metrics")

        commit_error: Exception | None = None
        stage_started = time.monotonic()
        try:
            resolve_all(data.commit_window(grad_accum))
        except Exception as error:  # noqa: BLE001 — applied optimizer/cursor divergence is fatal everywhere
            commit_error = error
        finally:
            stage_seconds["timer/source_commit_s"] += time.monotonic() - stage_started
        raise_if_any_rank_failed(commit_error, "online NFT source-window commit")

        source_state = _identical_metric_frame(
            data.state_dict(),
            operation="online NFT committed source state",
        )
        committed_rows = source_state.get("committed_rows")
        optimizer_step = source_state.get("optimizer_step")
        if committed_rows != (step + 1) * grad_accum or optimizer_step != step + 1:
            raise RuntimeError(
                "online NFT committed source clock differs from the applied model step: "
                f"rows={committed_rows!r}, optimizer_step={optimizer_step!r}, model_step={step + 1}"
            )
        aggregated["rollout/source_committed_rows"] = int(committed_rows)

        # Source publication and weight-version publication are both post-update
        # barriers. Any failure here is process-fatal; no later rollout may begin.
        stage_started = time.monotonic()
        weight_sync(rollout, policy)
        stage_seconds["timer/weight_sync_s"] += time.monotonic() - stage_started
        aggregated.update(stage_seconds)
        return aggregated
    except Exception:
        try:
            resolve_all(policy.abort_accumulation())
        except Exception:  # noqa: BLE001 — trainer may already have committed or rejected the window
            pass
        try:
            resolve_all(data.abort_window())
        except Exception:  # noqa: BLE001 — setup may fail before a source window exists
            pass
        raise


def loop(units: dict[str, Any], config: Any, out_dir: str) -> dict[str, Any]:
    """Drive synchronous online steps and checkpoint only committed boundaries."""
    from cosmos_framework.utils.lazy_config import instantiate
    from cosmos_framework.posttrain.control.checkpointing import CheckpointLifecycle
    from cosmos_framework.posttrain.control.termination import TerminationWatch
    from cosmos_framework.posttrain.utils.report import (
        emit_metric_frame,
        finish_best_effort,
        make_reporters,
        reportable_metrics,
        start_best_effort,
    )

    rl = instantiate(config.rl)
    checkpoint_lifecycle = CheckpointLifecycle.from_config(config, rl, run_out_dir=out_dir)
    start = int(resolve(units["policy"].iteration()))
    if start > int(rl.max_steps):
        raise RuntimeError(f"restored trainer iteration {start} exceeds configured total steps {rl.max_steps}")
    checkpoints = checkpoint_lifecycle.coordinator(
        out_dir,
        units["policy"],
        data=units["data"],
        initial_step=start,
    )
    if start:
        info0(f"resuming online NFT loop at step {start}/{rl.max_steps}")
    watch = TerminationWatch()
    reporters = make_reporters(
        config,
        out_dir,
        total_steps=int(rl.max_steps),
        headline_keys=_NFT_CONSOLE_KEYS,
    )
    failed_reporters: set[int] = set()
    start_best_effort(reporters, failed_reporters=failed_reporters)

    history: list[float] = []
    preempted: str | None = None
    try:
        for step in range(start, int(rl.max_steps)):
            started = time.monotonic()
            try:
                metrics = online_nft_step(units, rl, step=step)
            except RejectedNFTUpdate as error:
                failed_attempt = reportable_metrics({**error.metrics, "timer/e2e_step_s": time.monotonic() - started})
                logger.error(f"online NFT update rejected before model-step commit: {failed_attempt}")
                raise
            model_step = step + 1
            actual_model_step = int(resolve(units["policy"].iteration()))
            if actual_model_step != model_step:
                raise RuntimeError(
                    f"online NFT applied update reports model step {actual_model_step}, expected {model_step}"
                )
            loss = float(metrics["train/loss_avg"])
            if not math.isfinite(loss):
                raise RuntimeError(f"applied online NFT update returned non-finite loss {loss}")
            history.append(loss)
            metrics = reportable_metrics({**metrics, "timer/e2e_step_s": time.monotonic() - started})
            emit_metric_frame(reporters, metrics, model_step, failed_reporters=failed_reporters)
            if rl.save_every and model_step % int(rl.save_every) == 0:
                checkpoints.save_configured_if_needed(model_step)
            preempted = watch.should_stop()
            if preempted:
                # rl.emergency_checkpoint_components: "full" upgrades to the exact-resume plan; "run" writes
                # the run's own cadence plan plus the model -- see RunConfig for the trade.
                emergency_plan = str(rl.emergency_checkpoint_components)
                if checkpoints.save_emergency(model_step, components=emergency_plan):
                    logger.warning(f"termination trigger ({preempted}) — emergency checkpoint ({emergency_plan} plan)")
                else:
                    logger.warning(f"termination trigger ({preempted}) — already checkpointed, exiting")
                break
        else:
            if rl.save_every and int(rl.max_steps) > start:
                checkpoints.save_configured_if_needed(int(rl.max_steps))
    finally:
        finish_best_effort(reporters, failed_reporters=failed_reporters)
    summary = {
        "steps": int(rl.max_steps),
        "start": start,
        "final_loss": history[-1] if history else None,
        **({"preempted": preempted} if preempted else {}),
    }
    info0(f"DONE online NFT {summary} resolved={out_dir}/config_resolved.yaml")
    return summary


def run(config: Any, config_path: str, opts: list[str], out_dir: str) -> None:
    """Bind the production single-ActorGroup Ray GPU topology and run it."""
    from cosmos_framework.posttrain.control.bind import bind_check
    from cosmos_framework.posttrain.control.placement import ActorGroup, DeviceGroup, bind_device_groups
    from cosmos_framework.posttrain.control.recipe import nft_manifest
    from cosmos_framework.posttrain.utils.entry import pin_run_attempt

    report = bind_check(nft_manifest(ROLE_SPECS))
    info0(report.format())
    report.raise_for_status()
    if str(config.grid.backend) != "ray" or str(config.grid.grouping) != "single":
        raise ValueError(
            "production online NFT requires grid.backend=ray and grid.grouping=single; "
            "the Ray binder owns every GPU process and process group"
        )
    from cosmos_framework.utils.lazy_config import instantiate

    rl = instantiate(config.rl)
    preflight_errors = validate_nft_runtime_config(config, rl)
    if preflight_errors:
        raise ValueError("online NFT preflight failed: " + "; ".join(preflight_errors))
    from cosmos_framework.posttrain.config.build import resolve_policy_parallelism

    role_shapes = _nft_role_shapes(config, int(rl.group_size))
    policy_parallelism = resolve_policy_parallelism(config, role_shapes["policy"][1])
    attempt = pin_run_attempt(config, list(opts), out_dir)
    out_dir = attempt.out_dir
    opts_effective = list(attempt.opts)
    resume_checkpoint = attempt.resume_checkpoint
    dump_resolved(config, out_dir)
    wgs = declare_wgs(
        role_shapes["policy"][1],
        shapes=role_shapes,
        policy_parallelism=policy_parallelism,
    )
    actor_group = ActorGroup(
        tuple(wgs.values()),
        make_setup(config_path, opts_effective, resume_checkpoint),
        name="online_nft",
    )
    bind_device_groups(
        [DeviceGroup((actor_group,), device="gpu", name="online_nft")],
        backend="ray",
        threshold=0,
        weight_sync=True,
        gpu_node_domains=getattr(config.grid, "gpu_node_domains", None),
    )
    try:
        loop(wgs, config, out_dir)
    finally:
        from cosmos_framework.posttrain.control.checkpointing import CheckpointLifecycle

        CheckpointLifecycle.finalize_units(wgs)


def _validate_and_exit(config: Any) -> None:
    from cosmos_framework.utils.lazy_config import instantiate
    from cosmos_framework.posttrain.control.recipe import nft_manifest

    output, ok = render_manifest_validation(nft_manifest(ROLE_SPECS))
    print(output)
    rl = instantiate(config.rl)
    errors = validate_nft_runtime_config(config, rl)
    if ok and not errors:
        try:
            validate_nft_dataloader(
                config,
                modality=cast(Literal["image", "video"], str(rl.modality)),
                grad_accum=int(rl.grad_accum),
            )
        except Exception as error:  # noqa: BLE001 — report the source failure without a model allocation
            errors.append(f"dataloader one-batch smoke failed: {error}")
    success = ok and not errors
    if not success:
        print("NFT preflight: FAIL")
        for error in errors:
            print(f"  - {error}")
    else:
        print("NFT preflight: PASS")
    raise SystemExit(0 if success else 1)


def main() -> None:
    config_path, opts, validate = cli(
        "Cosmos3 colocated online DiffusionNFT",
        default_config="cosmos_framework/posttrain/config/config_nft.py",
    )
    from cosmos_framework.utils.config import load_config

    config = load_config(config_path, ["--", *opts])
    if validate:
        _validate_and_exit(config)
    run(config, config_path, opts, _out_dir(config))


if __name__ == "__main__":
    main()
