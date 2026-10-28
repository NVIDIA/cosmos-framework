# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Toy flow-matching model — a TEST FIXTURE implementing `FlowMatchingPolicy` (DESIGN §6).

Not a production backend and not registered in `model.registry` — production selects the
real `FlowMatchingPolicy` (the OmniMoT `model/cosmos3.py`) via the upstream config tree. This
is a tiny CPU Linear "velocity field", just enough to exercise the full DPO loop (loss + a
real optimizer step) on the `local` backend, so the model↔algorithm↔trainer decoupling can
be tested without a GPU / the 8B model / the training env. Tests construct it directly
(`ToyFlowModel(dim=..., seed=...)`); it lives under `tests/` so no test-only code sits next
to the production backends.

Sample shape: each side of a pair is `{"x": Tensor[D]}`. `encode_and_condition`
stacks the per-sample list into `x0 [B, D]`; `denoise_error` interpolates
`xt = (1-t)*eps + t*x0`, predicts velocity `v = net(xt)`, and returns the
per-sample fp32 mean-MSE against the rectified-flow target `x0 - eps`.
"""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from cosmos_framework.posttrain.algorithm.primitives.objective import OBJECTIVE_WEIGHT_SUM, normalization_scale
from cosmos_framework.posttrain.model.protocol import FlowCondition, FlowConditionMask, NFTVelocityPredictions

if TYPE_CHECKING:
    from cosmos_framework.posttrain.roles.trainer.llm import LLMTrainer

ToyItem = dict[str, int | bool]


@dataclass(frozen=True)
class _ToyFlowCondition:
    """Opaque toy condition; pair semantics remain on the model adapter."""


class ToyFlowModel:
    """Minimal `FlowMatchingPolicy` over a Linear net (+ frozen ref copy)."""

    def __init__(self, dim: int = 8, seed: int = 0) -> None:
        gen = torch.Generator().manual_seed(seed)
        self.net = torch.nn.Linear(dim, dim)
        with torch.no_grad():
            for p in self.net.parameters():
                p.copy_(torch.empty_like(p).normal_(generator=gen))
        # Frozen reference = a bit-equal copy held in eval (mirrors DPO net_ref).
        self.net_ref = torch.nn.Linear(dim, dim)
        self.net_ref.load_state_dict(self.net.state_dict())
        self.net_ref.eval()
        # A structural copy is bit-equal and does not consume the process-global RNG merely
        # because the NFT fixture owns a second frozen policy role.
        self.net_old = deepcopy(self.net)
        self.net_old.eval()
        self._dim = dim
        self.ema_iterations: list[int] = []

    # --- FlowMatchingPolicy ---
    def reference_semantics_identity(self) -> dict[str, int | str]:
        """Bind the toy adapter's only non-tensor forward construction value."""
        return {"adapter": "posttrain.toy-flow.v1", "dim": self._dim}

    def encode_and_condition(
        self,
        batch: dict,
        *,
        sample_encode_seeds: list[int] | None = None,
    ) -> tuple[torch.Tensor, FlowCondition]:
        del sample_encode_seeds  # The toy encoder is deterministic; accept the production capability contract.
        x_values = batch["x"]
        x0 = x_values.float() if isinstance(x_values, torch.Tensor) else torch.stack(x_values).float()  # [B, D]
        return x0, _ToyFlowCondition()

    def condition_mask(self, cond: FlowCondition) -> FlowConditionMask:
        if not isinstance(cond, _ToyFlowCondition):
            raise TypeError(f"ToyFlowModel requires _ToyFlowCondition, got {type(cond).__name__}")
        return None  # toy: no conditioning frames

    def assert_same_condition_context(self, winner: FlowCondition, loser: FlowCondition) -> None:
        """Require both preference sides to use this adapter's private condition type."""

        if not isinstance(winner, _ToyFlowCondition) or not isinstance(loser, _ToyFlowCondition):
            raise TypeError(
                "toy preference sides must both use toy conditioning, got "
                f"{type(winner).__name__} and {type(loser).__name__}"
            )

    def refresh_ref(self) -> None:
        """Refresh the moving-reference fixture from the current policy weights."""
        self.net_ref.load_state_dict(self.net.state_dict())
        self.net_ref.eval()

    def refresh_old(self) -> None:
        """Refresh the moving NFT behavior fixture from policy weights."""
        self.net_old.load_state_dict(self.net.state_dict())
        self.net_old.eval()

    def draw_sigma_epsilon(
        self,
        x0_shape: torch.Size,
        cond: FlowCondition,
        sample_seeds: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = next(self.net.parameters()).device  # follow the net (CPU or cuda/FSDP)
        b = x0_shape[0]
        if sample_seeds is None:
            t = torch.rand(b, 1, device=device)  # [B, 1] noise fraction (shared winner/loser)
            eps = torch.randn(x0_shape, device=device)  # [B, D]
            return t, eps
        t_rows: list[torch.Tensor] = []
        eps_rows: list[torch.Tensor] = []
        fork_devices = [device] if device.type == "cuda" else []
        for seed in sample_seeds:
            with torch.random.fork_rng(devices=fork_devices):
                torch.manual_seed(seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed(seed)
                t_rows.append(torch.rand(1, 1, device=device))  # [1, 1]
                eps_rows.append(torch.randn((1, *x0_shape[1:]), device=device))  # [1, D]
        t = torch.cat(t_rows, dim=0)  # [B, 1]
        eps = torch.cat(eps_rows, dim=0)  # [B, D]
        return t, eps

    def prepare_denoise_error(
        self,
        *,
        net: torch.nn.Module,
        x0: torch.Tensor,
        cond: FlowCondition,
        sigma: torch.Tensor,
        epsilon: torch.Tensor,
        condition_mask: FlowConditionMask,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del net, cond, condition_mask
        xt = (1.0 - sigma) * epsilon + sigma * x0  # [B, D]
        target = x0 - epsilon  # [B, D]
        return xt, target

    def denoise_prediction(
        self,
        *,
        net: torch.nn.Module,
        prepared: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        grad = net is self.net and torch.is_grad_enabled()
        with torch.set_grad_enabled(grad):
            prediction = net(prepared[0])  # [B, D]
        return prediction

    def finalize_denoise_error(
        self,
        *,
        prepared: tuple[torch.Tensor, torch.Tensor],
        prediction: torch.Tensor,
    ) -> torch.Tensor:
        target = prepared[1]  # [B, D]
        return (prediction.float() - target.float()).pow(2).mean(dim=1)  # [B]

    def denoise_error(
        self,
        *,
        net: torch.nn.Module,
        x0: torch.Tensor,
        cond: FlowCondition,
        sigma: torch.Tensor,
        epsilon: torch.Tensor,
        condition_mask: FlowConditionMask,
    ) -> torch.Tensor:
        """Run one complete staged denoiser call."""
        prepared = self.prepare_denoise_error(
            net=net,
            x0=x0,
            cond=cond,
            sigma=sigma,
            epsilon=epsilon,
            condition_mask=condition_mask,
        )
        prediction = self.denoise_prediction(net=net, prepared=prepared)
        return self.finalize_denoise_error(prepared=prepared, prediction=prediction)

    def draw_nft_sigma_epsilon(self, x0_shape: torch.Size, cond: FlowCondition) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw epsilon before timestep, matching the pinned NFT RNG contract."""
        del cond
        device = next(self.net.parameters()).device
        epsilon = torch.randn(x0_shape, device=device)  # [B, D]
        sigma = torch.rand(x0_shape[0], 1, device=device)  # [B, 1]
        return sigma, epsilon

    def update_ema(self, iteration: int) -> None:
        """Record the committed-update EMA clock for trainer contract tests."""
        self.ema_iterations.append(iteration)

    def nft_velocity_predictions(
        self,
        *,
        x0: torch.Tensor,
        cond: FlowCondition,
        sigma: torch.Tensor,
        epsilon: torch.Tensor,
        condition_mask: FlowConditionMask,
    ) -> NFTVelocityPredictions:
        del cond, condition_mask
        t_effective = sigma  # [B, 1]
        x_t = (1.0 - t_effective) * x0 + t_effective * epsilon  # [B, D]
        v_policy = self.net(x_t)  # [B, D]
        with torch.no_grad():
            v_old = self.net_old(x_t)  # [B, D]
            v_ref = self.net_ref(x_t)  # [B, D]
        return NFTVelocityPredictions(
            x_t=x_t,
            t_effective=t_effective,
            v_policy=v_policy,
            v_old=v_old,
            v_ref=v_ref,
        )

    def decode_final_latents(self, latents: list[torch.Tensor]) -> torch.Tensor:
        """Expose a deterministic toy decode for reward-contract tests."""
        batch = torch.cat(latents, dim=0)  # [B,D]
        return batch[:, None, None, None, :].repeat(1, 3, 1, 1, 1)  # [B,3,1,1,D]

    def video_reward_decoder(self) -> _ToyVideoRewardDecoder:
        """Expose the toy decoder with the same attested capability as Cosmos3."""
        return _ToyVideoRewardDecoder(self)

    def sample_final_latents(
        self,
        model_batch: dict,
        *,
        sampler: object,
        seeds: list[int],
        num_steps: int,
        shift: float,
        guidance: float,
    ) -> list[torch.Tensor]:
        """Return deterministic toy rollout samples for online-WFM tests."""
        del sampler, num_steps, shift, guidance
        rows = model_batch["x"]
        if len(rows) != len(seeds):
            raise ValueError(f"toy rollout rows/seeds mismatch: {len(rows)} vs {len(seeds)}")
        outputs: list[torch.Tensor] = []
        for row, seed in zip(rows, seeds, strict=True):
            generator = torch.Generator().manual_seed(seed)
            noise = torch.randn(row.shape, generator=generator, dtype=row.dtype)  # [D]
            outputs.append((row + 0.01 * noise).unsqueeze(0))  # [1,D]
        return outputs


class _ToyVideoRewardDecoder:
    """Model-owned decoder capability used by online-NFT integration tests."""

    output_range = "neg_one_to_one"

    def __init__(self, model: ToyFlowModel) -> None:
        self.model = model

    def __call__(self, latents: list[torch.Tensor]) -> torch.Tensor:
        return self.model.decode_final_latents(latents)


class ToyVideoFlowModel(ToyFlowModel):
    """Toy online-flow model whose generated actions obey the video latent contract."""

    def draw_nft_sigma_epsilon(self, x0_shape: torch.Size, cond: FlowCondition) -> tuple[torch.Tensor, torch.Tensor]:
        del cond
        device = next(self.net.parameters()).device
        epsilon = torch.randn(x0_shape, device=device)  # [B,C,T,H,W]
        sigma_shape = (x0_shape[0],) + (1,) * (len(x0_shape) - 1)
        sigma = torch.rand(sigma_shape, device=device)  # [B,1,1,1,1]
        return sigma, epsilon

    def decode_final_latents(self, latents: list[torch.Tensor]) -> torch.Tensor:
        batch = torch.cat(latents, dim=0)  # [B,1,1,1,D]
        # The production VAE declares decoded pixels in [-1, 1]. Keep that capability
        # load-bearing even when the synthetic source row grows across accumulation steps.
        return batch.clamp(-1.0, 1.0).repeat(1, 3, 1, 1, 1)  # [B,3,1,1,D]

    def sample_final_latents(
        self,
        model_batch: dict,
        *,
        sampler: object,
        seeds: list[int],
        num_steps: int,
        shift: float,
        guidance: float,
    ) -> list[torch.Tensor]:
        flat = super().sample_final_latents(
            model_batch,
            sampler=sampler,
            seeds=seeds,
            num_steps=num_steps,
            shift=shift,
            guidance=guidance,
        )
        return [latent[:, None, None, None, :] for latent in flat]  # each [1,1,1,1,D]


# Token-sum fixture for deferred normalization and poisoned-window tests.

_TOY_DIM = 4


def toy_item(idx: int, length: int, *, poison: bool = False) -> ToyItem:
    """Build a deterministic token-sum item, optionally poisoned with NaN."""
    return {"idx": idx, "length": length, "poison": poison}


def _item_features(idx: int, length: int, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return features independent of the item's chunk position."""
    gen = torch.Generator().manual_seed(20260813 + 97 * idx)
    x = torch.randn(length, dim, generator=gen)  # [T,D]
    target = torch.randn(length, dim, generator=gen)  # [T,D]
    return x, target


def _toy_token_sum_objective(
    model: torch.nn.Module,
    batch: list[ToyItem],
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Mirror GRPO's normalize-then-report token-sum contract."""
    dim = model.in_features
    if not batch:
        # Mirrors `algorithm/grpo.py`'s zeroed-forward trick for an empty shard: the loss must
        # stay ON the autograd graph (a detached constant makes `backward()` raise) while
        # contributing nothing, so every parameter still gets a defined (zero) gradient.
        zero = sum(parameter.sum() for parameter in model.parameters()) * 0.0  # []
        return zero, {OBJECTIVE_WEIGHT_SUM: 0.0, "grpo_n": 0, "grpo_valid_tokens": 0}
    numerator = torch.zeros(())  # []
    n_tokens = 0
    for item in batch:
        idx = int(item["idx"])
        length = int(item["length"])
        x, target = _item_features(idx, length, dim)  # [T,D], [T,D]
        pred = model(x)  # [T,D]
        squared_error = (pred - target).pow(2).sum()  # []
        if item.get("poison"):
            squared_error = squared_error * float("nan")  # []
        numerator = numerator + squared_error  # []
        n_tokens += length
    denom = float(n_tokens)
    loss = numerator * normalization_scale(denom)  # []
    return loss, {OBJECTIVE_WEIGHT_SUM: denom, "grpo_n": len(batch), "grpo_valid_tokens": n_tokens}


class _ToyTrainerHandle:
    """Small facade exposing only the deferred-normalization test seams."""

    _trainer: LLMTrainer

    def __init__(self, trainer: LLMTrainer) -> None:
        self._trainer = trainer

    def train(self, batch: list[ToyItem]) -> dict[str, Any]:
        return self._trainer.train(batch)

    def named_parameters(self) -> Iterator[tuple[str, torch.nn.Parameter]]:
        return self._trainer._net().named_parameters()

    def _require_checkpoint_boundary(self) -> None:
        self._trainer._require_checkpoint_boundary()

    @property
    def optimizer_step_count(self) -> int:
        return self._trainer.iteration()

    @property
    def _denom_at_last_step(self) -> float:
        return self._trainer._denom_at_last_step


def make_toy_trainer(
    grad_accum: int = 1,
    *,
    dim: int = _TOY_DIM,
    seed: int = 0,
    lr: float = 0.1,
) -> _ToyTrainerHandle:
    """Build a CPU trainer over the token-sum fixture."""
    from cosmos_framework.posttrain.algorithm.primitives.objective import Objective, Reduction
    from cosmos_framework.posttrain.roles.trainer.llm import LLMTrainer

    torch.manual_seed(seed)
    model = torch.nn.Linear(dim, dim)
    objective = Objective(_toy_token_sum_objective, reduction=Reduction.TOKEN_SUM)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    trainer = LLMTrainer(model, objective, optimizer, grad_accum=grad_accum)
    return _ToyTrainerHandle(trainer)
