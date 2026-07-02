# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from typing import Callable

import torch
import torch.distributed
from diffusers import FlowMatchEulerDiscreteScheduler

from cosmos_framework.model.generator.algorithm.loss.time_weight import TrainTimeWeight


class TrainTimeSampler:
    _WAVER_MODE_S = 1.29
    # 99.9th and 0.5th percentiles of the standard normal, used for ltx2 stretching.
    _LTX2_NORMAL_999_PCTILE = 3.0902
    _LTX2_NORMAL_005_PCTILE = -2.5758
    _LTX2_UNIFORM_PROB = 0.1

    def __init__(
        self,
        distribution: str = "uniform",
    ):
        self.distribution = distribution

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator | None = None,
        shifts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Sample sigma ∈ [0, 1] for training.

        Args:
            batch_size: Number of samples.
            device: Target device.
            dtype: Target dtype.
            shifts: Optional 1-D per-sample shift values, shape ``(batch_size,)``.  For non-ltx2
                distributions, the raw sample ``t`` is warped through
                ``sigma = shift * t / (1 + (shift-1) * t)``.  For ``ltx2``, ``shifts`` is
                required and used as the per-sample logit-normal mean.

        Returns:
            torch.Tensor: sigma ∈ [0, 1], shape (batch_size,).
        """
        if self.distribution == "uniform":
            t = torch.rand((batch_size,), generator=generator).to(device=device, dtype=dtype)  # [B]
        elif self.distribution == "logitnormal":
            t = torch.sigmoid(torch.randn((batch_size,), generator=generator)).to(device=device, dtype=dtype)  # [B]
        elif self.distribution == "waver":
            u = torch.rand((batch_size,), dtype=torch.float32, generator=generator)  # [B]
            t = 1.0 - u - self._WAVER_MODE_S * (torch.cos(torch.pi / 2.0 * u) ** 2 - 1 + u)  # [B]
            t = t.to(device=device, dtype=dtype)  # [B]
        elif self.distribution == "ltx2":
            # Shifted logit-normal with percentile-based stretching and 10% uniform fallback.
            assert shifts is not None, "'ltx2' distribution requires per-sample shifts."
            # shift(sigmoid(t), s) = sigmoid(t + ln(s))
            mu = torch.log(shifts.to(device=torch.device("cpu"), dtype=torch.float32))  # [B]
            std = 1.0
            eps = 1e-3

            normal_samples = torch.randn((batch_size,), dtype=torch.float32, generator=generator) * std + mu  # [B]
            logitnormal_samples = torch.sigmoid(normal_samples)  # [B]

            percentile_999 = torch.sigmoid(mu + self._LTX2_NORMAL_999_PCTILE * std)  # [B]
            percentile_005 = torch.sigmoid(mu + self._LTX2_NORMAL_005_PCTILE * std)  # [B]

            zero_terminal_raw = (logitnormal_samples - percentile_005) / (percentile_999 - percentile_005)
            stretched = torch.where(
                zero_terminal_raw >= eps,
                zero_terminal_raw,
                2 * eps - zero_terminal_raw,
            )
            stretched = torch.clamp(stretched, 0, 1)

            uniform = (1 - eps) * torch.rand((batch_size,), dtype=torch.float32, generator=generator) + eps
            prob = torch.rand((batch_size,), dtype=torch.float32, generator=generator)
            t = torch.where(prob > self._LTX2_UNIFORM_PROB, stretched, uniform).to(device=device, dtype=dtype)

            return t  # skip post-shift
        else:
            raise NotImplementedError(f"Time distribution '{self.distribution}' is not implemented.")

        if shifts is not None:
            shifts = shifts.to(device=device, dtype=dtype)  # [B]
            t = shifts * t / (1 + (shifts - 1) * t)  # [B], sigma ∈ [0,1]

        return t  # [B]


class RectifiedFlow:
    def __init__(
        self,
        velocity_field: Callable,
        train_time_distribution: TrainTimeSampler | str = "uniform",
        train_time_weight_method: str = "uniform",
        use_dynamic_shift: bool = False,
        shift: int = 3,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        r"""Initialize the RectifiedFlow class.

        Args:
            velocity_field (`Callable`):
                A function that predicts the velocity given the current state and time.
            train_time_distribution (`TrainTimeSampler` or `str`, *optional*, defaults to `"uniform"`):
                Distribution for sampling training times.
                Can be an instance of `TrainTimeSampler` or a string specifying the distribution type.
            train_time_weight (`TrainTimeWeight` or `str`, *optional*, defaults to `"uniform"`):
                Weight applied to training times.
                Can be an instance of `TrainTimeWeight` or a string specifying the weight type.
        """
        self.velocity_field = velocity_field
        self.train_time_sampler: TrainTimeSampler = (
            train_time_distribution
            if isinstance(train_time_distribution, TrainTimeSampler)
            else TrainTimeSampler(train_time_distribution)
        )

        if use_dynamic_shift:
            self.noise_scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=use_dynamic_shift)
        else:
            self.noise_scheduler = FlowMatchEulerDiscreteScheduler(shift=shift)
        self.train_time_weight = TrainTimeWeight(self.noise_scheduler, train_time_weight_method)

        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = torch.dtype(dtype) if isinstance(dtype, str) else dtype

    def sample_train_time(self, batch_size: int, iteration: int | None = None, shifts: torch.Tensor | None = None):
        r"""This method calls the `TrainTimeSampler` to sample training times.

        Args:
            batch_size: Number of samples.
            iteration: When provided, sampling uses a local generator seeded from
                ``(iteration, rank)`` so results are identical across independent runs
                regardless of prior global RNG state.
            shifts: Optional 1-D shift tensor, shape ``(batch_size,)``.  Forwarded to
                ``TrainTimeSampler.__call__``; see that docstring for details.

        Returns:
            t (`torch.Tensor`):
                A tensor of sampled sigmas with shape `(batch_size,)`,
                matching the class specified `device` and `dtype`.
        """
        generator = None
        if iteration is not None and torch.are_deterministic_algorithms_enabled():
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            generator = torch.Generator()
            generator.manual_seed(iteration * 65536 + rank)
        time = self.train_time_sampler(
            batch_size, device=self.device, dtype=self.dtype, generator=generator, shifts=shifts
        )
        return time

    def get_discrete_timestamp(self, u, tensor_kwargs):
        r"""This method map time from 0,1 to discrete steps"""

        indices = (u.squeeze() * self.noise_scheduler.config.num_train_timesteps).long()  # [B]
        timesteps = self.noise_scheduler.timesteps.to(**tensor_kwargs)[indices]  # [B]
        return timesteps.unsqueeze(0) if timesteps.ndim == 0 else timesteps  # [B]

    def get_sigmas(self, timesteps, tensor_kwargs):  # timesteps: [B], returns [B]
        sigmas = self.noise_scheduler.sigmas.to(**tensor_kwargs)  # [N_timesteps+1]
        schedule_timesteps = self.noise_scheduler.timesteps.to(**tensor_kwargs)  # [N_timesteps]
        step_indices = [(schedule_timesteps == t).nonzero().squeeze().tolist() for t in timesteps]
        assert len(step_indices) == timesteps.shape[0], "Number of indices do not match the given timesteps."
        sigma = sigmas[step_indices].flatten()  # [B]

        return sigma  # [B]

    def get_interpolation(
        self,
        x_0: list[torch.Tensor],  # each element: [B,C,T,H,W] or [B,D1,...,Dn]
        x_1: list[torch.Tensor],  # each element: [B,C,T,H,W] or [B,D1,...,Dn]
        t: list[torch.Tensor],  # each element: [B] or [B,1,1,1,1]
    ):
        r"""
        This method computes interpolation `X_t` and their time derivatives `dotX_t` at the specified time points `t`.
        Note that `x_0` is the noise, and `x_1` is the clean data. This is aligned with the notation in the recified flow community,
        but different from the notation in the diffusion community.

        Args:
            x_0 (`torch.Tensor`):
                noise, shape `(B, D1, D2, ..., Dn)`, where `B` is the batch size, and `D1, D2, ..., Dn` are the data dimensions.
            x_1 (`torch.Tensor`):
                clean data, with the same shape as `x_0`
            t (`torch.Tensor`):
                A tensor of time steps with values in `[0, 1]`. Can be shape `(B,)` or
                pre-broadcast to `(B, 1, T, ..., 1)` matching `x_1`'s dimensionality along batch and temporal dimension.

        Returns:
            (x_t, dot_x_t) (`Tuple[torch.Tensor, torch.Tensor]`):
                - x_t (`torch.Tensor`): The interpolated state, with shape `(B, D1, D2, ..., Dn)`.
                - dot_x_t (torch.Tensor): The time derivative of the interpolated state, with the same shape as `x_t`.
        """
        assert len(x_0) == len(x_1), "x_0 and x_1 must have the same length."
        assert len(x_0) == len(t), "Batch size of x_0 and x_1 must match."
        assert len(t) == len(x_1), "Batch size of t must match x_1."

        x_t = []
        dot_x_t = []
        for i in range(len(x_0)):
            x_t.append(x_0[i] * t[i] + x_1[i] * (1 - t[i]))  # [B,C,T,H,W]; t[i] broadcasts [B] or [B,1,1,1,1]
            dot_x_t.append(x_0[i] - x_1[i])  # [B,C,T,H,W]

        return x_t, dot_x_t  # each list element: [B,C,T,H,W]
