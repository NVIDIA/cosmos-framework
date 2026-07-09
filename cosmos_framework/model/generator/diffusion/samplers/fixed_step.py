# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Fixed-step sampler for DMD2-distilled student models.

Uses an explicit, fixed sigma schedule (t_list) baked in at construction time.
Each step predicts x0 via a single velocity forward pass, then re-noises x0 to
sigma_next with fresh noise.

This is incompatible with multi-step solvers (UniPC, EDM) because DMD2 students
are trained as one-shot denoisers at specific discrete sigmas, not as smooth
score functions.
"""

import torch

from cosmos_framework.model.generator.diffusion.samplers.utils import run_multiseed


class FixedStepSampler:
    def __init__(
        self,
        t_list: list[float],
        sample_type: str = "sde",
        num_train_timesteps: float = 1000.0,
    ) -> None:
        assert len(t_list) >= 1, "t_list must have at least 1 entry"
        assert sample_type == "sde", f"FixedStepSampler only supports sample_type='sde', got {sample_type}"
        # Auto-append 0.0 if not present (convention: t_list in config excludes final step)
        self.t_list = t_list if t_list[-1] == 0.0 else t_list + [0.0]
        assert len(self.t_list) >= 2, "t_list must have at least 2 entries after appending 0.0"
        self.sample_type = sample_type
        self.num_train_timesteps = num_train_timesteps

    def __call__(
        self,
        velocity_fn,
        noise: torch.Tensor | list[torch.Tensor],
        num_steps: int | None = None,
        shift: float | None = None,
        seed: int | list[int] | None = None,
        condition_reference: torch.Tensor | list[torch.Tensor] | None = None,
        condition_mask: torch.Tensor | list[torch.Tensor] | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Run the fixed-step sampling loop.

        Matches the UniPC sampler call signature so both can be used
        interchangeably in ``generate_samples_from_batch``.

        ``noise`` and ``seed`` must both be single values or both be lists
        (of the same length).  When lists are provided, each element
        corresponds to one independent sample; the return value is then a
        list of denoised tensors.  When single values are provided, a
        single tensor is returned.

        Args:
            velocity_fn: ``velocity_fn(noise=..., timestep=...) -> velocity``.
            noise: Initial noise.  Either a single ``torch.Tensor`` of shape
                ``(D,)`` or a ``list[torch.Tensor]`` where each element has
                shape ``(D,)``.
            seed: RNG seed.  Either a single ``int`` or a
                ``list[int]`` with the same length as ``noise``.
            num_steps: Ignored. The number of denoising steps is defined by
                ``self.t_list``.
            shift: Ignored. Fixed-step distilled sampling always uses
                ``self.t_list`` from the experiment config.
            condition_reference: Optional clean reference tensor(s) to preserve
                where ``condition_mask`` is 1.
            condition_mask: Optional mask tensor(s), same shape as ``noise``,
                where 1 marks clean conditioning values and 0 marks generated
                values.

        Returns:
            Denoised sample(s).  A single ``torch.Tensor`` when ``noise`` is a
            tensor, or a ``list[torch.Tensor]`` when ``noise`` is a list.
        """
        assert (condition_reference is None) == (condition_mask is None), (
            "condition_reference and condition_mask must be both set or both None"
        )
        if isinstance(noise, list):
            device = noise[0].device
            if seed is None:
                seed = [None] * len(noise)
            assert isinstance(seed, list), "seed must be a list when noise is a list"
            if condition_reference is None:
                condition_reference = [None] * len(noise)
                condition_mask = [None] * len(noise)
            else:
                assert isinstance(condition_reference, list), "condition_reference must be a list when noise is a list"
                assert isinstance(condition_mask, list), "condition_mask must be a list when noise is a list"
        else:
            device = noise.device
            assert not isinstance(seed, list), "seed must not be a list when noise is a tensor"
            assert not isinstance(condition_reference, list), (
                "condition_reference must not be a list when noise is a tensor"
            )
            assert not isinstance(condition_mask, list), "condition_mask must not be a list when noise is a tensor"

        t_list = self.t_list

        latent = noise

        for step_idx, (sigma_cur, sigma_next) in enumerate(
            zip(t_list[:-1], t_list[1:]),
        ):
            timestep = torch.tensor(sigma_cur * self.num_train_timesteps, device=device)
            v_pred = velocity_fn(latent, timestep.reshape(1, 1))

            def _sde_step(
                seed: int | None,
                latent: torch.Tensor,
                v_pred: torch.Tensor,
                condition_reference: torch.Tensor | None,
                condition_mask: torch.Tensor | None,
            ) -> torch.Tensor:
                x0_pred = latent - sigma_cur * v_pred  # [...,D]

                if sigma_next > 0:
                    if seed is not None:
                        torch.manual_seed(seed + step_idx)
                    eps_fresh = torch.randn_like(x0_pred)  # [...,D]
                    latent_next = (1.0 - sigma_next) * x0_pred + sigma_next * eps_fresh  # [...,D]
                else:
                    latent_next = x0_pred  # [...,D]

                if condition_reference is not None:
                    assert condition_mask is not None, "condition_mask is required when condition_reference is set"
                    condition_reference = condition_reference.to(
                        dtype=latent_next.dtype, device=latent_next.device
                    )  # [...,D]
                    condition_mask = condition_mask.to(dtype=latent_next.dtype, device=latent_next.device)  # [...,D]
                    latent_next = condition_mask * condition_reference + (1.0 - condition_mask) * latent_next  # [...,D]
                return latent_next

            latent = run_multiseed(
                _sde_step,
                seed=seed,
                latent=latent,
                v_pred=v_pred,
                condition_reference=condition_reference,
                condition_mask=condition_mask,
            )

        return latent
