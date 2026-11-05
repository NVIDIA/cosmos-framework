# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Adapt source draw probabilities toward the configured token mix.

A loader that draws one source per batch spends the whole batch on that source, so
over ``N`` batches source ``i`` contributes ``N * p_i * T_i`` tokens, where ``p_i`` is
its draw probability and ``T_i`` its mean tokens per batch. Its share of the tokens
the job trains on is therefore proportional to ``p_i * T_i``, not to ``p_i``.

Under a token ceiling every ``T_i`` is about that ceiling, so the two coincide and a
configured ratio reads as a token share. An iteration-time ceiling parts them:
attention is quadratic, so an expensive source runs out of time long before it runs
out of tokens and its ``T_i`` falls. On the 30B fit, stills stop on tokens near 49,700
while 480p/61f clips stop on time at 33,770 -- a third of the video mix, silently
reassigned to images.

Restoring the configured share means drawing with ``p_i`` proportional to
``ratio_i / T_i``, which is what this module estimates ``T_i`` for. The estimate is an
EMA over the batches each source actually produced, so it needs no cost model and no
cooperation from the packer: it prices what the packer did, drops, look-ahead misses,
token ceiling and all.

The same estimator supports sample-level mixing, where ``T_i`` is mean tokens per
sample. ``RandomMixedJointDataLoader`` observes each sample only when accepted into
a packed batch. Deferred candidates do not update the estimate or warmup until
accepted. ``TokenMixedDataset`` instead observes packable samples before yielding
them to its parent loader. Packing statistics are reported by loader callbacks
rather than this controller.

The default correction estimates ``T_i``, a property of a source's data under the
packer's ceilings that does not depend on how often that source is drawn. It has no
feedback path into the quantity it measures. Estimator noise is traded against
tracking speed by ``half_life_batches``.

The mixed loader also enables best-effort feedback: multiply the size correction
by target token share / observed token share since this controller was created.
Overrepresented sources get smaller weights on subsequent draws. Its observations
count accepted tokens only, so deferral does not charge a source before acceptance.
This is a soft correction: every eligible source keeps a positive weight,
and fallback to the only eligible source still draws it regardless of its surplus.
Thus packing constraints can prevent the realized mix from reaching its target.

Every rank runs its own controller over its own batches, which keeps it out of the
collective path -- the packer's daemon thread makes any collective there a hazard.
Both ``RandomJointDataLoader`` and ``RandomMixedJointDataLoader`` draw independently
across ranks and converge to the same weights up to noise, whereas
``IterativeJointDataLoader`` seeds its draw so that every rank picks the same source
at a given iteration, and rank-local weights would break that.

The state is not checkpointed. A resumed job re-estimates from its first batches;
its sample-size estimates settle within a few half-lives and its feedback history
starts over.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass
class TokenMixControlConfig:
    """Config for the token-mix controller.

    The historical ``*_batches`` names remain compatible with batch-loader configs.
    EMA half-life and warmup count observations: the mixed loader maps its sample
    settings to these fields and disables correction clipping explicitly.

    Attributes:
        enabled: Whether to adapt the draw at all. When false the loader draws on
            the configured ratios and the token mix skews by whatever the packer's
            ceilings do to each source, which is the behaviour before this existed.
        half_life_batches: EMA half-life for a source's mean tokens per batch,
            counted in batches drawn from THAT source rather than in iterations, so
            a rarely drawn source smooths over as many of its own observations as a
            common one. Lower tracks a changing mixture sooner and passes on more
            noise to the draw.
        warmup_batches_per_source: Batches every source must have produced before
            the draw is adapted at all. Until then the configured ratios stand,
            since one batch of an unlucky shape is a poor estimate of a source.
        max_weight_ratio: Widest factor by which a draw probability may depart from
            its configured share, in either direction, and equivalently the widest
            spread in tokens per batch the default correction will fully invert.
            When feedback is enabled, this bounds the combined correction. A guard rail
            against a bad estimate starving a source rather than a tuning knob, but
            one with a cost: a mixture whose tokens per batch spread wider than this
            is only partly corrected, and the residual shows up as a realized share
            that sits off target with the source pinned at the bound. ``None``
            disables clipping, allowing arbitrary sample-length differences.
        feedback: Apply target share / cumulative observed token share after
            warmup, in addition to the sample-size correction. Observed tokens
            count accepted samples in ``RandomMixedJointDataLoader``.
            This history is independent of the EMA half-life.
            It provides soft feedback without excluding overrepresented sources.
    """

    enabled: bool = True
    half_life_batches: float = 50.0
    warmup_batches_per_source: int = 8
    max_weight_ratio: float | None = 8.0
    feedback: bool = False

    def build(self, dataset_names: Sequence[str], target_shares: Sequence[float]) -> TokenMixController | None:
        """Resolve into a controller, or ``None`` when there is nothing to control.

        Args:
            dataset_names: Source names, in the loader's index order.
            target_shares: Configured ratios in that same order. Normalised here,
                so they may be given as ratios rather than as shares.

        Returns:
            The controller, or ``None`` when disabled or when a single source
            leaves no mix to rebalance.

        Raises:
            ValueError: If a parameter is out of range, or if the names and shares
                disagree in length, or if no share is positive.
        """
        if self.half_life_batches <= 0.0:
            raise ValueError(f"half_life_batches must be positive, got {self.half_life_batches}.")
        if self.warmup_batches_per_source < 1:
            raise ValueError(f"warmup_batches_per_source must be at least 1, got {self.warmup_batches_per_source}.")
        if self.max_weight_ratio is not None and self.max_weight_ratio < 1.0:
            raise ValueError(f"max_weight_ratio must be at least 1, got {self.max_weight_ratio}.")
        if len(dataset_names) != len(target_shares):
            raise ValueError(f"Got {len(dataset_names)} dataset names and {len(target_shares)} target shares.")

        if not self.enabled or len(dataset_names) < 2:
            return None
        return TokenMixController(dataset_names, target_shares, self)


class TokenMixController:
    """Draw probabilities that target each source's configured token share.

    Attributes:
        sampling_probs: The distribution over all sources after the last
            observation. Use ``probabilities(active)`` when sources can exhaust.
        draw_counts: Number of positive-token observations from each source.
        observed_tokens: Cumulative token exposure, counting each observation
            once. ``RandomMixedJointDataLoader`` counts accepted samples only.
    """

    def __init__(
        self,
        dataset_names: Sequence[str],
        target_shares: Sequence[float],
        config: TokenMixControlConfig,
    ) -> None:
        """See :meth:`TokenMixControlConfig.build`, which validates and calls this."""
        shares = np.asarray(target_shares, dtype=np.float64)
        if shares.ndim != 1 or not shares.size or not np.isfinite(shares).all() or np.any(shares <= 0.0):
            raise ValueError(f"Every target share must be positive and finite, got {target_shares}.")
        if len(dataset_names) != len(shares):
            raise ValueError(f"Got {len(dataset_names)} dataset names and {len(shares)} target shares.")

        self.names: list[str] = list(dataset_names)
        self.target_shares: np.ndarray = shares / shares.sum()
        self.config: TokenMixControlConfig = config
        self.sampling_probs: np.ndarray = self.target_shares.copy()
        # Per observation rather than per iteration, matching half_life_batches.
        self._decay: float = 0.5 ** (1.0 / config.half_life_batches)
        self._mean_tokens: np.ndarray = np.zeros(len(self.names), dtype=np.float64)
        self.draw_counts: np.ndarray = np.zeros(len(self.names), dtype=np.int64)
        self.observed_tokens: np.ndarray = np.zeros(len(self.names), dtype=np.int64)

    def observe(self, index: int, tokens: int) -> None:
        """Fold one source draw into the estimate and refresh the probabilities.

        Args:
            index: Source of this observation, in the loader's index order.
            tokens: Tokens in the batch or candidate. Non-positive counts are
                ignored, an empty draw being a fact about the input stream.
        """
        if tokens <= 0:
            return

        self.draw_counts[index] += 1
        self.observed_tokens[index] += tokens
        if self.draw_counts[index] == 1:
            # Seed rather than decay from zero, which would read a source's first
            # observations as cheap and draw it even harder while the error was largest.
            self._mean_tokens[index] = float(tokens)
        else:
            self._mean_tokens[index] *= self._decay
            self._mean_tokens[index] += (1.0 - self._decay) * tokens
        self.sampling_probs = self.probabilities()

    def probabilities(self, active: np.ndarray | None = None) -> np.ndarray:
        """``ratio_i / T_i``, normalised over the active sources after warmup.

        With feedback, multiply by target share / cumulative observed share.
        Shares are normalized over the same active sources, so exhausted or
        temporarily excluded sources do not change their relative correction.
        Warmup guarantees positive exposure for every corrected source.

        Disabled controllers return the configured ratios over active sources.
        An all-false mask returns zeros; the caller must stop drawing in that case.
        Inactive sources affect neither warmup nor correction clipping.

        When configured, the combined size and feedback correction is clipped in
        log space around its geometric mean rather than after normalising. Ratios
        between corrections survive normalisation where absolute values do not, so bounding their spread there is
        what makes ``max_weight_ratio`` an exact bound on the departure of every
        ``p_i`` from its configured share, instead of one that normalising undoes.
        """
        if active is None:
            active = np.ones(len(self.names), dtype=bool)
        weights = np.where(active, self.target_shares, 0.0)
        if not active.any():
            return weights
        if not self.config.enabled or (self.draw_counts[active] < self.config.warmup_batches_per_source).any():
            return weights / weights.sum()

        correction = 1.0 / self._mean_tokens[active]
        if self.config.feedback:
            exposure = self.observed_tokens[active]
            observed_shares = exposure / exposure.sum()
            target_shares = weights[active] / weights.sum()
            correction *= target_shares / observed_shares

        if self.config.max_weight_ratio is None:
            weights[active] *= correction
        else:
            correction = np.log(correction)
            correction -= correction.mean()
            half_band = 0.5 * np.log(self.config.max_weight_ratio)
            weights[active] *= np.exp(np.clip(correction, -half_band, half_band))
        return weights / weights.sum()
