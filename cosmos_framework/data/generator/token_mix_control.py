# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Holds the realized token mix on the configured ratios when packing is cost-based.

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

The same estimator supports sample-level mixing: each observation is one fresh,
packable candidate and ``T_i`` is mean tokens per sample. Deferred candidates must
be observed only once. Their eventual acceptance is accounted for by the loader,
separately from the candidate estimate, using ``record_realized=False`` here.

This is a plug-in rather than a servo. ``T_i`` is a property of a source's data under
the packer's ceilings and does not depend on how often that source is drawn, so the
loop has no feedback path into the quantity it measures: no gain to tune, nothing to
ring, and the mix lands on target as soon as the EMA has warmed up. What is left is
estimator noise, traded against tracking speed by ``half_life_batches``.

Every rank runs its own controller over its own batches, which keeps it out of the
collective path -- the packer's daemon thread makes any collective there a hazard.
Both ``RandomJointDataLoader`` and ``RandomMixedJointDataLoader`` draw independently
across ranks and converge to the same weights up to noise, whereas
``IterativeJointDataLoader`` seeds its draw so that every rank picks the same source
at a given iteration, and rank-local weights would break that.

The state is not checkpointed. A resumed job re-estimates from its first batches and
is back on target within a few half-lives.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from cosmos_framework.utils import log


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
            spread in tokens per batch the draw will fully invert. A guard rail
            against a bad estimate starving a source rather than a tuning knob, but
            one with a cost: a mixture whose tokens per batch spread wider than this
            is only partly corrected, and the residual shows up as a realized share
            that sits off target with the source pinned at the bound. ``None``
            disables clipping, allowing arbitrary sample-length differences.
        log_every_n_batches: How often to log target against realized share. The
            realized figure is measured over that window, so it is also how long
            the diagnostic takes to reflect a change.
    """

    enabled: bool = True
    half_life_batches: float = 50.0
    warmup_batches_per_source: int = 8
    max_weight_ratio: float | None = 8.0
    log_every_n_batches: int = 500

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
        if self.log_every_n_batches < 1:
            raise ValueError(f"log_every_n_batches must be at least 1, got {self.log_every_n_batches}.")
        if len(dataset_names) != len(target_shares):
            raise ValueError(f"Got {len(dataset_names)} dataset names and {len(target_shares)} target shares.")

        if not self.enabled or len(dataset_names) < 2:
            return None
        return TokenMixController(dataset_names, target_shares, self)


class TokenMixController:
    """Draw probabilities that keep each source's token share on its configured one.

    Attributes:
        sampling_probs: The distribution over all sources after the last
            observation. Use ``probabilities(active)`` when sources can exhaust.
        draw_counts: Number of positive-token observations from each source.
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
        self._window_tokens: np.ndarray = np.zeros(len(self.names), dtype=np.float64)
        self._window_batches: int = 0
        self._at_band: np.ndarray = np.zeros(len(self.names), dtype=bool)

    def observe(self, index: int, tokens: int, *, record_realized: bool = True) -> None:
        """Fold one source draw into the estimate and refresh the probabilities.

        Args:
            index: Source of this observation, in the loader's index order.
            tokens: Tokens in the batch or candidate. Non-positive counts are
                ignored, an empty draw being a fact about the input stream.
            record_realized: Record this observation in the accepted-token window.
                Batch draws are already packed. Sample candidates may be deferred,
                so their loader passes false and tracks accepted tokens separately.
        """
        if tokens <= 0:
            return

        self.draw_counts[index] += 1
        if self.draw_counts[index] == 1:
            # Seed rather than decay from zero, which would read a source's first
            # observations as cheap and draw it even harder while the error was largest.
            self._mean_tokens[index] = float(tokens)
        else:
            self._mean_tokens[index] *= self._decay
            self._mean_tokens[index] += (1.0 - self._decay) * tokens
        self.sampling_probs = self.probabilities()
        if not record_realized:
            return

        self._window_tokens[index] += tokens
        self._window_batches += 1

        if self._window_batches >= self.config.log_every_n_batches:
            # Rank 0 only, for volume: every rank holds its own controller, but they see
            # the same data and differ by estimator noise, so one rank's view is the mix.
            log.info(self.describe())
            self._window_tokens[:] = 0.0
            self._window_batches = 0

    def realized_shares(self) -> np.ndarray:
        """Token share each source won since the last log, or zeros before any batch."""
        total = self._window_tokens.sum()
        if total <= 0.0:
            return np.zeros_like(self._window_tokens)
        return self._window_tokens / total

    def describe(self) -> str:
        """Target against realized share and the draw in force, one line per source."""
        realized = self.realized_shares()
        warming = int((self.draw_counts < self.config.warmup_batches_per_source).sum())
        header = f"token mix over {self._window_batches} batches"
        if warming:
            header += f" (warming up: {warming} source(s) below {self.config.warmup_batches_per_source} observations)"
        lines = [header]
        for index, name in enumerate(self.names):
            # Naming the sources at the band is the whole diagnostic for a mix that is
            # off target and staying there: it says to widen max_weight_ratio rather
            # than to distrust the ratios.
            at_band = " (at the band, so only partly corrected)" if self._at_band[index] else ""
            lines.append(
                f"  {name}: target {self.target_shares[index]:.1%}, realized {realized[index]:.1%}, "
                f"drawn at {self.sampling_probs[index]:.1%}, {self._mean_tokens[index]:,.0f} tokens/observation{at_band}"
            )
        return "\n".join(lines)

    def probabilities(self, active: np.ndarray | None = None) -> np.ndarray:
        """``ratio_i / T_i``, normalised over the active sources after warmup.

        Disabled controllers return the configured ratios over active sources.
        An all-false mask returns zeros; the caller must stop drawing in that case.
        Inactive sources affect neither warmup nor correction clipping.

        When configured, ``1 / T_i`` is clipped in log space around its geometric mean
        rather than after normalising. Ratios between corrections survive
        normalisation where absolute values do not, so bounding their spread there is
        what makes ``max_weight_ratio`` an exact bound on the departure of every
        ``p_i`` from its configured share, instead of one that normalising undoes.

        Also records which sources the clip caught, for :meth:`describe`.
        """
        if active is None:
            active = np.ones(len(self.names), dtype=bool)
        weights = np.where(active, self.target_shares, 0.0)
        self._at_band[:] = False
        if not active.any():
            return weights
        if not self.config.enabled or (self.draw_counts[active] < self.config.warmup_batches_per_source).any():
            return weights / weights.sum()

        if self.config.max_weight_ratio is None:
            weights[active] /= self._mean_tokens[active]
        else:
            correction = -np.log(self._mean_tokens[active])
            correction -= correction.mean()
            half_band = 0.5 * np.log(self.config.max_weight_ratio)
            self._at_band[active] = np.abs(correction) > half_band
            weights[active] *= np.exp(np.clip(correction, -half_band, half_band))
        return weights / weights.sum()
