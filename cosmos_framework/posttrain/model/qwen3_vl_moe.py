# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Native Qwen3-VL-MoE policy differences.

Shared Qwen3-VL construction, forward compatibility, and weight semantics live in
``model/qwen3_vl.py``. This module owns the sparse-MoE batch adaptation, the routing-replay
capability (``ForwardReplayPolicy``: pin the per-token top-k expert selection of every adapted block
to the routing the policy batch carries) and the MoE forward-noise tolerance (``ForwardNoisePolicy``).

The vendored sparse block exposes one algorithm-neutral seam, ``select_experts`` (the dispatch
selection and its combine weights); every replay semantic -- what a pin is, how the combine weights
are formed over pinned experts, where the pin state lives -- is installed on the adapted instances
from here and nowhere else.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from types import MethodType
from typing import Any

import torch
import torch.nn as nn

from cosmos_framework.posttrain.model.protocol import STRICT_FORWARD_NOISE_TOLERANCE, ForwardNoiseTolerance
from cosmos_framework.posttrain.model.qwen3_vl import Qwen3VLPolicy
from cosmos_framework.posttrain.schema.routing import FORWARD_REPLAY_BATCH_KEY

# Pinned routing still leaves ~1e-3 kernel jitter in the combine weights and the dense layers, and
# on Qwen3-VL-MoE-30B-A3B a few tokens per chunk move past 0.05 nats even with the dispatch fixed
# (worst pinned token 0.115-0.178 on a real arm). The FRACTION of moved tokens is stable in the
# chunk length where a max-abs bound is not, and dropout at p=0.1 would move ~100% of tokens.
PINNED_MOE_FORWARD_NOISE_TOLERANCE = ForwardNoiseTolerance(per_token_nats=0.05, moved_fraction=0.05)


def native_qwen3_vl_moe_classes() -> tuple[type[Any], type[nn.Module]]:
    """Resolve the vendored MoE config/model without importing it on dense-only paths."""
    from cosmos_framework.model.generator.reasoner.qwen3_vl_moe.configuration_qwen3_vl_moe import (
        Qwen3VLMoeConfig,
    )
    from cosmos_framework.model.generator.reasoner.qwen3_vl_moe.qwen3_vl_moe import (
        Qwen3VLMoeForConditionalGeneration,
    )

    return Qwen3VLMoeConfig, Qwen3VLMoeForConditionalGeneration


def override_qwen3_vl_moe_for_padded_batches(net: nn.Module) -> int:
    """Adapt native sparse-MoE blocks from packed ``[N,H]`` to policy ``[B,T,H]`` and install the
    posttrain routing-replay selection on them.

    The vendored block deliberately returns ``(tensor, LBLMetadata)`` and requires rank-2 input,
    while its conditional-generation decoder calls it like an HF MLP with padded rank-3 hidden
    states. The adapter flattens/restores rows and deliberately discards the unused graph-bearing
    LBL metadata without changing parameters or state-dict names. It also binds `pinned_select_experts`
    over the block's own ``select_experts`` so a staged pin decides the dispatch. Both overrides are
    bound on the instance (no class swap, so ``type(module)`` and the state dict are untouched) and
    must run before activation checkpointing/FSDP wrapping.
    """
    from cosmos_framework.model.generator.reasoner.qwen3_vl_moe.qwen3_vl_moe import (
        Qwen3VLMoeTextSparseMoeBlock,
    )

    overridden = 0
    for module in net.modules():
        if not isinstance(module, Qwen3VLMoeTextSparseMoeBlock):
            continue
        if bool(getattr(module, "_posttrain_padded_forward", False)):
            continue
        original_forward = module.forward
        original_select_experts = module.select_experts

        def padded_forward(
            self: Any,
            hidden_states: torch.Tensor,
            _original_forward: Callable[..., Any] = original_forward,
        ) -> torch.Tensor:  # hidden_states: [B,T,H] or [N,H], returns same shape
            original_shape = hidden_states.shape
            flat_hidden = hidden_states.reshape(-1, original_shape[-1])  # [N,H]
            # Written only by `Qwen3VLMoePolicy.prepare_policy_batch` and only READ on the forward
            # path (here and in `pinned_select_experts`): an activation-checkpoint recompute re-runs
            # this forward and must see the same mask and the same selection.
            token_mask = getattr(self, "_posttrain_token_mask", None)
            routed, metadata = _original_forward(flat_hidden, token_mask=token_mask)  # [N,H]
            # The LBL metadata is kept only for an objective that asked for it (`set_lbl_collection`, the
            # SFT load-balancing term) and only by the REAL forward: non-reentrant activation checkpointing
            # re-runs this forward inside backward, after the objective has popped the stash, and a
            # re-stash would pin that layer's recomputed graph for the rest of the backward (the same
            # rule `cosmos3/utils/monkey_patch.py`'s patched block follows). OPD never asks, so its
            # path stays byte-identical: nothing is stashed across optimizer steps.
            if getattr(self, "_posttrain_collect_lbl", False) and not _in_autograd_backward():
                self._posttrain_lbl_metadata = metadata
            return routed.reshape(original_shape)  # [B,T,H] or [N,H]

        module.forward = MethodType(padded_forward, module)
        module.select_experts = MethodType(
            functools.partial(pinned_select_experts, natural_select_experts=original_select_experts), module
        )
        module._posttrain_padded_forward = True
        module._posttrain_token_mask = None
        module._posttrain_routing_replay_indices = None
        module._posttrain_collect_lbl = False
        module._posttrain_lbl_metadata = None
        overridden += 1
    return overridden


def _in_autograd_backward() -> bool:
    """True inside an autograd backward pass (the activation-checkpoint recompute), False in the real forward and
    while Dynamo traces; `torch.utils.checkpoint` draws the line the same way (graph task id -1 = no backward)."""
    if torch.compiler.is_compiling():
        return False
    return torch._C._current_graph_task_id() != -1


def set_lbl_collection(net: nn.Module, enabled: bool) -> int:
    """Switch the per-forward LBL stash of every adapted sparse block; returns the number of blocks switched (0 for a
    dense network, which lets a caller refuse a load-balancing coefficient the model cannot honour)."""
    count = 0
    for module in net.modules():
        if bool(getattr(module, "_posttrain_padded_forward", False)):
            module._posttrain_collect_lbl = bool(enabled)
            if not enabled:
                module._posttrain_lbl_metadata = None
            count += 1
    return count


def collect_lbl_metadata(net: nn.Module) -> list[Any]:
    """Pop the stashed per-layer ``LBLMetadata`` in module order. Popping is the point: the mean router probability
    carries the router's autograd graph, and a stash that outlives the objective would pin a forward's activations."""
    per_layer: list[Any] = []
    for module in net.modules():
        if bool(getattr(module, "_posttrain_padded_forward", False)):
            metadata = getattr(module, "_posttrain_lbl_metadata", None)
            if metadata is not None:
                module._posttrain_lbl_metadata = None
                per_layer.append(metadata)
    return per_layer


def pinned_select_experts(
    block: Any,
    selection_logits: torch.Tensor,  # [N,num_experts]
    selection_scores: torch.Tensor,  # [N,num_experts]
    natural_indices: torch.Tensor,  # [N,top_k]
    natural_weights: torch.Tensor,  # [N,top_k]
    *,
    natural_select_experts: Callable[..., tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """The posttrain ``select_experts``: the staged pin, if any, decides the dispatch.

    Rollout Routing Replay (Ma et al., arXiv 2510.11370, section 4.1) pins the top-k SELECTION of
    every token to the experts the rollout engine used, while the combine weights are still gathered
    from THIS forward's router scores and normalized over the pinned experts, so the router keeps
    receiving gradients and only the dispatch set is replayed. Pinned experts may hold no softmax
    mass at all in this forward (a row can underflow to exactly zero), so the renormalization goes
    through the router's ``normalize_scores``: its clamp keeps the forward finite and its uniform
    fallback keeps the zero row's gradient finite, where a plain division would produce NaN.

    Without a staged pin this defers to the block's own selection, so an unpinned batch runs the
    vendored default path exactly.

    Args:
        block: the adapted sparse block (bound as ``self``).
        selection_logits: (num_tokens, num_experts) router logits the selection is made on.
        selection_scores: (num_tokens, num_experts) the activated ``selection_logits``.
        natural_indices: (num_tokens, top_k) the gate's own top-k over ``selection_scores``.
        natural_weights: (num_tokens, top_k) the ``selection_scores`` at ``natural_indices``.
        natural_select_experts: the block's original bound ``select_experts``.

    Returns:
        ``(expert_indices, expert_weights)``, both (num_tokens, top_k): the dispatch and its
        row-normalized combine weights.

    Raises:
        ValueError: a staged pin does not describe this forward's rows; unreachable after
            `Qwen3VLMoePolicy._pin` validated the record against the padded batch.
    """
    pinned = getattr(block, "_posttrain_routing_replay_indices", None)  # [N,top_k] | None
    if pinned is None:
        return natural_select_experts(selection_logits, selection_scores, natural_indices, natural_weights)
    if pinned.shape != natural_indices.shape:
        raise ValueError(
            f"the staged routing pin has shape {tuple(pinned.shape)} but this forward routes "
            f"{tuple(natural_indices.shape)} (num_tokens, top_k); the replayed routing does not describe this batch"
        )
    expert_indices = pinned.to(device=selection_scores.device, dtype=torch.long)  # [N,top_k]
    expert_weights = selection_scores.gather(1, expert_indices)  # [N,top_k]
    return expert_indices, block.cosine_router.normalize_scores(expert_weights)


class Qwen3VLMoePolicy(Qwen3VLPolicy):
    """Qwen3-VL policy adapter that publishes the padded-token mask to sparse experts and owns the
    routing-replay and load-balancing-collection capabilities (``ForwardReplayPolicy``, ``LoadBalancingPolicy``)
    over the same blocks."""

    def __init__(self, net: nn.Module) -> None:
        super().__init__(net, "qwen3_vl_moe")
        self._replay_pinned = False
        self._replay_pinned_extent = (0, 0)  # (unpadded token rows, blocks) of the current pin

    def _adapted_blocks(self) -> list[nn.Module]:
        from cosmos_framework.model.generator.reasoner.qwen3_vl_moe.qwen3_vl_moe import (
            Qwen3VLMoeTextSparseMoeBlock,
        )

        blocks = [
            module
            for module in self.net.modules()
            if isinstance(module, Qwen3VLMoeTextSparseMoeBlock)
            and bool(getattr(module, "_posttrain_padded_forward", False))
        ]
        if not blocks:
            raise ValueError("Qwen3-VL-MoE policy holds no padded-batch-adapted sparse-MoE block")
        return blocks

    def prepare_policy_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move the common VLM batch, publish its valid rows to every sparse block, and stage the
        routing it carries (``FORWARD_REPLAY_BATCH_KEY``) as the pin of every adapted block.

        Every prepared batch rewrites the pin (to its routing, or to ``None``), so a pin never
        outlives the forwards of the batch that carried it and needs no release call. This is the
        single writer of the pin (``ForwardReplayPolicy``): the routing the batch carries is the
        rollout engine's record, aligned by the packer, and nothing else stages one. The routing key
        is consumed here and never forwarded into the network.

        Raises:
            ValueError: the routing does not describe this model (block count, top-k, expert ids
                outside ``[0, num_experts)``, an expert repeated within a row) or this batch (padded
                token rows), or is not an integer tensor. Nothing is staged when this raises.
        """
        prepared = super().prepare_policy_batch(batch)
        from cosmos_framework.model.generator.reasoner.qwen3_vl_moe.qwen3_vl_moe import (
            Qwen3VLMoeTextSparseMoeBlock,
        )

        attention_mask = prepared.get("attention_mask")
        token_mask = None if attention_mask is None else attention_mask.reshape(-1).bool()  # [B*T]
        expected_rows = int(prepared["input_ids"].numel())
        if token_mask is not None and token_mask.numel() != expected_rows:
            raise ValueError(f"Qwen3-VL-MoE token mask has {token_mask.numel()} rows for {expected_rows} padded tokens")
        routing = prepared.pop(FORWARD_REPLAY_BATCH_KEY, None)  # [B,T,L,top_k] | None
        for module in self.net.modules():
            if isinstance(module, Qwen3VLMoeTextSparseMoeBlock):
                module._posttrain_token_mask = token_mask
                module._posttrain_routing_replay_indices = None
        self._replay_pinned = False
        self._replay_pinned_extent = (0, 0)
        if routing is not None:
            blocks = self._pin(routing, expected_rows)
            # Rows the blocks dispatch through the pin: the mask's rows when the packer supplies one,
            # every padded row otherwise (the vendored block routes all rows without a mask).
            pinned_rows = expected_rows if token_mask is None else int(token_mask.sum())
            self._replay_pinned = True
            self._replay_pinned_extent = (pinned_rows, blocks)
        return prepared

    def _replay_geometry(self) -> tuple[list[nn.Module], int, int]:
        """The adapted blocks and the ``(top_k, num_experts)`` they all share.

        Raises:
            ValueError: no adapted block, or the blocks disagree, so no one record can describe them.
        """
        blocks = self._adapted_blocks()
        geometries = {(int(block.top_k), int(block.num_experts)) for block in blocks}
        if len(geometries) != 1:
            raise ValueError(
                f"Qwen3-VL-MoE sparse blocks disagree on (top_k, num_experts): {sorted(geometries)}; one "
                "recorded routing cannot describe them"
            )
        ((top_k, num_experts),) = geometries
        return blocks, top_k, num_experts

    def _pin(self, routing: torch.Tensor, rows: int) -> int:
        """Validate ``routing`` against this model and this batch, then stage it on every adapted block.

        Every check runs before any block state is touched: the record's type, its shape against the
        padded rows and the block geometry, every id inside ``[0, num_experts)`` and no expert repeated
        within a row. An out-of-range or repeated id would otherwise reach the gather inside the
        collective-bearing forward, where one rank fails while its peers walk into FSDP and hang.

        Returns:
            The number of blocks pinned.
        """
        blocks, top_k, num_experts = self._replay_geometry()
        if not isinstance(routing, torch.Tensor) or routing.is_floating_point() or routing.dtype == torch.bool:
            raise ValueError(
                f"{FORWARD_REPLAY_BATCH_KEY} must be an integer tensor of expert ids, got "
                f"{type(routing).__name__}{'' if not isinstance(routing, torch.Tensor) else f' of {routing.dtype}'}"
            )
        expected = (rows, len(blocks), top_k)
        actual = (
            (routing.shape[0] * routing.shape[1], routing.shape[2], routing.shape[3]) if routing.ndim == 4 else None
        )
        if actual != expected:
            raise ValueError(
                f"{FORWARD_REPLAY_BATCH_KEY} has shape {tuple(routing.shape)} but this batch routes "
                f"{rows} padded tokens through {len(blocks)} sparse-MoE blocks at top_k={top_k} "
                "([B,T,blocks,top_k]); the recorded routing does not describe this model or this batch"
            )
        flat = routing.reshape(-1, top_k)  # [rows*blocks,top_k]
        if flat.numel():
            lowest, highest = int(flat.min()), int(flat.max())
            if lowest < 0 or highest >= num_experts:
                raise ValueError(
                    f"{FORWARD_REPLAY_BATCH_KEY} holds expert ids from {lowest} to {highest} but this model "
                    f"routes over {num_experts} experts (valid ids 0..{num_experts - 1})"
                )
            sorted_rows = flat.sort(dim=-1).values  # [rows*blocks,top_k]
            if bool((sorted_rows[:, 1:] == sorted_rows[:, :-1]).any()):
                raise ValueError(
                    f"{FORWARD_REPLAY_BATCH_KEY} repeats an expert within a top-{top_k} row; a token is "
                    f"dispatched to {top_k} distinct experts"
                )
        for layer, block in enumerate(blocks):
            # Staged as delivered (the packer's narrow dtype); `pinned_select_experts` widens on use.
            block._posttrain_routing_replay_indices = routing[:, :, layer, :].reshape(rows, top_k)  # [N,top_k]
        return len(blocks)

    # ---------------------------------------------------------------- ForwardReplayPolicy

    def replay_pinned(self) -> bool:
        return self._replay_pinned

    def replay_pinned_extent(self) -> tuple[int, int]:
        return self._replay_pinned_extent

    def replay_identity(self) -> dict[str, Any]:
        blocks, top_k, num_experts = self._replay_geometry()
        return {
            "kind": "qwen3_vl_moe_top_k_experts",
            "blocks": len(blocks),
            "top_k": top_k,
            "num_experts": num_experts,
        }

    # ---------------------------------------------------------------- ForwardNoisePolicy

    def forward_noise_tolerance(self, *, replay_pinned: bool) -> ForwardNoiseTolerance:
        return PINNED_MOE_FORWARD_NOISE_TOLERANCE if replay_pinned else STRICT_FORWARD_NOISE_TOLERANCE

    # ---------------------------------------------------------------- LoadBalancingPolicy

    def begin_lbl_collection(self) -> int:
        """Switch the per-forward LBL stash on over the adapted sparse blocks (:func:`set_lbl_collection`); the count
        lets the objective refuse a coefficient a network without sparse blocks cannot honour (the objective reads
        this capability, never this module)."""
        return set_lbl_collection(self.net, True)

    def collect_lbl_metadata(self) -> list[Any]:
        """Pop the stashed per-layer ``LBLMetadata`` of the last forward, in module order
        (:func:`collect_lbl_metadata`)."""
        return collect_lbl_metadata(self.net)
