# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import importlib
from collections.abc import Callable

import torch
import transformers
from transformers.cache_utils import Cache
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLModel,
    Qwen3VLVisionAttention,
    Qwen3VLVisionModel,
)
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeModel,
    Qwen3VLMoeTextSparseMoeBlock,
    Qwen3VLMoeVisionAttention,
    Qwen3VLMoeVisionModel,
)
from transformers.utils.import_utils import is_torchdynamo_compiling

from cosmos_framework.utils import log

_EXPECTED_TRANSFORMERS_VERSION_PREFIX = "4.57."


def _in_autograd_backward() -> bool:
    """True when the caller is executing inside an autograd backward pass.

    Used to tell a module's real forward apart from the re-execution that non-reentrant
    activation checkpointing performs during backward. ``torch.utils.checkpoint`` draws the
    same distinction the same way: a graph task id of ``-1`` means "not inside a backward"
    (see its ``unpack_hook``).

    Reports False while Dynamo traces, for two reasons. ``torch._C._current_graph_task_id``
    has no Dynamo handler, so leaving it in the traced region costs a graph break — a hard
    error under ``fullgraph=True``, which is why this probe alone would keep the VLM path
    pinned to ``fullgraph=False`` (see ``parallelize_vlm._compile_kwargs``). And there is
    nothing to suppress: with the checkpoint inside the compiled region
    (``parallelize_vlm.apply_compile``) Dynamo traces this forward ONCE and the
    activation-checkpoint higher-order op replays that subgraph during backward instead of
    re-entering Python, so the stash below cannot run twice. Should the HOP fail to form and
    AC fall back to eager checkpointing, the recompute runs untraced and takes the real probe.
    """
    if torch.compiler.is_compiling():
        return False
    return torch._C._current_graph_task_id() != -1


def patch_qwen3_vl_forward(model):
    """Monkey-patch a ``Qwen3VLModel`` / ``Qwen3VLMoeModel`` **instance's** forward:
       **Single visual forward per batch**: Under FSDP, every rank must call
       ``self.visual(...)`` the same number of times each forward step so that
       collective all-gather operations stay in sync. Image and video inputs are
       encoded together in one visual call. When a batch contains only text, a
       lightweight dummy image (16x16 zeros) is pushed through the full ViT ->
       merger -> deepstack pipeline, then outputs are sliced to ``[0:0]`` so
       they carry ``grad_fn`` but contribute no features.

       Handles both the dense (``Qwen3VLModel``) and MoE (``Qwen3VLMoeModel``)
       backbones: their ``self.visual`` return signature, ``get_placeholder_mask``,
       ``get_rope_index`` and language-model call signature are identical for the
       purposes of this patch; only the output dataclass differs (resolved below).

    Args:
        model: The ``Qwen3VLModel`` / ``Qwen3VLMoeModel`` instance (i.e.
            ``model.model.model`` when the outer model is ``HFModel``).
    """
    if not transformers.__version__.startswith(_EXPECTED_TRANSFORMERS_VERSION_PREFIX):
        raise ValueError(f"monkey patching transformers version {transformers.__version__} is not supported")

    if not isinstance(model, (Qwen3VLModel, Qwen3VLMoeModel)):
        raise ValueError(
            f"Trying to monkey patch a model that is not a Qwen3VLModel/Qwen3VLMoeModel instance: {type(model)}"
        )

    # Resolve the output dataclass from the actual runtime module. Both dense and
    # MoE name it "<ModelClass>OutputWithPast" (Qwen3VLModel -> Qwen3VLModelOutputWithPast,
    # Qwen3VLMoeModel -> Qwen3VLMoeModelOutputWithPast) and both expose the only fields
    # this patch populates: (last_hidden_state, past_key_values, rope_deltas).
    model_module = importlib.import_module(type(model).__module__)
    output_cls = getattr(model_module, f"{type(model).__name__}OutputWithPast")

    # Replaces Qwen3VLModel/Qwen3VLMoeModel.forward from:
    #   transformers.models.qwen3_vl.modeling_qwen3_vl          (transformers v4.57.1)
    #   transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe  (transformers v4.57.1)
    def patched_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ):
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None
        deepstack_image_embeds = None
        deepstack_video_embeds = None

        visual_pixel_values_list: list[torch.Tensor] = []
        visual_grid_thw_list: list[torch.Tensor] = []
        image_embed_len = 0
        video_embed_len = 0
        has_image = pixel_values is not None
        has_video = pixel_values_videos is not None

        if has_image:
            if image_grid_thw is None:
                raise ValueError("image_grid_thw must be provided when pixel_values is provided")
            visual_pixel_values_list.append(pixel_values)
            visual_grid_thw_list.append(image_grid_thw)
            image_embed_len = int((image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).sum().item())

        if has_video:
            if video_grid_thw is None:
                raise ValueError("video_grid_thw must be provided when pixel_values_videos is provided")
            visual_pixel_values_list.append(pixel_values_videos)
            visual_grid_thw_list.append(video_grid_thw)
            video_embed_len = int((video_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).sum().item())

        if has_image or has_video:
            visual_pixel_values = torch.cat(visual_pixel_values_list, dim=0).type(self.visual.dtype)  # [N_patch,D]
            visual_grid_thw = torch.cat(visual_grid_thw_list, dim=0)  # [N_media,3]
            visual_embeds, deepstack_visual_feature_lists = self.visual(
                visual_pixel_values, grid_thw=visual_grid_thw
            )  # visual_embeds: [N_visual,C]
            image_embeds = visual_embeds[:image_embed_len]  # [N_image_visual,C]
            video_embeds = visual_embeds[image_embed_len : image_embed_len + video_embed_len]  # [N_video_visual,C]
            deepstack_image_embeds = [
                deepstack_visual_embeds[:image_embed_len] for deepstack_visual_embeds in deepstack_visual_feature_lists
            ]  # each: [N_image_visual,C]
            deepstack_video_embeds = [
                deepstack_visual_embeds[image_embed_len : image_embed_len + video_embed_len]
                for deepstack_visual_embeds in deepstack_visual_feature_lists
            ]  # each: [N_video_visual,C]

            if has_image:
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)  # [N_image_visual,C]
                image_mask, _ = self.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)  # [B,N_token,C]

            if has_video:
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)  # [N_video_visual,C]
                _, video_mask = self.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
                )
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)  # [B,N_token,C]

        # Dummy visual forward for text-only data
        else:
            dummy_h, dummy_w = 16, 16
            dummy_pixels = torch.zeros(
                dummy_h * dummy_w,
                self.visual.config.temporal_patch_size * self.visual.config.patch_size**2 * 3,
                device=inputs_embeds.device,
                dtype=self.visual.dtype,
            )  # [N_dummy_patch,D]
            dummy_thw = torch.tensor([[1, dummy_h, dummy_w]], device=inputs_embeds.device)  # [1,3]
            image_embeds, deepstack_image_embeds = self.visual(dummy_pixels, grid_thw=dummy_thw)
            image_embeds = image_embeds[0:0]  # [0,C]
            deepstack_image_embeds = [e[0:0] for e in deepstack_image_embeds]  # each: [0,C]

            # no-op to mask scatter empty embeddings into inputs to preserve computation graph
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)  # [0,C]
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)  # [B,N_token,C]

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            # aggregate visual_pos_masks and deepstack_visual_embeds
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        if position_ids is None:
            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                # Only apply conversion for floating point tensors (inverted masks)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do asssisted decoding
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device) if cache_position is not None else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        return output_cls(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        )

    # Replace the forward method
    model.forward = patched_forward.__get__(model, type(model))
    log.critical(f"Patched {type(model).__name__} instance forward with one visual call per forward")


def patch_qwen3_vl_moe_grouped_mm_experts(model: torch.nn.Module) -> int:
    """Monkey-patch every ``Qwen3VLMoeTextSparseMoeBlock`` **instance** under ``model`` to run
    its experts through ``torch._grouped_mm`` instead of HF's per-expert Python loop.

    HF's ``Qwen3VLMoeTextExperts.forward`` loops over every *hit* expert in training mode,
    issuing two small GEMMs plus a gather/``index_add_`` per expert (128 experts for
    30B-A3B, in all 48 MoE layers). ``Qwen3VLMoeTextExpertsGroupedMm`` instead sorts tokens
    into expert-contiguous order once and runs the whole layer as two ``torch._grouped_mm``
    calls.

    Patched in place, per MoE block:
    - ``block.experts.forward`` (and its ``_reorder_tokens`` helper) is bound from
      ``Qwen3VLMoeTextExpertsGroupedMm``. Those methods read ``gate_up_proj``
      ``[num_experts,hidden_size,2*moe_intermediate_size]``, ``down_proj``
      ``[num_experts,moe_intermediate_size,hidden_size]``, ``act_fn`` and ``top_k`` (the
      expert count comes from the ``num_tokens_per_expert`` argument, not from ``self``).
      The HF experts module already owns the parameters and ``act_fn`` under identical
      names, so the module object is reused as-is and only ``top_k`` is attached. Nothing is
      re-parametrized: state-dict keys, the fused ``mlp.experts.*`` mapping in
      ``safetensors_loader.load_vlm_model``, and the FSDP dim-0 (expert-axis) shard plan are
      all untouched.
    - ``block.forward`` keeps HF's routing math (fp32 softmax → top-k → renormalize → cast)
      but hands the top-k scores/indices straight to the experts instead of materializing
      the dense ``[num_tokens,num_experts]`` routing-weight matrix. It also stashes this
      layer's load-balancing statistics on the block for
      ``moe_utils.collect_hf_moe_lbl_metadata``; that is far cheaper than HF's
      ``output_router_logits`` route, which keeps every layer's ``[num_tokens,num_experts]``
      logits alive to recompute the same softmax. The tuple's second slot — HF's
      ``router_logits``, read only by ``output_router_logits=True`` — is therefore ``None``:
      that path is unsupported here and must fail rather than diverge from the statistics
      training actually uses. HF's decoder layer unpacks and discards it, so the training
      forward is unaffected.
    - ``block.forward`` takes no mask argument (HF's signature is ``forward(hidden_states)``
      and its decoder layer calls it that way), so it reads the step's token mask off
      ``block.token_mask``, which ``moe_utils.set_hf_moe_token_mask`` publishes from the
      collate's ``attention_mask``. The mask keeps the padded rows out of the expert compute
      and out of every routing statistic; with no mask published, every row counts, which is
      what an unpadded stream wants.

    Requires CUDA (the permutation-index kernel is Triton) and bf16 activations, which
    ``torch._grouped_mm`` requires. Must run BEFORE ``parallelize()`` so FSDP captures the
    patched forwards.

    Args:
        model: Module containing ``Qwen3VLMoeTextSparseMoeBlock`` submodules — i.e. the raw
            HF model (``HFModel.model``, a ``Qwen3VLMoeForConditionalGeneration``).

    Returns:
        Number of MoE blocks patched.

    Raises:
        ValueError: On an unsupported transformers version, or when ``model`` contains no
            ``Qwen3VLMoeTextSparseMoeBlock`` (wrong model family — patching nothing would
            silently leave the slow path in place).
    """
    if not transformers.__version__.startswith(_EXPECTED_TRANSFORMERS_VERSION_PREFIX):
        raise ValueError(f"monkey patching transformers version {transformers.__version__} is not supported")

    # Local imports: moe_kernels imports triton at module scope, and the dense (non-MoE)
    # backbones must not pay that import.
    from cosmos_framework.model.generator.reasoner.qwen3_vl_moe.moe import Qwen3VLMoeTextExpertsGroupedMm
    from cosmos_framework.model.generator.reasoner.qwen3_vl_moe.qwen3_vl_moe import (
        LBLMetadata,
        _weighted_expert_counts,
        _weighted_mean,
    )

    # Replaces Qwen3VLMoeTextSparseMoeBlock.forward from:
    #   transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe  (transformers v4.57.1)
    def patched_moe_block_forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, None]:
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)  # [num_tokens,hidden_size]

        # Which rows are real tokens, published for this step by moe_utils.set_hf_moe_token_mask.
        # Absent (None) means an unpadded stream, where every row counts. A stale mask from an
        # earlier step would silently mask the wrong rows, so its length is checked rather than
        # trusted: the collate pads each step to its own multiple of 16, so lengths differ.
        token_mask = getattr(self, "token_mask", None)  # [num_tokens] or None
        if token_mask is not None and token_mask.shape[0] != hidden_states.shape[0]:
            raise ValueError(
                f"token_mask has {token_mask.shape[0]} rows but this forward has "
                f"{hidden_states.shape[0]}: moe_utils.set_hf_moe_token_mask must be called with "
                "this step's attention_mask before every forward."
            )
        token_weight = None if token_mask is None else token_mask.to(torch.float32)  # [num_tokens] or None

        router_logits = self.gate(hidden_states)  # [num_tokens,num_experts]
        routing_weights = torch.nn.functional.softmax(
            router_logits, dim=-1, dtype=torch.float
        )  # [num_tokens,num_experts]
        topk_scores, expert_indices = torch.topk(
            routing_weights, self.top_k, dim=-1
        )  # [num_tokens,top_k], [num_tokens,top_k]
        topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)  # [num_tokens,top_k]
        topk_scores = topk_scores.to(hidden_states.dtype)  # [num_tokens,top_k]
        if token_weight is not None:
            # A zero combine weight is the implementation-independent half of making a padding
            # row inert, as it is in the cosmos3 gen-tower block: the row contributes nothing to
            # the output and no gradient to the experts it was nominally dispatched to, whether
            # or not the experts skip its compute.
            topk_scores = topk_scores * token_weight.unsqueeze(1).to(topk_scores.dtype)  # [num_tokens,top_k]

        # The dispatch histogram, shared with the cosmos3 gen-tower block so both MoE paths
        # count the same thing: exactly the rows that reach the GEMM. Unmasked it is a histc
        # rather than a bincount, which would size its output from a device-side max and sync
        # the GPU on every MoE layer; masked it is a scatter-add of the 0/1 weights, which drops
        # the padding without an index-select — that would make the count's shape depend on how
        # many rows are real, the data-dependent shape this padded path exists to avoid.
        num_tokens_per_expert = _weighted_expert_counts(
            expert_indices, self.num_experts, token_weight
        )  # int32 [num_experts]

        # Per-layer load-balancing statistics, popped once per step by
        # moe_utils.collect_hf_moe_lbl_metadata and turned into the auxiliary loss. Only
        # mean_router_prob_per_expert carries gradient (the counts are a histogram), so the
        # aux loss trains the router alone — the standard Switch/GShard formulation.
        #
        # Real forward only. Under non-reentrant activation checkpointing the backward
        # re-executes this forward once per layer, and the collector has already popped the
        # stash for this step, so a re-stash is never cleared. The module's reference to
        # mean_router_prob_per_expert — a tensor of the RECOMPUTED graph — would then pin
        # that layer's recomputed activations for the rest of the backward. With every layer
        # doing it the recomputations accumulate instead of being freed one at a time, which
        # is what OOMs the 94-layer Qwen3-VL-235B-A22B in its first backward.
        #
        # Every term is over real tokens only, so the loss a step reports does not move when
        # the collate happens to pad the batch further: the counts already skip the padding,
        # num_tokens is the number of rows behind them, and the mean router probability is
        # taken over the same rows.
        #
        # torch.full, not torch.tensor([...], device=...), for the scalars built here: the
        # latter builds the value in pageable host memory and copies it over, and a blocking
        # pageable H2D copy is cudaMemcpyAsync followed by cudaStreamSynchronize. That
        # synchronize drains everything this layer just queued — milliseconds of CPU stall
        # for 8 bytes, once per layer per step. full() fills on device, passing the scalar
        # as a kernel argument, so no host memory is involved and nothing waits. The masked
        # token count is a device-side sum of the mask for the same reason.
        if not _in_autograd_backward():
            self.lbl_metadata = LBLMetadata(
                num_tokens_per_expert=num_tokens_per_expert.to(dtype=torch.int64),  # [num_experts]
                num_tokens=(
                    torch.full((1,), hidden_states.shape[0], dtype=torch.int64, device=hidden_states.device)
                    if token_weight is None
                    else token_weight.sum().to(torch.int64).reshape(1)
                ),  # [1]
                mean_router_prob_per_expert=_weighted_mean(routing_weights, token_weight).squeeze(0),  # [num_experts]
                top_k=torch.full((1,), self.top_k, dtype=torch.int64, device=hidden_states.device),  # [1]
            )

        routed_out = self.experts(
            hidden_states=hidden_states,
            topk_scores=topk_scores,
            expert_indices=expert_indices,
            num_tokens_per_expert=num_tokens_per_expert,
            token_mask=token_mask,
        )  # [num_tokens,hidden_size]
        # The routed output is reshaped like HF's 3-D input. HF puts the flat router logits
        # in the second slot, which its decoder layer discards and only
        # Qwen3VLMoePreTrainedModel._can_record_outputs reads (as `router_logits`,
        # OutputRecorder index=1) under `output_router_logits=True`, to feed HF's own
        # load_balancing_loss_func. This patch does not support that path — the aux loss is
        # built from the per-layer stash above — so return None and let it fail loudly
        # instead of silently diverging from the statistics we actually train on.
        return routed_out.view(batch_size, -1, self.hidden_size), None

    blocks = [module for module in model.modules() if isinstance(module, Qwen3VLMoeTextSparseMoeBlock)]
    if not blocks:
        raise ValueError(
            f"Found no Qwen3VLMoeTextSparseMoeBlock to patch under {type(model).__name__}; "
            "grouped_mm experts apply to Qwen3-VL-MoE backbones only"
        )

    for block in blocks:
        experts = block.experts
        # top_k is the only attribute the bound methods need that HF's experts module lacks
        # (HF keeps it on the block); the parameters and act_fn already match by name.
        experts.top_k = block.top_k
        # Seeded here so the attribute always exists: the patched forward reads it on every
        # call, and a first trace that saw no attribute at all would bake in the unmasked
        # branch and only notice the mask on a later recompile.
        block.token_mask = None

        experts.forward = Qwen3VLMoeTextExpertsGroupedMm.forward.__get__(experts, type(experts))
        experts._reorder_tokens = Qwen3VLMoeTextExpertsGroupedMm._reorder_tokens.__get__(experts, type(experts))
        block.forward = patched_moe_block_forward.__get__(block, type(block))

    log.info(f"Patched {len(blocks)} Qwen3VLMoeTextSparseMoeBlock instance(s) with grouped_mm experts")
    return len(blocks)


def patch_qwen3_vl_vision_varlen_attention(model: torch.nn.Module) -> int:
    """Monkey-patch every Qwen3-VL vision tower **instance** under ``model`` so each block runs
    ONE varlen attention call over the packed patches instead of one call per image/frame.

    HF's ``Qwen3VL{,Moe}VisionAttention.forward`` reserves its varlen path for
    ``_attn_implementation == "flash_attention_2"``. Every other implementation — including
    the ``cosmos`` adapter onto ``cosmos_framework.model.attention`` — takes a fallback that chops the
    packed sequence up and attends to each piece on its own::

        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        splits = [torch.split(t, lengths.tolist(), dim=2) for t in (q, k, v)]

    ``lengths.tolist()`` copies a device tensor to the host, which blocks the CPU until
    everything already queued on the stream has run. It costs one stall per vision block —
    27 on the 235B tower, doubled by the activation-checkpoint recompute in backward — and,
    being data-dependent control flow, it also graph-breaks the block so the pieces run
    eager. ``cosmos_framework.model.attention`` takes packed sequences directly, so none of that is
    needed: the ``cosmos`` adapter forwards ``cu_seq_lens_{q,k}`` / ``max_length_{q,k}`` on to
    ``cumulative_seqlen_{Q,KV}`` / ``max_seqlen_{Q,KV}``, and backend selection routes varlen
    to flash2/flash3 (cuDNN has no varlen integration and rejects it).

    The adapter is called directly rather than resolved out of ``ALL_ATTENTION_FUNCTIONS``,
    since name-based dispatch would be a hazard rather than a convenience here: this forward
    always passes cu_seqlens, and the other implementations in that registry swallow
    unrecognized kwargs, so a tower whose ``_attn_implementation`` changed after patching
    would quietly attend ACROSS image boundaries instead of failing. By the same token, apply
    this only to a tower already configured for ``cosmos`` — patching one that asked for
    sdpa/eager overrides that choice silently. ``hf_model`` gates the call on it.

    Patched in place, per vision tower:
    - Each ``Qwen3VL{,Moe}VisionAttention.forward`` becomes the FA2-shaped single call.
    - The tower's own ``forward`` gains a wrapper that computes ``max_seqlen`` — an int, as
      the kernels need a host value to size their scheduling — and passes it down through
      the ``**kwargs`` that the tower already threads into every block. That leaves ONE
      device-to-host read per vision forward in place of one per block, and none at all in
      backward: recompute replays the int the forward captured rather than recomputing it.

    Note the int lands inside the compiled region as a guard, so a step whose largest image
    differs from the last one recompiles the vision block. Dynamo's automatic-dynamic
    promotes such an int to a symbol after it changes once, so this settles rather than
    running to ``recompile_limit``; ``TORCH_LOGS=recompiles`` is the way to confirm it on a
    new data mix.

    Args:
        model: Module containing the vision tower(s), i.e. the outer
            ``Qwen3VL{,Moe}ForConditionalGeneration`` (``HFModel.model``).

    Returns:
        Number of patched attention modules.

    Raises:
        ValueError: if no vision tower is found.
    """
    if not transformers.__version__.startswith(_EXPECTED_TRANSFORMERS_VERSION_PREFIX):
        raise ValueError(f"monkey patching transformers version {transformers.__version__} is not supported")

    # Local import: the adapter pulls in cosmos_framework.model.attention, which probes for flash/NATTEN
    # backends at module scope, and backbones that never reach this patch must not pay for that.
    from cosmos_framework.utils.generator.hf_attention_cosmos import hf_attention_cosmos

    towers = [module for module in model.modules() if isinstance(module, (Qwen3VLVisionModel, Qwen3VLMoeVisionModel))]
    if not towers:
        raise ValueError(
            f"Found no Qwen3VLVisionModel/Qwen3VLMoeVisionModel to patch under {type(model).__name__}; "
            "varlen vision attention applies to Qwen3-VL backbones only"
        )

    def make_patched_vision_attention_forward(apply_rotary_pos_emb_vision: Callable) -> Callable:
        """Bind the patched forward to the rope helper of the attention's own modeling module.

        Dense and MoE ship separate copies of ``apply_rotary_pos_emb_vision``; resolving it per
        class (rather than importing one of them here) keeps each patched block calling the
        function its own module defines, as HF's unpatched code does.
        """

        # Replaces Qwen3VLVisionAttention/Qwen3VLMoeVisionAttention.forward from:
        #   transformers.models.qwen3_vl.modeling_qwen3_vl          (transformers v4.57.1)
        #   transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe  (transformers v4.57.1)
        def patched_vision_attention_forward(
            self,
            hidden_states: torch.Tensor,  # [N_patches,hidden_size]
            cu_seqlens: torch.Tensor,  # [num_sequences+1]
            rotary_pos_emb: torch.Tensor | None = None,
            position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
            max_seqlen: int | None = None,
            **kwargs,
        ) -> torch.Tensor:
            if max_seqlen is None:
                raise ValueError(
                    "varlen vision attention needs max_seqlen, which the patched vision tower "
                    "forward injects. Reaching here without it means the attention module was "
                    "patched but its tower was not, or a caller invoked the block directly."
                )
            if position_embeddings is None:
                raise ValueError(
                    "varlen vision attention needs position_embeddings; the vision tower builds the "
                    "(cos, sin) pair once and passes it to every block. HF keeps the parameter "
                    "optional for the superseded rotary_pos_emb calling convention, which this "
                    "forward does not implement (neither does the unpatched one — it unpacks the "
                    "pair unconditionally)."
                )

            seq_length = hidden_states.shape[0]
            query_states, key_states, value_states = (
                self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
            )
            # query_states, key_states, value_states: [N,num_heads,head_dim]
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)
            # query_states, key_states: [N,num_heads,head_dim]

            query_states = query_states.transpose(0, 1).unsqueeze(0)  # [1,num_heads,N,head_dim]
            key_states = key_states.transpose(0, 1).unsqueeze(0)  # [1,num_heads,N,head_dim]
            value_states = value_states.transpose(0, 1).unsqueeze(0)  # [1,num_heads,N,head_dim]

            # The adapter itself, not ALL_ATTENTION_FUNCTIONS[...]. Name-based dispatch would be
            # a hazard rather than a convenience here: this forward always passes cu_seqlens, and
            # the other implementations in that registry swallow unrecognized kwargs, so a tower
            # whose _attn_implementation changed after patching would drop them on the floor and
            # attend ACROSS image boundaries — wrong features, no error.
            attn_output, _ = hf_attention_cosmos(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
            )  # [1,N,num_heads,head_dim]

            attn_output = attn_output.reshape(seq_length, -1).contiguous()  # [N,hidden_size]
            attn_output = self.proj(attn_output)  # [N,hidden_size]
            return attn_output

        return patched_vision_attention_forward

    def make_patched_vision_tower_forward(tower_forward: Callable) -> Callable:
        """Wrap (rather than replace) the tower's bound forward: the body is HF's, unchanged.

        ``tower_forward`` comes in already bound, so the wrapper takes no ``self`` and is
        assigned as a plain instance attribute — binding it again would pass ``self`` twice.
        """

        def patched_vision_tower_forward(
            hidden_states: torch.Tensor,  # [N_patch,patch_dim]
            grid_thw: torch.Tensor,  # [N_media,3]
            **kwargs,
        ) -> tuple[torch.Tensor, list[torch.Tensor]]:
            # Packed sequence lengths are h*w per grid row (a t-frame video contributes t
            # sequences of the same length), so the longest sequence is max(h*w) — the same
            # value HF's FA2 branch derives from cu_seqlens, read here once for the whole
            # tower. int() is the device-to-host sync; it sits outside the compiled blocks.
            kwargs.setdefault("max_seqlen", int((grid_thw[:, 1] * grid_thw[:, 2]).max()))
            return tower_forward(hidden_states, grid_thw, **kwargs)

        return patched_vision_tower_forward

    n_patched = 0
    for tower in towers:
        attentions = [
            module
            for module in tower.modules()
            if isinstance(module, (Qwen3VLVisionAttention, Qwen3VLMoeVisionAttention))
        ]
        if not attentions:
            raise ValueError(f"Found no vision attention module under {type(tower).__name__}")

        # One tower's blocks are all the same class, so the rope helper is resolved once here.
        attention_module = importlib.import_module(type(attentions[0]).__module__)
        patched_forward = make_patched_vision_attention_forward(attention_module.apply_rotary_pos_emb_vision)
        for attention in attentions:
            attention.forward = patched_forward.__get__(attention, type(attention))
        n_patched += len(attentions)

        tower.forward = make_patched_vision_tower_forward(tower.forward)

    log.info(f"Patched {n_patched} Qwen3-VL vision attention module(s) with varlen attention")
    return n_patched


def patch_siglip2_pos_embed_antialias_off(vision_transformer) -> None:
    """Force the SigLIP2 vision tower's position-embedding resize to ``antialias=False``.

    The stock ``get_position_embedding`` interpolates the learned position embedding with
    ``F.interpolate(mode="bilinear", antialias=True)``. The antialiased backward
    (``upsample_bilinear2d_aa_backward``) has no deterministic CUDA implementation, so under
    ``torch.use_deterministic_algorithms(warn_only=True)`` it runs nondeterministically and
    perturbs vision-encoder grads run-to-run. ``antialias=False`` has a deterministic backward.
    Reimplements the method verbatim except for the flag; slightly changes pos-embed numerics
    vs the antialias=True baseline, so apply only for deterministic-repro runs.
    """
    import torch.nn.functional as F

    def get_position_embedding(self, grid_thw: torch.Tensor) -> torch.Tensor:
        positional_embedding = (
            self.embeddings.position_embedding.weight.reshape(
                self.embeddings.position_embedding_size, self.embeddings.position_embedding_size, -1
            )
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        embed_dim = self.embeddings.embed_dim
        resized_positional_embeddings = torch.empty(
            (total_tokens, embed_dim), dtype=positional_embedding.dtype, device=grid_thw.device
        )
        offset = 0
        for t, height, width in grid_thw:
            resized_embeddings = F.interpolate(
                positional_embedding,
                size=(height.cpu().item(), width.cpu().item()),
                mode="bilinear",
                align_corners=False,
                antialias=False,
            )
            resized_embeddings = resized_embeddings.reshape(embed_dim, -1).transpose(0, 1)
            num_spatial_tokens = height * width
            total_block_tokens = t * num_spatial_tokens
            resized_positional_embeddings[offset : offset + total_block_tokens] = resized_embeddings.repeat(t, 1)
            offset += total_block_tokens
        assert offset == resized_positional_embeddings.shape[0]
        return resized_positional_embeddings

    vision_transformer.get_position_embedding = get_position_embedding.__get__(
        vision_transformer, type(vision_transformer)
    )
    log.critical(f"Patched {type(vision_transformer).__name__} get_position_embedding: antialias=False (deterministic)")
