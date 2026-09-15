# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``Cosmos3VFMNetwork.forward`` over a real multiview pack, down both attention routes.

The branch this covers decides, per batch, whether the multiview GEN attention runs as the
FlexAttention mask or as ``multiview_dense_attention``'s three merged passes. Its two
halves are each unit-tested -- ``_multiview_dense_geometry`` for the decision and
``dispatch_attention`` for the routing -- but nothing joined them: the geometry the network
reads off a ``PackedSequence`` had never been handed to the attention function that folds by
it. A test built on hand-written packs cannot join them either, since the thing at risk is
exactly whether the packer's own layout and that geometry agree.

So the pack here comes from ``pack_input_sequence``, the entry the training and inference
paths both pack through, and the forward is the network's own. What is stubbed is the
reasoner: ``_StubLanguageModel`` stands in for the Qwen3-VL backbone, whose weights are an
S3 download and whose depth the attention routing does not turn on. It still runs the real
``dispatch_attention`` over the real pack, so the geometry is exercised against the kernels
rather than merely recorded, and it keeps the ``SplitInfo`` it was handed so the tests can
assert which route was taken rather than infer it from the numbers.
"""

from __future__ import annotations

import random
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.model.generator.mot.attention import SplitInfo, dispatch_attention
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean
from cosmos_framework.data.generator.sequence_packing import PackedSequence
from cosmos_framework.data.generator.sequence_packing.packers import pack_input_sequence
from cosmos_framework.data.generator.sequence_packing.runtime import SequencePack
from cosmos_framework.data.generator.sequence_packing.sequence import SequencePlan

# The rig this harness packs: two cameras, two latent frames each, on a 4x4 latent grid that
# one patch step halves to 2x2. Sixteen GEN tokens in all -- small enough that the flex mask's
# 128-token block padding is almost the whole stream, which is the case the decomposition has
# to trim rather than attend.
NUM_VIEWS = 2
FRAMES_PER_VIEW = 2
LATENT_T = NUM_VIEWS * FRAMES_PER_VIEW
LATENT_HW = 4
PATCH_SPATIAL = 2
LATENT_CHANNELS = 16
TEXT_LEN = 8

HIDDEN_SIZE = 64
NUM_HEADS = 4
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS

SPECIAL_TOKENS = {"eos_token_id": 0, "start_of_generation": 1, "end_of_generation": 2}


class _StubLanguageModel(torch.nn.Module):
    """The reasoner, reduced to the two things the multiview attention branch runs through it.

    ``model.embed_tokens`` because ``_encode_text`` embeds the caption through it, and a
    forward that runs ``dispatch_attention`` over the pack it is handed. Those are what carry
    the network's routing decision into the kernels; the decoder stack in between changes
    which numbers come out and not which attention runs, so it is a projection here.

    The ``SplitInfo`` is kept rather than copied: the tests assert on the same object the
    forward annotated, which is what makes "the mask was not built" an assertion rather than
    an inference from a shape.
    """

    def __init__(self, vocab_size: int = 32) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=HIDDEN_SIZE,
            num_attention_heads=NUM_HEADS,
            num_key_value_heads=NUM_HEADS,
            head_dim=HEAD_DIM,
            num_hidden_layers=1,
        )
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(vocab_size, HIDDEN_SIZE)
        self.seen_attention_mask: SplitInfo | None = None

    def forward(
        self,
        input_pack: SequencePack,
        attention_mask: SplitInfo | None = None,
        position_ids: torch.Tensor | None = None,
        natten_metadata_list: list | None = None,
        memory: object | None = None,
    ) -> tuple[SequencePack, dict]:
        self.seen_attention_mask = attention_mask

        # One pack serving as q, k and v. Only the two token streams are reshaped from
        # [N,hidden] to the [N,heads,head_dim] the attention paths read; every other entry --
        # the offsets, the pad segments, the caption boundaries -- is the network's own, which
        # is the point of running the real pack through rather than rebuilding one.
        as_heads = dict(input_pack)
        for key in ("causal_seq", "full_only_seq"):
            stream = input_pack[key]  # [N,hidden]
            as_heads[key] = stream.view(stream.shape[0], NUM_HEADS, HEAD_DIM)  # [N,heads,head_dim]

        output_pack, kv_to_store = dispatch_attention(as_heads, as_heads, as_heads, attention_mask)
        assert kv_to_store is None
        return output_pack, {}


def _multiview_packed_sequence() -> PackedSequence:
    """One multiview sample, packed the way the training and inference paths pack one.

    ``num_views_per_vision_item`` is what ``enable_per_camera_vae_encoding`` records, and it is
    the field the whole multiview attention path keys off: without it the network cannot say
    where one camera's latent frames end, and both the mask and the decomposition refuse the
    pack. The item's latent axis is camera-major, ``num_views * frames_per_view``, which is the
    layout ``multiview_dense_attention`` folds by.

    Packed on CPU and moved with ``to_cuda`` afterwards, which is the order ``OmniMoTModel``
    packs in: the packer builds its index tensors on the host (it rejects CUDA text indexes
    outright), so the move is a step of the production path rather than a convenience here.
    """
    random.seed(0)
    text_indexes = [[random.randint(3, 31) for _ in range(TEXT_LEN)]]
    x0_tokens_vision = [torch.randn(1, LATENT_CHANNELS, LATENT_T, LATENT_HW, LATENT_HW)]  # [1,C,T,H,W]
    gen_data_clean = GenerationDataClean(
        batch_size=1,
        is_image_batch=False,
        raw_state_vision=[torch.randn(1, 3, 1 + 4 * (LATENT_T - 1), 64, 64)],
        x0_tokens_vision=x0_tokens_vision,
        num_vision_items_per_sample=[1],
        num_views_per_vision_item=[NUM_VIEWS],
    )
    sequence_plans = [
        SequencePlan(
            has_text=True,
            has_vision=True,
            has_action=False,
            condition_frame_indexes_vision=[],
            condition_frame_indexes_action=[],
        )
    ]
    packed_seq = pack_input_sequence(
        sequence_plans=sequence_plans,
        input_text_indexes=text_indexes,
        gen_data_clean=gen_data_clean,
        input_timesteps=torch.tensor([0.5], dtype=torch.float32),
        special_tokens=SPECIAL_TOKENS,
        latent_patch_size=PATCH_SPATIAL,
    )
    packed_seq.to_cuda()
    return packed_seq


def _multiview_network(*, dense_attention: bool, device: torch.device):
    """The network under test, on the stub reasoner, with the multiview mask configured."""
    from cosmos_framework.configs.base.defaults.multiview_attention import (
        MultiviewAttentionConfig,
        MultiviewAttentionMaskConfig,
    )
    from cosmos_framework.model.generator.mot.cosmos3_vfm_network import (
        Cosmos3VFMNetwork,
        Cosmos3VFMNetworkConfig,
    )

    language_model = _StubLanguageModel()
    config = Cosmos3VFMNetworkConfig(
        vlm_config=language_model.config,
        vision_gen=True,
        latent_channel_size=LATENT_CHANNELS,
        latent_patch_size=PATCH_SPATIAL,
        latent_downsample_factor=16,
        max_latent_h=LATENT_HW,
        max_latent_w=LATENT_HW,
        max_latent_t=LATENT_T,
        joint_attn_implementation="multiview",
        multiview_attention_config=MultiviewAttentionConfig(
            # Pinned rather than "auto" so the stream padding and the mask's block size are the
            # same on every host this runs on, FlashAttention-4 present or not.
            backend="dense" if dense_attention else "flex_triton",
            mask=MultiviewAttentionMaskConfig(
                # The scope the folds are the maskless alternative to, so the flex route this
                # harness compares against is the one a caller would be choosing between.
                attention_scope="decomposed",
                # "dense" requires it, and it is inert on a batch with no control item.
                control_attends_sensor=True,
            ),
        ),
    )
    return Cosmos3VFMNetwork(language_model, config).to(device=device, dtype=torch.float32)


def _run_forward(*, dense_attention: bool, device: torch.device) -> tuple[dict, SplitInfo]:
    """One inference forward, returning its outputs and the metadata the reasoner was handed."""
    network = _multiview_network(dense_attention=dense_attention, device=device)
    packed_seq = _multiview_packed_sequence()
    with torch.no_grad():
        output_dict = network(packed_seq)
    attention_mask = network.language_model.seen_attention_mask
    assert isinstance(attention_mask, SplitInfo)
    return output_dict, attention_mask


@pytest.mark.L0
@pytest.mark.GPU
@pytest.mark.skipif(not torch.cuda.is_available(), reason="The attention kernels require a GPU.")
def test_forward_routes_a_multiview_inference_pack_through_the_decomposition() -> None:
    """The flag on, and a pack it accepts: the geometry travels and no mask is built.

    Both halves are asserted because either alone would pass a broken wiring. The geometry
    alone would not catch a forward that annotated it and built the mask anyway, and
    ``two_way_attention`` prefers the mask when it has one -- so the decomposition would never
    run and nothing would fail.
    """
    device = torch.device("cuda")
    output_dict, attention_mask = _run_forward(dense_attention=True, device=device)

    assert attention_mask.multiview_dense is not None
    # Per-sample tuples: this batch holds one sample.
    assert attention_mask.multiview_dense.num_views == (NUM_VIEWS,)
    assert attention_mask.multiview_dense.token_shapes == (
        (LATENT_T, LATENT_HW // PATCH_SPATIAL, LATENT_HW // PATCH_SPATIAL),
    )
    assert attention_mask.flex_block_mask is None, "The decomposition needs no mask, so none is built."
    assert attention_mask.flex_backend is None

    # The forward completed through decode, which is what says the attention output kept the
    # pack's own layout: the decomposition trims the padded stream and re-pads it, and a fold
    # that returned the tokens in any other order would land the wrong latents here.
    preds = output_dict["preds_vision"]
    assert len(preds) == 1
    assert preds[0].shape == (1, LATENT_CHANNELS, LATENT_T, LATENT_HW, LATENT_HW)
    assert torch.isfinite(preds[0]).all()


@pytest.mark.L0
@pytest.mark.GPU
@pytest.mark.skipif(not torch.cuda.is_available(), reason="The attention kernels require a GPU.")
def test_forward_keeps_the_flex_mask_when_the_decomposition_is_off() -> None:
    """The same pack with the flag off takes the mask, which is the fallback every other pack takes."""
    device = torch.device("cuda")
    output_dict, attention_mask = _run_forward(dense_attention=False, device=device)

    assert attention_mask.multiview_dense is None
    assert attention_mask.flex_block_mask is not None
    assert attention_mask.flex_backend is not None
    assert output_dict["preds_vision"][0].shape == (1, LATENT_CHANNELS, LATENT_T, LATENT_HW, LATENT_HW)


@pytest.mark.L0
@pytest.mark.GPU
@pytest.mark.skipif(not torch.cuda.is_available(), reason="The attention kernels require a GPU.")
def test_the_two_routes_are_different_attention_over_the_same_pack() -> None:
    """The decomposition is its own pattern, and the forward is where that becomes observable.

    Everything but the flag is held fixed -- same seed, same weights, same pack -- so the gap
    is the attention and nothing else. Asserting that it is large is what makes the routing
    tests above load-bearing: an annotation that reached no kernel, or a flex route that
    quietly ran the same passes, would land the two outputs on top of each other.

    The size of the gap is the ``(view, frame)`` cell both sensor passes take, which the merge
    keeps twice and the mask counts once. On this 2x2 rig that cell is a large share of the key
    set, so the two disagree by tens of percent rather than by rounding -- the same reason
    ``multiview_dense_attention`` documents itself as unable to serve a checkpoint trained
    under the mask.
    """
    device = torch.device("cuda")
    torch.manual_seed(0)
    dense, _ = _run_forward(dense_attention=True, device=device)
    torch.manual_seed(0)
    flex, _ = _run_forward(dense_attention=False, device=device)

    dense_preds, flex_preds = dense["preds_vision"][0], flex["preds_vision"][0]
    assert dense_preds.shape == flex_preds.shape
    relative_gap = (dense_preds - flex_preds).abs().mean() / flex_preds.abs().mean()
    assert relative_gap > 0.05, (
        f"The two routes came out {float(relative_gap):.1%} apart, which is close enough that the "
        "the dense annotation may not have reached a kernel at all."
    )


@pytest.mark.L0
@pytest.mark.GPU
@pytest.mark.skipif(not torch.cuda.is_available(), reason="The attention kernels require a GPU.")
def test_forward_trains_through_the_decomposition() -> None:
    """A training step takes the decomposition too, and its gradient reaches the parameters.

    Grad mode used to be a gate here: the merged backward was wrong, so training fell back to
    the mask. It is no longer, both fold-backs now going through ``MergeAttentionsBridge``
    (``attention_test`` pins the gradients themselves against a float64 reference). What this
    covers is the network's half -- that the route is taken under grad at all, and that the loss
    it produces differentiates all the way back to the projections rather than detaching
    somewhere in the fold.
    """
    device = torch.device("cuda")
    network = _multiview_network(dense_attention=True, device=device)
    packed_seq = _multiview_packed_sequence()

    with torch.enable_grad():
        output_dict = network(packed_seq)
        output_dict["preds_vision"][0].square().mean().backward()

    attention_mask = network.language_model.seen_attention_mask
    assert isinstance(attention_mask, SplitInfo)
    assert attention_mask.multiview_dense is not None, "Training takes the decomposition too."
    assert attention_mask.flex_block_mask is None

    # vae2llm feeds the GEN tokens the two sensor passes attend, so a fold that dropped the
    # branch backward would leave it without a gradient.
    assert network.vae2llm.weight.grad is not None
    assert torch.isfinite(network.vae2llm.weight.grad).all()
    assert float(network.vae2llm.weight.grad.abs().sum()) > 0.0
