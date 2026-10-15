# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``Cosmos3VFMNetwork.forward`` over a real multiview pack, down both attention routes.

The branch this covers decides, per batch, whether the multiview GEN attention runs as the
FlexAttention mask or as ``multiview_maskless_attention``'s three merged passes. Its two
halves are each unit-tested -- ``_multiview_maskless_geometry`` for the decision and
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
from typing import TYPE_CHECKING

import attrs
import pytest
import torch
from omegaconf import OmegaConf

from cosmos_framework.configs.base.defaults.model_config import MultiviewActionConditioningConfig
from cosmos_framework.configs.base.defaults.multiview_attention import MultiviewAttentionConfig
from cosmos_framework.data.generator.action.utils.transforms import build_sequence_plan_from_mode
from cosmos_framework.model.generator.mot.attention import SplitInfo, dispatch_attention
from cosmos_framework.model.generator.mot.multiview_attention import resolve_multiview_backend
from cosmos_framework.model.generator.mot.parallelize_unified_mot import _to_empty_preserving_buffers
from cosmos_framework.model.generator.mot.rigrope import RopeRigPE
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean
from cosmos_framework.data.generator.sequence_packing import PackedSequence
from cosmos_framework.data.generator.sequence_packing.packers import (
    pack_input_sequence,
    pack_multiview_action_conditioning,
    replicate_multiview_actions,
)
from cosmos_framework.data.generator.sequence_packing.runtime import SequencePack
from cosmos_framework.data.generator.sequence_packing.sequence import SequencePlan

if TYPE_CHECKING:
    from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork

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


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("window", [None, 0, 0.4, (-0.2, 0.2)])
@pytest.mark.parametrize(
    "backend,deduplicate_cross_view",
    [("flex_triton", None), ("maskless", None), ("maskless", False), ("maskless", True)],
)
def test_network_config_migrates_saved_unstructured_window(
    window: float | tuple[float, float] | None, backend: str, deduplicate_cross_view: bool | None
) -> None:
    from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetworkConfig

    # Exported checkpoint JSON need not carry the attrs object metadata.
    multiview = OmegaConf.create(attrs.asdict(MultiviewAttentionConfig()))
    multiview.backend = backend
    multiview.mask.attention_scope = "decomposed"
    multiview.mask.decomposed_temporal_window_seconds = window
    if deduplicate_cross_view is None:
        del multiview.deduplicate_cross_view
    else:
        multiview.deduplicate_cross_view = deduplicate_cross_view
    config = Cosmos3VFMNetworkConfig(multiview_attention_config=multiview)
    restored = config.multiview_attention_config.mask.decomposed_temporal_window_seconds
    expected = (-window, 0.0) if isinstance(window, (int, float)) else window
    assert (None if restored is None else tuple(restored)) == expected
    assert config.multiview_attention_config.deduplicate_cross_view == (deduplicate_cross_view or False)
    resolved_backend, _ = resolve_multiview_backend(
        torch.device("cpu"), backend, config=config.multiview_attention_config
    )
    assert resolved_backend == backend


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

    def init_weights(self, buffer_device: torch.device | None) -> None:
        pass

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
    layout ``multiview_maskless_attention`` folds by.

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


def _multiview_network(
    *,
    maskless_attention: bool,
    device: torch.device,
    temporal_window_seconds: tuple[float, float] | None = None,
    deduplicate_cross_view: bool = False,
    action_conditioning: bool = False,
    geometry_position_encoding: str = "baseline",
    materialize: bool = True,
    pre_geometry_export: bool = False,
) -> Cosmos3VFMNetwork:
    """The network under test, on the stub reasoner, with the multiview mask configured.

    ``materialize=False`` returns the network as built, so a caller can build it on meta.
    ``pre_geometry_export`` hands the attention config over as a checkpoint exported before
    the geometry modes and bands saved it: plain JSON with only its three original keys.
    """
    from cosmos_framework.configs.base.defaults.multiview_attention import (
        MultiviewAttentionConfig,
        MultiviewAttentionMaskConfig,
    )
    from cosmos_framework.model.generator.mot.cosmos3_vfm_network import (
        Cosmos3VFMNetwork,
        Cosmos3VFMNetworkConfig,
    )

    attention_config = MultiviewAttentionConfig(
        geometry_position_encoding=geometry_position_encoding,
        deduplicate_cross_view=deduplicate_cross_view,
        # Pinned rather than "auto" so the stream padding and the mask's block size are the
        # same on every host this runs on, FlashAttention-4 present or not.
        backend="maskless" if maskless_attention else "flex_triton",
        mask=MultiviewAttentionMaskConfig(
            # The scope the folds are the maskless alternative to, so the flex route this
            # harness compares against is the one a caller would be choosing between.
            attention_scope="decomposed",
            decomposed_temporal_window_seconds=temporal_window_seconds,
            # "maskless" requires it, and it is inert on a batch with no control item.
            control_attends_sensor=True,
        ),
    )
    exported = OmegaConf.create(attrs.asdict(attention_config))
    pre_geometry = OmegaConf.create({key: exported[key] for key in ("backend", "mask", "deduplicate_cross_view")})
    language_model = _StubLanguageModel()
    config = Cosmos3VFMNetworkConfig(
        vlm_config=language_model.config,
        vision_gen=True,
        action_gen=action_conditioning,
        action_dim=64 if action_conditioning else 32,
        multiview_action_conditioning=action_conditioning,
        latent_channel_size=LATENT_CHANNELS,
        latent_patch_size=PATCH_SPATIAL,
        latent_downsample_factor=16,
        max_latent_h=LATENT_HW,
        max_latent_w=LATENT_HW,
        max_latent_t=LATENT_T,
        joint_attn_implementation="multiview",
        multiview_attention_config=pre_geometry if pre_geometry_export else attention_config,
    )
    network = Cosmos3VFMNetwork(language_model, config)
    return network.to(device=device, dtype=torch.float32) if materialize else network


def _run_forward(
    *,
    maskless_attention: bool,
    device: torch.device,
    temporal_window_seconds: tuple[float, float] | None = None,
    deduplicate_cross_view: bool = False,
) -> tuple[dict, SplitInfo]:
    """One inference forward, returning its outputs and the metadata the reasoner was handed."""
    network = _multiview_network(
        maskless_attention=maskless_attention,
        device=device,
        temporal_window_seconds=temporal_window_seconds,
        deduplicate_cross_view=deduplicate_cross_view,
    )
    packed_seq = _multiview_packed_sequence()
    with torch.no_grad():
        output_dict = network(packed_seq)
    attention_mask = network.language_model.seen_attention_mask
    assert isinstance(attention_mask, SplitInfo)
    return output_dict, attention_mask


@pytest.mark.L0
@pytest.mark.GPU
@pytest.mark.skipif(not torch.cuda.is_available(), reason="The attention kernels require a GPU.")
@pytest.mark.parametrize("window", [None, (-0.4, 0.0), (-0.4, 0.2)])
@pytest.mark.parametrize("deduplicate_cross_view", [True, False])
def test_forward_routes_a_multiview_inference_pack_through_the_decomposition(
    window: tuple[float, float] | None,
    deduplicate_cross_view: bool,
) -> None:
    """The flag on, and a pack it accepts: the geometry travels and no mask is built.

    Both halves are asserted because either alone would pass a broken wiring. The geometry
    alone would not catch a forward that annotated it and built the mask anyway, and
    ``two_way_attention`` prefers the mask when it has one -- so the decomposition would never
    run and nothing would fail.
    """
    device = torch.device("cuda")
    output_dict, attention_mask = _run_forward(
        maskless_attention=True,
        device=device,
        temporal_window_seconds=window,
        deduplicate_cross_view=deduplicate_cross_view,
    )

    assert attention_mask.multiview_maskless is not None
    assert attention_mask.multiview_maskless.decomposed_temporal_window_seconds == window
    assert attention_mask.multiview_maskless.deduplicate_cross_view == deduplicate_cross_view
    # Per-sample tuples: this batch holds one sample.
    assert attention_mask.multiview_maskless.num_views == (NUM_VIEWS,)
    assert attention_mask.multiview_maskless.token_shapes == (
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
@pytest.mark.parametrize("window", [None, (-0.4, 0.0), (-0.4, 0.2)])
def test_forward_keeps_the_flex_mask_when_the_decomposition_is_off(window: tuple[float, float] | None) -> None:
    """The same pack with the flag off takes the mask, which is the fallback every other pack takes."""
    device = torch.device("cuda")
    output_dict, attention_mask = _run_forward(maskless_attention=False, device=device, temporal_window_seconds=window)

    assert attention_mask.multiview_maskless is None
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
    ``multiview_maskless_attention`` documents itself as unable to serve a checkpoint trained
    under the mask.
    """
    device = torch.device("cuda")
    torch.manual_seed(0)
    maskless, _ = _run_forward(maskless_attention=True, device=device)
    torch.manual_seed(0)
    flex, _ = _run_forward(maskless_attention=False, device=device)

    maskless_preds, flex_preds = maskless["preds_vision"][0], flex["preds_vision"][0]
    assert maskless_preds.shape == flex_preds.shape
    relative_gap = (maskless_preds - flex_preds).abs().mean() / flex_preds.abs().mean()
    assert relative_gap > 0.05, (
        f"The two routes came out {float(relative_gap):.1%} apart, which is close enough that the "
        "the maskless annotation may not have reached a kernel at all."
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
    network = _multiview_network(maskless_attention=True, device=device)
    packed_seq = _multiview_packed_sequence()

    with torch.enable_grad():
        output_dict = network(packed_seq)
        output_dict["preds_vision"][0].square().mean().backward()

    attention_mask = network.language_model.seen_attention_mask
    assert isinstance(attention_mask, SplitInfo)
    assert attention_mask.multiview_maskless is not None, "Training takes the decomposition too."
    assert attention_mask.flex_block_mask is None

    # vae2llm feeds the GEN tokens the two sensor passes attend, so a fold that dropped the
    # branch backward would leave it without a gradient.
    assert network.vae2llm.weight.grad is not None
    assert torch.isfinite(network.vae2llm.weight.grad).all()
    assert float(network.vae2llm.weight.grad.abs().sum()) > 0.0


@pytest.mark.L0
@pytest.mark.GPU
@pytest.mark.skipif(not torch.cuda.is_available(), reason="The attention kernels require a GPU.")
def test_opt_in_action_conditioning_train_all_ragged_views_and_action_encoder() -> None:
    network = _multiview_network(maskless_attention=True, device=torch.device("cuda"), action_conditioning=True)
    views = [torch.randn(1, LATENT_CHANNELS, 3, h, w) for h, w in [(4, 6), (2, 2), (2, 4)]]
    data = GenerationDataClean(
        batch_size=1,
        is_image_batch=False,
        x0_tokens_vision=views,
        x0_tokens_action=[torch.randn(8, 64)],
        action_domain_id=[torch.tensor([2])],
        fps_vision=torch.tensor([30.0]),
        num_vision_items_per_sample=[3],
    )
    data = replicate_multiview_actions(
        data,
        num_views=3,
        temporal_factor=4,
        action_dim=64,
    )
    packed = pack_multiview_action_conditioning(
        [build_sequence_plan_from_mode("forward_dynamics", video_length=9, action_length=8)],
        [[3, 4]],
        data,
        torch.tensor([0.5]),
        config=MultiviewActionConditioningConfig(view_code_start=59, view_codes=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        special_tokens=SPECIAL_TOKENS,
        patch_spatial=PATCH_SPATIAL,
        temporal_factor=4,
        temporal_margin=15_000,
        base_fps=24.0,
    )
    packed.to_cuda()
    output = network(packed)
    assert [pred.shape for pred in output["preds_vision"]] == [view.shape for view in views]
    torch.stack([pred.square().mean() for pred in output["preds_vision"]]).sum().backward()
    attention = network.language_model.seen_attention_mask
    assert isinstance(attention, SplitInfo)
    assert attention.flex_block_mask is None
    assert attention.multiview_maskless is not None
    assert attention.multiview_maskless.is_control == (False,) * 6
    assert network.vae2llm.weight.grad is not None and torch.isfinite(network.vae2llm.weight.grad).all()
    grads = [parameter.grad for parameter in network.action2llm.parameters() if parameter.grad is not None]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)
    assert sum(grad.abs().sum() for grad in grads) > 0


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("cpu_offload", [False, True], ids=["to_empty_first", "cpu_offload"])
def test_init_weights_materializes_the_rigrope_frequency_table(cpu_offload: bool) -> None:
    """A meta-built network gets its fixed frequencies back, as ``build_net`` materializes it.

    The table is a non-persistent buffer, so the checkpoint load that follows cannot fill it.
    The ordinary path runs ``to_empty`` and then ``init_weights``; the CPU-offload path runs
    ``init_weights`` on meta and then materializes, keeping every buffer no longer on meta.
    """
    device = torch.device("cpu")
    with torch.device("meta"):
        network = _multiview_network(
            maskless_attention=True,
            device=device,
            geometry_position_encoding="rigrope_cross_view",
            materialize=False,
        )
    assert network.rigrope is not None and network.rigrope.freq_matrix.is_meta
    if cpu_offload:
        network.init_weights(buffer_device=device)
        _to_empty_preserving_buffers(network, device=device, recurse=True)
    else:
        network.to_empty(device=device)
        network.rigrope.freq_matrix.fill_(float("nan"))  # [8,D/2]
        network.init_weights(buffer_device=device)

    rigrope = network.rigrope
    reference = RopeRigPE(
        rigrope.head_dim, max_freq_exponent=rigrope.max_freq_exponent, min_freq_exponent=rigrope.min_freq_exponent
    )
    torch.testing.assert_close(network.rigrope.freq_matrix, reference.freq_matrix, rtol=0, atol=0)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("maskless_attention", [True, False], ids=["maskless", "flex_triton"])
def test_a_checkpoint_exported_before_the_geometry_modes_builds_with_them_off(maskless_attention: bool) -> None:
    """Every field added since then is restored to its default, which leaves the baseline running."""
    network = _multiview_network(
        maskless_attention=maskless_attention, device=torch.device("cpu"), pre_geometry_export=True
    )

    restored = network.config.multiview_attention_config
    defaults = MultiviewAttentionConfig()
    for field in attrs.fields(MultiviewAttentionConfig):
        if field.name not in ("backend", "mask"):
            assert getattr(restored, field.name) == getattr(defaults, field.name), field.name
    assert restored.backend == ("maskless" if maskless_attention else "flex_triton")
    assert network.rigrope is None
