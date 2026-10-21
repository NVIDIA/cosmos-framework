# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""@L0 @CPU: the advantage channel — `Transition.advantage` as a scalar OR a per-token tensor.

`Transition.advantage` was born a `float` because GRPO's group-relative estimate is one number per
sample. It is now the general per-token type, with the scalar as the degenerate case: OPD's
advantage genuinely varies per position, PPO's GAE will too, and the shared policy-gradient
surrogate consumes a per-token tensor.

Three properties are pinned here.

- **GRPO is not perturbed.** A scalar advantage still travels inline through the codec and still
  takes the same expression in both objectives. Widening a channel for a new algorithm must not
  change the numerics of the merged, working one.
- **A per-token advantage is transported as bulk.** `comm/codec.py` externalizes it, and the
  read-only prefetch fold (`collect_bulk_refs`) sees exactly the refs the rewriting walk created.
  Those two mirror each other by comment; here they mirror each other by test.
- **A per-token advantage is applied per token.** Not summed, averaged, or broadcast from element
  zero — position `t`'s value multiplies position `t`'s logprob.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cosmos_framework.posttrain.algorithm.annotators.advantage import advantage_to_tokens, require_advantages
from cosmos_framework.posttrain.comm.bulk import BulkRef, BulkStore
from cosmos_framework.posttrain.comm.codec import collect_bulk_refs, materialize_payload, pack_payload
from cosmos_framework.posttrain.comm.transfer.movers import LocalStoreMover
from cosmos_framework.posttrain.schema import ObsBundle, Status, Trajectory, Transition


def _traj(advantage, action=None, text: str = "1+2=", traj_id: str = "g0_k0") -> Trajectory:
    """A minimal trajectory carrying `advantage`, with NO tensor fields other than `action`.

    Keeping the rest tensor-free is what lets the store-size assertions below attribute every
    externalized entry to the field under test."""
    tr = Transition(
        obs=ObsBundle(representations={"prompt": text}, id="p0"),
        action=action,
        policy_info={"completion_text": " 3"},
        advantage=advantage,
        done=True,
    )
    return Trajectory(transitions=[tr], id=traj_id, group_id="g0", status=Status.COMPLETED)


def _tiny():
    """Replicated tiny random GPT2 + gpt2 packer (posttrain tests never import each other).

    Mirrors `tests/test_algorithm_grpo_clipped.py::_tiny`: a 2-layer random `GPT2LMHeadModel` on
    CPU plus the `DecoderOnlyLLMDataPacker` that owns the tokenizer. `.eval()` disables dropout so
    repeated forwards of the same batch are comparable."""
    from transformers import GPT2Config, GPT2LMHeadModel

    from cosmos_framework.posttrain.data.packer import init_data_packer

    model = GPT2LMHeadModel(GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=50257, n_positions=64)).eval()
    return model, init_data_packer("gpt2")


# ── transport: the scalar path is untouched, the tensor path becomes bulk ─────────────────────


@pytest.mark.L0
@pytest.mark.CPU
def test_scalar_advantage_still_travels_inline():
    """A float is not bulk-eligible, so `_map_transition` walking it changes nothing.

    This is the regression that matters for GRPO: adding `advantage` to the codec's externalized
    field set must leave every existing payload byte-identical."""
    store = BulkStore()

    packed = pack_payload([_traj(0.75)], store, threshold=0)

    assert packed[0].transitions[0].advantage == 0.75  # inline, not a handle
    assert len(store) == 0  # nothing externalized
    assert collect_bulk_refs(packed) == []


@pytest.mark.L0
@pytest.mark.CPU
def test_per_token_advantage_is_externalized_and_round_trips():
    """A `[T]` advantage goes to the producer's `BulkStore` instead of riding the control plane.

    The `collect_bulk_refs` assertion is the point: `_map_transition` (the rewriting walk) and
    `collect_bulk_refs::visit` (the read-only prefetch fold) enumerate the same field set, and a
    field added to one but not the other makes prefetch silently miss those refs — no error, just
    an extra round trip per field under load."""
    adv = torch.tensor([0.5, -0.25, 1.0])
    store = BulkStore()

    packed = pack_payload([_traj(adv)], store, threshold=0)

    packed_advantage = packed[0].transitions[0].advantage
    assert isinstance(packed_advantage, BulkRef)
    assert len(store) == 1
    assert collect_bulk_refs(packed) == [packed_advantage]  # the two walks agree

    out = materialize_payload(packed, LocalStoreMover())

    assert torch.equal(out[0].transitions[0].advantage, adv)


# ── validation: `require_advantages` accepts both shapes ──────────────────────────────────────


@pytest.mark.L0
@pytest.mark.CPU
def test_require_advantages_accepts_a_finite_per_token_tensor():
    """The check used to be `isfinite(float(tr.advantage))`, which raises outright on `[T]`."""
    require_advantages([_traj(torch.tensor([0.0, 1.5, -2.0]))])
    require_advantages([_traj(torch.zeros(0))])  # empty completion: vacuously finite
    require_advantages([_traj(0.5)])  # the scalar path still passes


@pytest.mark.L0
@pytest.mark.CPU
def test_require_advantages_rejects_a_non_finite_tensor_as_it_rejects_a_float():
    """One poisoned position poisons the whole batch's gradient, exactly as one bad scalar does, so
    the elementwise check has to be as strict as the scalar one it replaces."""
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="non-finite advantage"):
            require_advantages([_traj(torch.tensor([0.0, bad, 1.0]))])
    with pytest.raises(ValueError, match="non-finite advantage"):
        require_advantages([_traj(float("nan"))])


@pytest.mark.L0
@pytest.mark.CPU
def test_require_advantages_reports_an_unmaterialized_advantage():
    """A newly reachable mistake: now that the advantage can be bulk, a consumer that validates
    before `materialize_payload` holds a `BulkRef`. That must name itself, not die inside `float()`
    with a message about the wrong thing."""
    packed = pack_payload([_traj(torch.tensor([1.0, 2.0]))], BulkStore(), threshold=0)

    with pytest.raises(ValueError, match="neither a scalar nor a per-token tensor"):
        require_advantages(packed)


# ── expansion: where the two shapes meet ──────────────────────────────────────────────────────


@pytest.mark.L0
@pytest.mark.CPU
def test_advantage_to_tokens_broadcasts_a_scalar_and_refuses_a_wrong_width():
    """A wrong-width tensor is refused rather than broadcast.

    A 1-element tensor would otherwise spread silently over the whole completion — the exact
    scalar-vs-per-token confusion this channel now has to keep straight."""
    cpu = torch.device("cpu")
    assert torch.equal(advantage_to_tokens(0.5, 3, device=cpu), torch.full((3,), 0.5))
    assert advantage_to_tokens(0.5, 3, device=cpu).dtype is torch.float32  # matches the scalar path

    vector = torch.tensor([1.0, 2.0, 3.0])
    assert torch.equal(advantage_to_tokens(vector, 3, device=cpu), vector)
    with pytest.raises(ValueError, match="has 3 element"):
        advantage_to_tokens(vector, 4, device=cpu)
    with pytest.raises(ValueError, match="has 1 element"):
        advantage_to_tokens(torch.tensor([2.0]), 4, device=cpu)


# ── the objectives: scalar unchanged, per-token genuinely per-token ───────────────────────────


@pytest.mark.L0
@pytest.mark.CPU
def test_constant_per_token_advantage_matches_the_scalar_it_broadcasts_from():
    """A per-token advantage that happens to be constant IS the scalar case, in both objectives.

    Approximate for the legacy_pg objective on purpose: it keeps `adv · Σ logπ` for a scalar and
    takes `Σ adv_t · logπ_t` for a tensor, which differ in summation order. The clipped objective
    builds one flat `[n_tok]` vector either way, so there the two must agree EXACTLY — that is the
    assertion which would catch a dtype or ordering slip in the new `torch.cat` branch."""
    pytest.importorskip("transformers")
    from cosmos_framework.posttrain.algorithm.grpo import make_clipped_objective, make_objective
    from cosmos_framework.posttrain.config.rl import GRPOConfig

    torch.manual_seed(0)
    model, packer = _tiny()
    tok = packer.tokenizer

    # Unequal TOTAL lengths, so the clipped objective's single batched forward genuinely pads —
    # that is the case where the per-sequence widths recorded during preparation could drift from
    # the collated `counts` the scalar path uses, and the exact-equality assertion below is what
    # would catch it.
    specs = [("3+4=", " seven exactly", 1.0), ("100+200=", " three hundred exactly", -1.0)]

    def _batch(per_token: bool):
        trajs = []
        for i, (text, completion, advantage) in enumerate(specs):
            ids = torch.tensor(tok(completion, add_special_tokens=False).input_ids, dtype=torch.long)
            value = torch.full((ids.numel(),), advantage) if per_token else advantage
            trajs.append(_traj(value, action=ids, text=text, traj_id=f"g0_k{i}"))
        return trajs

    padded_lengths = {
        len(tok(text, add_special_tokens=False).input_ids) + int(traj.transitions[0].action.numel())
        for (text, _, _), traj in zip(specs, _batch(per_token=False))
    }
    assert len(padded_lengths) == len(specs), f"fixture must produce unequal lengths, got {padded_lengths}"

    scalar_loss, _ = make_objective(packer)(model, _batch(per_token=False))
    tensor_loss, _ = make_objective(packer)(model, _batch(per_token=True))
    assert torch.allclose(scalar_loss, tensor_loss, atol=1e-6), (scalar_loss.item(), tensor_loss.item())

    cfg = GRPOConfig(epsilon_low=10.0, epsilon_high=10.0, kl_beta=0.0, temperature=1.0, group_size=2)
    scalar_clipped, _ = make_clipped_objective(packer, cfg)(model, _batch(per_token=False))
    tensor_clipped, _ = make_clipped_objective(packer, cfg)(model, _batch(per_token=True))
    assert torch.equal(scalar_clipped, tensor_clipped), (scalar_clipped.item(), tensor_clipped.item())


@pytest.mark.L0
@pytest.mark.CPU
def test_per_token_advantage_weights_each_position_independently():
    """Position `t`'s advantage multiplies position `t`'s logprob — nothing collapses it.

    Proven by decomposition rather than by a hand-computed number: the loss is linear in the
    advantage over a fixed denominator, so the all-ones advantage must equal the SUM of the
    one-hots, and each one-hot must differ from the others (different positions have different
    logprobs). Together those two pin that every token gets exactly its own weight and that the
    weights are not permuted — an implementation that averaged, or read element zero, or reversed
    the vector, fails one or the other."""
    pytest.importorskip("transformers")
    from cosmos_framework.posttrain.algorithm.grpo import make_objective

    torch.manual_seed(0)
    model, packer = _tiny()
    completion = torch.tensor(
        packer.tokenizer(" seven exactly now", add_special_tokens=False).input_ids, dtype=torch.long
    )
    n_tokens = completion.numel()
    assert n_tokens >= 3, "need several completion tokens for a positionwise test"
    objective = make_objective(packer)

    def _loss(advantage):
        loss, _ = objective(model, [_traj(advantage, action=completion.clone())])
        return float(loss.detach())

    ones = _loss(torch.ones(n_tokens))
    one_hots = [_loss(torch.eye(n_tokens)[t]) for t in range(n_tokens)]

    assert ones == pytest.approx(sum(one_hots), abs=1e-6)  # linear, and every position contributes
    assert len(set(round(v, 9) for v in one_hots)) == n_tokens  # positions are distinct, not permuted
    assert ones == pytest.approx(_loss(1.0), abs=1e-6)  # and the scalar 1.0 is the same thing


@pytest.mark.L0
@pytest.mark.CPU
def test_objective_refuses_a_per_token_advantage_of_the_wrong_width():
    """The width check runs during PREPARATION, above the first forward.

    That placement is the point, not a detail: under per-sample dp sharding a malformed trajectory
    lands on ONE rank, and a raise between forwards would abort it while its peers block in the
    FSDP all-gathers of forwards it will never issue — a hang instead of an error."""
    pytest.importorskip("transformers")
    from cosmos_framework.posttrain.algorithm.grpo import make_objective

    torch.manual_seed(0)
    model, packer = _tiny()
    completion = torch.tensor(packer.tokenizer(" seven", add_special_tokens=False).input_ids, dtype=torch.long)

    with pytest.raises(ValueError, match="scored token"):
        make_objective(packer)(model, [_traj(torch.ones(completion.numel() + 1), action=completion)])
