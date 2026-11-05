# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""@L0 @CPU: `schema/trajectory.py::TeacherSignal` — the OPD teacher's raw per-transition output.

Two things are pinned here, and the second is the one that is easy to get wrong:

1. The construction contract — an all-None signal is refused, each payload pair is all-or-nothing,
   and shapes must agree.
2. **The codec round trip.** `TeacherSignal` rides `Transition.targets`, which
   `comm/codec.py::_map_transition` externalizes, and `_walk` rebuilds every dataclass with
   `dataclasses.replace()` — so `__post_init__` runs AGAIN on the packed side, where the tensor
   fields are `BulkRef` handles with no `.shape`. A validator that asserted shapes unconditionally
   would pass every test above and then fail the first time a signal crossed a worker boundary.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cosmos_framework.posttrain.comm.bulk import BulkRef, BulkStore
from cosmos_framework.posttrain.comm.codec import collect_bulk_refs, materialize_payload, pack_payload
from cosmos_framework.posttrain.comm.transfer.movers import LocalStoreMover
from cosmos_framework.posttrain.schema import ObsBundle, Status, TeacherSignal, Trajectory, Transition


def _topk(n_tokens: int = 4, k: int = 3):
    """A well-formed shared-vocab (v1) payload: `[T, k]` ids + the teacher's logprobs for them."""
    ids = torch.arange(n_tokens * k, dtype=torch.long).reshape(n_tokens, k)
    logprobs = torch.log_softmax(torch.randn(n_tokens, k), dim=-1)
    return ids, logprobs


def _traj(signal: TeacherSignal) -> Trajectory:
    """One trajectory carrying `signal` where the teacher role puts it: `Transition.targets`."""
    tr = Transition(
        obs=ObsBundle(representations={"prompt": "1+2="}, id="p0"),
        action=torch.tensor([7, 8], dtype=torch.long),
        policy_info={"completion_text": " 3"},
        targets=signal,
        done=True,
    )
    return Trajectory(transitions=[tr], id="g0_k0", group_id="g0", status=Status.COMPLETED)


@pytest.mark.L0
@pytest.mark.CPU
def test_all_none_signal_is_rejected():
    """A signal with nothing in it would distil the student against no teacher at all.

    That failure is silent by construction — the objective would read four Nones, contribute
    nothing, and report a healthy loss — so it has to be refused where the object is built."""
    with pytest.raises(ValueError, match="all four fields are None"):
        TeacherSignal()


@pytest.mark.L0
@pytest.mark.CPU
def test_each_payload_pair_is_all_or_nothing():
    """ids without logprobs (or the reverse) is not a partial signal, it is a broken one."""
    ids, logprobs = _topk()
    with pytest.raises(ValueError, match="topk_ids is set but topk_logprobs is None"):
        TeacherSignal(topk_ids=ids)
    with pytest.raises(ValueError, match="topk_logprobs is set but topk_ids is None"):
        TeacherSignal(topk_logprobs=logprobs)
    with pytest.raises(ValueError, match="chunk_bounds is set but chunk_teacher_logprob is None"):
        TeacherSignal(chunk_bounds=torch.zeros(2, 2, dtype=torch.long))
    with pytest.raises(ValueError, match="chunk_teacher_logprob is set but chunk_bounds is None"):
        TeacherSignal(chunk_teacher_logprob=torch.zeros(2))


@pytest.mark.L0
@pytest.mark.CPU
def test_topk_and_chunk_payloads_are_mutually_exclusive() -> None:
    """A signal has one interpretation; carrying both modes makes every consumer ambiguous."""
    ids, logprobs = _topk(n_tokens=2, k=2)  # [T,K], [T,K]
    bounds = torch.tensor([[0, 2]], dtype=torch.long)  # [C,2]
    chunk_logprobs = torch.tensor([-1.0], dtype=torch.float32)  # [C]

    with pytest.raises(ValueError, match="top-k and chunk payloads are both set"):
        TeacherSignal(
            topk_ids=ids,
            topk_logprobs=logprobs,
            chunk_bounds=bounds,
            chunk_teacher_logprob=chunk_logprobs,
        )


@pytest.mark.L0
@pytest.mark.CPU
def test_topk_shapes_must_agree_and_be_two_dimensional():
    """Entry (t, j) of `topk_ids` names the token entry (t, j) of `topk_logprobs` scores.

    Mismatched shapes silently pair a token with another token's logprob wherever broadcasting
    happens to work, which is a wrong gradient rather than a crash."""
    ids, logprobs = _topk(n_tokens=4, k=3)
    with pytest.raises(ValueError, match="identically shaped"):
        TeacherSignal(topk_ids=ids, topk_logprobs=logprobs[:, :2])
    with pytest.raises(ValueError, match="identically shaped"):
        TeacherSignal(topk_ids=ids.reshape(-1), topk_logprobs=logprobs.reshape(-1))  # 1-D: no k axis
    TeacherSignal(topk_ids=ids, topk_logprobs=logprobs)  # the well-formed pair constructs


@pytest.mark.L0
@pytest.mark.CPU
def test_chunk_payload_shapes_are_checked_against_each_other():
    """The RESERVED cross-tokenizer (DPCA) pair: `[C, 2]` spans and one teacher logprob per span."""
    with pytest.raises(ValueError, match=r"chunk_bounds must be \[C, 2\]"):
        TeacherSignal(chunk_bounds=torch.zeros(3, 3, dtype=torch.long), chunk_teacher_logprob=torch.zeros(3))
    with pytest.raises(ValueError, match=r"chunk_teacher_logprob must be \[C\]"):
        TeacherSignal(chunk_bounds=torch.zeros(3, 2, dtype=torch.long), chunk_teacher_logprob=torch.zeros(2))
    TeacherSignal(chunk_bounds=torch.zeros(3, 2, dtype=torch.long), chunk_teacher_logprob=torch.zeros(3))


@pytest.mark.L0
@pytest.mark.CPU
def test_signal_survives_the_codec_round_trip():
    """THE load-bearing case: pack to `BulkRef`s and back, with `__post_init__` running on both.

    `pack_payload` alone would raise if the validator demanded `.shape` from a handle, so reaching
    the assertions at all is half the test; the other half is that the tensors really went to the
    producer's store rather than riding inline through the control plane."""
    ids, logprobs = _topk(n_tokens=5, k=4)
    signal = TeacherSignal(topk_ids=ids, topk_logprobs=logprobs)
    store = BulkStore()

    packed = pack_payload([_traj(signal)], store, threshold=0)

    packed_signal = packed[0].transitions[0].targets
    assert isinstance(packed_signal, TeacherSignal)  # the dataclass shell survives, rebuilt fresh
    assert isinstance(packed_signal.topk_ids, BulkRef)
    assert isinstance(packed_signal.topk_logprobs, BulkRef)
    assert packed_signal.chunk_bounds is None  # reserved fields stay absent, not invented
    # The prefetch fold must see the same refs `_walk` created — a `_walk`/`_visit` divergence would
    # make prefetch silently miss them, which shows up only as extra round trips under load.
    assert set(collect_bulk_refs(packed)) >= {packed_signal.topk_ids, packed_signal.topk_logprobs}

    out = materialize_payload(packed, LocalStoreMover())

    got = out[0].transitions[0].targets
    assert torch.equal(got.topk_ids, ids)
    assert torch.equal(got.topk_logprobs, logprobs)


@pytest.mark.L0
@pytest.mark.CPU
def test_reserved_chunk_payload_also_round_trips():
    """The DPCA fields are declared now so the schema does not move when that path lands; they must
    already survive transport, or landing it would be a schema change plus a codec change."""
    bounds = torch.tensor([[0, 2], [2, 5]], dtype=torch.long)
    chunk_logprob = torch.tensor([-1.5, -0.25])
    store = BulkStore()

    packed = pack_payload([_traj(TeacherSignal(chunk_bounds=bounds, chunk_teacher_logprob=chunk_logprob))], store, 0)
    out = materialize_payload(packed, LocalStoreMover())

    got = out[0].transitions[0].targets
    assert torch.equal(got.chunk_bounds, bounds)
    assert torch.equal(got.chunk_teacher_logprob, chunk_logprob)
    assert got.topk_ids is None
