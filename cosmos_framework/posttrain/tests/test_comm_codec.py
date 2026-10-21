# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Codec metadata/bulk seam (DESIGN §3.2/§3.4): tensor-bearing Trajectory fields are
externalized to a producer BulkStore as BulkRefs (controller sees only metadata),
then pulled back — zero-copy same-process (LocalStoreMover) or a fresh copy
cross-process (LocalStoreMover(cross_process=True)). Pure CPU, no torch: a `_Blob`
with an `nbytes` attribute stands in for a tensor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import pytest

from cosmos_framework.posttrain.comm.bulk import BulkRef, BulkStore
from cosmos_framework.posttrain.comm.codec import materialize_obs, materialize_payload, pack_payload
from cosmos_framework.posttrain.comm.transfer.movers import LocalStoreMover, resolve_mover
from cosmos_framework.posttrain.schema import ObsBundle, Status, Trajectory, Transition


class _Pair(NamedTuple):
    """A NamedTuple carrying a tensor field — its type must survive the pack/materialize round trip."""

    a: object
    b: object


@dataclass
class _Blob:
    """Stands in for a tensor: has an int `nbytes` (what the codec keys off)."""

    nbytes: int
    tag: str = "x"


def _traj(action, prompt_ids):
    tr = Transition(
        obs=ObsBundle(representations={"prompt": "1+2="}),
        action=action,
        policy_info={"prompt_ids": prompt_ids, "completion_text": "3", "weight_version": 0},
        done=True,
    )
    return Trajectory(transitions=[tr], id="g0_k0", group_id="g0", status=Status.COMPLETED)


@dataclass
class _Wrap:
    """A non-tensor dataclass carrying a tensor field — exercises dataclass-leaf recursion."""

    media: object


@dataclass
class _DerivedWrap:
    """A dataclass with an `init=False` field derived in `__post_init__` alongside a tensor field —
    the walker must not pass the derived field to `dataclasses.replace`."""

    media: object
    kind: str = field(init=False, default="")

    def __post_init__(self) -> None:
        self.kind = "video"


def _dpo_traj(targets):
    """A DPO-shaped trajectory whose `targets` carry the (possibly nested) preference payload."""
    tr = Transition(
        obs=ObsBundle(representations={"prompt": "prompt"}),
        action=None,
        policy_info={"weight_version": 0},
        targets=targets,
        done=True,
    )
    return Trajectory(transitions=[tr], id="g0_k0", group_id="g0", status=Status.COMPLETED)


@pytest.mark.L0
@pytest.mark.CPU
def test_pack_externalizes_tensors_keeps_small_fields_inline():
    store = BulkStore()
    action, pid = _Blob(2000), _Blob(3000)
    trajs = [_traj(action, pid)]
    packed = pack_payload(trajs, store, threshold=0)

    t = packed[0].transitions[0]
    assert isinstance(t.action, BulkRef)  # bulk → metadata handle
    assert isinstance(t.policy_info["prompt_ids"], BulkRef)
    assert t.policy_info["completion_text"] == "3"  # small field stays inline
    assert t.policy_info["weight_version"] == 0
    assert t.obs.representations["prompt"] == "1+2="
    assert len(store) == 2  # both blobs resident in producer store
    # non-mutating: the producer's original objects are untouched
    assert trajs[0].transitions[0].action is action


@pytest.mark.L0
@pytest.mark.CPU
def test_materialize_direct_is_zero_copy_and_consumes():
    store = BulkStore()
    action, pid = _Blob(2000), _Blob(3000)
    packed = pack_payload([_traj(action, pid)], store, threshold=0)

    out = materialize_payload(packed, LocalStoreMover())
    t = out[0].transitions[0]
    assert t.action is action  # SAME object — zero-copy (§3.3)
    assert t.policy_info["prompt_ids"] is pid
    assert len(store) == 0  # consume-once drained the store (§3.5)
    assert store.fetch_count == 2


@pytest.mark.L0
@pytest.mark.CPU
def test_tuple_batch_preserves_container_and_externalizes_nested_bulk():
    """PPO roles return tuple[Trajectory, ...]; that outer tuple must not bypass the codec."""
    store = BulkStore()
    actions = (_Blob(2000, "a0"), _Blob(2000, "a1"))
    payload = tuple(_traj(action, (1, 2, 3)) for action in actions)

    packed = pack_payload(payload, store, threshold=1024)

    assert isinstance(packed, tuple)
    assert all(isinstance(item.transitions[0].action, BulkRef) for item in packed)
    assert len(store) == 2

    materialized = materialize_payload(packed, LocalStoreMover())
    assert isinstance(materialized, tuple)
    assert tuple(item.transitions[0].action for item in materialized) == actions
    assert len(store) == 0


@pytest.mark.L0
@pytest.mark.CPU
def test_copy_transport_is_a_fresh_copy():
    store = BulkStore()
    action, pid = _Blob(2000), _Blob(3000)
    packed = pack_payload([_traj(action, pid)], store, threshold=0)

    out = materialize_payload(packed, LocalStoreMover(cross_process=True))
    t = out[0].transitions[0]
    assert t.action is not action  # fresh copy, no aliasing (§3.5)
    assert t.action.nbytes == action.nbytes


@pytest.mark.L0
@pytest.mark.CPU
def test_threshold_keeps_small_tensors_inline():
    store = BulkStore()
    small, big = _Blob(100), _Blob(3000)
    packed = pack_payload([_traj(small, big)], store, threshold=1024)

    t = packed[0].transitions[0]
    assert t.action is small  # below threshold → inline
    assert isinstance(t.policy_info["prompt_ids"], BulkRef)  # above → externalized
    assert len(store) == 1


@pytest.mark.L0
@pytest.mark.CPU
def test_resolve_mover_same_process_is_local_store():
    assert isinstance(resolve_mover(relationship="same_process", plane="data"), LocalStoreMover)
    assert isinstance(resolve_mover(relationship="cross_node", plane="data"), LocalStoreMover)


@pytest.mark.L0
@pytest.mark.CPU
def test_obsbundle_payload_packs_bulk_keeps_text_inline():
    # multimodal prompts: text inline, image-like blob externalized (DESIGN §3.4)
    store = BulkStore()
    img = _Blob(5000)
    prompts = [ObsBundle(representations={"prompt": "1+2=", "image": img})]
    packed = pack_payload(prompts, store, threshold=1024)

    rep = packed[0].representations
    assert rep["prompt"] == "1+2="  # small field inline
    assert isinstance(rep["image"], BulkRef)  # bulk externalized
    assert len(store) == 1
    out = materialize_payload(packed, LocalStoreMover())
    assert out[0].representations["image"] is img  # zero-copy on the way back


@pytest.mark.L0
@pytest.mark.CPU
def test_obsbundle_id_survives_pack_and_materialize():
    # The GRPO group_id namespace rides on ObsBundle.id (stamped by the data role). The codec rebuilds
    # ObsBundle FRESH per pack/unpack, so id must be threaded through both — else it is silently dropped
    # on the Ray-grid dispatch and the rollout's group_ids_for_prompt fails loud on the empty id.
    store = BulkStore()
    prompts = [ObsBundle(representations={"prompt": "1+2=", "image": _Blob(5000)}, id="uuid-xyz")]
    packed = pack_payload(prompts, store, threshold=1024)
    assert packed[0].id == "uuid-xyz"  # preserved across pack (bulk externalized)
    out = materialize_payload(packed, LocalStoreMover())
    assert out[0].id == "uuid-xyz"  # preserved across materialize (round-trip intact)


@pytest.mark.L0
@pytest.mark.CPU
def test_non_bulk_payloads_pass_through():
    store = BulkStore()
    assert pack_payload(0.5, store) == 0.5  # loss scalar
    assert pack_payload(None, store) is None  # push_weights result
    assert pack_payload([1, 2, 3], store) == [1, 2, 3]  # not Trajectory/ObsBundle
    assert len(store) == 0


@pytest.mark.L0
@pytest.mark.CPU
def test_pack_recurses_into_nested_dpo_targets():
    # A DPO video nested at targets['winner']['video'][0] (dict→dict→list→tensor) must be
    # externalized, not ride ~20 MB inline on the control plane.
    store = BulkStore()
    win, los = _Blob(2000, "win"), _Blob(2000, "los")
    targets = {"winner": {"video": [win]}, "loser": {"video": [los]}}
    trajs = [_dpo_traj(targets)]
    packed = pack_payload(trajs, store, threshold=0)

    pt = packed[0].transitions[0].targets
    assert isinstance(pt["winner"]["video"][0], BulkRef)  # nested tensor externalized
    assert isinstance(pt["loser"]["video"][0], BulkRef)
    assert set(pt) == {"winner", "loser"}  # dict/list structure preserved
    assert isinstance(pt["winner"]["video"], list)
    assert len(store) == 2  # both bulks resident in the producer store, not inline
    # non-mutating: the producer's original nested objects are untouched
    assert trajs[0].transitions[0].targets["winner"]["video"][0] is win


@pytest.mark.L0
@pytest.mark.CPU
def test_nested_targets_round_trip():
    store = BulkStore()
    win = _Blob(2000, "win")
    targets = {"winner": {"video": [win]}, "loser": {"video": [_Blob(2000, "los")]}}
    packed = pack_payload([_dpo_traj(targets)], store, threshold=0)
    # The round-trip is only meaningful if the nested tensor actually left the control plane.
    assert isinstance(packed[0].transitions[0].targets["winner"]["video"][0], BulkRef)
    assert len(store) == 2

    out = materialize_payload(packed, LocalStoreMover(cross_process=True))  # cross-process → fresh copy
    got = out[0].transitions[0].targets["winner"]["video"][0]
    assert got is not win  # a fresh reconstruction, not the original object
    assert (got.nbytes, got.tag) == (win.nbytes, win.tag)  # byte-equal round-trip
    assert len(store) == 0  # consume-once drained both nested refs


@pytest.mark.L0
@pytest.mark.CPU
def test_collect_refs_finds_nested_refs():
    from cosmos_framework.posttrain.comm.codec import collect_bulk_refs

    store = BulkStore()
    targets = {"winner": {"video": [_Blob(2000)]}, "loser": {"video": [_Blob(2000)]}}
    packed = pack_payload([_dpo_traj(targets)], store, threshold=0)

    refs = collect_bulk_refs(packed[0])  # prefetch must see refs at any depth
    assert len(refs) == 2
    assert all(isinstance(r, BulkRef) for r in refs)


@pytest.mark.L0
@pytest.mark.CPU
def test_small_nested_tensor_stays_inline_per_leaf():
    store = BulkStore()
    small = _Blob(100)
    targets = {"winner": {"video": [small]}, "loser": {"video": [_Blob(3000)]}}
    packed = pack_payload([_dpo_traj(targets)], store, threshold=1024)

    pt = packed[0].transitions[0].targets
    assert pt["winner"]["video"][0] is small  # below threshold → inline
    assert isinstance(pt["loser"]["video"][0], BulkRef)  # above → externalized
    assert len(store) == 1


@pytest.mark.L0
@pytest.mark.CPU
def test_nested_string_stays_inline():
    store = BulkStore()
    targets = {"winner": {"caption": "a cat", "video": [_Blob(2000)]}}
    packed = pack_payload([_dpo_traj(targets)], store, threshold=0)

    pt = packed[0].transitions[0].targets
    assert pt["winner"]["caption"] == "a cat"  # strings are never externalized/pickled
    assert isinstance(pt["winner"]["video"][0], BulkRef)
    assert len(store) == 1


@pytest.mark.L0
@pytest.mark.CPU
def test_dataclass_with_tensor_field_externalizes():
    store = BulkStore()
    blob = _Blob(2000)
    trajs = [_dpo_traj({"winner": _Wrap(media=blob)})]
    packed = pack_payload(trajs, store, threshold=0)

    pt = packed[0].transitions[0].targets
    assert isinstance(pt["winner"], _Wrap)  # dataclass rebuilt fresh
    assert isinstance(pt["winner"].media, BulkRef)  # tensor field externalized
    assert len(store) == 1
    # non-mutating: the producer's original dataclass + tensor are untouched
    assert trajs[0].transitions[0].targets["winner"] is not pt["winner"]
    assert trajs[0].transitions[0].targets["winner"].media is blob


@pytest.mark.L0
@pytest.mark.CPU
def test_dataclass_with_init_false_field_packs():
    # A payload dataclass with an init=False field (derived in __post_init__) must pack without
    # tripping dataclasses.replace, which rejects init=False fields passed as changes.
    store = BulkStore()
    blob = _Blob(2000)
    trajs = [_dpo_traj({"winner": _DerivedWrap(media=blob)})]
    packed = pack_payload(trajs, store, threshold=0)

    pt = packed[0].transitions[0].targets["winner"]
    assert isinstance(pt, _DerivedWrap)
    assert isinstance(pt.media, BulkRef)  # init tensor field externalized
    assert pt.kind == "video"  # init=False field recomputed by __post_init__
    assert len(store) == 1


@pytest.mark.L0
@pytest.mark.CPU
def test_collect_refs_deep_mixed_shape():
    # dict → dataclass → list → tensor: prefetch must find a ref through the deepest mixed nesting.
    from cosmos_framework.posttrain.comm.codec import collect_bulk_refs

    store = BulkStore()
    packed = pack_payload([_dpo_traj({"winner": _Wrap(media=[_Blob(2000)])})], store, threshold=0)
    refs = collect_bulk_refs(packed[0])
    assert len(refs) == 1 and isinstance(refs[0], BulkRef)


@pytest.mark.L0
@pytest.mark.CPU
def test_nested_obsbundle_is_loud():
    # An ObsBundle belongs in obs/next_obs (the field-selective path); nesting one in a generic field
    # is out of contract and must fail loud rather than silently bypass materialize_obs.
    store = BulkStore()
    trajs = [_dpo_traj({"o": ObsBundle(representations={"x": _Blob(2000)})})]
    with pytest.raises(TypeError, match="ObsBundle nested"):
        pack_payload(trajs, store, threshold=0)


@pytest.mark.L0
@pytest.mark.CPU
def test_namedtuple_field_preserved_round_trip():
    store = BulkStore()
    win = _Blob(2000, "win")
    packed = pack_payload([_dpo_traj({"pair": _Pair(a=win, b="caption")})], store, threshold=0)
    pt = packed[0].transitions[0].targets["pair"]
    assert isinstance(pt, _Pair)  # type preserved, not downgraded to a plain tuple
    assert isinstance(pt.a, BulkRef) and pt.b == "caption"

    out = materialize_payload(packed, LocalStoreMover(cross_process=True))
    got = out[0].transitions[0].targets["pair"]
    assert isinstance(got, _Pair)  # type preserved across the full round trip
    assert (got.a.nbytes, got.a.tag) == (win.nbytes, win.tag) and got.b == "caption"


@pytest.mark.L0
@pytest.mark.CPU
def test_pack_recurses_into_policy_info_tuple_round_trip():
    # The _walk path runs on every _map_transition field, not just targets: a tensor nested in a
    # tuple inside a policy_info value must externalize and round-trip (tuple type preserved).
    store = BulkStore()
    a, b = _Blob(2000, "a"), _Blob(2000, "b")
    tr = Transition(
        obs=ObsBundle(representations={"prompt": "p"}),
        action=None,
        policy_info={"weight_version": 0, "pair": (a, b)},
        done=True,
    )
    trajs = [Trajectory(transitions=[tr], id="g0_k0", group_id="g0", status=Status.COMPLETED)]
    packed = pack_payload(trajs, store, threshold=0)
    pi = packed[0].transitions[0].policy_info
    assert isinstance(pi["pair"], tuple) and all(isinstance(x, BulkRef) for x in pi["pair"])
    assert len(store) == 2

    out = materialize_payload(packed, LocalStoreMover(cross_process=True))
    got = out[0].transitions[0].policy_info["pair"]
    assert isinstance(got, tuple)
    assert (got[0].nbytes, got[0].tag) == (a.nbytes, a.tag)


@pytest.mark.L0
@pytest.mark.CPU
def test_pack_recurses_into_algo_extra_round_trip():
    # algo_extra rides the same _map_transition fn_val as policy_info — mirror the tuple-nesting
    # round-trip so both first-class tensor-bearing dict fields are pinned (guards a future
    # _map_transition change that stops routing one of them through the walk).
    store = BulkStore()
    a, b = _Blob(2000, "a"), _Blob(2000, "b")
    tr = Transition(
        obs=ObsBundle(representations={"prompt": "p"}),
        action=None,
        policy_info={"weight_version": 0},
        algo_extra={"pair": (a, b)},
        done=True,
    )
    trajs = [Trajectory(transitions=[tr], id="g0_k0", group_id="g0", status=Status.COMPLETED)]
    packed = pack_payload(trajs, store, threshold=0)
    ae = packed[0].transitions[0].algo_extra
    assert isinstance(ae["pair"], tuple) and all(isinstance(x, BulkRef) for x in ae["pair"])
    assert len(store) == 2

    out = materialize_payload(packed, LocalStoreMover(cross_process=True))
    got = out[0].transitions[0].algo_extra["pair"]
    assert isinstance(got, tuple)
    assert (got[0].nbytes, got[0].tag) == (a.nbytes, a.tag)


@pytest.mark.L0
@pytest.mark.CPU
def test_dataclass_field_round_trip():
    # pairs with test_dataclass_with_tensor_field_externalizes: the materialize half, byte-equal.
    store = BulkStore()
    blob = _Blob(2000, "m")
    packed = pack_payload([_dpo_traj({"winner": _Wrap(media=blob)})], store, threshold=0)
    assert isinstance(packed[0].transitions[0].targets["winner"].media, BulkRef)

    out = materialize_payload(packed, LocalStoreMover(cross_process=True))
    got = out[0].transitions[0].targets["winner"]
    assert isinstance(got, _Wrap)  # dataclass type preserved
    assert (got.media.nbytes, got.media.tag) == (blob.nbytes, blob.tag)  # byte-equal round trip


@pytest.mark.L0
@pytest.mark.CPU
def test_materialize_obs_deep_and_field_selective():
    # Pack recurses into container-valued representations; materialize_obs must resolve them for the
    # selected key while leaving a non-selected bulk field as a lazy BulkRef (§3.4).
    store = BulkStore()
    f0, f1, vid = _Blob(2000, "f0"), _Blob(2000, "f1"), _Blob(5000, "v")
    packed = pack_payload([ObsBundle(representations={"frames": [f0, f1], "video": vid})], store, threshold=0)
    rep = packed[0].representations
    assert all(isinstance(x, BulkRef) for x in rep["frames"]) and isinstance(rep["video"], BulkRef)

    out = materialize_obs(packed[0], LocalStoreMover(), fields={"frames"})
    assert isinstance(out.representations["frames"], list)
    assert [x.tag for x in out.representations["frames"]] == ["f0", "f1"]  # deep-resolved
    assert isinstance(out.representations["video"], BulkRef)  # non-selected → still lazy
