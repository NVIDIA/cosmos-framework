# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The `Rollout` Protocol pins the training-rollout surface (generate + apply_weights): every
sampling rollout body satisfies it and shares ONE interchangeable `generate(prompts, **knobs)`
signature. PolicyBody is intentionally a DIFFERENT body kind (infer-based policy-eval server)."""

from __future__ import annotations

import inspect

import pytest

torch = pytest.importorskip("torch")


@pytest.mark.L0
@pytest.mark.CPU
def test_sampling_rollouts_satisfy_protocol():
    import torch.nn as nn

    from cosmos_framework.posttrain.roles.rollout import LocalRollout, Rollout

    # LocalRollout conforms structurally (generate + apply_weights) — runtime_checkable Protocol.
    lr = LocalRollout(nn.Linear(2, 2))
    assert isinstance(lr, Rollout)


@pytest.mark.L0
@pytest.mark.CPU
def test_generate_surface_is_interchangeable():
    """Both sampling bodies accept the SAME call — `generate(prompts, **knobs)` — so a caller
    written against one works with the other (the widening the review flagged is gone)."""
    from cosmos_framework.posttrain.roles.rollout import HFRollout, LocalRollout

    for body in (LocalRollout, HFRollout):
        sig = inspect.signature(body.generate)
        params = sig.parameters
        # exactly (self, prompts, **knobs): no per-call knob widens the surface.
        assert list(params)[:2] == ["self", "prompts"], (body.__name__, list(params))
        assert any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()), body.__name__
        # group/seed are NOT positional/keyword params (they ride in **knobs).
        assert "group" not in params and "seed" not in params, body.__name__


@pytest.mark.L0
@pytest.mark.CPU
def test_group_ids_for_prompt_uses_prompt_id_as_group():
    """The GRPO group_id is the prompt's own stable id (obs.id), NOT the batch-local index — so
    completions of DISTINCT prompts never share a group when fan-in concatenates multiple rollout
    replicas' outputs (the replica-index collision the review flagged). Each completion gets a
    globally-unique traj id `{obs.id}_k{ki}` and all `group` completions share `group_id == obs.id`."""
    from cosmos_framework.posttrain.roles.rollout.base import group_ids_for_prompt
    from cosmos_framework.posttrain.schema import ObsBundle

    gid_a, tids_a = group_ids_for_prompt(ObsBundle(id="uuid-a"), group=3)
    gid_b, tids_b = group_ids_for_prompt(ObsBundle(id="uuid-b"), group=3)
    assert gid_a == "uuid-a" and gid_b == "uuid-b"
    assert tids_a == ["uuid-a_k0", "uuid-a_k1", "uuid-a_k2"]
    # distinct prompts -> disjoint group_ids AND disjoint traj ids (no cross-replica collision).
    assert gid_a != gid_b
    assert set(tids_a).isdisjoint(tids_b)


@pytest.mark.L0
@pytest.mark.CPU
def test_group_ids_for_prompt_raises_on_unstamped_prompt():
    """An empty obs.id would make EVERY prompt collide into group_id "" — the exact silent
    advantage-corruption the fix prevents. Fail loud (repo convention) so the data role's missing
    stamp surfaces immediately instead of poisoning the GRPO mean/std."""
    from cosmos_framework.posttrain.roles.rollout.base import group_ids_for_prompt
    from cosmos_framework.posttrain.schema import ObsBundle

    with pytest.raises(ValueError, match="ObsBundle.id"):
        group_ids_for_prompt(ObsBundle(), group=3)


@pytest.mark.L0
@pytest.mark.CPU
def test_policybody_is_not_a_generate_rollout():
    """PolicyBody deliberately has a different surface (infer/infer_batch/backend), so it is NOT
    the generate-based Rollout — documented in roles/rollout/base.py."""
    try:
        from cosmos_framework.posttrain.roles.rollout import cosmos3

        policy_body = cosmos3.PolicyBody
    except Exception as exc:  # noqa: BLE001 — heavy i4/OSS policy deps may be absent on CI
        pytest.skip(f"PolicyBody import unavailable: {exc!r}")

    assert hasattr(policy_body, "infer_batch")
    assert not hasattr(policy_body, "generate")
