# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Small-scale proof of the reversed second hop against a REAL vLLM-Omni worker, before the six-combination runs.

One GPU, one `Gr00tPolicyBody`, no trainer and no Ray. The worker exports its live parameters when the body is bound, rather than when it is
built and the body maps them; `_engine_sink` is driven directly with two buckets of different values through that
mapping, and the worker's own digests -- an independent reader in another process -- are the evidence. Then the
bracket must refuse a stale epoch, a bad wire name must be refused before any copy, `rebuild_engine()` must give a
new worker with a fresh mapping that loads and serves, and a worker that died on its own must be replaceable the
same way (the mapping is released before the engine in both cases). L1/GPU: needs the gr00t venv, a GB200 and the
DROID checkpoint (`GR00T_CKPT` or the HF default); `GR00T_DIT_SDPA_MODE=math` for the serving call.
"""

from __future__ import annotations

import os
import signal
import time
import weakref
from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest

from cosmos_framework.posttrain.comm.transfer.mover import MoveHandle
from cosmos_framework.posttrain.comm.transfer.movers import CopyMover
from cosmos_framework.posttrain.comm.weight_transfer import WeightSync, bucket_plan

if TYPE_CHECKING:
    import torch

    from cosmos_framework.posttrain.model.gr00t.serving.body import Gr00tPolicyBody

pytestmark = [pytest.mark.L1, pytest.mark.GPU]


@pytest.fixture
def body(monkeypatch: pytest.MonkeyPatch) -> Iterator[Gr00tPolicyBody]:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU: the mapping opens one process's CUDA storages in another")
    pytest.importorskip("vllm_omni")
    from cosmos_framework.posttrain.configs.gr00t_n1d7_droid import make_model_config
    from cosmos_framework.posttrain.model.gr00t.serving.body import Gr00tPolicyBody

    monkeypatch.setenv("GR00T_DIT_SDPA_MODE", "math")
    checkpoint = os.environ.get("GR00T_CKPT", "nvidia/GR00T-N1.7-DROID")
    body = Gr00tPolicyBody(
        make_model_config(checkpoint_path=checkpoint), transfer_dtype="bfloat16", verify_weight_values=True
    )
    try:
        assert body._live_params is None
        body.bind_weight_backend(WeightSync(_CudaCopyMover()))
        mapping = body._mapping
        body.initialize_weight_receiver()
        assert body._mapping is mapping
        yield body
    finally:
        body.close()


def _small_live_names(body, n: int = 3) -> list[str]:
    layout = body.describe_action_head()
    by_size = sorted(layout, key=lambda name: (sum(int(d) for d in layout[name]["shape"]), name))
    return by_size[:n]


def _values(body, names: list[str], fill: float) -> dict:
    import torch

    layout = body.describe_action_head()
    return {
        name: torch.full(
            tuple(layout[name]["shape"]),
            fill,
            dtype=getattr(torch, str(layout[name]["dtype"]).split(".")[-1]),
            device="cuda",
        )
        for name in names
    }


def _expected(state: dict) -> dict:
    from cosmos_framework.posttrain.comm.weight_digest import tensor_digest_metadata

    return {name: tensor_digest_metadata(t) for name, t in state.items()}


def test_two_buckets_land_bit_exactly_through_the_worker_exported_mapping(body) -> None:
    names = _small_live_names(body)
    pid = body.worker_device["pid"]
    assert body._mapping["pid"] == pid and set(body._live_params) == set(body.describe_action_head())
    assert not hasattr(body, "_hop2"), "the body owns no staging buffer any more: it writes into the mapping"

    first = _values(body, names, 0.5)
    body._engine_sink(first)
    assert body._worker_rpc("digest_live_parameters", names) == _expected(first)

    second = _values(body, names, -1.25)
    body._engine_sink(second)
    assert body._worker_rpc("digest_live_parameters", names) == _expected(second)
    assert body._worker_rpc("describe_device")["pid"] == pid, "a normal load never restarts the worker"
    assert body._expected_digests == _expected(second), "the body's expectation is the wire bytes it copied"


def test_the_verification_switch_decides_what_reads_the_bytes_back(body) -> None:
    """With `verify_weight_values` off the bucket carries no compare and no digest, and `finish` still closes the
    epoch with no expectation; with it on, the worker re-reads every live parameter. The bytes land either way."""
    # This body has no weight backend bound (no trainer, no mover), so the publication API (`mark_weight_version_ready`,
    # `commit_weight_version`) refuses by design; the bracket is driven through the worker RPCs, as the stale-epoch
    # test below does, and the per-stream records are reset the way `apply_bucket` resets them at bucket 0 --
    # otherwise this body would carry the preceding stream's rows and expectation into the next stream.
    names = _small_live_names(body)
    off = _values(body, names, 0.25)
    body.verify_weight_values = False
    epoch = body._update_epoch + 1
    body._worker_rpc("begin_weight_update", epoch)
    body._update_epoch = epoch
    body._coverage_reset(0)
    body._engine_sink(off)
    assert [t.compare_s for t in body.bucket_timings] == [0.0]
    assert [t.digest_s for t in body.bucket_timings] == [0.0] and body._expected_digests == {}
    closed = body._worker_rpc("finish_weight_update", epoch, None)  # what `_finish_weight_update` sends when off
    assert closed["epoch"] == epoch and closed["verified"] is None and closed["digest_s"] is None
    assert body._worker_rpc("digest_live_parameters", names) == _expected(off), "the copies landed unverified"

    # With the switch on, the same three-parameter bucket is compared and digested. The worker's verified `finish`
    # demands a digest for EVERY live parameter (a partial expectation is refused, see the L0 coverage tests), which
    # only a full stream provides, so this epoch is closed bare and the evidence is the sink's records plus the
    # worker's independent read-back.
    on = _values(body, names, -0.75)
    body.verify_weight_values = True
    epoch += 1
    body._worker_rpc("begin_weight_update", epoch)
    body._update_epoch = epoch
    body._coverage_reset(0)
    body._engine_sink(on)
    assert [t.compare_s > 0.0 for t in body.bucket_timings] == [True]
    assert [t.digest_s > 0.0 for t in body.bucket_timings] == [True]
    assert set(body._expected_digests) == set(names)
    assert body._worker_rpc("finish_weight_update", epoch, None)["epoch"] == epoch
    assert body._worker_rpc("digest_live_parameters", names) == _expected(on)


def test_begin_finish_bracket_refuses_a_stale_epoch(body) -> None:
    epoch = body._update_epoch + 1
    opened = body._worker_rpc("begin_weight_update", epoch)
    assert opened["epoch"] == epoch and opened["superseded"] is None and opened["pid"] == body.worker_device["pid"]
    body._update_epoch = epoch
    with pytest.raises(RuntimeError, match="the open update is epoch"):
        body._worker_rpc("finish_weight_update", epoch + 5, None)
    closed = body._worker_rpc("finish_weight_update", epoch, None)
    assert closed["epoch"] == epoch and closed["verified"] is None and closed["digest_s"] is None
    with pytest.raises(RuntimeError, match="does not advance"):
        body._worker_rpc("begin_weight_update", epoch)


def test_bad_wire_name_fails_before_any_copy_and_rebuild_re_imports_a_fresh_worker_that_serves(body) -> None:
    import torch

    names = _small_live_names(body)
    before = body._worker_rpc("digest_live_parameters", names)
    old_pid = body.worker_device["pid"]
    with pytest.raises(ValueError, match="not a live parameter"):
        body._engine_sink({"action_head.does_not_exist": torch.zeros(4, dtype=torch.bfloat16, device="cuda")})
    # The whole bucket is checked against the mapping before the first copy, so a rejected bucket changed nothing.
    assert body._worker_rpc("digest_live_parameters", names) == before

    body.engine_invalidated = True  # what `apply_bucket` records for a failed stream
    body.rebuild_engine()
    assert body.worker_device["pid"] != old_pid, "rebuild_engine must start a new worker process"
    assert body._mapping["pid"] == body.worker_device["pid"], "and map THAT worker's parameters"
    assert body.engine_invalidated is False and body._engine_names == set(body.describe_action_head())
    state = _values(body, names, 0.75)
    body._engine_sink(state)
    assert body._worker_rpc("digest_live_parameters", names) == _expected(state)


class _CudaCopyMover(CopyMover):
    """Use the real snapshot transport, uploading its owned tensors to the sink's device."""

    def fetch(self, handle: MoveHandle) -> dict[str, torch.Tensor]:
        return {name: tensor.cuda() for name, tensor in super().fetch(handle).items()}


def _publish(body: Gr00tPolicyBody, snapshot: dict[str, torch.Tensor], version: int) -> None:
    """Exercise the complete public bucket/readiness/publication protocol against the real worker."""
    plan = bucket_plan([(name, tuple(t.shape), str(t.dtype)) for name, t in sorted(snapshot.items())])
    producer = WeightSync(CopyMover())
    body.set_bucket_plan(plan)
    body.check_plan_coverage()
    for index, bucket in enumerate(plan):
        state = {name: snapshot[name] for name, _shape, _dtype in bucket}
        body.apply_bucket(producer.stage_bucket_state(state, version=version), index)
    body.mark_weight_version_ready(version)
    body.prepare_weight_version(version, None)
    body.commit_weight_version(version)


@pytest.mark.parametrize("verify", [False, True])
def test_rebuild_after_the_worker_was_killed_requires_resync_before_serving(
    body: Gr00tPolicyBody, verify: bool
) -> None:
    """A replacement worker must not serve its base checkpoint under an already-trained version."""
    import numpy as np

    from cosmos_framework.posttrain.tests.helpers.gr00t_synthetic import build_observations

    name = "action_head.action_decoder.layer2.b"
    base_digest = body._worker_rpc("digest_live_parameters", [name])[name]
    snapshot = {key: tensor.detach().to("cpu", copy=True) for key, tensor in body._live_params.items()}
    snapshot[name].add_(0.015625)  # the existing bf16-representable perturbation of the serving head
    expected = _expected(snapshot)
    assert expected[name] != base_digest
    body.verify_weight_values = verify
    _publish(body, snapshot, version=1)
    obs = build_observations(body.config, 1)
    before = body.infer_batch(obs, seed=1234, weight_version=1)
    assert body._worker_rpc("digest_live_parameters", [name])[name] == expected[name]

    pid = body.worker_device["pid"]
    os.kill(pid, signal.SIGKILL)
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.2)

    body.rebuild_engine()
    assert body.worker_device["pid"] != pid and body._mapping["pid"] == body.worker_device["pid"]
    assert body.weight_version() == 1 and body._stream_open and not body.engine_invalidated
    assert body._worker_rpc("digest_live_parameters", [name])[name] == base_digest
    for version in (None, 1):
        with pytest.raises(RuntimeError, match="refuses to serve"):
            body.infer_batch(obs, seed=1234, weight_version=version)
    _publish(body, snapshot, version=1)
    assert body._worker_rpc("digest_live_parameters") == expected
    after = body.infer_batch(obs, seed=1234, weight_version=1)
    assert np.array_equal(before[0]["action"], after[0]["action"])


@pytest.mark.parametrize("verify", [False, True])
def test_retained_bucket_error_drops_real_ipc_views_before_rebuild(body, monkeypatch, verify):
    import torch

    body.verify_weight_values = verify
    refs = [weakref.ref(view) for view in body._live_params.values()]
    name = _small_live_names(body, 1)[0]
    layout = body.describe_action_head()[name]
    wrong = torch.zeros(
        (int(torch.tensor(layout["shape"]).prod()) + 1,), device="cuda", dtype=body._live_params[name].dtype
    )
    with pytest.raises(ValueError, match="live parameter") as error:
        body._engine_sink({name: wrong})
    before_close = []
    close = body._omni.close

    def closing():
        before_close.append(sum(ref() is not None for ref in refs))
        close()

    monkeypatch.setattr(body._omni, "close", closing)
    body.rebuild_engine()
    assert error.value.__traceback__ is not None
    assert before_close == [0] and all(ref() is None for ref in refs)
    with pytest.raises(RuntimeError, match="refuses to serve"):
        body.infer_batch([{}])
    snapshot = {name: value.detach().cpu().clone() for name, value in body._live_params.items()}
    _publish(body, snapshot, version=1)
    assert not body._stream_open and body.weight_version() == 1
