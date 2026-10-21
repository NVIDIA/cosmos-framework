# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""VLLMRollout weight-sink: structural contract (no vLLM needed) + a real load-into-engine check
(skipped without vLLM). The COLLECTIVE path fetches FULL tensors over the weight plane and hands
them to vLLM's load_weights, which slices to the local TP shard (§2.4/§4)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from cosmos_framework.posttrain.tests.helpers.vllm_engine import release_in_process_engines

# Same model as the sibling vLLM rigs (test_roles_rollout_vllm{,_tp,_sync_correctness}.py). Its
# `qwen2` model_type is registered, so the rig builds its packer from the engine's own model rather
# than substituting an unrelated one.
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(autouse=True)
def _release_engines_after_each_test():
    yield
    release_in_process_engines()


# MODULE-LEVEL callables for apply_model (V1 pickles them to the EngineCore worker; return
# PRIMITIVES — tensors are msgpack'd to lists across the boundary).
def _norm_param_meta(m):
    # A layernorm param: 1:1 name+shape between HF and vLLM (not fused, not TP-sharded, and NOT the
    # vocab-parallel embedding, whose live shape is padded past org_vocab_size — its weight_loader
    # asserts the CHECKPOINT shape, so a zeros_like of the live param would be rejected).
    name, p = next((n, q) for n, q in m.named_parameters() if "norm" in n and q.dim() == 1)
    return (name, list(p.shape), str(p.dtype))


def _param_abs_sum(m, name):
    return float(dict(m.named_parameters())[name].detach().float().abs().sum().cpu())


class _LoopComm:
    """Records producer broadcasts (rank==src) and replays them on the consumer pass."""

    rank = 0
    world_size = 2

    def __init__(self):
        self._buf, self._i = [], 0

    def broadcast(self, t, src):
        if self.rank == src:
            self._buf.append(t.clone())
        else:
            t.copy_(self._buf[self._i])
            self._i += 1

    def send(self, *a): ...
    def recv(self, *a): ...
    def close(self): ...


@pytest.mark.L0
@pytest.mark.CPU
def test_vllm_rollout_satisfies_weight_sink_contract():
    # Structural (no vLLM import needed): orchestration stays on the rollout while version-specific
    # live-engine mechanics remain isolated in the temporary compatibility module.
    from cosmos_framework.posttrain.roles.rollout import vllm_weights
    from cosmos_framework.posttrain.roles.rollout.vllm import VLLMRollout

    for name in ("apply_weights", "prepare_recv", "shard_map"):
        assert callable(getattr(VLLMRollout, name)), f"VLLMRollout missing {name}"
    for name in ("_apply_to_model", "_vllm_load_weights", "_vllm_begin_moe_reload", "resolve_vllm_moe_layout"):
        assert callable(getattr(vllm_weights, name)), f"vllm_weights missing {name}"


@pytest.mark.L0
@pytest.mark.CPU
def test_apply_weights_no_backend_is_version_bump_noop():
    # No bound backend ⇒ apply_weights only bumps the staleness stamp (static-weight fallback),
    # never touching the engine (so it is safe to call before a weight plane exists). No vLLM needed.
    from cosmos_framework.posttrain.roles.rollout.vllm import VLLMRollout

    obj = VLLMRollout.__new__(VLLMRollout)  # bypass __init__ (which needs a real vllm.LLM)
    obj._weight_version = 3
    obj.apply_weights()  # no _weight_backend attr
    assert obj._weight_version == 4


@pytest.mark.L0
@pytest.mark.CPU
def test_apply_weights_none_with_real_backend_raises_clearly():
    # apply_weights is PEER-ONLY: None with a bound non-shared backend is the retired batch/
    # collective-recv form — fail loud with the remedy, and never move the staleness stamp.
    from cosmos_framework.posttrain.roles.rollout.vllm import VLLMRollout

    class _StagedMover:
        lifetime = "staged"

    class _BE:
        mover = _StagedMover()

    obj = VLLMRollout.__new__(VLLMRollout)
    obj._weight_version = 5
    obj._weight_backend = _BE()
    with pytest.raises(ValueError, match="peer"):
        obj.apply_weights(None)
    assert obj._weight_version == 5  # unchanged


@pytest.mark.L1
@pytest.mark.GPU
def test_vllm_apply_weights_loads_into_engine():
    # Real ingestion: overwrite one engine param via the COLLECTIVE weight-sink path and read it
    # back. Requires vLLM + GPU; skipped otherwise. Executor-agnostic: reads/writes go through
    # vllm_weights._apply_to_model (vLLM V1 dispatches into the EngineCore worker via apply_model;
    # V0 falls back to the in-process model path) — never the V0-only attribute directly.
    vllm = pytest.importorskip("vllm")
    import functools

    import torch

    from cosmos_framework.posttrain.comm.transfer.movers import NcclMover
    from cosmos_framework.posttrain.comm.weight_transfer import WeightSync
    from cosmos_framework.posttrain.config.build import _configure_vllm_runtime
    from cosmos_framework.posttrain.data.packer import init_data_packer
    from cosmos_framework.posttrain.roles.rollout import vllm_weights
    from cosmos_framework.posttrain.roles.rollout.vllm import VLLMRollout

    # Production runs EngineCore in-process; a multiprocess EngineCore would have to msgpack the
    # callable, which vLLM refuses without VLLM_ALLOW_INSECURE_SERIALIZATION.
    _configure_vllm_runtime()
    llm = vllm.LLM(model=MODEL, enforce_eager=True, gpu_memory_utilization=0.3)
    # Packer is REQUIRED at construction even though this rig is weight-only (it never generates —
    # the packer is unused here). Built from the engine's OWN model so the two agree: MODEL's
    # `qwen2` is registered, whereas the previous `facebook/opt-125m` engine had no registered
    # packer at all and forced an unrelated tokenizer in as a stand-in.
    rollout = VLLMRollout(llm, packer=init_data_packer(MODEL))
    name, shape, dtype_str = vllm_weights._apply_to_model(rollout.model, _norm_param_meta)
    assert (
        vllm_weights._apply_to_model(rollout.model, functools.partial(_param_abs_sum, name=name)) != 0.0
    )  # kills a vacuous pass
    target = torch.zeros(*shape, dtype=getattr(torch, dtype_str.split(".")[-1]))

    comm = _LoopComm()
    be = WeightSync(NcclMover(comm, src=0))
    be.mover.recv_plan = [(name, tuple(target.shape), str(target.dtype))]  # injected full manifest
    rollout._weight_backend = be
    comm.rank = 0  # producer stages the target as the "full" tensor
    NcclMover(comm, src=0).stage({name: target})
    comm.rank = 1  # consumer path — the body surface is peer-only, so drive the backend batch
    # layer directly (what manual/NCCL rigs do): collective recv via the injected recv_plan, put
    # to the engine through the same pluggable sink the streaming path uses.
    rollout._weight_backend.apply(None, None, sink=rollout._engine_sink)

    got = vllm_weights._apply_to_model(rollout.model, functools.partial(_param_abs_sum, name=name))
    assert got == 0.0, f"engine param not overwritten by the sink (abs sum {got})"


class _FakeVllmModel:
    """CPU stand-in for the engine-side model: load_weights returns the set of ENGINE names it
    loaded (vLLM's contract), or None when report=False (engines that don't report)."""

    def __init__(self, names, report=True):
        import torch

        self._p = {n: torch.zeros(2) for n in names}
        self._report = report
        self.load_calls = 0

    def named_parameters(self):
        return list(self._p.items())

    def get_parameter(self, name):
        if name not in self._p:
            raise AttributeError(name)
        return self._p[name]

    def load_weights(self, named):
        self.load_calls += 1
        return {n for n, _t in named if n in self._p} if self._report else None


class _FakeLLM:
    def __init__(self, m):
        self._m = m
        self.apply_model_calls = 0  # each call models one cross-process EngineCore RPC (vLLM V1)

    def apply_model(self, fn):
        self.apply_model_calls += 1
        return [fn(self._m)]


@pytest.mark.L0
@pytest.mark.CPU
def test_engine_sink_forwards_shared_storage_views_in_one_inprocess_call() -> None:
    """In-process vLLM receives the original tensor views without RPC compaction or batching."""
    import torch

    from cosmos_framework.posttrain.roles.rollout import vllm as vllm_role

    class _CopyingModel:
        def __init__(self) -> None:
            self.values = {"a": torch.zeros(4), "b": torch.zeros(4)}  # values: [4]
            self.received: list[tuple[str, torch.Tensor]] = []

        def load_weights(self, named: list[tuple[str, torch.Tensor]]) -> set[str]:
            self.received = list(named)
            for name, tensor in named:  # tensor: [4]
                self.values[name].copy_(tensor)  # [4]
            return {name for name, _tensor in named}

    class _InprocessLLM:
        def __init__(self, model: _CopyingModel) -> None:
            self.model = model
            self.calls = 0

        def apply_model(self, fn: Any) -> list[Any]:
            self.calls += 1
            return [fn(self.model)]

    backing = torch.arange(8, dtype=torch.float32)  # [8]
    state = {"a": backing[:4], "b": backing[4:]}  # values: [4]
    engine_model = _CopyingModel()
    rollout = vllm_role.VLLMRollout.__new__(vllm_role.VLLMRollout)
    rollout.model = _InprocessLLM(engine_model)
    rollout._loaded_names = set()

    loaded = rollout._engine_sink(state)

    assert torch.equal(engine_model.values["a"], torch.arange(4, dtype=torch.float32))  # [4]
    assert torch.equal(engine_model.values["b"], torch.arange(4, 8, dtype=torch.float32))  # [4]
    assert loaded == {"a", "b"} and rollout._loaded_names == set()
    assert rollout.model.calls == 1
    assert engine_model.received[0][1] is state["a"]
    assert engine_model.received[1][1] is state["b"]


@pytest.mark.L0
@pytest.mark.CPU
def test_engine_sink_coverage_becomes_unverifiable_when_load_weights_returns_none() -> None:
    import torch

    from cosmos_framework.posttrain.roles.rollout import vllm as vllm_role

    class _NonReportingModel:
        def load_weights(self, named: list[tuple[str, torch.Tensor]]) -> set[str] | None:
            return None

    rollout = vllm_role.VLLMRollout.__new__(vllm_role.VLLMRollout)
    rollout.model = _FakeLLM(_NonReportingModel())
    rollout._loaded_names = set()

    loaded = rollout._engine_sink({"a": torch.zeros(4), "b": torch.zeros(4)})  # values: [4]

    assert rollout.model.apply_model_calls == 1
    assert loaded is None and rollout._loaded_names == set()


@pytest.mark.L0
@pytest.mark.CPU
def test_engine_sink_witness_loader_receives_the_whole_state_exactly_once(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    from cosmos_framework.posttrain.comm.weight_digest import tensor_digest_metadata
    from cosmos_framework.posttrain.comm.weight_transfer import WeightWitnessSpec
    from cosmos_framework.posttrain.roles.rollout import vllm as vllm_role

    expected_digest = tensor_digest_metadata(torch.zeros(4))
    calls: list[tuple[str, list[str]]] = []

    def _load_with_witness(
        _model: Any,
        named: list[tuple[str, torch.Tensor]],
        spec: WeightWitnessSpec,
    ) -> tuple[set[str], dict[str, Any]]:
        assert spec.wire_name == "witness"
        names = [name for name, _tensor in named]
        calls.append(("witness", names))
        return set(names), expected_digest

    monkeypatch.setattr(vllm_role.weight_compat, "_vllm_load_weights_with_witness", _load_with_witness)
    rollout = vllm_role.VLLMRollout.__new__(vllm_role.VLLMRollout)
    rollout.model = _FakeLLM(object())
    rollout._loaded_names = set()
    rollout._weight_witness_spec = WeightWitnessSpec(
        wire_name="witness",
        destination_name="packed",
        shape=(4,),
        dtype="torch.float32",
        layout="packed_qkv",
        packed_part="q",
    )

    loaded = rollout._engine_sink(
        {
            "a": torch.zeros(4),
            "witness": torch.zeros(4),
            "b": torch.zeros(4),
        }
    )

    assert calls == [("witness", ["a", "witness", "b"])]
    assert rollout.model.apply_model_calls == 1
    assert loaded == {"a", "witness", "b"} and rollout._loaded_names == set()
    assert rollout._weight_witness_expected == (expected_digest,)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "expert_name",
    [
        "model.language_model.layers.0.mlp.experts.down_proj",
        "model.language_model.layers.0.mlp.experts.gate_up_proj",
    ],
)
def test_fused_moe_checkpoint_weights_own_storage_across_stream_buckets(expert_name: str) -> None:
    import torch

    from cosmos_framework.posttrain.roles.rollout.vllm_weights import _vllm_load_weights

    dense_name = "model.language_model.layers.0.self_attn.q_proj.weight"
    expert = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)  # [E,I,H]
    dense = torch.arange(16, dtype=torch.float32).reshape(4, 4)  # [Q,H]

    class _RetainingModel:
        def __init__(self) -> None:
            self.received: dict[str, torch.Tensor] = {}

        def load_weights(self, named: list[tuple[str, torch.Tensor]]) -> set[str]:
            self.received = dict(named)
            return set(self.received)

    model = _RetainingModel()
    assert _vllm_load_weights(model, [(expert_name, expert), (dense_name, dense)]) == {expert_name, dense_name}
    retained_expert = model.received[expert_name]  # [E,I,H]
    retained_dense = model.received[dense_name]  # [Q,H]
    assert retained_expert.untyped_storage().data_ptr() != expert.untyped_storage().data_ptr()
    assert retained_dense is dense

    expert.zero_()  # [E,I,H]
    assert torch.count_nonzero(retained_expert) > 0


@pytest.mark.L1
@pytest.mark.CPU
def test_vllm_moe_reload_helpers_scope_and_finalize_with_frontend_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check the installed vLLM reload API; its engine dependencies are outside the required CPU CI image."""
    import torch

    owner_name = "language_model.model.layers.0.mlp.experts"  # vLLM 0.23: the container is the RoutedExperts

    fused_moe = pytest.importorskip("vllm.model_executor.layers.fused_moe")
    reload_module = pytest.importorskip("vllm.model_executor.model_loader.reload")
    from cosmos_framework.posttrain.roles.rollout.vllm_weights import (
        _vllm_begin_moe_reload,
        _vllm_finish_moe_reload,
        _vllm_routed_expert_owner_names,
    )

    class _RoutedExperts:
        pass

    routed = _RoutedExperts()
    dense = object()

    class _Model:
        def modules(self) -> list[object]:
            return [self, dense, routed]

        def named_modules(self) -> list[tuple[str, object]]:
            return [("", self), ("language_model.model.layers.0.mlp.gate", dense), (owner_name, routed)]

    initialized: list[object] = []
    finalized: list[tuple[object, object]] = []
    monkeypatch.setattr(fused_moe, "RoutedExperts", _RoutedExperts)
    monkeypatch.setattr(reload_module, "initialize_layerwise_reload", initialized.append)
    monkeypatch.setattr(
        reload_module, "finalize_layerwise_reload", lambda module, config: finalized.append((module, config))
    )

    model = _Model()
    assert _vllm_routed_expert_owner_names(model) == [owner_name]
    assert _vllm_begin_moe_reload(model) == 1
    assert initialized == [routed]
    assert _vllm_finish_moe_reload(model, torch.bfloat16) == 1
    assert len(finalized) == 1
    assert finalized[0][0] is routed
    assert finalized[0][1].dtype is torch.bfloat16


class _FakeStreamBackend:
    """Minimal streaming backend: bucket i is a prebuilt state dict handed straight to the sink."""

    def __init__(self, buckets, destination_names=None):
        self._buckets = buckets
        self._destination_names = destination_names

    def bucket_count(self):
        return len(self._buckets)

    def destination_names(self) -> tuple[str, ...] | None:
        return self._destination_names

    def apply_bucket(self, model, i, payload, sink=None):
        sink(self._buckets[i])


def _stream_rollout(engine_names, buckets, report=True, destination_names=None):
    import torch

    from cosmos_framework.posttrain.roles.rollout.vllm import VLLMRollout

    obj = VLLMRollout.__new__(VLLMRollout)  # bypass __init__ (needs a real vllm.LLM)
    obj.model = _FakeLLM(_FakeVllmModel(engine_names, report=report))
    obj._weight_backend = _FakeStreamBackend(
        [{n: torch.zeros(2) for n in bk} for bk in buckets],
        destination_names=destination_names,
    )
    return obj


@pytest.mark.L0
@pytest.mark.CPU
def test_mapped_moe_stream_brackets_all_buckets_before_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    import torch

    from cosmos_framework.posttrain.roles.rollout import vllm_weights as weight_compat

    events: list[str] = []

    def _begin(_model: object) -> int:
        events.append("begin")
        return 1

    def _load(_model: object, named: list[tuple[str, object]]) -> set[str]:
        names = {name for name, _tensor in named}
        events.append("load:" + ",".join(sorted(names)))
        return names

    def _finish(_model: object, *, model_dtype: object) -> int:
        assert model_dtype is torch.float32
        events.append("finish")
        return 1

    original_named_shapes = weight_compat._vllm_named_shapes

    def _named_shapes(model: object) -> list:
        events.append("coverage")
        return original_named_shapes(model)

    monkeypatch.setattr(weight_compat, "_vllm_begin_moe_reload", _begin)
    monkeypatch.setattr(weight_compat, "_vllm_load_weights", _load)
    monkeypatch.setattr(weight_compat, "_vllm_finish_moe_reload", _finish)
    monkeypatch.setattr(weight_compat, "_vllm_named_shapes", _named_shapes)
    # The engine owns its routed experts on the `experts` module itself (flat layout); the stream's
    # destinations are live names by the time apply_bucket reads them.
    monkeypatch.setattr(
        weight_compat,
        "_vllm_routed_expert_owner_names",
        lambda _model: ["language_model.model.layers.0.mlp.experts"],
    )

    obj = _stream_rollout(
        ["a", "b"],
        [["a"], ["b"]],
        destination_names=("language_model.model.layers.0.mlp.experts.w2_weight",),
    )
    obj.model.llm_engine = SimpleNamespace(model_config=SimpleNamespace(dtype=torch.float32))
    obj.apply_bucket(None, 0)
    obj.apply_bucket(None, 1)

    assert events == ["begin", "load:a", "load:b", "finish", "coverage"]
    assert obj._moe_reload_active is False


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("mapped", "fails"),
    [
        (("a", "b"), False),
        (("a",), True),
        (("a", "b", "extra"), True),
    ],
)
def test_mapped_destination_preflight_is_exact_and_precedes_load(mapped: tuple[str, ...], fails: bool) -> None:
    obj = _stream_rollout(["a", "b"], [["a", "b"]], destination_names=mapped)
    if fails:
        with pytest.raises(ValueError, match="cover.*exactly"):
            obj.check_plan_coverage()
    else:
        obj.check_plan_coverage()
    assert obj.model._m.load_calls == 0


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(("tp_rank", "tp_world"), [(0, 1), (0, 2), (1, 2)])
def test_qkv_inline_witness_runs_after_existing_vllm_load(
    monkeypatch: pytest.MonkeyPatch,
    tp_rank: int,
    tp_world: int,
) -> None:
    import sys
    from types import ModuleType

    import torch

    from cosmos_framework.posttrain.comm.weight_transfer import WeightWitnessSpec
    from cosmos_framework.posttrain.roles.rollout.vllm_weights import (
        _tensor_digest_metadata,
        _vllm_load_weights_with_witness,
    )

    distributed = ModuleType("vllm.distributed")
    distributed.get_tensor_model_parallel_rank = lambda: tp_rank
    distributed.get_tensor_model_parallel_world_size = lambda: tp_world
    vllm = ModuleType("vllm")
    vllm.distributed = distributed
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.distributed", distributed)

    wire_name = "model.language_model.layers.0.self_attn.q_proj.weight"
    target = "language_model.model.layers.0.self_attn.qkv_proj.weight"
    source_q = torch.arange(16, dtype=torch.float32).reshape(8, 2)  # [Q,H]
    q_rows = source_q.shape[0] // tp_world

    class _PackedModel:
        def __init__(self) -> None:
            self.live_qkv = torch.full((q_rows + 4, 2), -1.0)  # [Q_local+K_local+V_local,H]

        def load_weights(self, named: list) -> set[str]:
            source = dict(named)[wire_name]  # [Q,H]
            shard = source.narrow(0, tp_rank * q_rows, q_rows)  # [Q_local,H]
            self.live_qkv[:q_rows].copy_(shard)  # [Q_local,H]
            return {target}

        def get_parameter(self, name: str) -> torch.Tensor:  # returns [Q_local+K_local+V_local,H]
            if name != target:
                raise AttributeError(name)
            return self.live_qkv

    model = _PackedModel()
    spec = WeightWitnessSpec(wire_name, target, (8, 2), "torch.float32", "packed_qkv", "q")
    loaded, expected = _vllm_load_weights_with_witness(model, [(wire_name, source_q)], spec)

    assert loaded == {target}
    assert expected["shape"] == [q_rows, 2]
    assert expected == _tensor_digest_metadata(source_q.narrow(0, tp_rank * q_rows, q_rows))


@pytest.mark.L0
@pytest.mark.CPU
def test_json_round_tripped_witness_shape_is_stable() -> None:
    import torch

    from cosmos_framework.posttrain.comm.weight_transfer import WeightWitnessSpec
    from cosmos_framework.posttrain.roles.rollout.vllm_weights import _tensor_digest_metadata

    target = "language_model.model.layers.0.self_attn.qkv_proj.weight"
    obj = _stream_rollout([target], [["wire.q_proj.weight"]])
    live_qkv = torch.arange(12, dtype=torch.float32).reshape(6, 2)  # [Q_local+K_local+V_local,H]
    obj.model._m._p[target] = live_qkv
    spec = WeightWitnessSpec(
        wire_name="wire.q_proj.weight",
        destination_name=target,
        shape=(4, 2),
        dtype="torch.float32",
        layout="packed_qkv",
        packed_part="q",
    )
    obj.set_weight_witness_spec(spec)
    initial_q = live_qkv[:2]  # [Q_local,H]
    metadata = _tensor_digest_metadata(initial_q)
    assert metadata["shape"] == [2, 2]
    obj._weight_witness_expected = (json.loads(json.dumps(metadata)),)

    obj.require_weight_witness_verified(spec)


@pytest.mark.L0
@pytest.mark.CPU
def test_later_bucket_q_corruption_fails_final_witness() -> None:
    import torch

    from cosmos_framework.posttrain.comm.weight_transfer import WeightWitnessSpec
    from cosmos_framework.posttrain.roles.rollout.vllm_weights import _tensor_digest_metadata

    target = "language_model.model.layers.0.self_attn.qkv_proj.weight"
    obj = _stream_rollout([target], [["wire.q_proj.weight"]])
    live_qkv = torch.arange(12, dtype=torch.float32).reshape(6, 2)  # [Q_local+K_local+V_local,H]
    obj.model._m._p[target] = live_qkv
    spec = WeightWitnessSpec(
        wire_name="wire.q_proj.weight",
        destination_name=target,
        shape=(4, 2),
        dtype="torch.float32",
        layout="packed_qkv",
        packed_part="q",
    )
    obj.set_weight_witness_spec(spec)
    initial_q = live_qkv[:2]  # [Q_local,H]
    obj._weight_witness_expected = (_tensor_digest_metadata(initial_q),)
    live_qkv[0, 0].add_(1.0)  # [Q_local+K_local+V_local,H]

    with pytest.raises(ValueError, match="changed after inline verification"):
        obj.require_weight_witness_verified(spec)


@pytest.mark.L0
@pytest.mark.CPU
def test_apply_bucket_stream_coverage_ok():
    # Every engine param appears somewhere in the stream ⇒ the final coverage check passes.
    obj = _stream_rollout(["a", "b", "c"], [["a", "b"], ["c"]])
    for i in range(2):
        obj.apply_bucket(None, i)  # payload unused by the fake backend


@pytest.mark.L0
@pytest.mark.CPU
def test_apply_bucket_stream_missing_engine_param_raises():
    # The generic strict check is skipped for engine sinks and load_weights only raises on UNKNOWN
    # names — a plan MISSING an engine param would silently keep initial-checkpoint weights. The
    # per-stream coverage check must fail loud after the LAST bucket.
    obj = _stream_rollout(["a", "b", "c"], [["a"], ["b"]])  # engine's "c" never streamed
    obj.apply_bucket(None, 0)
    with pytest.raises(ValueError, match="never .*loaded|did not cover"):
        obj.apply_bucket(None, 1)


@pytest.mark.L0
@pytest.mark.CPU
def test_apply_bucket_stream_coverage_resets_per_stream():
    # Names loaded by an EARLIER stream must not mask a gap in the current one (reset on bucket 0).
    import torch

    obj = _stream_rollout(["a", "b"], [["a", "b"]])
    obj.apply_bucket(None, 0)  # stream 1: full coverage
    obj._weight_backend = _FakeStreamBackend([{"a": torch.zeros(2)}])  # stream 2: "b" missing
    with pytest.raises(ValueError, match="never .*loaded|did not cover"):
        obj.apply_bucket(None, 0)


@pytest.mark.L0
@pytest.mark.CPU
def test_apply_bucket_coverage_engine_names_cached_across_streams():
    # The engine's param set is fixed at construction, so the coverage check must fetch it ONCE
    # per body — under vLLM V1 that fetch is a cross-process apply_model RPC, and paying it at the
    # tail of every sync stream would accumulate in a per-step GRPO loop.
    obj = _stream_rollout(["a"], [["a"]])
    llm = obj.model
    obj.apply_bucket(None, 0)  # stream 1: one load RPC + one names RPC (cache fill)
    assert llm.apply_model_calls == 2
    obj.apply_bucket(None, 0)  # stream 2: one load RPC only — names come from the cache
    assert llm.apply_model_calls == 3


@pytest.mark.L0
@pytest.mark.CPU
def test_apply_bucket_stream_nonreporting_engine_skips_check():
    # An engine whose load_weights returns None cannot be verified — the check degrades to a no-op
    # rather than raising spuriously.
    obj = _stream_rollout(["a", "b", "c"], [["a"]], report=False)
    obj.apply_bucket(None, 0)  # engine params b, c never loaded, but coverage is unverifiable


@pytest.mark.L0
@pytest.mark.CPU
def test_apply_weights_shared_mover_is_noop():
    # weight_sync=True with a single colocation resolves to SharedMover (same-instance). The peer
    # form must short-circuit to a stamp-only no-op BEFORE any peer traffic (a real PeerClient here
    # would self-call the busy actor) and never touch the engine.
    from cosmos_framework.posttrain.comm.transfer.movers import SharedMover
    from cosmos_framework.posttrain.comm.weight_transfer import WeightSync
    from cosmos_framework.posttrain.roles.rollout.vllm import VLLMRollout

    class _ExplodingPeer:
        def __getattr__(self, name):
            raise AssertionError(f"peer RPC {name!r} on a shared edge (would self-deadlock)")

    obj = VLLMRollout.__new__(VLLMRollout)
    obj._weight_version = 2
    obj._weight_backend = WeightSync(SharedMover())
    obj.apply_weights(_ExplodingPeer())
    assert obj._weight_version == 3  # local stamp; no engine touch, no peer RPC
