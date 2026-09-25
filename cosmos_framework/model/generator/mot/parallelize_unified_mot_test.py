# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Unit tests for the ``mark_unbacked`` pack-marking helpers in parallelize_unified_mot.

Coverage focuses on the correctness guard added around ``torch.compile`` recompiles caused by
sequence packing's per-step-varying pack tensors (``causal_seq``, ``sample_offsets``, etc.):

- ``_mark_pack_unbacked`` must call ``mark_unbacked(tensor, 0)`` -- without ``strict`` -- for
  every dynamic-length key with dim-0 size > 1. See
  ``test_marks_normal_sized_dims_non_strict`` for why ``strict=True`` defeats the marking.
- It must NOT mark a dim-0 size of 0 or 1: ``mark_unbacked`` hard-codes the assumption that the
  size is "always not equal to zero or one", and ``sequence_packing/runtime.py`` documents real
  batches (AR no-text packs) whose ``causal_seq`` is genuinely length 0. Marking those would risk
  silently wrong compiled output instead of a guard failure.
- Non-tensor values, 0-dim tensors, and keys outside ``_PACK_DYNAMIC_LEN_KEYS`` must be ignored.
- ``_wrap_forward_with_unbacked_pack`` -- not ``_mark_pack_unbacked`` -- must mark nothing when
  grad is disabled. Inference has a fixed pack layout and wants the specialization, and an unbacked
  length that reaches a slice offset gives the resulting view a data-dependent ``storage_offset``
  that Inductor's post-grad ``add(mm, residual) -> addmm`` pattern cannot guard on. The condition
  lives on the wrapper because it describes the call, not the pack, and because grad state flips
  within a process -- the periodic sampling callbacks run these blocks under ``no_grad``. See
  ``test_delegates_unchanged_without_grad``.
- ``_wrap_forward_with_unbacked_pack`` must mark both ``input`` and every element of
  ``packed_position_embeddings`` before delegating to the wrapped forward, and must pass through
  its return value unchanged -- including when grad is off and there is nothing to mark.
- It must mark nothing when the attention mask carries a FlexAttention block mask. That path's
  fused key stream gives Inductor's layout pass a stride it cannot sort without a ShapeEnv its
  callers do not thread through. See ``test_marks_nothing_when_the_mask_carries_a_flex_block_mask``.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch.distributed.fsdp import CPUOffloadPolicy, OffloadPolicy

from cosmos_framework.configs.base.defaults.compile import CompileConfig
from cosmos_framework.model.generator.mot.parallelize_unified_mot import (
    _PACK_DYNAMIC_LEN_KEYS,
    _mark_pack_unbacked,
    _wrap_forward_with_unbacked_pack,
    apply_compile,
    apply_fsdp,
    materialize_non_offloaded_state,
    parallelize_unified_mot,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _pack_with_causal_len(causal_len: int) -> dict:
    """Build a minimal SequencePack-shaped dict with ``causal_seq`` at the given length.

    Every other dynamic-length key gets an unrelated, always-safe length so the test can isolate
    the behaviour to the one key under test.
    """
    pack = {key: torch.zeros(5, 3) for key in _PACK_DYNAMIC_LEN_KEYS}
    pack["causal_seq"] = torch.zeros(causal_len, 3)
    return pack


class TestMarkPackUnbacked:
    @pytest.mark.parametrize("causal_len", [0, 1])
    def test_skips_zero_and_one_sized_dims(self, causal_len: int) -> None:
        pack = _pack_with_causal_len(causal_len)
        with patch("cosmos_framework.model.generator.mot.parallelize_unified_mot.mark_unbacked") as mock_mark:
            _mark_pack_unbacked(pack)

        # Identity (`is`), not `==`/`in`: tensor equality is elementwise and raises on multi-
        # element tensors, so list/set membership checks must not be trusted here.
        marked_ids = [id(c.args[0]) for c in mock_mark.call_args_list]
        assert id(pack["causal_seq"]) not in marked_ids, (
            f"a dim-0 size of {causal_len} was marked unbacked, violating mark_unbacked's "
            "'never zero or one' assumption"
        )

    def test_marks_normal_sized_dims_non_strict(self) -> None:
        pack = _pack_with_causal_len(45)
        with patch("cosmos_framework.model.generator.mot.parallelize_unified_mot.mark_unbacked") as mock_mark:
            _mark_pack_unbacked(pack)

        # Every dynamic-length key (all length-5 here, plus causal_seq at 45) must be marked, each
        # with dim 0 and *without* strict.
        #
        # strict is not a stronger form of the marking: mark_unbacked returns early under it,
        # recording the index in ``_dynamo_strict_unbacked_indices`` and never in
        # ``_dynamo_unbacked_indices``, which is the set that selects DimDynamic.UNBACKED. The dim
        # then falls through to ordinary automatic-dynamic and gets a *backed* symbol carrying a
        # hint -- exactly the thing being marked against, since a hint is what lets Inductor answer
        # size questions and install the guards that recompile. Verified on torch 2.9 and 2.13:
        # ``mark_unbacked(x, 0)`` yields u0, ``strict=True`` yields s77.
        assert mock_mark.call_count == len(_PACK_DYNAMIC_LEN_KEYS)
        calls_by_tensor_id = {id(c.args[0]): c for c in mock_mark.call_args_list}
        for key in _PACK_DYNAMIC_LEN_KEYS:
            c = calls_by_tensor_id[id(pack[key])]
            assert c.args[1] == 0
            assert c.kwargs == {}

    def test_ignores_non_tensor_and_scalar_and_unknown_keys(self) -> None:
        pack = _pack_with_causal_len(45)
        pack["max_num_tokens"] = 128  # non-tensor, present in real SequencePacks
        pack["is_sharded"] = False
        pack["_unrelated_scalar_tensor"] = torch.tensor(3.0)  # dim() == 0, not in the key list
        with patch("cosmos_framework.model.generator.mot.parallelize_unified_mot.mark_unbacked") as mock_mark:
            _mark_pack_unbacked(pack)

        marked_ids = {id(c.args[0]) for c in mock_mark.call_args_list}
        assert id(pack["max_num_tokens"]) not in marked_ids
        assert id(pack["is_sharded"]) not in marked_ids
        assert id(pack["_unrelated_scalar_tensor"]) not in marked_ids

    def test_missing_keys_do_not_raise(self) -> None:
        # A pack that only has a subset of the dynamic-length keys (e.g. AR no-text packs,
        # which carry full splits only) must not crash on the missing ones.
        _mark_pack_unbacked({"full_only_seq": torch.zeros(10, 3)})

    def test_marks_regardless_of_grad(self) -> None:
        """Whether to mark on a given call belongs to the wrapper, not here.

        Grad state gates the marking, but it is a property of the call rather than of the pack, so
        the test for it lives with the wrapper that owns the decision
        (``TestWrapForwardWithUnbackedPack.test_delegates_unchanged_without_grad``). This one pins
        the complementary half: called directly, this marks whatever it is handed.
        """
        pack = _pack_with_causal_len(45)
        with patch("cosmos_framework.model.generator.mot.parallelize_unified_mot.mark_unbacked") as mock_mark:
            with torch.no_grad():
                _mark_pack_unbacked(pack)

        assert mock_mark.call_count == len(_PACK_DYNAMIC_LEN_KEYS)


class TestWrapForwardWithUnbackedPack:
    def test_marks_input_and_position_embeddings_then_delegates(self) -> None:
        input_pack = _pack_with_causal_len(45)
        cos_pack = _pack_with_causal_len(45)
        sin_pack = _pack_with_causal_len(45)
        sentinel_output = object()
        inner_forward = lambda *args, **kwargs: sentinel_output  # noqa: E731

        wrapped = _wrap_forward_with_unbacked_pack(inner_forward)

        with patch("cosmos_framework.model.generator.mot.parallelize_unified_mot.mark_unbacked") as mock_mark:
            result = wrapped(input_pack, "attn_mask_sentinel", (cos_pack, sin_pack), "extra_arg")

        assert result is sentinel_output
        marked_ids = {id(c.args[0]) for c in mock_mark.call_args_list}
        for pack in (input_pack, cos_pack, sin_pack):
            for key in _PACK_DYNAMIC_LEN_KEYS:
                assert id(pack[key]) in marked_ids

    def test_marks_nothing_when_the_mask_carries_a_flex_block_mask(self) -> None:
        """The FlexAttention path cannot be compiled with unbacked lengths, so it opts out.

        Its fused ``[UND | GEN]`` stream has an outer stride of ``heads * head_dim * (u_und +
        u_gen)``, and Inductor's layout pass sorts strides with a plain ``sorted()`` whenever its
        caller omits the ShapeEnv, which raises GuardOnDataDependentSymNode from inside an Inductor
        pass where this repo cannot annotate it. Skipping costs the recompiles on multiview
        configurations and nothing else.
        """
        input_pack = _pack_with_causal_len(45)
        pos = (_pack_with_causal_len(45), _pack_with_causal_len(45))
        attention_mask = SimpleNamespace(flex_block_mask=object())
        sentinel = object()
        wrapped = _wrap_forward_with_unbacked_pack(lambda *a, **k: sentinel)

        with patch("cosmos_framework.model.generator.mot.parallelize_unified_mot.mark_unbacked") as mock_mark:
            result = wrapped(input_pack, attention_mask, pos)

        assert result is sentinel
        assert mock_mark.call_count == 0

    def test_marks_when_the_mask_has_no_flex_block_mask(self) -> None:
        # The complement: a SplitInfo whose flex_block_mask is None is the non-multiview path,
        # which is the case the marking exists for and must not be suppressed.
        input_pack = _pack_with_causal_len(45)
        pos = (_pack_with_causal_len(45), _pack_with_causal_len(45))
        attention_mask = SimpleNamespace(flex_block_mask=None)
        wrapped = _wrap_forward_with_unbacked_pack(lambda *a, **k: None)

        with patch("cosmos_framework.model.generator.mot.parallelize_unified_mot.mark_unbacked") as mock_mark:
            wrapped(input_pack, attention_mask, pos)

        marked_ids = {id(c.args[0]) for c in mock_mark.call_args_list}
        for pack in (input_pack, *pos):
            for key in _PACK_DYNAMIC_LEN_KEYS:
                assert id(pack[key]) in marked_ids

    def test_delegates_unchanged_without_grad(self) -> None:
        # Skipping the marking must not disturb the call itself: the wrapped forward still runs
        # with the same arguments and its return value still passes straight through.
        input_pack = _pack_with_causal_len(45)
        cos_pack = _pack_with_causal_len(45)
        sin_pack = _pack_with_causal_len(45)
        sentinel_output = object()
        seen = []

        def inner_forward(*args, **kwargs):
            seen.append((args, kwargs))
            return sentinel_output

        wrapped = _wrap_forward_with_unbacked_pack(inner_forward)

        with patch("cosmos_framework.model.generator.mot.parallelize_unified_mot.mark_unbacked") as mock_mark:
            with torch.no_grad():
                result = wrapped(input_pack, "attn_mask_sentinel", (cos_pack, sin_pack), "extra_arg")

        assert result is sentinel_output
        assert mock_mark.call_count == 0
        # Identity (`is`), not `==`: these arguments hold tensors, whose equality is elementwise.
        assert len(seen) == 1
        args, kwargs = seen[0]
        assert kwargs == {}
        assert args[0] is input_pack
        assert args[1] == "attn_mask_sentinel"
        assert args[2][0] is cos_pack and args[2][1] is sin_pack
        assert args[3] == "extra_arg"


class TestMarkUnbackedFlag:
    """``CompileConfig.mark_unbacked`` has to reach ``apply_compile``'s wrapping decision.

    The flag exists so a run that hits a ``Could not guard on data-dependent expression`` the
    stated invariants do not cover can fall back to recompiling instead of failing. That is only
    an escape hatch if turning it off actually leaves the compiled block unwrapped, so both
    directions are pinned here.
    """

    @staticmethod
    def _model_with_one_layer() -> torch.nn.Module:
        block = torch.nn.Identity()
        layers = torch.nn.Module()
        layers.register_module("0", block)
        inner = torch.nn.Module()
        inner.layers = layers
        model = torch.nn.Module()
        model.model = inner
        return model

    @pytest.mark.parametrize("mark_unbacked", [True, False])
    def test_flag_controls_whether_the_block_forward_is_wrapped(self, mark_unbacked: bool) -> None:
        model = self._model_with_one_layer()
        compiled_forward = object()

        def fake_compile(block, **kwargs):
            block.forward = compiled_forward
            return block

        with patch("cosmos_framework.model.generator.mot.parallelize_unified_mot.torch.compile", fake_compile):
            apply_compile(model, CompileConfig(enabled=True, mark_unbacked=mark_unbacked))

        forward = model.model.layers.get_submodule("0").forward
        if mark_unbacked:
            # Wrapped: the marking frame sits between the caller and the compiled forward.
            assert forward is not compiled_forward
            assert callable(forward)
        else:
            # Unwrapped: the compiled forward is called directly, as it was before the marking.
            assert forward is compiled_forward

    def test_defaults_to_not_marking(self) -> None:
        # Off by default: the marking is incompatible with the DTensor all-to-all every
        # context-parallel config goes through, so it is opted into per configuration after
        # verifying that configuration compiles under it. See CompileConfig.mark_unbacked.
        assert CompileConfig().mark_unbacked is False


class TestFSDPCPUOffload:
    @staticmethod
    def _model_with_two_layers() -> torch.nn.Module:
        layers = torch.nn.Module()
        layers.register_module("0", torch.nn.Linear(4, 4))
        layers.register_module("1", torch.nn.Linear(4, 4))
        inner = torch.nn.Module()
        inner.layers = layers
        model = torch.nn.Module()
        model.model = inner
        return model

    @pytest.mark.parametrize(
        ("cpu_offload", "expected_policy"),
        [(False, OffloadPolicy), (True, CPUOffloadPolicy)],
    )
    def test_wraps_every_complete_block_with_selected_offload_policy(
        self,
        cpu_offload: bool,
        expected_policy: type[OffloadPolicy],
    ) -> None:
        model = self._model_with_two_layers()
        parallel_dims = SimpleNamespace(fsdp_cpu_offload=cpu_offload)
        mesh = object()

        with (
            patch(
                "cosmos_framework.model.generator.mot.parallelize_unified_mot.fsdp_mesh",
                return_value=mesh,
            ),
            patch("cosmos_framework.model.generator.mot.parallelize_unified_mot.fully_shard") as mock_shard,
            patch(
                "cosmos_framework.model.generator.mot.parallelize_unified_mot.register_fsdp_forward_method"
            ) as mock_register,
        ):
            apply_fsdp(model, parallel_dims)

        blocks = list(model.model.layers.children())
        assert mock_shard.call_count == len(blocks)
        assert mock_register.call_count == len(blocks)
        for block, call in zip(blocks, mock_shard.call_args_list, strict=True):
            assert call.kwargs["module"] is block
            assert call.kwargs["mesh"] is mesh
            assert isinstance(call.kwargs["offload_policy"], expected_policy)
            assert call.kwargs["reshard_after_forward"] is True
            assert getattr(block, "_fsdp_cpu_offloaded") is cpu_offload

    def test_materializes_meta_blocks_before_fsdp_wrapping(self) -> None:
        with torch.device("meta"):
            model = self._model_with_two_layers()
        first_block = model.model.layers.get_submodule("0")
        first_block.register_buffer("positions", torch.arange(3))  # [3]
        parallel_dims = SimpleNamespace(fsdp_cpu_offload=True)
        mesh = SimpleNamespace(device_type="cpu")
        blocks = list(model.model.layers.children())
        wrapped_blocks: list[torch.nn.Module] = []

        def record_wrap_order(*, module: torch.nn.Module, **_: object) -> None:
            block_index = len(wrapped_blocks)
            assert module is blocks[block_index]
            assert all(not parameter.is_meta for parameter in module.parameters())
            for future_block in blocks[block_index + 1 :]:
                assert all(parameter.is_meta for parameter in future_block.parameters())
            wrapped_blocks.append(module)

        with (
            patch(
                "cosmos_framework.model.generator.mot.parallelize_unified_mot.fsdp_mesh",
                return_value=mesh,
            ),
            patch(
                "cosmos_framework.model.generator.mot.parallelize_unified_mot.fully_shard",
                side_effect=record_wrap_order,
            ),
            patch("cosmos_framework.model.generator.mot.parallelize_unified_mot.register_fsdp_forward_method"),
        ):
            apply_fsdp(model, parallel_dims)

        assert wrapped_blocks == blocks
        for block in blocks:
            assert all(parameter.device.type == "cpu" for parameter in block.parameters())
        assert torch.equal(first_block.positions, torch.arange(3))

    def test_materializes_remaining_state_without_revisiting_offloaded_blocks(self) -> None:
        root = torch.nn.Module()
        root.register_parameter("root_weight", torch.nn.Parameter(torch.empty(2, 2, device="meta")))
        block = torch.nn.Module()
        block.register_parameter("weight", torch.nn.Parameter(torch.ones(3, 2)))  # [3,2]
        block.register_buffer("positions", torch.arange(3))  # [3]
        setattr(block, "_fsdp_cpu_offloaded", True)
        root.register_module("block", block)

        materialize_non_offloaded_state(root, device="cpu")

        assert root.root_weight.device.type == "cpu"
        assert block.weight.device.type == "cpu"
        assert torch.equal(block.weight, torch.ones(3, 2))
        assert torch.equal(block.positions, torch.arange(3))

    def test_compile_runs_before_fsdp_wrapping(self) -> None:
        calls: list[str] = []
        model = torch.nn.Module()
        compile_config = SimpleNamespace(enabled=True)
        parallel_dims = SimpleNamespace(cp_enabled=False, dp_enabled=True)

        with (
            patch(
                "cosmos_framework.model.generator.mot.parallelize_unified_mot.apply_ac",
                side_effect=lambda *_: calls.append("ac"),
            ),
            patch(
                "cosmos_framework.model.generator.mot.parallelize_unified_mot.apply_compile",
                side_effect=lambda *_: calls.append("compile"),
            ),
            patch(
                "cosmos_framework.model.generator.mot.parallelize_unified_mot.apply_fsdp",
                side_effect=lambda *_: calls.append("fsdp"),
            ),
        ):
            parallelize_unified_mot(model, parallel_dims, compile_config, SimpleNamespace())

        assert calls == ["ac", "compile", "fsdp"]


class TestCudaGraphScope:
    """``cuda_graph_scope`` decides whether inductor's CUDA-graph trees wrap each block.

    With scope "forward" the AR loop captures one explicit graph around the whole forward, and a
    capture cannot nest the per-block graph trees, so the blocks must compile with ``mode=None``.
    """

    @staticmethod
    def _model_with_one_layer() -> torch.nn.Module:
        block = torch.nn.Identity()
        layers = torch.nn.Module()
        layers.register_module("0", block)
        inner = torch.nn.Module()
        inner.layers = layers
        model = torch.nn.Module()
        model.model = inner
        return model

    @pytest.mark.parametrize(
        ("use_cuda_graphs", "scope", "expected_mode"),
        [(True, "block", "reduce-overhead"), (True, "forward", None), (False, "block", None), (False, "forward", None)],
    )
    def test_block_scope_alone_uses_reduce_overhead(self, use_cuda_graphs: bool, scope: str, expected_mode) -> None:
        model = self._model_with_one_layer()
        seen_modes: list[object] = []

        def fake_compile(block, **kwargs):
            seen_modes.append(kwargs["mode"])
            return block

        with patch("cosmos_framework.model.generator.mot.parallelize_unified_mot.torch.compile", fake_compile):
            apply_compile(model, CompileConfig(enabled=True, use_cuda_graphs=use_cuda_graphs, cuda_graph_scope=scope))

        assert seen_modes == [expected_mode]

    @pytest.mark.parametrize("scope", ["block", "forward"])
    def test_forward_scope_forces_static_shapes(self, scope: str) -> None:
        model = self._model_with_one_layer()
        seen_dynamic: list[object] = []

        def fake_compile(block, **kwargs):
            seen_dynamic.append(kwargs["dynamic"])
            return block

        with patch("cosmos_framework.model.generator.mot.parallelize_unified_mot.torch.compile", fake_compile):
            apply_compile(
                model, CompileConfig(enabled=True, use_cuda_graphs=True, cuda_graph_scope=scope, compile_dynamic=True)
            )

        # Symbolic shapes stage Python floats through pinned host memory inside the compiled wrapper,
        # which a whole-forward capture would replay from freed memory; forward scope pins static shapes.
        assert seen_dynamic == [scope == "block"]

    def test_scope_is_validated_and_defaults_to_block(self) -> None:
        assert CompileConfig().cuda_graph_scope == "block"
        with pytest.raises(ValueError):
            CompileConfig(cuda_graph_scope="whole")  # type: ignore[arg-type]
